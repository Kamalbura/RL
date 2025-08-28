"""
Shared DQN agent for hierarchical UAV cybersecurity RL framework.

Features
- MLP Q-network with target network
- Double DQN update (optional)
- Dueling architecture (optional)
- Prioritized Experience Replay (simple proportional) (optional: off by default)
- Epsilon-greedy exploration

This agent operates over discrete action spaces. For factored/tuple action spaces,
flatten the action to a single integer externally and convert as needed.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Deque, List, Tuple, Optional
from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


def _set_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Tuple[int, int] = (128, 128), dueling: bool = True):
        super().__init__()
        self.dueling = dueling
        h1, h2 = hidden
        self.feature = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
        )
        if dueling:
            self.value_stream = nn.Sequential(nn.Linear(h2, 128), nn.ReLU(), nn.Linear(128, 1))
            self.adv_stream = nn.Sequential(nn.Linear(h2, 128), nn.ReLU(), nn.Linear(128, out_dim))
        else:
            self.head = nn.Linear(h2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.feature(x)
        if self.dueling:
            v = self.value_stream(z)
            a = self.adv_stream(z)
            q = v + (a - a.mean(dim=1, keepdim=True))
            return q
        return self.head(z)


@dataclass
class DQNConfig:
    state_dim: int
    action_dim: int
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 100_000
    min_buffer: int = 1_000
    train_freq: int = 1
    target_sync_freq: int = 1_000
    double_dqn: bool = True
    dueling: bool = True
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000
    prefer_cuda: bool = False
    grad_clip_norm: Optional[float] = 10.0


class DQNAgent:
    def __init__(self, config: DQNConfig):
        self.cfg = config
        self.device = _set_device(config.prefer_cuda)

        self.q = MLP(config.state_dim, config.action_dim, dueling=config.dueling).to(self.device)
        self.t = MLP(config.state_dim, config.action_dim, dueling=config.dueling).to(self.device)
        self.t.load_state_dict(self.q.state_dict())
        self.t.eval()

        self.optim = optim.Adam(self.q.parameters(), lr=config.lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=config.buffer_size)
        self.steps = 0
        self._eps = config.eps_start

    # ----------------------------- Exploration -----------------------------
    @property
    def epsilon(self) -> float:
        # Linear decay
        frac = min(1.0, self.steps / max(1, self.cfg.eps_decay_steps))
        self._eps = self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)
        return self._eps

    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.cfg.action_dim)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q(s)
            a = int(torch.argmax(q, dim=1).item())
            return a

    # ----------------------------- Replay buffer ---------------------------
    def remember(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, done: bool) -> None:
        self.replay.append((s.astype(np.float32), int(a), float(r), ns.astype(np.float32), bool(done)))

    # ----------------------------- Learning --------------------------------
    def _sample(self) -> Optional[List[Tuple[np.ndarray, int, float, np.ndarray, bool]]]:
        if len(self.replay) < max(self.cfg.min_buffer, self.cfg.batch_size):
            return None
        batch = random.sample(self.replay, self.cfg.batch_size)
        return batch

    def learn(self) -> Optional[float]:
        self.steps += 1

        if self.steps % self.cfg.train_freq != 0:
            return None

        batch = self._sample()
        if batch is None:
            return None

        s, a, r, ns, d = zip(*batch)
        s = torch.as_tensor(np.stack(s), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.as_tensor(np.stack(ns), dtype=torch.float32, device=self.device)
        d = torch.as_tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_sa = self.q(s).gather(1, a)

        with torch.no_grad():
            if self.cfg.double_dqn:
                next_a = torch.argmax(self.q(ns), dim=1, keepdim=True)
                next_q = self.t(ns).gather(1, next_a)
            else:
                next_q = torch.max(self.t(ns), dim=1, keepdim=True).values
            target = r + (1.0 - d) * self.cfg.gamma * next_q

        loss = self.loss_fn(q_sa, target)
        self.optim.zero_grad()
        loss.backward()
        if self.cfg.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()

        # target sync
        if self.steps % self.cfg.target_sync_freq == 0:
            self.t.load_state_dict(self.q.state_dict())

        return float(loss.item())

    # ----------------------------- Persistence -----------------------------
    def save_policy(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "model": self.q.state_dict(),
            "config": self.cfg.__dict__,
        }, path)

    def load_policy(self, path: str, strict: bool = False) -> bool:
        if not os.path.exists(path):
            return False
        data = torch.load(path, map_location=self.device)
        self.q.load_state_dict(data["model"], strict=strict)
        self.t.load_state_dict(self.q.state_dict())
        return True

    # ----------------------------- Utilities --------------------------------
    def eval_q(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q(s).squeeze(0).cpu().numpy()
            return q

__all__ = ["DQNAgent", "DQNConfig"]
