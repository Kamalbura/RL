"""
Strategic (GCS-side) Crypto RL agent and minimal environment.

Design goals
- Minimal state: [Threat, AvgFleetBattery, MissionPhase]
- Actions: 4 crypto algorithms from config.crypto_config.CRYPTO_ALGORITHMS
- Reward: Balance security strength vs. battery impact and latency under mission pressure

This module integrates with the existing crypto_rl package and config.
It reuses the generic QLearningAgent already present in crypto_rl.rl_agent.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

# Make sure we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from config.crypto_config import CRYPTO_ALGORITHMS, CRYPTO_RL  # type: ignore
except Exception:
    # Fallback loader to avoid top-level config.py name clash
    import runpy
    cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'crypto_config.py'))
    data = runpy.run_path(cfg_path)
    CRYPTO_ALGORITHMS = data.get('CRYPTO_ALGORITHMS', {})
    CRYPTO_RL = data.get('CRYPTO_RL', {})
from crypto_rl.rl_agent import QLearningAgent


class StrategicCryptoEnv:
    """
    Lightweight environment for GCS-side crypto selection.

    State: [Threat(0-2), AvgBattery(0-2), MissionPhase(0-3)]
      - Threat: 0=LOW, 1=ELEVATED, 2=CRITICAL
      - AvgBattery: 0=CRITICAL, 1=DEGRADING, 2=HEALTHY
      - MissionPhase: 0=IDLE, 1=TRANSIT, 2=TASK, 3=CRITICAL_TASK

    Action: index into CRYPTO_ALGORITHMS (0..3) mapping to algorithms.

    Reward: security_reward(threat) - power_penalty(battery) - latency_penalty(mission)
    """

    def __init__(self):
        self.state_dims = [3, 3, 4]
        self.action_dim = 4
        self.max_steps = 200
        self.steps = 0
        self._rng = np.random.default_rng()
        self.reset()

    def reset(self) -> List[int]:
        self.steps = 0
        # Start reasonably safe defaults
        self.threat = int(self._rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15]))
        self.battery = 2  # HEALTHY
        self.mission = int(self._rng.choice([0, 1, 2, 3], p=[0.35, 0.3, 0.25, 0.1]))
        return [self.threat, self.battery, self.mission]

    def _security_reward(self, algo_security: float) -> float:
        # Weight security more when threat increases
        weights = [0.6, 1.0, 1.6]
        return algo_security * weights[self.threat]

    def _power_penalty(self, power_mult: float) -> float:
        # Penalize power more when battery is low
        weights = [2.0, 1.0, 0.5]
        return power_mult * 10.0 * weights[self.battery]

    def _latency_penalty(self, latency_ms: float) -> float:
        # Penalize latency more in critical missions
        weights = [0.25, 0.5, 1.0, 1.5]
        return (latency_ms / 10.0) * weights[self.mission]

    def step(self, action: int) -> Tuple[List[int], float, bool, Dict]:
        assert 0 <= action < self.action_dim, f"Invalid action {action}"
        algo = CRYPTO_ALGORITHMS[action]
        security = float(algo.get("security_rating", 5))
        power_mult = float(algo.get("power_multiplier", 1.0))
        latency_ms = float(algo.get("latency_ms", 5.0))

        reward = self._security_reward(security)
        reward -= self._power_penalty(power_mult)
        reward -= self._latency_penalty(latency_ms)

        # Simple drift: threat and mission can change; battery degrades slowly if power heavy
        if np.random.random() < 0.1:
            self.threat = int(np.clip(self.threat + self._rng.choice([-1, 0, 1], p=[0.15, 0.7, 0.15]), 0, 2))
        if np.random.random() < 0.12:
            self.mission = int(np.clip(self.mission + self._rng.choice([-1, 0, 1], p=[0.25, 0.5, 0.25]), 0, 3))

        # Battery trend
        drain = 0.02 * power_mult  # abstract units per step
        if drain > 0.03 and np.random.random() < 0.7:
            self.battery = max(0, self.battery - 1) if np.random.random() < 0.3 else self.battery

        self.steps += 1
        done = self.steps >= self.max_steps
        state = [self.threat, self.battery, self.mission]
        info = {"algorithm": algo.get("name", str(action)), "security_rating": security,
                "power_multiplier": power_mult, "latency_ms": latency_ms, "reward": reward}
        return state, float(reward), done, info


class StrategicCryptoAgent:
    """
    Thin wrapper using the shared QLearningAgent with a fixed strategic state/action space.
    """

    def __init__(self,
                 learning_rate: float | None = None,
                 discount_factor: float | None = None,
                 exploration_rate: float | None = None,
                 exploration_decay: float | None = None,
                 min_exploration_rate: float | None = None):
        self.state_dims = [3, 3, 4]
        self.action_dim = 4
        self.agent = QLearningAgent(
            state_dims=self.state_dims,
            action_dim=self.action_dim,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_rate=exploration_rate,
            exploration_decay=exploration_decay,
            min_exploration_rate=min_exploration_rate,
        )

    def choose_action(self, state: List[int], training: bool = True) -> int:
        return int(self.agent.choose_action(state, training=training))

    def learn(self, state: List[int], action: int, reward: float, next_state: List[int], done: bool):
        self.agent.learn(state, action, reward, next_state, done)

    def save_policy(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.agent.save_policy(path)

    def load_policy(self, path: str) -> bool:
        return self.agent.load_policy(path)


def train_strategic_agent(episodes: int = 5000, eval_every: int = 250, out_dir: str = "output") -> StrategicCryptoAgent:
    env = StrategicCryptoEnv()
    agent = StrategicCryptoAgent(
        learning_rate=CRYPTO_RL.get("LEARNING_RATE", 0.1),
        discount_factor=CRYPTO_RL.get("DISCOUNT_FACTOR", 0.99),
        exploration_rate=CRYPTO_RL.get("EXPLORATION_RATE", 1.0),
        exploration_decay=CRYPTO_RL.get("EXPLORATION_DECAY", 0.9995),
        min_exploration_rate=CRYPTO_RL.get("MIN_EXPLORATION_RATE", 0.01),
    )

    rewards = []
    eval_scores = []
    for ep in range(episodes):
        s = env.reset()
        done = False
        total = 0.0
        while not done:
            a = agent.choose_action(s, training=True)
            ns, r, done, _ = env.step(a)
            agent.learn(s, a, r, ns, done)
            s = ns
            total += r
        rewards.append(total)
        if (ep + 1) % eval_every == 0:
            eval_scores.append(evaluate(agent, env, 20))

    os.makedirs(out_dir, exist_ok=True)
    agent.save_policy(os.path.join(out_dir, "strategic_crypto_q_table.npy"))
    return agent


def evaluate(agent: StrategicCryptoAgent, env: StrategicCryptoEnv, episodes: int = 10) -> float:
    scores: List[float] = []
    for _ in range(episodes):
        s = env.reset()
        done = False
        total = 0.0
        while not done:
            a = agent.choose_action(s, training=False)
            ns, r, done, _ = env.step(a)
            s = ns
            total += r
        scores.append(total)
    return float(np.mean(scores))


__all__ = [
    "StrategicCryptoEnv",
    "StrategicCryptoAgent",
    "train_strategic_agent",
    "evaluate",
]
