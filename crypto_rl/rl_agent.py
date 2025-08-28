import os
import numpy as np
from shared.dqn_agent import DQNAgent, DQNConfig


class CryptoDQNAgent:
    """DQN-based agent for cryptographic algorithm selection (4 discrete actions)."""

    def __init__(self, state_dim: int = 4, action_dim: int = 4):
        cfg = DQNConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=0.99,
            lr=1e-3,
            batch_size=64,
            buffer_size=100_000,
            min_buffer=1000,
            train_freq=1,
            target_sync_freq=1000,
            double_dqn=True,
            dueling=True,
            eps_start=1.0,
            eps_end=0.05,
            eps_decay_steps=50_000,
            prefer_cuda=True,
        )
        self.agent = DQNAgent(cfg)

    @property
    def epsilon(self) -> float:
        return self.agent.epsilon

    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        return int(self.agent.choose_action(state, training=training))

    def remember(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, done: bool) -> None:
        self.agent.remember(s, a, r, ns, done)

    def learn(self) -> None:
        self.agent.learn()

    def save_policy(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.agent.save_policy(path)

    def load_policy(self, path: str) -> bool:
        return self.agent.load_policy(path)
