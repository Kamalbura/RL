"""
Tactical agent using shared DQNAgent with flattened tuple action space.
"""

from typing import Any, Dict, Tuple
import os
import numpy as np

from shared.dqn_agent import DQNAgent, DQNConfig


def flatten_action(model_choice: int, freq_choice: int) -> int:
	"""Map (model_choice,freq_choice) -> int in [0, 11]."""
	return model_choice * 4 + freq_choice


def unflatten_action(action: int) -> Tuple[int, int]:
	"""Map int [0,11] -> (model_choice,freq_choice)."""
	return action // 4, action % 4


class TacticalAgent:
	def __init__(self, state_dim: int = 5, action_dim: int = 12):
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
			prefer_cuda=False,
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

	def get_policy_summary(self) -> Dict[str, Any]:
		return {
			"epsilon": self.epsilon,
		}

__all__ = ["TacticalAgent", "flatten_action", "unflatten_action"]
