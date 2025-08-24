"""
QLearningAgent (moved from top-level rl_agent.py)
"""

import numpy as np
import os
from typing import List, Tuple, Dict, Any


class QLearningAgent:
	"""
	Q-Learning agent for UAV cybersecurity decision making
	"""

	def __init__(self, state_dims: List[int], action_dim: int, learning_rate: float = 0.1,
				 discount_factor: float = 0.99, exploration_rate: float = 1.0,
				 exploration_decay: float = 0.995, min_exploration_rate: float = 0.01):
		self.state_dims = state_dims
		self.action_dim = action_dim
		self.learning_rate = learning_rate
		self.discount_factor = discount_factor
		self.epsilon = exploration_rate
		self.epsilon_decay = exploration_decay
		self.min_epsilon = min_exploration_rate

		# Initialize Q-table
		self.q_table = np.zeros(state_dims + [action_dim])

		# Training metrics
		self.training_episodes = 0
		self.training_steps = 0

	def _state_to_index(self, state: List[int]) -> Tuple[int, ...]:
		return tuple(state)

	def choose_action(self, state: List[int], training: bool = True) -> int:
		state_index = self._state_to_index(state)
		if training and np.random.random() < self.epsilon:
			return np.random.randint(self.action_dim)
		else:
			return int(np.argmax(self.q_table[state_index]))

	def learn(self, state: List[int], action: int, reward: float, next_state: List[int], done: bool) -> None:
		state_index = self._state_to_index(state)
		next_state_index = self._state_to_index(next_state)
		current_q = self.q_table[state_index][action]
		max_next_q = np.max(self.q_table[next_state_index]) if not done else 0
		new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
		self.q_table[state_index][action] = new_q
		if self.epsilon > self.min_epsilon:
			self.epsilon *= self.epsilon_decay
		self.training_steps += 1

	def save_policy(self, filepath: str) -> None:
		os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
		np.save(filepath, self.q_table)
		print(f"Policy saved to {filepath}")

	def load_policy(self, filepath: str) -> bool:
		if os.path.exists(filepath):
			self.q_table = np.load(filepath)
			print(f"Policy loaded from {filepath}")
			return True
		else:
			print(f"Policy file {filepath} not found")
			return False

	def get_best_action_for_state(self, state: List[int]) -> Tuple[int, float]:
		state_index = self._state_to_index(state)
		action = int(np.argmax(self.q_table[state_index]))
		q_value = float(self.q_table[state_index][action])
		return action, q_value

	def get_policy_summary(self) -> Dict[str, Any]:
		num_states = int(np.prod(self.state_dims))
		num_nonzero = int(np.count_nonzero(self.q_table))
		coverage = (num_nonzero / (num_states * self.action_dim)) * 100
		positive_q = int(np.sum(self.q_table > 0))
		positive_ratio = (positive_q / (num_states * self.action_dim)) * 100
		avg_q = float(np.mean(self.q_table))
		return {
			"state_dimensions": self.state_dims,
			"action_dimension": self.action_dim,
			"total_state_action_pairs": num_states * self.action_dim,
			"nonzero_q_values": num_nonzero,
			"coverage_percentage": coverage,
			"positive_q_values": positive_q,
			"positive_q_percentage": positive_ratio,
			"average_q_value": avg_q,
			"training_episodes": self.training_episodes,
			"training_steps": self.training_steps,
		}

__all__ = ["QLearningAgent"]
