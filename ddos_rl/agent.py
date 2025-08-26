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
				 exploration_decay: float = 0.9995, min_exploration_rate: float = 0.01):
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
		
		# State visitation tracking
		self.state_visits = np.zeros(state_dims)
		self.action_counts = np.zeros(action_dim)

	def _state_to_index(self, state: List[int]) -> Tuple[int, ...]:
		return tuple(state)

	def choose_action(self, state: List[int], training: bool = True) -> int:
		state_index = self._state_to_index(state)
		
		# Track state visitation during training
		if training:
			self.state_visits[state_index] += 1
		
		if training and np.random.random() < self.epsilon:
			action = np.random.randint(self.action_dim)
		else:
			action = int(np.argmax(self.q_table[state_index]))
		
		# Track action selection
		if training:
			self.action_counts[action] += 1
			
		return action

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
		
		# State visitation statistics
		visited_states = int(np.count_nonzero(self.state_visits))
		state_coverage = (visited_states / num_states) * 100
		min_visits = int(np.min(self.state_visits[self.state_visits > 0])) if visited_states > 0 else 0
		max_visits = int(np.max(self.state_visits))
		
		# Action distribution statistics
		action_entropy = -np.sum((self.action_counts / np.sum(self.action_counts)) * 
								np.log(self.action_counts / np.sum(self.action_counts) + 1e-10))
		
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
			"visited_states": visited_states,
			"state_coverage_percentage": state_coverage,
			"min_state_visits": min_visits,
			"max_state_visits": max_visits,
			"action_entropy": float(action_entropy),
			"action_distribution": self.action_counts.tolist(),
		}

	def get_underexplored_states(self, threshold: int = 5) -> List[Tuple[int, ...]]:
		"""Get states that have been visited fewer than threshold times"""
		underexplored = []
		for idx in np.ndindex(self.state_visits.shape):
			if 0 < self.state_visits[idx] < threshold:
				underexplored.append(idx)
		return underexplored

	def get_unvisited_states(self) -> List[Tuple[int, ...]]:
		"""Get states that have never been visited"""
		unvisited = []
		for idx in np.ndindex(self.state_visits.shape):
			if self.state_visits[idx] == 0:
				unvisited.append(idx)
		return unvisited

__all__ = ["QLearningAgent"]
