"""
Prioritized Experience Replay Buffer

This module implements a Prioritized Experience Replay (PER) buffer,
which allows for more efficient training of RL agents by replaying
important transitions more frequently.

This implementation is inspired by the approach in Google's Dopamine framework
and the original PER paper (https://arxiv.org/abs/1511.05952).
"""

import numpy as np
import random

class SumTree:
    """
    A SumTree data structure for efficient prioritized sampling.
    The tree stores priorities, and each parent node is the sum of its children.
    This allows for O(log n) sampling and updating.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.size = 0

    def _propagate(self, idx: int, change: float):
        """Propagate a change in priority up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Find the sample index for a given priority value."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Return the total sum of priorities."""
        return self.tree[0]

    def add(self, priority: float, data: object):
        """Add a new experience with its priority."""
        tree_idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(tree_idx, priority)

        self.write_idx += 1
        if self.write_idx >= self.capacity:
            self.write_idx = 0
        
        if self.size < self.capacity:
            self.size += 1

    def update(self, tree_idx: int, priority: float):
        """Update the priority of an experience."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s: float) -> tuple[int, float, object]:
        """Get an experience based on a priority value."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])


class PrioritizedReplayBuffer:
    """
    A Prioritized Experience Replay buffer.
    """
    epsilon = 0.01  # Small value to ensure all experiences have some chance of being sampled
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        """
        Initialize the replay buffer with prioritization parameters.
        
        Args:
            capacity: The maximum number of experiences to store.
            alpha: The prioritization exponent (0 for uniform, 1 for full prioritization).
            beta: The importance-sampling exponent.
            beta_increment: The amount to increment beta at each sampling step.
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer with maximum priority to ensure
        it is sampled at least once.
        
        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state.
            done: Whether the episode has ended.
        """
        experience = (state, action, reward, next_state, done)
        self.tree.add(self.max_priority, experience)

    def sample(self, batch_size: int) -> tuple[list, np.ndarray, np.ndarray]:
        """
        Sample a batch of experiences proportional to their priority.
        
        Args:
            batch_size: The number of experiences to sample.
            
        Returns:
            A tuple containing the batch of experiences, their indices in the tree,
            and their importance-sampling weights.
        """
        batch = []
        indices = np.empty((batch_size,), dtype=np.int32)
        weights = np.empty((batch_size,), dtype=np.float32)
        
        segment = self.tree.total() / batch_size
        self.beta = np.min([1., self.beta + self.beta_increment])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            (idx, p, data) = self.tree.get(s)
            
            sampling_probability = p / self.tree.total()
            weights[i] = np.power(self.tree.size * sampling_probability, -self.beta)
            indices[i] = idx
            batch.append(data)
        
        # Normalize weights
        if self.tree.total() > 0:
            weights /= weights.max()

        return batch, indices, weights

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray):
        """
        Update the priorities of experiences based on their TD errors.
        
        Args:
            tree_indices: The indices of the experiences in the SumTree.
            td_errors: The TD errors for each experience.
        """
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        
        for idx, p in zip(tree_indices, priorities):
            self.tree.update(idx, p)
            
        # Update max priority
        self.max_priority = max(self.max_priority, np.max(priorities))

    def __len__(self) -> int:
        return self.tree.size
