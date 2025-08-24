import numpy as np
import os
import sys

# Add config directory to path and import with fallback to avoid name clash with root config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from config.crypto_config import CRYPTO_RL  # type: ignore
except Exception:
    import runpy
    _cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'crypto_config.py'))
    _data = runpy.run_path(_cfg_path)
    CRYPTO_RL = _data.get('CRYPTO_RL', {})

class QLearningAgent:
    """
    Q-Learning agent for cryptographic algorithm selection
    """
    
    def __init__(self, state_dims, action_dim, learning_rate=None, discount_factor=None, 
                 exploration_rate=None, exploration_decay=None, min_exploration_rate=None):
        """
        Initialize the Q-Learning agent
        
        Args:
            state_dims: List of dimensions for each state variable
            action_dim: Number of possible actions
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            exploration_rate: Initial exploration rate (epsilon)
            exploration_decay: Exploration rate decay factor
            min_exploration_rate: Minimum exploration rate
        """
        self.state_dims = state_dims
        self.action_dim = action_dim
        
        # Use default values from config if not provided
        self.learning_rate = learning_rate if learning_rate is not None else CRYPTO_RL.get("LEARNING_RATE", 0.1)
        self.discount_factor = discount_factor if discount_factor is not None else CRYPTO_RL.get("DISCOUNT_FACTOR", 0.99)
        self.epsilon = exploration_rate if exploration_rate is not None else CRYPTO_RL.get("EXPLORATION_RATE", 1.0)
        self.epsilon_decay = exploration_decay if exploration_decay is not None else CRYPTO_RL.get("EXPLORATION_DECAY", 0.9995)
        self.min_epsilon = min_exploration_rate if min_exploration_rate is not None else CRYPTO_RL.get("MIN_EXPLORATION_RATE", 0.01)
        
        # Initialize Q-table
        self.q_table = np.zeros(state_dims + [action_dim])
        
        # Statistics
        self.training_episodes = 0
        self.training_steps = 0
    
    def _state_to_index(self, state):
        """Convert state vector to index tuple for Q-table"""
        return tuple(state)
    
    def choose_action(self, state, training=True):
        """
        Choose an action based on current state
        
        Args:
            state: Current state vector
            training: If True, use epsilon-greedy policy; if False, use greedy policy
            
        Returns:
            action: Selected action
        """
        state_index = self._state_to_index(state)
        
        # Exploration-exploitation trade-off
        if training and np.random.random() < self.epsilon:
            # Exploration: choose random action
            return np.random.randint(self.action_dim)
        else:
            # Exploitation: choose best action
            return np.argmax(self.q_table[state_index])
    
    def learn(self, state, action, reward, next_state, done):
        """
        Update Q-value based on experience
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        state_index = self._state_to_index(state)
        next_state_index = self._state_to_index(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_index][action]
        
        # Maximum Q-value for next state
        max_next_q = np.max(self.q_table[next_state_index]) if not done else 0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        # Update Q-table
        self.q_table[state_index][action] = new_q
        
        # Update exploration rate
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        
        # Update statistics
        self.training_steps += 1
    
    def save_policy(self, filepath):
        """Save the Q-table to a file"""
        np.save(filepath, self.q_table)
        print(f"Policy saved to {filepath}")
    
    def load_policy(self, filepath):
        """Load the Q-table from a file"""
        if os.path.exists(filepath):
            self.q_table = np.load(filepath)
            print(f"Policy loaded from {filepath}")
            return True
        else:
            print(f"Policy file {filepath} not found")
            return False
