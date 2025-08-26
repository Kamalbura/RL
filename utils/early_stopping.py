"""
Early stopping utilities for RL training to detect convergence and prevent overfitting.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque


class EarlyStopping:
    """Early stopping based on reward convergence and Q-value stability"""
    
    def __init__(self, 
                 patience: int = 50,
                 min_delta: float = 0.01,
                 window_size: int = 20,
                 convergence_threshold: float = 0.005):
        """
        Args:
            patience: Number of episodes to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            window_size: Window size for moving average calculation
            convergence_threshold: Threshold for reward variance to consider converged
        """
        self.patience = patience
        self.min_delta = min_delta
        self.window_size = window_size
        self.convergence_threshold = convergence_threshold
        
        self.reward_history = deque(maxlen=window_size)
        self.q_value_history = deque(maxlen=window_size)
        self.best_reward = -np.inf
        self.episodes_without_improvement = 0
        self.converged = False
        
    def update(self, episode_reward: float, avg_q_value: float) -> Dict[str, any]:
        """Update early stopping state with new episode data"""
        self.reward_history.append(episode_reward)
        self.q_value_history.append(avg_q_value)
        
        # Check for improvement
        if episode_reward > self.best_reward + self.min_delta:
            self.best_reward = episode_reward
            self.episodes_without_improvement = 0
        else:
            self.episodes_without_improvement += 1
            
        # Check convergence if we have enough data
        should_stop = False
        convergence_info = {}
        
        if len(self.reward_history) >= self.window_size:
            # Calculate reward stability
            reward_variance = np.var(list(self.reward_history))
            reward_mean = np.mean(list(self.reward_history))
            
            # Calculate Q-value stability  
            q_variance = np.var(list(self.q_value_history))
            q_mean = np.mean(list(self.q_value_history))
            
            # Check convergence criteria
            reward_converged = reward_variance < self.convergence_threshold
            patience_exceeded = self.episodes_without_improvement >= self.patience
            
            convergence_info = {
                "reward_variance": reward_variance,
                "reward_mean": reward_mean,
                "q_variance": q_variance, 
                "q_mean": q_mean,
                "reward_converged": reward_converged,
                "patience_exceeded": patience_exceeded,
                "episodes_without_improvement": self.episodes_without_improvement,
                "best_reward": self.best_reward
            }
            
            should_stop = reward_converged or patience_exceeded
            self.converged = should_stop
            
        return {
            "should_stop": should_stop,
            "converged": self.converged,
            **convergence_info
        }
        
    def reset(self):
        """Reset early stopping state"""
        self.reward_history.clear()
        self.q_value_history.clear()
        self.best_reward = -np.inf
        self.episodes_without_improvement = 0
        self.converged = False


class PerformanceMonitor:
    """Monitor training performance and detect issues"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.reward_history = deque(maxlen=window_size)
        self.loss_history = deque(maxlen=window_size) 
        self.exploration_history = deque(maxlen=window_size)
        
    def update(self, reward: float, loss: float = None, exploration_rate: float = None):
        """Update performance metrics"""
        self.reward_history.append(reward)
        if loss is not None:
            self.loss_history.append(loss)
        if exploration_rate is not None:
            self.exploration_history.append(exploration_rate)
            
    def get_performance_summary(self) -> Dict[str, any]:
        """Get current performance summary"""
        if not self.reward_history:
            return {}
            
        rewards = list(self.reward_history)
        summary = {
            "reward_mean": np.mean(rewards),
            "reward_std": np.std(rewards),
            "reward_trend": self._calculate_trend(rewards),
            "episodes_tracked": len(rewards)
        }
        
        if self.loss_history:
            losses = list(self.loss_history)
            summary.update({
                "loss_mean": np.mean(losses),
                "loss_std": np.std(losses),
                "loss_trend": self._calculate_trend(losses)
            })
            
        if self.exploration_history:
            exploration = list(self.exploration_history)
            summary.update({
                "exploration_mean": np.mean(exploration),
                "exploration_current": exploration[-1] if exploration else 0
            })
            
        return summary
        
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend using linear regression slope"""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)
        
    def detect_issues(self) -> Dict[str, bool]:
        """Detect common training issues"""
        issues = {
            "reward_plateau": False,
            "reward_collapse": False,
            "high_variance": False,
            "insufficient_exploration": False
        }
        
        if len(self.reward_history) < self.window_size // 2:
            return issues
            
        rewards = list(self.reward_history)
        
        # Detect reward plateau
        trend = self._calculate_trend(rewards)
        issues["reward_plateau"] = abs(trend) < 0.001 and len(rewards) > 20
        
        # Detect reward collapse  
        recent_rewards = rewards[-10:] if len(rewards) >= 10 else rewards
        early_rewards = rewards[:10] if len(rewards) >= 20 else []
        if early_rewards and recent_rewards:
            issues["reward_collapse"] = np.mean(recent_rewards) < np.mean(early_rewards) * 0.5
            
        # Detect high variance
        issues["high_variance"] = np.std(rewards) > abs(np.mean(rewards))
        
        # Detect insufficient exploration
        if self.exploration_history:
            current_exploration = self.exploration_history[-1]
            issues["insufficient_exploration"] = current_exploration < 0.01
            
        return issues


def create_early_stopping_config(agent_type: str = "tactical") -> Dict[str, any]:
    """Create default early stopping configuration for different agent types"""
    configs = {
        "tactical": {
            "patience": 100,
            "min_delta": 0.05,
            "window_size": 30,
            "convergence_threshold": 0.01
        },
        "strategic": {
            "patience": 75,
            "min_delta": 0.02, 
            "window_size": 25,
            "convergence_threshold": 0.005
        }
    }
    return configs.get(agent_type, configs["tactical"])
