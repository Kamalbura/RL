import numpy as np
import os
from typing import Dict, List, Tuple
try:
    import gym
    from gym import spaces
except ImportError:
    import gymnasium as gym
    from gymnasium import spaces

# Import data-driven profiles (single source of truth)
from shared.crypto_profiles import (
    get_algorithm_performance,
)


class StrategicCryptoEnv(gym.Env):
    """
    Enhanced strategic cryptographic algorithm selection environment for UAV swarms.
    Uses empirical performance data and includes swarm consensus threat assessment.
    """

    def __init__(self):
        super(StrategicCryptoEnv, self).__init__()

        # Discrete action over 4 algorithms (0..3)
        self.action_space = spaces.Discrete(4)
        # Continuous normalized observation
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        # Initialize enhanced state variables
        self.threat_level = 0.0
        self.avg_fleet_battery = 100.0
        self.mission_phase = 0  # 0..3
        self.swarm_consensus_threat = 0.0

        # Fleet simulation (kept for potential extensions)
        self.fleet_size = 5
        self.fleet_batteries = [100.0] * self.fleet_size
        self.swarm_threat_reports = [0] * self.fleet_size  # Individual threat assessments

        # Episode tracking
        self.episode_steps = 0
        self.max_steps = 200  # Maximum steps per episode

        # Time-based/system variables
        self.time_step = 30  # seconds per step (not directly used in reward)
        self.cpu_cores = 4
        self.cpu_frequency = 1800

        # Reset the environment
        self.reset()

    def _get_state(self):
        return np.array([
            np.clip(self.threat_level, 0.0, 1.0),
            np.clip(self.avg_fleet_battery / 100.0, 0.0, 1.0),
            self.mission_phase / 3.0,
            np.clip(self.swarm_consensus_threat, 0.0, 1.0),
        ], dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"
        algo = ["KYBER", "DILITHIUM", "SPHINCS", "FALCON"][int(action)]
        perf = get_algorithm_performance(algo, self.cpu_cores, self.cpu_frequency)
        latency_ms = float(perf["latency_ms"])
        power_w = float(perf["power_watts"])
        sec = float(perf["security_rating"])  # 1..10

        # Normalized reward components
        sec_term = float(np.clip(sec / 10.0, 0.0, 1.0)) * (0.5 + 0.5 * self.threat_level)
        power_term = -float(np.clip((power_w - 3.0) / (20.0 - 3.0), 0.0, 1.0)) * (1.2 if self.avg_fleet_battery < 40.0 else 1.0)
        latency_term = -float(np.clip(latency_ms / 1000.0, 0.0, 1.0)) * (0.3 + 0.7 * (self.mission_phase / 3.0))
        reward = float(np.clip(sec_term + power_term + latency_term, -1.0, 1.0))

        # Battery drain proportional to power
        self.avg_fleet_battery = float(max(0.0, self.avg_fleet_battery - power_w / 20.0))

        # Drift
        if np.random.random() < 0.12:
            self.threat_level = float(np.clip(self.threat_level + np.random.normal(0, 0.05), 0.0, 1.0))
        if np.random.random() < 0.15:
            self.swarm_consensus_threat = float(np.clip(0.8 * self.swarm_consensus_threat + 0.2 * self.threat_level + np.random.normal(0, 0.05), 0.0, 1.0))

        self.episode_steps += 1
        done = self.episode_steps >= self.max_steps or self.avg_fleet_battery <= 0.0
        state = self._get_state()
        info = {
            "algorithm": algo,
            "latency_ms": latency_ms,
            "power_watts": power_w,
            "security_rating": sec,
            "reward": reward,
        }
        return state, reward, done, info

    # Deprecated reward function removed in favor of normalized components above.

    def _update_environment_state(self):
        # Deprecated in normalized model; drift handled in step()
        pass

    def reset(self):
        """Reset the environment state for a new episode"""
        self.avg_fleet_battery = 100.0
        self.episode_steps = 0
        self.threat_level = float(np.random.uniform(0.0, 0.4))
        self.mission_phase = int(np.random.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.3, 0.1]))
        self.swarm_consensus_threat = float(np.random.uniform(0.0, 0.3))
        return self._get_state()

    def render(self, mode: str = 'human'):
        """Render the environment state"""
        print(f"Step: {self.episode_steps}/{self.max_steps}")
        print(f"Fleet Battery: {self.avg_fleet_battery:.1f}%")
        print(f"Threat Level (0-1): {self.threat_level:.2f}")
        print(f"Mission Phase (0-3): {self.mission_phase}")
        print(f"Swarm Consensus Threat (0-1): {self.swarm_consensus_threat:.2f}")
        print("-" * 40)


# Alias for backward compatibility
CryptoEnv = StrategicCryptoEnv
