import numpy as np
import sys
import os
from typing import Dict, List, Tuple
try:
    import gym
    from gym import spaces
except ImportError:
    import gymnasium as gym
    from gymnasium import spaces

# Import data-driven profiles
from ..shared.crypto_profiles import (
    CRYPTO_PROFILES, MissionPhase, get_algorithm_performance,
    calculate_mission_algorithm_score, MISSION_ALGORITHM_PREFERENCES
)

# Legacy crypto algorithms mapping for compatibility
CRYPTO_ALGORITHMS = {
    0: {"name": "KYBER", "latency_ms": 156.8, "security_rating": 9, "power_multiplier": 1.5},
    1: {"name": "DILITHIUM", "latency_ms": 192.6, "security_rating": 8, "power_multiplier": 1.7},
    2: {"name": "SPHINCS", "latency_ms": 442.8, "security_rating": 10, "power_multiplier": 2.1},
    3: {"name": "FALCON", "latency_ms": 134.7, "security_rating": 7, "power_multiplier": 1.4}
}

CRYPTO_RL = {
    "REWARDS": {
        "SECURITY_MATCH_BONUS": 20,
        "UNDERKILL_PENALTY": -30,
        "OVERKILL_PENALTY": -15,
        "POWER_EFFICIENCY_FACTOR": 10,
        "LATENCY_PENALTY_FACTOR": 15,
        "BATTERY_PRESERVATION_BONUS": 25
    }
}

class StrategicCryptoEnv(gym.Env):
    """
    Enhanced strategic cryptographic algorithm selection environment for UAV swarms.
    Uses empirical performance data and includes swarm consensus threat assessment.
    """
    
    def __init__(self):
        super(StrategicCryptoEnv, self).__init__()
        
        # Define action and observation space
        # Actions: Select from available crypto algorithms (KYBER, DILITHIUM, SPHINCS, FALCON)
        self.action_space = spaces.Discrete(4)
        
        # Enhanced state space dimensions:
        # Threat Level (4), Avg Fleet Battery (4), Mission Phase (4), Swarm Consensus Threat (5)
        # Total: 4 * 4 * 4 * 5 = 320 possible states
        self.observation_space = spaces.MultiDiscrete([4, 4, 4, 5])
        
        # Initialize enhanced state variables
        self.threat_level_idx = 0  # LOW
        self.avg_fleet_battery_idx = 3  # HIGH
        self.mission_phase_idx = 0  # IDLE
        self.swarm_consensus_threat_idx = 0  # NO_THREAT
        
        # Fleet simulation
        self.fleet_size = 5
        self.fleet_batteries = [100.0] * self.fleet_size
        self.swarm_threat_reports = [0] * self.fleet_size  # Individual threat assessments
        
        # Episode tracking
        self.episode_steps = 0
        self.max_steps = 200  # Maximum steps per episode
        
        # Time-based variables
        self.time_step = 30  # Each step represents 30 seconds
        
        # CPU configuration for empirical data lookup
        self.cpu_cores = 4  # GCS has more cores than drone
        self.cpu_frequency = 1800  # MHz - GCS can run at higher frequency
        
        # Reset the environment
        self.reset()
    
    def _get_state(self):
        """Convert current state to discrete state vector for Q-learning"""
        return [
            self.security_risk_idx,
            self.battery_state_idx,
            self.computation_capacity_idx,
            self.mission_criticality_idx,
            self.communication_intensity_idx,
            self.threat_context_idx
        ]
    
    def step(self, action):
        """
        Take an action in the environment and return the next state, reward, done, info
        
        Actions:
        0: Use ASCON_128 (Baseline: Low latency, low power, low security)
        1: Use KYBER_CRYPTO (Balanced PQC: Medium latency, medium power, medium security)
        2: Use SPHINCS (High Security PQC: Very high latency, high power, maximum security)
        3: Use FALCON512 (Fast PQC: Lowest latency, low power, high security)
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Get algorithm properties
        algo_props = CRYPTO_ALGORITHMS[action]
        
        # Calculate battery drain based on algorithm power requirements
        power_multiplier = algo_props["power_multiplier"]
        base_drain = 0.2  # Base battery drain per step (percentage points)
        
        # Adjust drain based on computation capacity
        if self.computation_capacity_idx == 0:  # CONSTRAINED
            comp_factor = 1.5
        elif self.computation_capacity_idx == 1:  # NORMAL
            comp_factor = 1.0
        else:  # ABUNDANT
            comp_factor = 0.7
        
        # Adjust drain based on communication intensity
        comm_factors = [1.0, 1.3, 1.8]  # LOW, MEDIUM, HIGH
        comm_factor = comm_factors[self.communication_intensity_idx]
        
        # Calculate total battery drain
        total_drain = base_drain * power_multiplier * comp_factor * comm_factor
        self.battery_percentage = max(0, self.battery_percentage - total_drain)
        
        # Update battery state index based on battery percentage
        if self.battery_percentage < 20:
            self.battery_state_idx = 0  # CRITICAL
        elif self.battery_percentage < 50:
            self.battery_state_idx = 1  # LOW
        elif self.battery_percentage < 80:
            self.battery_state_idx = 2  # MEDIUM
        else:
            self.battery_state_idx = 3  # HIGH
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Update environment state probabilistically
        self._update_environment_state()
        
        # Increment step counter
        self.episode_steps += 1
        
        # Determine if episode is done
        done = False
        if self.battery_percentage <= 0:
            done = True
            reward += CRYPTO_RL["REWARDS"]["BATTERY_PRESERVATION_BONUS"] * -10  # Large penalty
        
        if self.episode_steps >= self.max_steps:
            done = True
        
        # Get the new discretized state
        state = self._get_state()
        
        # Create info dict
        info = {
            "battery_percentage": self.battery_percentage,
            "algorithm": CRYPTO_ALGORITHMS[action]["name"],
            "latency": CRYPTO_ALGORITHMS[action]["latency_ms"],
            "security_rating": CRYPTO_ALGORITHMS[action]["security_rating"],
            "power_multiplier": CRYPTO_ALGORITHMS[action]["power_multiplier"]
        }
        
        return state, reward, done, info
    
    def _calculate_reward(self, action):
        """Calculate the reward based on the action taken and the current state"""
        reward = 0
        
        # Get algorithm properties
        algo = CRYPTO_ALGORITHMS[action]
        latency = algo["latency_ms"]
        security = algo["security_rating"]
        power = algo["power_multiplier"]
        
        # Get reward parameters
        rewards = CRYPTO_RL["REWARDS"]
        
        # Security match reward/penalty
        security_needed = self.security_risk_idx * 3 + 1  # Maps [0,1,2,3] to [1,4,7,10]
        security_match = min(security, security_needed) / max(security, security_needed)
        
        # If security is too low for the risk (underkill)
        if security < security_needed:
            reward += rewards["UNDERKILL_PENALTY"] * (1 - security_match)
        # If security is too high for the risk (overkill)
        elif security > security_needed + 2:  # Some buffer for "just right" security
            reward += rewards["OVERKILL_PENALTY"] * (1 - security_match)
        # If security is just right
        else:
            reward += rewards["SECURITY_MATCH_BONUS"] * security_match
        
        # Power efficiency reward
        if self.battery_state_idx <= 1:  # CRITICAL or LOW
            # More reward for power efficiency when battery is low
            power_efficiency = 1.5 / power
            reward += rewards["POWER_EFFICIENCY_FACTOR"] * power_efficiency
        else:
            power_efficiency = 1.2 / power
            reward += rewards["POWER_EFFICIENCY_FACTOR"] * power_efficiency * 0.5
        
        # Latency reward/penalty
        if self.mission_criticality_idx >= 2:  # HIGH or CRITICAL mission
            # Penalize high latency in critical missions
            latency_factor = latency / 20.0  # Normalize, SPHINCS is ~20ms
            reward -= rewards["LATENCY_PENALTY_FACTOR"] * latency_factor * self.mission_criticality_idx
        
        # Battery preservation bonus
        if self.battery_state_idx <= 1:  # CRITICAL or LOW
            if action in [0, 3]:  # ASCON or FALCON (efficient)
                reward += rewards["BATTERY_PRESERVATION_BONUS"]
        
        # Threat context rewards
        if self.threat_context_idx == 2:  # HOSTILE
            if security >= 8:  # High security algorithms
                reward += rewards["SECURITY_MATCH_BONUS"] * 1.5
        
        return reward
    
    def _update_environment_state(self):
        """Update the environment state probabilistically to simulate real-world conditions"""
        
        # Update security risk level (changes less frequently)
        if np.random.random() < 0.1:  # 10% chance of change
            # Security risk tends to change gradually
            change = np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
            self.security_risk_idx = max(0, min(3, self.security_risk_idx + change))
        
        # Update computation capacity
        if np.random.random() < 0.15:  # 15% chance of change
            self.computation_capacity_idx = np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])
        
        # Update mission criticality (changes with mission phases)
        if np.random.random() < 0.12:  # 12% chance of change
            # Mission criticality tends to change gradually
            change = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
            self.mission_criticality_idx = max(0, min(3, self.mission_criticality_idx + change))
        
        # Update communication intensity
        if np.random.random() < 0.2:  # 20% chance of change
            self.communication_intensity_idx = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
        
        # Update threat context
        if np.random.random() < 0.15:  # 15% chance of change
            # Threat context is correlated with security risk
            if self.security_risk_idx >= 2:  # HIGH or CRITICAL risk
                self.threat_context_idx = np.random.choice([0, 1, 2], p=[0.1, 0.3, 0.6])
            else:  # LOW or MEDIUM risk
                self.threat_context_idx = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
    
    def reset(self):
        """Reset the environment state for a new episode"""
        self.battery_percentage = 100.0
        self.episode_steps = 0
        
        # Randomize initial state
        self.security_risk_idx = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
        self.battery_state_idx = 3  # Start with HIGH battery
        self.computation_capacity_idx = np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])
        self.mission_criticality_idx = np.random.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.3, 0.1])
        self.communication_intensity_idx = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
        
        # Set threat context correlated with security risk
        if self.security_risk_idx >= 2:  # HIGH or CRITICAL risk
            self.threat_context_idx = np.random.choice([0, 1, 2], p=[0.1, 0.4, 0.5])
        else:  # LOW or MEDIUM risk
            self.threat_context_idx = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
        
        return self._get_state()
    
    def render(self, mode='human'):
        """Render the environment state"""
        threat_labels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        battery_labels = ["CRITICAL", "LOW", "MEDIUM", "HIGH"]
        mission_labels = ["IDLE", "PATROL", "ENGAGEMENT", "CRITICAL_TASK"]
        swarm_threat_labels = ["NO_THREAT", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        
        print(f"Step: {self.episode_steps}/{self.max_steps}")
        print(f"Fleet Battery: {np.mean(self.fleet_batteries):.1f}% ({battery_labels[self.avg_fleet_battery_idx]})")
        print(f"Threat Level: {threat_labels[self.threat_level_idx]}")
        print(f"Mission Phase: {mission_labels[self.mission_phase_idx]}")
        print(f"Swarm Consensus Threat: {swarm_threat_labels[self.swarm_consensus_threat_idx]}")
        print(f"Individual Threat Reports: {self.swarm_threat_reports}")
        print("-" * 40)

# Alias for backward compatibility
CryptoEnv = StrategicCryptoEnv
