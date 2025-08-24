import numpy as np
import gym
from gym import spaces
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class StrategicSwarmEnv(gym.Env):
    """
    A simulator for the Strategic Swarm environment.
    This environment simulates a fleet of drones and makes decisions about cryptographic policies.
    """
    
    def __init__(self, num_drones=3):
        super(StrategicSwarmEnv, self).__init__()
        
        self.num_drones = num_drones
        
        # Define action and observation space
        # Actions: ASCON_128, KYBER_CRYPTO, SPHINCS, FALCON512
        self.action_space = spaces.Discrete(4)
        
        # State space: Swarm_Threat_Level (3), Fleet_Battery_Status (3), Mission_Phase (4)
        self.observation_space = spaces.MultiDiscrete([3, 3, 4])
        
        # Crypto algorithm performance characteristics (ms)
        self.crypto_latency = {
            0: 5.0,     # ASCON_128
            1: 2.8,     # KYBER_CRYPTO
            2: 19.6,    # SPHINCS
            3: 0.7      # FALCON512
        }
        
        # Security strength rating (1-10, higher is better)
        self.security_rating = {
            0: 3,    # ASCON_128
            1: 6,    # KYBER_CRYPTO
            2: 10,   # SPHINCS
            3: 8     # FALCON512
        }
        
        # Power consumption relative to base (multiplier)
        self.power_multiplier = {
            0: 1.05,   # ASCON_128 (low power)
            1: 1.15,   # KYBER_CRYPTO (medium power)
            2: 1.40,   # SPHINCS (very high power)
            3: 1.10    # FALCON512 (low power)
        }
        
        # Time-based variables
        self.time_step = 30  # Each step represents 30 seconds in the strategic context
        self.max_steps = 100  # Maximum steps per episode
        
        # Reset environment
        self.reset()
    
    def reset(self):
        """Reset the environment to start a new episode"""
        # Initialize fleet state
        self.fleet_battery = np.full(self.num_drones, 100.0)  # All drones start at 100% battery
        self.fleet_operational = np.ones(self.num_drones, dtype=bool)  # All drones start operational
        
        # Initialize strategic state variables
        self.swarm_threat_level_idx = 0  # NONE
        self.fleet_battery_status_idx = 0  # HEALTHY
        self.mission_phase_idx = 0  # IDLE
        
        self.steps = 0
        self.current_crypto = 0  # Start with ASCON_128
        
        return self._get_state()
    
    def _get_state(self):
        """Get the current strategic state"""
        return [self.swarm_threat_level_idx, self.fleet_battery_status_idx, self.mission_phase_idx]
    
    def _update_fleet_battery_status(self):
        """Update the fleet battery status based on individual drone batteries"""
        avg_battery = np.mean(self.fleet_battery[self.fleet_operational])
        
        if avg_battery < 30:
            self.fleet_battery_status_idx = 2  # CRITICAL
        elif avg_battery < 60:
            self.fleet_battery_status_idx = 1  # DEGRADING
        else:
            self.fleet_battery_status_idx = 0  # HEALTHY
    
    def _update_environment(self):
        """Update the environment state probabilistically"""
        # Update mission phase with some probability
        if np.random.random() < 0.1:  # 10% chance of mission phase change
            # Prefer logical mission progression
            if self.mission_phase_idx == 0:  # IDLE
                self.mission_phase_idx = 1 if np.random.random() < 0.8 else 0  # Likely move to INGRESS
            elif self.mission_phase_idx == 1:  # INGRESS
                self.mission_phase_idx = 2 if np.random.random() < 0.7 else 1  # Likely move to LOITER_ON_TARGET
            elif self.mission_phase_idx == 2:  # LOITER_ON_TARGET
                self.mission_phase_idx = 3 if np.random.random() < 0.3 else 2  # Might move to EGRESS
            else:  # EGRESS
                self.mission_phase_idx = 0 if np.random.random() < 0.2 else 3  # Might return to IDLE
        
        # Update swarm threat level with some probability
        if np.random.random() < 0.2:  # 20% chance of threat level change
            # Threat level more likely during INGRESS and EGRESS
            if self.mission_phase_idx in [1, 3]:  # INGRESS or EGRESS
                self.swarm_threat_level_idx = np.random.choice([0, 1, 2], p=[0.2, 0.5, 0.3])
            else:
                self.swarm_threat_level_idx = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
    
    def _calculate_reward(self, action):
        """Calculate reward based on the action and state"""
        reward = 0
        
        # Fleet endurance reward/penalty
        if self.fleet_battery_status_idx == 2:  # CRITICAL
            reward -= 10  # Penalty for critical battery
        
        # Latency-based penalties
        latency = self.crypto_latency[action]
        # High latency during INGRESS or EGRESS is bad
        if self.mission_phase_idx in [1, 3]:  # INGRESS or EGRESS
            if latency > 10:  # SPHINCS is the only one > 10ms
                reward -= 20  # Severe penalty for high latency during critical phases
        
        # Security-based rewards
        security = self.security_rating[action]
        # Higher security when under threat is good
        if self.swarm_threat_level_idx == 2:  # CRITICAL threat
            reward += security * 2  # Reward scales with security level
        
        # Efficiency rewards - using low-power crypto during safe phases
        if self.mission_phase_idx in [0, 2]:  # IDLE or LOITER
            if action in [0, 3]:  # ASCON or FALCON (low power)
                reward += 5
        
        # Strategic coherence rewards
        if self.swarm_threat_level_idx == 2:  # CRITICAL threat
            if action in [2, 3]:  # SPHINCS or FALCON (high security)
                reward += 10
        
        return reward
    
    def step(self, action):
        """
        Take an action in the environment
        
        Actions:
        0: Command ASCON_128 (Baseline: Low latency, low power)
        1: Command KYBER_CRYPTO (Balanced PQC)
        2: Command SPHINCS (High Security PQC, VERY high latency)
        3: Command FALCON512 (High Security PQC, low latency)
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Apply the crypto policy to all drones
        self.current_crypto = action
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Update drone batteries based on crypto power requirements
        power_factor = self.power_multiplier[action]
        
        # Base battery drain per step (percentage points)
        base_drain = 0.5
        
        # Apply drain to each operational drone
        for i in range(self.num_drones):
            if self.fleet_operational[i]:
                # Apply power factor to base drain
                drain = base_drain * power_factor
                
                # Additional drain during active mission phases
                if self.mission_phase_idx in [1, 3]:  # INGRESS or EGRESS
                    drain *= 1.5
                
                self.fleet_battery[i] -= drain
                
                # Check if drone is still operational
                if self.fleet_battery[i] <= 0:
                    self.fleet_battery[i] = 0
                    self.fleet_operational[i] = False
        
        # Update fleet battery status
        self._update_fleet_battery_status()
        
        # Update environment state
        self._update_environment()
        
        # Increment step counter
        self.steps += 1
        
        # Check if episode is done
        done = False
        
        # Episode ends if all drones are depleted or max steps reached
        if not np.any(self.fleet_operational) or self.steps >= self.max_steps:
            done = True
        
        # Get new state
        state = self._get_state()
        
        # Create info dict
        info = {
            "fleet_battery_mean": np.mean(self.fleet_battery),
            "operational_drones": np.sum(self.fleet_operational),
            "crypto_latency": self.crypto_latency[action],
            "security_rating": self.security_rating[action]
        }
        
        return state, reward, done, info
    
    def render(self, mode='human'):
        """Render the environment state"""
        threat_levels = ["NONE", "CAUTION", "CRITICAL"]
        battery_status = ["HEALTHY", "DEGRADING", "CRITICAL"]
        mission_phases = ["IDLE", "INGRESS", "LOITER_ON_TARGET", "EGRESS"]
        crypto_names = ["ASCON_128", "KYBER_CRYPTO", "SPHINCS", "FALCON512"]
        
        print(f"Step: {self.steps}/{self.max_steps}")
        print(f"Swarm Threat Level: {threat_levels[self.swarm_threat_level_idx]}")
        print(f"Fleet Battery Status: {battery_status[self.fleet_battery_status_idx]}")
        print(f"Mission Phase: {mission_phases[self.mission_phase_idx]}")
        print(f"Current Crypto: {crypto_names[self.current_crypto]}")
        print(f"Mean Fleet Battery: {np.mean(self.fleet_battery):.1f}%")
        print(f"Operational Drones: {np.sum(self.fleet_operational)}/{self.num_drones}")
        print("-" * 30)
