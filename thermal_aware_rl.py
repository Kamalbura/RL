"""
Thermal-Aware RL Environment for UAV Systems
Integrates RPi temperature monitoring, resource utilization, and swarm coordination
"""

import numpy as np
import psutil
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json

try:
    # RPi-specific thermal monitoring
    import subprocess
    RPi_AVAILABLE = True
except ImportError:
    RPi_AVAILABLE = False

from crypto_profiles import (
    POST_QUANTUM_FREQUENCY_RULES, 
    get_algorithm_latency,
    get_algorithm_power,
    is_algorithm_realtime_safe
)

class ThermalState(Enum):
    OPTIMAL = 0      # < 60°C
    WARM = 1         # 60-70°C  
    HOT = 2          # 70-80°C
    CRITICAL = 3     # > 80°C

class SwarmAlertLevel(Enum):
    GREEN = 0        # No threats detected
    YELLOW = 1       # Low-level threats
    ORANGE = 2       # Medium threats
    RED = 3          # High threats - coordinated response needed

@dataclass
class DroneState:
    drone_id: str
    alert_level: SwarmAlertLevel
    battery_level: float
    last_detection: Optional[str]  # XGBoost/TST detection result
    connection_status: bool
    timestamp: float

@dataclass
class SystemResources:
    cpu_utilization: float      # 0-100%
    memory_utilization: float   # 0-100%
    temperature: float          # Celsius
    thermal_state: ThermalState
    available_cores: int
    current_frequency: int      # MHz

class ThermalAwareRLEnvironment:
    """
    Advanced RL environment with thermal awareness, resource monitoring,
    and swarm coordination capabilities.
    """
    
    def __init__(self, drone_id: str, swarm_size: int = 5):
        self.drone_id = drone_id
        self.swarm_size = swarm_size
        
        # Enhanced state space: [Threat, Battery, CPU_Load, Task_Priority, 
        #                       Thermal_State, Resource_Usage, Swarm_Alert, Connection_Status]
        self.state_dims = [4, 4, 3, 3, 4, 4, 4, 2]  # 6,144 states
        
        # Enhanced action space: [DDoS_Model, CPU_Freq, Crypto_Algo, Emergency_Mode]
        self.action_space_size = 24  # 2*4*3*1 combinations
        
        # Swarm state tracking
        self.swarm_states: Dict[str, DroneState] = {}
        self.last_swarm_update = 0
        
        # Thermal management
        self.thermal_history = []
        self.thermal_throttle_active = False
        
        # Anti-greedy exploration parameters
        self.exploration_bonus = 0.1
        self.curiosity_threshold = 10  # Encourage exploration of less-visited states
        self.state_visit_counts = np.zeros(self.state_dims)
        
        # XGBoost failure tracking
        self.xgboost_failure_count = 0
        self.max_xgboost_failures = 3
        self.fallback_detection_active = False
        
        # Connection loss handling
        self.connection_lost_time = None
        self.autonomous_mode = False
        
    def get_system_resources(self) -> SystemResources:
        """Get current system resource utilization and thermal state."""
        
        # CPU utilization
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory utilization  
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Temperature monitoring
        temperature = self._get_cpu_temperature()
        thermal_state = self._classify_thermal_state(temperature)
        
        # Available cores
        available_cores = psutil.cpu_count(logical=False)
        
        # Current CPU frequency
        current_freq = self._get_current_cpu_frequency()
        
        return SystemResources(
            cpu_utilization=cpu_percent,
            memory_utilization=memory_percent,
            temperature=temperature,
            thermal_state=thermal_state,
            available_cores=available_cores,
            current_frequency=current_freq
        )
    
    def _get_cpu_temperature(self) -> float:
        """Get RPi CPU temperature."""
        if RPi_AVAILABLE:
            try:
                # RPi-specific temperature reading
                result = subprocess.run(['vcgencmd', 'measure_temp'], 
                                      capture_output=True, text=True)
                temp_str = result.stdout.strip()
                # Extract temperature from "temp=XX.X'C"
                temp = float(temp_str.split('=')[1].split("'")[0])
                return temp
            except:
                pass
        
        # Fallback: estimate from CPU usage (for testing)
        cpu_percent = psutil.cpu_percent()
        base_temp = 45.0  # Base temperature
        load_temp = cpu_percent * 0.4  # 0.4°C per % CPU usage
        return base_temp + load_temp
    
    def _classify_thermal_state(self, temperature: float) -> ThermalState:
        """Classify thermal state based on temperature."""
        if temperature < 60:
            return ThermalState.OPTIMAL
        elif temperature < 70:
            return ThermalState.WARM
        elif temperature < 80:
            return ThermalState.HOT
        else:
            return ThermalState.CRITICAL
    
    def _get_current_cpu_frequency(self) -> int:
        """Get current CPU frequency in MHz."""
        try:
            freq_info = psutil.cpu_freq()
            return int(freq_info.current) if freq_info else 1200
        except:
            return 1200  # Default fallback
    
    def get_enhanced_state(self) -> List[int]:
        """
        Get enhanced state vector including thermal and swarm information.
        
        Returns:
            List of discrete state values for RL agent
        """
        # Get base tactical state
        threat_level = self._get_current_threat_level()
        battery_level = self._get_battery_level_discrete()
        cpu_load = self._get_cpu_load_discrete()
        task_priority = self._get_task_priority()
        
        # Get system resources
        resources = self.get_system_resources()
        thermal_state = resources.thermal_state.value
        resource_usage = self._discretize_resource_usage(resources)
        
        # Get swarm coordination state
        swarm_alert = self._get_swarm_alert_level().value
        connection_status = 1 if self._check_swarm_connection() else 0
        
        state = [
            threat_level,
            battery_level, 
            cpu_load,
            task_priority,
            thermal_state,
            resource_usage,
            swarm_alert,
            connection_status
        ]
        
        # Update state visit counts for anti-greedy exploration
        state_tuple = tuple(state)
        if len(state_tuple) == len(self.state_dims):
            # Safely increment visit count
            try:
                self.state_visit_counts[state_tuple] += 1
            except IndexError:
                # State out of bounds, use default
                pass
        
        return state
    
    def _discretize_resource_usage(self, resources: SystemResources) -> int:
        """Convert resource usage to discrete state."""
        # Combine CPU and memory usage
        combined_usage = (resources.cpu_utilization + resources.memory_utilization) / 2
        
        if combined_usage < 25:
            return 0  # LOW
        elif combined_usage < 50:
            return 1  # MEDIUM
        elif combined_usage < 75:
            return 2  # HIGH
        else:
            return 3  # CRITICAL
    
    def calculate_thermal_aware_reward(self, state: List[int], action: int, 
                                     next_state: List[int], info: Dict) -> float:
        """
        Calculate reward with thermal awareness and swarm coordination.
        """
        base_reward = 0
        
        # Get system resources
        resources = self.get_system_resources()
        
        # 1. Thermal penalty/bonus
        thermal_reward = self._calculate_thermal_reward(resources, action)
        base_reward += 0.25 * thermal_reward
        
        # 2. Post-quantum frequency optimization
        pq_reward = self._calculate_post_quantum_reward(action, resources)
        base_reward += 0.20 * pq_reward
        
        # 3. Anti-greedy exploration bonus
        exploration_reward = self._calculate_exploration_bonus(state)
        base_reward += 0.10 * exploration_reward
        
        # 4. Swarm coordination reward
        swarm_reward = self._calculate_swarm_coordination_reward(state, action)
        base_reward += 0.15 * swarm_reward
        
        # 5. XGBoost failure handling
        detection_reward = self._calculate_detection_reliability_reward(action, info)
        base_reward += 0.15 * detection_reward
        
        # 6. Base security-power-performance reward
        base_security_reward = self._calculate_base_reward(state, action, next_state)
        base_reward += 0.15 * base_security_reward
        
        # Critical safety constraints
        safety_penalty = self._calculate_safety_penalties(resources, action)
        base_reward += safety_penalty  # Can be large negative values
        
        return base_reward
    
    def _calculate_thermal_reward(self, resources: SystemResources, action: int) -> float:
        """Calculate thermal-based reward/penalty."""
        thermal_state = resources.thermal_state
        
        if thermal_state == ThermalState.CRITICAL:
            # Force immediate thermal throttling
            self.thermal_throttle_active = True
            return -50.0  # Massive penalty
        elif thermal_state == ThermalState.HOT:
            # Encourage lower-power actions
            return -10.0 if self._is_high_power_action(action) else 5.0
        elif thermal_state == ThermalState.WARM:
            return -2.0 if self._is_high_power_action(action) else 2.0
        else:  # OPTIMAL
            self.thermal_throttle_active = False
            return 5.0  # Bonus for optimal thermal state
    
    def _calculate_post_quantum_reward(self, action: int, resources: SystemResources) -> float:
        """Reward optimal frequency selection for post-quantum crypto."""
        crypto_algo = self._extract_crypto_algorithm_from_action(action)
        current_freq = resources.current_frequency
        
        if crypto_algo in POST_QUANTUM_FREQUENCY_RULES:
            rules = POST_QUANTUM_FREQUENCY_RULES[crypto_algo]
            optimal_freq = rules["optimal_frequency"]
            
            # Reward proximity to optimal frequency
            freq_diff = abs(current_freq - optimal_freq)
            if freq_diff < 100:  # Within 100MHz of optimal
                return 10.0
            elif freq_diff < 300:  # Within 300MHz
                return 5.0
            else:
                return -5.0  # Penalty for suboptimal frequency
        
        return 0.0
    
    def _calculate_exploration_bonus(self, state: List[int]) -> float:
        """Anti-greedy exploration bonus for less-visited states."""
        state_tuple = tuple(state)
        
        try:
            visit_count = self.state_visit_counts[state_tuple]
            if visit_count < self.curiosity_threshold:
                # Bonus inversely proportional to visit count
                return self.exploration_bonus * (self.curiosity_threshold - visit_count)
        except (IndexError, TypeError):
            # New or invalid state - give exploration bonus
            return self.exploration_bonus * self.curiosity_threshold
        
        return 0.0
    
    def _calculate_swarm_coordination_reward(self, state: List[int], action: int) -> float:
        """Reward coordinated swarm behavior."""
        swarm_alert = SwarmAlertLevel(state[6])  # swarm_alert from state
        
        if swarm_alert == SwarmAlertLevel.RED:
            # High threat - reward aggressive detection
            if self._is_aggressive_detection_action(action):
                return 15.0
            else:
                return -5.0  # Penalty for passive response during high threat
        elif swarm_alert == SwarmAlertLevel.GREEN:
            # No threats - reward power-efficient actions
            if self._is_power_efficient_action(action):
                return 10.0
        
        return 0.0
    
    def _calculate_detection_reliability_reward(self, action: int, info: Dict) -> float:
        """Handle XGBoost failure scenarios."""
        detection_method = self._extract_detection_method_from_action(action)
        
        # Check if XGBoost failed recently
        if self.xgboost_failure_count >= self.max_xgboost_failures:
            if detection_method == "TST":
                return 20.0  # Strong reward for switching to TST
            elif detection_method == "XGBOOST":
                return -15.0  # Penalty for continuing with failed method
        
        # Reward method diversity
        if detection_method == "TST" and self.xgboost_failure_count > 0:
            return 5.0  # Encourage TST when XGBoost is unreliable
        
        return 0.0
    
    def _calculate_safety_penalties(self, resources: SystemResources, action: int) -> float:
        """Apply critical safety constraints."""
        penalty = 0.0
        
        # SPHINCS safety constraint
        crypto_algo = self._extract_crypto_algorithm_from_action(action)
        if crypto_algo == "SPHINCS_SIGNATURE":
            if resources.thermal_state == ThermalState.CRITICAL:
                penalty -= 100.0  # Never run SPHINCS when overheating
            elif not self._is_mission_phase_safe_for_sphincs():
                penalty -= 200.0  # MASSIVE penalty for SPHINCS during critical flight
        
        # Thermal throttling constraint
        if self.thermal_throttle_active and self._is_high_power_action(action):
            penalty -= 75.0  # Force low-power actions during thermal throttling
        
        # Connection loss constraint
        if self.autonomous_mode and self._requires_swarm_coordination(action):
            penalty -= 50.0  # Penalty for swarm-dependent actions when isolated
        
        return penalty
    
    def update_swarm_state(self, drone_states: List[DroneState]):
        """Update swarm state information."""
        self.swarm_states = {drone.drone_id: drone for drone in drone_states}
        self.last_swarm_update = time.time()
        
        # Check for connection loss
        if not drone_states:
            if self.connection_lost_time is None:
                self.connection_lost_time = time.time()
            
            # Enter autonomous mode after 30 seconds of connection loss
            if time.time() - self.connection_lost_time > 30:
                self.autonomous_mode = True
        else:
            self.connection_lost_time = None
            self.autonomous_mode = False
    
    def handle_xgboost_failure(self):
        """Handle XGBoost detection failure."""
        self.xgboost_failure_count += 1
        
        if self.xgboost_failure_count >= self.max_xgboost_failures:
            self.fallback_detection_active = True
            print(f"[ALERT] XGBoost failure count: {self.xgboost_failure_count}. "
                  f"Activating TST fallback detection.")
    
    def reset_xgboost_failure_count(self):
        """Reset failure count after successful detection."""
        if self.xgboost_failure_count > 0:
            self.xgboost_failure_count = max(0, self.xgboost_failure_count - 1)
            
            if self.xgboost_failure_count == 0:
                self.fallback_detection_active = False
    
    # Helper methods for action interpretation
    def _extract_crypto_algorithm_from_action(self, action: int) -> str:
        """Extract crypto algorithm from action encoding."""
        # Simplified action decoding - adjust based on your action space
        crypto_actions = ["KYBER_HYBRID", "DILITHIUM_SIGNATURE", "SPHINCS_SIGNATURE", "FALCON_SIGNATURE"]
        crypto_idx = (action // 6) % 4  # Assuming 6 actions per crypto type
        return crypto_actions[crypto_idx]
    
    def _extract_detection_method_from_action(self, action: int) -> str:
        """Extract detection method from action."""
        return "XGBOOST" if action % 2 == 0 else "TST"
    
    def _is_high_power_action(self, action: int) -> bool:
        """Check if action involves high power consumption."""
        # Actions involving 1800MHz or SPHINCS are high-power
        return (action % 8) >= 6  # Assuming last 2 actions are high-frequency
    
    def _is_aggressive_detection_action(self, action: int) -> bool:
        """Check if action involves aggressive threat detection."""
        return self._extract_detection_method_from_action(action) == "TST"
    
    def _is_power_efficient_action(self, action: int) -> bool:
        """Check if action is power-efficient."""
        return not self._is_high_power_action(action)
    
    def _is_mission_phase_safe_for_sphincs(self) -> bool:
        """Check if current mission phase allows SPHINCS usage."""
        # Never use SPHINCS during takeoff, landing, or emergency
        return False  # For safety, always return False in this implementation
    
    def _requires_swarm_coordination(self, action: int) -> bool:
        """Check if action requires swarm coordination."""
        # Actions that depend on swarm state
        return (action // 12) == 1  # Assuming certain actions require coordination
    
    def _get_swarm_alert_level(self) -> SwarmAlertLevel:
        """Get current swarm alert level."""
        if not self.swarm_states:
            return SwarmAlertLevel.YELLOW  # Assume caution when no swarm data
        
        # Aggregate alert levels from swarm
        max_alert = SwarmAlertLevel.GREEN
        for drone_state in self.swarm_states.values():
            if drone_state.alert_level.value > max_alert.value:
                max_alert = drone_state.alert_level
        
        return max_alert
    
    def _check_swarm_connection(self) -> bool:
        """Check if connected to swarm."""
        return not self.autonomous_mode and len(self.swarm_states) > 0
    
    # Placeholder methods - implement based on your existing system
    def _get_current_threat_level(self) -> int:
        return 1  # Placeholder
    
    def _get_battery_level_discrete(self) -> int:
        return 2  # Placeholder
    
    def _get_cpu_load_discrete(self) -> int:
        return 1  # Placeholder
    
    def _get_task_priority(self) -> int:
        return 1  # Placeholder
    
    def _calculate_base_reward(self, state: List[int], action: int, next_state: List[int]) -> float:
        return 5.0  # Placeholder - use your existing reward calculation

def get_thermal_aware_frequency_recommendation(temperature: float, algorithm: str, 
                                             battery_level: float) -> int:
    """
    Get frequency recommendation based on thermal state and algorithm requirements.
    
    Args:
        temperature: Current CPU temperature in Celsius
        algorithm: Cryptographic algorithm name
        battery_level: Battery level (0.0-1.0)
    
    Returns:
        Recommended CPU frequency in MHz
    """
    if algorithm not in POST_QUANTUM_FREQUENCY_RULES:
        return 1200  # Default safe frequency
    
    rules = POST_QUANTUM_FREQUENCY_RULES[algorithm]
    optimal_freq = rules["optimal_frequency"]
    thermal_limit_freq = rules["thermal_limit_frequency"]
    min_safe_freq = rules["min_safe_frequency"]
    
    # Thermal constraints override optimal frequency
    if temperature > 80:  # CRITICAL
        return min(min_safe_freq, 1000)  # Emergency throttling
    elif temperature > 70:  # HOT
        return min(thermal_limit_freq, optimal_freq)
    elif temperature > 60:  # WARM
        return thermal_limit_freq
    else:  # OPTIMAL
        # Consider battery level
        if battery_level < 0.3:  # Low battery
            return min(thermal_limit_freq, optimal_freq)
        else:
            return optimal_freq
