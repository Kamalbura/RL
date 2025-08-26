"""
System Coordinator for Dual-Agent RL UAV System
Integrates tactical and strategic agents with hardware and communication layers
"""

import time
import threading
import logging
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

# Import system components
try:
    from hardware.rpi_interface import get_hardware_interface, BatteryMonitor, SystemHealthMonitor
    from communication.mavlink_interface import (
        MAVLinkInterface, UAVCoordinator, MessageType,
        TacticalStatus, StrategicCommand, ThreatAlert, SystemHealth
    )
    from ddos_rl.agent import QLearningAgent
    from ddos_rl.env import TacticalUAVEnv
    from crypto_rl.strategic_agent import StrategicCryptoEnv, QLearningAgent as StrategicAgent
except ImportError as e:
    logging.warning(f"Import warning: {e}")

@dataclass
class SystemState:
    """Current system state for coordination"""
    timestamp: float
    tactical_state: Optional[Tuple] = None
    strategic_state: Optional[Tuple] = None
    hardware_metrics: Optional[Dict] = None
    communication_status: Optional[Dict] = None
    active_threats: int = 0
    system_health: str = "UNKNOWN"

class TacticalAgentController:
    """Controller for tactical UAV agent with hardware integration"""
    
    def __init__(self, uav_id: str):
        self.uav_id = uav_id
        self.logger = logging.getLogger(f"{__name__}.tactical.{uav_id}")
        
        # Initialize components
        self.env = TacticalUAVEnv()
        self.agent = QLearningAgent(
            state_dims=(4, 4, 3, 3),
            action_dim=9,
            learning_rate=0.1,
            discount_factor=0.99,
            exploration_rate=0.1,  # Lower for deployment
            exploration_decay=0.999,
            min_exploration_rate=0.01
        )
        
        # Hardware interfaces
        self.hardware = get_hardware_interface()
        self.battery = BatteryMonitor()
        self.health_monitor = SystemHealthMonitor()
        
        # Communication
        self.mavlink = MAVLinkInterface(uav_id, "UAV")
        self.mavlink.register_handler(MessageType.STRATEGIC_COMMAND, self._handle_strategic_command)
        
        # State tracking
        self.current_state = None
        self.last_action = None
        self.current_crypto_algorithm = "KYBER"
        self.running = False
        
    def start(self):
        """Start tactical agent controller"""
        self.mavlink.start()
        self.running = True
        
        # Start main control loop
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.status_thread = threading.Thread(target=self._status_loop, daemon=True)
        
        self.control_thread.start()
        self.status_thread.start()
        
        self.logger.info(f"Tactical agent controller started for {self.uav_id}")
        
    def stop(self):
        """Stop tactical agent controller"""
        self.running = False
        self.mavlink.stop()
        self.logger.info(f"Tactical agent controller stopped for {self.uav_id}")
        
    def load_policy(self, policy_path: str) -> bool:
        """Load trained policy"""
        try:
            self.agent.load_policy(policy_path)
            self.logger.info(f"Policy loaded from {policy_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load policy: {e}")
            return False
            
    def _control_loop(self):
        """Main tactical control loop"""
        while self.running:
            try:
                # Get current system state
                state = self._get_current_state()
                if state is None:
                    time.sleep(1.0)
                    continue
                    
                # Get action from agent
                action = self.agent.choose_action(state, training=False)
                
                # Execute action
                self._execute_action(action)
                
                # Update state
                self.current_state = state
                self.last_action = action
                
                time.sleep(5.0)  # 5-second control cycle
                
            except Exception as e:
                self.logger.error(f"Control loop error: {e}")
                time.sleep(5.0)
                
    def _status_loop(self):
        """Status reporting loop"""
        while self.running:
            try:
                # Send tactical status update
                self._send_status_update()
                time.sleep(10.0)  # 10-second status updates
                
            except Exception as e:
                self.logger.error(f"Status loop error: {e}")
                time.sleep(10.0)
                
    def _get_current_state(self) -> Optional[Tuple]:
        """Get current tactical state from hardware"""
        try:
            # Get hardware metrics
            metrics = self.hardware.get_system_metrics()
            battery_pct = self.battery.get_battery_percentage()
            
            if battery_pct is None:
                return None
                
            # Map to discrete state space
            # Threat level (would come from threat detection system)
            threat_level = self._assess_threat_level()
            
            # Battery state (4 levels)
            if battery_pct > 75:
                battery_state = 3  # HIGH
            elif battery_pct > 50:
                battery_state = 2  # MEDIUM
            elif battery_pct > 25:
                battery_state = 1  # LOW
            else:
                battery_state = 0  # CRITICAL
                
            # CPU load (3 levels)
            cpu_percent = metrics.get('cpu_percent', 0)
            if cpu_percent > 80:
                cpu_load = 2  # HIGH
            elif cpu_percent > 40:
                cpu_load = 1  # MEDIUM
            else:
                cpu_load = 0  # LOW
                
            # Task priority (would come from mission system)
            task_priority = self._get_task_priority()
            
            return (threat_level, battery_state, cpu_load, task_priority)
            
        except Exception as e:
            self.logger.error(f"Error getting current state: {e}")
            return None
            
    def _execute_action(self, action: int):
        """Execute tactical action on hardware"""
        try:
            if action == 8:  # De-escalate action
                self.logger.info("De-escalating DDoS detection")
                return
                
            # Decode action
            model_idx = action // 4
            freq_idx = action % 4
            
            model = ["XGBOOST", "TST"][model_idx]
            frequencies = [600, 1200, 1800, 2000]
            frequency = frequencies[freq_idx]
            
            # Set CPU frequency
            success = self.hardware.set_cpu_frequency(frequency)
            if success:
                self.logger.info(f"Set CPU frequency to {frequency}MHz for {model} model")
            else:
                self.logger.error(f"Failed to set CPU frequency to {frequency}MHz")
                
        except Exception as e:
            self.logger.error(f"Error executing action: {e}")
            
    def _send_status_update(self):
        """Send tactical status update via MAVLink"""
        try:
            metrics = self.hardware.get_system_metrics()
            battery_pct = self.battery.get_battery_percentage() or 0
            
            status = TacticalStatus(
                timestamp=time.time(),
                uav_id=self.uav_id,
                threat_level=self._assess_threat_level(),
                battery_percentage=battery_pct,
                cpu_load=metrics.get('cpu_percent', 0),
                current_model="TST",  # Would track actual model
                current_frequency=metrics.get('cpu_frequency_mhz', 0),
                detection_count=0,  # Would track detections
                power_consumption=metrics.get('power_estimate', 0),
                temperature=metrics.get('cpu_temperature_c', 0)
            )
            
            self.mavlink.send_tactical_status(status)
            
        except Exception as e:
            self.logger.error(f"Error sending status update: {e}")
            
    def _handle_strategic_command(self, message: Dict):
        """Handle strategic command from GCS"""
        try:
            data = message['data']
            crypto_algorithm = data.get('crypto_algorithm')
            
            if crypto_algorithm:
                self.current_crypto_algorithm = crypto_algorithm
                self.logger.info(f"Updated crypto algorithm to {crypto_algorithm}")
                
        except Exception as e:
            self.logger.error(f"Error handling strategic command: {e}")
            
    def _assess_threat_level(self) -> int:
        """Assess current threat level (placeholder)"""
        # In practice, this would interface with threat detection system
        return 1  # MEDIUM threat level
        
    def _get_task_priority(self) -> int:
        """Get current task priority (placeholder)"""
        # In practice, this would interface with mission planning system
        return 1  # MEDIUM priority


class StrategicAgentController:
    """Controller for strategic GCS agent"""
    
    def __init__(self, gcs_id: str = "GCS_MAIN"):
        self.gcs_id = gcs_id
        self.logger = logging.getLogger(f"{__name__}.strategic.{gcs_id}")
        
        # Initialize components
        self.env = StrategicCryptoEnv()
        self.agent = StrategicAgent(
            state_dims=(3, 3, 4),
            action_dim=4,
            learning_rate=0.1,
            discount_factor=0.99,
            exploration_rate=0.05,  # Lower for deployment
            exploration_decay=0.999,
            min_exploration_rate=0.01
        )
        
        # Communication and coordination
        self.coordinator = UAVCoordinator(gcs_id)
        
        # State tracking
        self.fleet_status = {}
        self.current_state = None
        self.last_action = None
        self.running = False
        
    def start(self):
        """Start strategic agent controller"""
        self.coordinator.start()
        self.running = True
        
        # Start main control loop
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        
        self.logger.info(f"Strategic agent controller started for {self.gcs_id}")
        
    def stop(self):
        """Stop strategic agent controller"""
        self.running = False
        self.coordinator.stop()
        self.logger.info(f"Strategic agent controller stopped for {self.gcs_id}")
        
    def load_policy(self, policy_path: str) -> bool:
        """Load trained policy"""
        try:
            self.agent.load_policy(policy_path)
            self.logger.info(f"Policy loaded from {policy_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load policy: {e}")
            return False
            
    def _control_loop(self):
        """Main strategic control loop"""
        while self.running:
            try:
                # Get current fleet state
                state = self._get_current_state()
                if state is None:
                    time.sleep(10.0)
                    continue
                    
                # Get action from agent
                action = self.agent.choose_action(state, training=False)
                
                # Execute action (broadcast crypto algorithm)
                self._execute_action(action)
                
                # Update state
                self.current_state = state
                self.last_action = action
                
                time.sleep(30.0)  # 30-second strategic control cycle
                
            except Exception as e:
                self.logger.error(f"Strategic control loop error: {e}")
                time.sleep(30.0)
                
    def _get_current_state(self) -> Optional[Tuple]:
        """Get current strategic state from fleet"""
        try:
            fleet_status = self.coordinator.get_fleet_status()
            
            # Map to discrete state space
            # Threat level (3 levels)
            threat_level = min(2, fleet_status.get('threat_level', 0))
            
            # Average battery (3 levels)
            avg_battery = fleet_status.get('average_battery', 100)
            if avg_battery > 66:
                battery_state = 2  # HEALTHY
            elif avg_battery > 33:
                battery_state = 1  # MEDIUM
            else:
                battery_state = 0  # CRITICAL
                
            # Mission phase (4 levels)
            mission_phase = self._get_mission_phase()
            
            return (threat_level, battery_state, mission_phase)
            
        except Exception as e:
            self.logger.error(f"Error getting strategic state: {e}")
            return None
            
    def _execute_action(self, action: int):
        """Execute strategic action (crypto algorithm selection)"""
        try:
            algorithms = ["KYBER", "DILITHIUM", "SPHINCS", "FALCON"]
            selected_algorithm = algorithms[action]
            
            # Broadcast to fleet
            success_count = self.coordinator.broadcast_crypto_update(selected_algorithm)
            self.logger.info(f"Broadcasted {selected_algorithm} to {success_count} UAVs")
            
        except Exception as e:
            self.logger.error(f"Error executing strategic action: {e}")
            
    def _get_mission_phase(self) -> int:
        """Get current mission phase (placeholder)"""
        # In practice, this would interface with mission planning system
        return 1  # NORMAL phase


class SystemCoordinator:
    """Main system coordinator integrating all components"""
    
    def __init__(self, uav_id: str = "UAV_001", gcs_id: str = "GCS_MAIN"):
        self.uav_id = uav_id
        self.gcs_id = gcs_id
        self.logger = logging.getLogger(__name__)
        
        # Initialize controllers
        self.tactical_controller = TacticalAgentController(uav_id)
        self.strategic_controller = StrategicAgentController(gcs_id)
        
        # System state
        self.system_state = SystemState(timestamp=time.time())
        self.running = False
        
    def start(self, tactical_policy: Optional[str] = None, 
              strategic_policy: Optional[str] = None):
        """Start complete system"""
        try:
            # Load policies if provided
            if tactical_policy:
                self.tactical_controller.load_policy(tactical_policy)
            if strategic_policy:
                self.strategic_controller.load_policy(strategic_policy)
                
            # Start controllers
            self.tactical_controller.start()
            self.strategic_controller.start()
            
            self.running = True
            self.logger.info("System coordinator started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start system coordinator: {e}")
            self.stop()
            
    def stop(self):
        """Stop complete system"""
        self.running = False
        
        try:
            self.tactical_controller.stop()
            self.strategic_controller.stop()
            self.logger.info("System coordinator stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping system coordinator: {e}")
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'timestamp': time.time(),
            'running': self.running,
            'uav_id': self.uav_id,
            'gcs_id': self.gcs_id,
            'tactical_status': {
                'running': self.tactical_controller.running,
                'current_state': self.tactical_controller.current_state,
                'last_action': self.tactical_controller.last_action
            },
            'strategic_status': {
                'running': self.strategic_controller.running,
                'current_state': self.strategic_controller.current_state,
                'last_action': self.strategic_controller.last_action
            },
            'communication': {
                'tactical_mavlink': self.tactical_controller.mavlink.get_connection_status(),
                'strategic_coordinator': self.strategic_controller.coordinator.get_fleet_status()
            }
        }


# Global system coordinator
_system_coordinator = None

def get_system_coordinator() -> SystemCoordinator:
    """Get singleton system coordinator"""
    global _system_coordinator
    if _system_coordinator is None:
        _system_coordinator = SystemCoordinator()
    return _system_coordinator
