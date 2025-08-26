"""
Swarm-Aware Decision Making for UAV Networks
Handles distributed coordination, connection loss scenarios, and priority re-evaluation
"""

import numpy as np
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import socket
import queue

class ConnectionStatus(Enum):
    CONNECTED = 0
    DEGRADED = 1      # Intermittent connection
    LOST = 2          # No connection
    AUTONOMOUS = 3    # Operating independently

class PriorityLevel(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3
    EMERGENCY = 4

@dataclass
class SwarmDroneStatus:
    drone_id: str
    position: Tuple[float, float, float]  # x, y, altitude
    battery_level: float
    threat_detected: bool
    detection_method: str  # "XGBOOST", "TST", "NONE"
    detection_confidence: float
    alert_level: int  # 0-3
    connection_quality: float  # 0.0-1.0
    last_update: float
    autonomous_mode: bool
    
class SwarmCoordinator:
    """
    Manages swarm-wide coordination and handles connection loss scenarios.
    Implements distributed decision making with fallback strategies.
    """
    
    def __init__(self, drone_id: str, swarm_size: int = 5):
        self.drone_id = drone_id
        self.swarm_size = swarm_size
        
        # Swarm state management
        self.swarm_status: Dict[str, SwarmDroneStatus] = {}
        self.connection_status = ConnectionStatus.CONNECTED
        self.last_swarm_update = time.time()
        
        # Priority management
        self.current_priority = PriorityLevel.MEDIUM
        self.priority_history = []
        
        # XGBoost failure tracking across swarm
        self.swarm_xgboost_failures: Dict[str, int] = {}
        self.global_detection_reliability = 1.0
        
        # Connection loss handling
        self.connection_lost_timestamp = None
        self.autonomous_decision_count = 0
        self.reconnection_attempts = 0
        
        # Distributed consensus
        self.consensus_threshold = 0.6  # 60% agreement needed
        self.voting_window = 30  # seconds
        self.pending_votes: Dict[str, List] = {}
        
        # Communication threads
        self.communication_active = True
        self.message_queue = queue.Queue()
        
    def update_swarm_status(self, drone_statuses: List[SwarmDroneStatus]):
        """Update swarm status with latest drone information."""
        current_time = time.time()
        
        # Update known drones
        for status in drone_statuses:
            self.swarm_status[status.drone_id] = status
            
            # Track XGBoost failures
            if status.detection_method == "XGBOOST" and not status.threat_detected:
                self.swarm_xgboost_failures[status.drone_id] = \
                    self.swarm_xgboost_failures.get(status.drone_id, 0) + 1
            elif status.threat_detected:
                # Reset failure count on successful detection
                self.swarm_xgboost_failures[status.drone_id] = 0
        
        # Check for missing drones (connection loss)
        missing_drones = []
        for drone_id, status in self.swarm_status.items():
            if current_time - status.last_update > 60:  # 1 minute timeout
                missing_drones.append(drone_id)
        
        # Handle connection degradation
        if missing_drones:
            self._handle_connection_loss(missing_drones)
        
        self.last_swarm_update = current_time
        self._update_global_detection_reliability()
    
    def _handle_connection_loss(self, missing_drones: List[str]):
        """Handle scenarios where connection to other drones is lost."""
        missing_ratio = len(missing_drones) / len(self.swarm_status)
        
        if missing_ratio > 0.7:  # Lost connection to majority
            self.connection_status = ConnectionStatus.LOST
            if self.connection_lost_timestamp is None:
                self.connection_lost_timestamp = time.time()
                print(f"[SWARM] Drone {self.drone_id}: Lost connection to {len(missing_drones)} drones. "
                      f"Entering autonomous mode.")
            
            # Enter autonomous mode after 30 seconds
            if time.time() - self.connection_lost_timestamp > 30:
                self.connection_status = ConnectionStatus.AUTONOMOUS
                self._activate_autonomous_mode()
                
        elif missing_ratio > 0.3:  # Degraded connection
            self.connection_status = ConnectionStatus.DEGRADED
            self._handle_degraded_connection()
        else:
            self.connection_status = ConnectionStatus.CONNECTED
            self.connection_lost_timestamp = None
    
    def _activate_autonomous_mode(self):
        """Activate autonomous decision making when isolated."""
        self.autonomous_decision_count += 1
        
        # Re-evaluate priorities for autonomous operation
        self.current_priority = self._calculate_autonomous_priority()
        
        print(f"[AUTONOMOUS] Drone {self.drone_id}: Operating independently. "
              f"Priority level: {self.current_priority.name}")
        
        # Increase detection sensitivity when isolated
        self._increase_detection_sensitivity()
    
    def _calculate_autonomous_priority(self) -> PriorityLevel:
        """Calculate priority level for autonomous operation."""
        # Get current drone status
        battery_level = self._get_current_battery_level()
        threat_detected = self._get_current_threat_status()
        
        # Autonomous priority logic
        if battery_level < 0.2:  # Critical battery
            return PriorityLevel.EMERGENCY
        elif threat_detected:
            return PriorityLevel.CRITICAL
        elif battery_level < 0.4:  # Low battery
            return PriorityLevel.HIGH
        else:
            return PriorityLevel.MEDIUM
    
    def _update_global_detection_reliability(self):
        """Update global detection reliability based on swarm XGBoost failures."""
        if not self.swarm_xgboost_failures:
            self.global_detection_reliability = 1.0
            return
        
        total_failures = sum(self.swarm_xgboost_failures.values())
        total_drones = len(self.swarm_status)
        
        # Calculate reliability (0.0 = all failing, 1.0 = none failing)
        failure_rate = total_failures / (total_drones * 10)  # Normalize to 10 max failures
        self.global_detection_reliability = max(0.1, 1.0 - failure_rate)
        
        # Trigger swarm-wide TST fallback if reliability drops below 50%
        if self.global_detection_reliability < 0.5:
            self._trigger_swarm_tst_fallback()
    
    def _trigger_swarm_tst_fallback(self):
        """Trigger swarm-wide switch to TST detection."""
        message = {
            "type": "DETECTION_FALLBACK",
            "source": self.drone_id,
            "method": "TST",
            "reason": f"XGBoost reliability: {self.global_detection_reliability:.2f}",
            "timestamp": time.time()
        }
        
        self._broadcast_swarm_message(message)
        print(f"[SWARM] Drone {self.drone_id}: Triggered swarm-wide TST fallback. "
              f"XGBoost reliability: {self.global_detection_reliability:.2f}")
    
    def get_swarm_aware_action_weights(self) -> Dict[str, float]:
        """
        Get action weights based on swarm status and connection state.
        
        Returns:
            Dictionary of action type weights for RL agent
        """
        weights = {
            "detection_aggressive": 1.0,
            "detection_conservative": 1.0,
            "crypto_high_security": 1.0,
            "crypto_balanced": 1.0,
            "power_save": 1.0,
            "performance": 1.0
        }
        
        # Adjust weights based on connection status
        if self.connection_status == ConnectionStatus.AUTONOMOUS:
            # Increase conservative actions when isolated
            weights["detection_conservative"] *= 1.5
            weights["power_save"] *= 1.3
            weights["crypto_balanced"] *= 1.2
            
        elif self.connection_status == ConnectionStatus.CONNECTED:
            # Increase coordinated actions when connected
            swarm_alert_level = self._get_max_swarm_alert_level()
            
            if swarm_alert_level >= 3:  # High threat across swarm
                weights["detection_aggressive"] *= 2.0
                weights["crypto_high_security"] *= 1.8
                weights["performance"] *= 1.5
            elif swarm_alert_level <= 1:  # Low threat across swarm
                weights["power_save"] *= 1.4
                weights["crypto_balanced"] *= 1.2
        
        # Adjust for XGBoost reliability
        if self.global_detection_reliability < 0.7:
            weights["detection_aggressive"] *= 1.3  # Favor TST over XGBoost
        
        # Adjust for battery coordination
        avg_swarm_battery = self._get_average_swarm_battery()
        if avg_swarm_battery < 0.4:  # Swarm low on battery
            weights["power_save"] *= 1.6
            weights["crypto_balanced"] *= 1.3
        
        return weights
    
    def vote_on_swarm_decision(self, decision_type: str, proposal: Dict) -> bool:
        """
        Participate in swarm consensus voting.
        
        Args:
            decision_type: Type of decision ("THREAT_RESPONSE", "FORMATION_CHANGE", etc.)
            proposal: Proposed action/decision
            
        Returns:
            True if vote cast successfully
        """
        vote_id = f"{decision_type}_{proposal.get('id', time.time())}"
        
        # Calculate vote based on local state and proposal
        vote = self._calculate_vote(decision_type, proposal)
        
        vote_message = {
            "type": "CONSENSUS_VOTE",
            "vote_id": vote_id,
            "voter": self.drone_id,
            "vote": vote,  # True/False
            "confidence": self._calculate_vote_confidence(proposal),
            "timestamp": time.time()
        }
        
        # Store vote locally
        if vote_id not in self.pending_votes:
            self.pending_votes[vote_id] = []
        self.pending_votes[vote_id].append(vote_message)
        
        # Broadcast vote to swarm
        self._broadcast_swarm_message(vote_message)
        
        return True
    
    def _calculate_vote(self, decision_type: str, proposal: Dict) -> bool:
        """Calculate vote on swarm proposal based on local conditions."""
        if decision_type == "THREAT_RESPONSE":
            # Vote based on local threat assessment
            local_threat = self._get_current_threat_status()
            proposed_response = proposal.get("response_level", 0)
            
            # Vote yes if proposed response matches local assessment
            if local_threat and proposed_response >= 2:
                return True
            elif not local_threat and proposed_response <= 1:
                return True
            else:
                return False
                
        elif decision_type == "FORMATION_CHANGE":
            # Vote based on battery and position
            battery_level = self._get_current_battery_level()
            if battery_level < 0.3:
                return False  # Don't support formation changes when low battery
            return True
            
        elif decision_type == "DETECTION_METHOD":
            # Vote based on XGBoost reliability
            proposed_method = proposal.get("method", "XGBOOST")
            if proposed_method == "TST" and self.global_detection_reliability < 0.6:
                return True
            elif proposed_method == "XGBOOST" and self.global_detection_reliability > 0.8:
                return True
            return False
        
        return False  # Default: abstain
    
    def _calculate_vote_confidence(self, proposal: Dict) -> float:
        """Calculate confidence in vote decision."""
        base_confidence = 0.7
        
        # Increase confidence based on local data quality
        if self.connection_status == ConnectionStatus.CONNECTED:
            base_confidence += 0.2
        
        # Decrease confidence if operating autonomously
        if self.connection_status == ConnectionStatus.AUTONOMOUS:
            base_confidence -= 0.3
        
        return max(0.1, min(1.0, base_confidence))
    
    def check_consensus_reached(self, vote_id: str) -> Optional[bool]:
        """Check if consensus has been reached on a vote."""
        if vote_id not in self.pending_votes:
            return None
        
        votes = self.pending_votes[vote_id]
        if len(votes) < max(2, int(self.swarm_size * 0.5)):  # Need minimum participation
            return None
        
        # Calculate weighted consensus
        yes_votes = sum(1 for vote in votes if vote["vote"])
        total_votes = len(votes)
        
        consensus_ratio = yes_votes / total_votes
        
        if consensus_ratio >= self.consensus_threshold:
            return True
        elif consensus_ratio <= (1 - self.consensus_threshold):
            return False
        
        return None  # No consensus yet
    
    def handle_priority_reevaluation(self) -> PriorityLevel:
        """
        Re-evaluate priority based on current swarm state and connection status.
        Called when connection status changes or major events occur.
        """
        old_priority = self.current_priority
        
        # Base priority calculation
        battery_level = self._get_current_battery_level()
        threat_detected = self._get_current_threat_status()
        
        if self.connection_status == ConnectionStatus.AUTONOMOUS:
            # Autonomous mode - conservative priority
            if battery_level < 0.15:
                new_priority = PriorityLevel.EMERGENCY
            elif threat_detected:
                new_priority = PriorityLevel.CRITICAL
            elif battery_level < 0.3:
                new_priority = PriorityLevel.HIGH
            else:
                new_priority = PriorityLevel.MEDIUM
                
        elif self.connection_status == ConnectionStatus.CONNECTED:
            # Connected mode - consider swarm state
            max_swarm_alert = self._get_max_swarm_alert_level()
            avg_swarm_battery = self._get_average_swarm_battery()
            
            if max_swarm_alert >= 3 or battery_level < 0.15:
                new_priority = PriorityLevel.EMERGENCY
            elif max_swarm_alert >= 2 or threat_detected:
                new_priority = PriorityLevel.CRITICAL
            elif avg_swarm_battery < 0.25 or battery_level < 0.3:
                new_priority = PriorityLevel.HIGH
            else:
                new_priority = PriorityLevel.MEDIUM
        else:
            # Degraded connection - moderate priority
            new_priority = PriorityLevel.HIGH if threat_detected else PriorityLevel.MEDIUM
        
        # Update priority if changed
        if new_priority != old_priority:
            self.current_priority = new_priority
            self.priority_history.append({
                "old_priority": old_priority.name,
                "new_priority": new_priority.name,
                "reason": self._get_priority_change_reason(),
                "timestamp": time.time()
            })
            
            print(f"[PRIORITY] Drone {self.drone_id}: Priority changed from "
                  f"{old_priority.name} to {new_priority.name}")
        
        return new_priority
    
    def _get_priority_change_reason(self) -> str:
        """Get reason for priority change."""
        reasons = []
        
        if self.connection_status == ConnectionStatus.AUTONOMOUS:
            reasons.append("autonomous_mode")
        
        if self._get_current_battery_level() < 0.3:
            reasons.append("low_battery")
        
        if self._get_current_threat_status():
            reasons.append("threat_detected")
        
        if self.global_detection_reliability < 0.6:
            reasons.append("detection_unreliable")
        
        return ",".join(reasons) if reasons else "routine_evaluation"
    
    # Helper methods for swarm state queries
    def _get_max_swarm_alert_level(self) -> int:
        """Get maximum alert level across swarm."""
        if not self.swarm_status:
            return 1  # Default moderate alert when no swarm data
        
        return max(status.alert_level for status in self.swarm_status.values())
    
    def _get_average_swarm_battery(self) -> float:
        """Get average battery level across swarm."""
        if not self.swarm_status:
            return 0.5  # Default when no swarm data
        
        battery_levels = [status.battery_level for status in self.swarm_status.values()]
        return sum(battery_levels) / len(battery_levels)
    
    def _broadcast_swarm_message(self, message: Dict):
        """Broadcast message to swarm (placeholder for actual implementation)."""
        # In real implementation, this would use UDP multicast or mesh networking
        self.message_queue.put(message)
        print(f"[SWARM] Broadcasting: {message['type']}")
    
    # Placeholder methods - implement based on your system
    def _get_current_battery_level(self) -> float:
        return 0.7  # Placeholder
    
    def _get_current_threat_status(self) -> bool:
        return False  # Placeholder
    
    def _handle_degraded_connection(self):
        """Handle degraded connection scenarios."""
        print(f"[SWARM] Drone {self.drone_id}: Connection degraded. Reducing coordination dependency.")
    
    def _increase_detection_sensitivity(self):
        """Increase detection sensitivity for autonomous operation."""
        print(f"[AUTONOMOUS] Drone {self.drone_id}: Increasing detection sensitivity.")

class SwarmAwareRLAgent:
    """
    RL Agent that integrates swarm coordination into decision making.
    """
    
    def __init__(self, drone_id: str, swarm_size: int = 5):
        self.drone_id = drone_id
        self.swarm_coordinator = SwarmCoordinator(drone_id, swarm_size)
        
        # Enhanced action space with swarm-aware actions
        self.action_space_size = 32  # Expanded for swarm coordination
        
    def choose_swarm_aware_action(self, state: List[int], q_values: np.ndarray) -> int:
        """
        Choose action considering swarm coordination weights.
        
        Args:
            state: Current RL state
            q_values: Q-values for all actions
            
        Returns:
            Selected action index
        """
        # Get swarm-aware weights
        weights = self.swarm_coordinator.get_swarm_aware_action_weights()
        
        # Apply weights to Q-values
        weighted_q_values = q_values.copy()
        
        # Map weights to action indices (simplified mapping)
        for i, q_val in enumerate(q_values):
            action_type = self._classify_action_type(i)
            if action_type in weights:
                weighted_q_values[i] *= weights[action_type]
        
        # Choose action with highest weighted Q-value
        return np.argmax(weighted_q_values)
    
    def _classify_action_type(self, action_idx: int) -> str:
        """Classify action type for weight application."""
        # Simplified action classification - adjust based on your action space
        if action_idx < 8:
            return "detection_aggressive" if action_idx % 2 == 1 else "detection_conservative"
        elif action_idx < 16:
            return "crypto_high_security" if action_idx % 2 == 1 else "crypto_balanced"
        elif action_idx < 24:
            return "performance"
        else:
            return "power_save"
    
    def update_with_swarm_feedback(self, state: List[int], action: int, reward: float, 
                                 next_state: List[int], swarm_feedback: Dict):
        """
        Update RL agent with swarm coordination feedback.
        
        Args:
            state: Previous state
            action: Action taken
            reward: Base reward received
            next_state: Resulting state
            swarm_feedback: Feedback from swarm coordination
        """
        # Adjust reward based on swarm coordination success
        coordination_bonus = 0.0
        
        if swarm_feedback.get("consensus_reached", False):
            coordination_bonus += 5.0
        
        if swarm_feedback.get("swarm_threat_resolved", False):
            coordination_bonus += 10.0
        
        if swarm_feedback.get("coordination_failed", False):
            coordination_bonus -= 5.0
        
        # Apply swarm coordination bonus to reward
        adjusted_reward = reward + coordination_bonus
        
        # Update Q-values with adjusted reward
        # (This would integrate with your existing Q-learning update)
        return adjusted_reward
