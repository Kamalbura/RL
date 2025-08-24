"""
Tactical Agent for UAV DDoS Defense and CPU Management

This module provides a trained RL agent that makes tactical decisions
for an individual UAV, balancing security, performance, and battery life.
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Any

class TacticalAgent:
    """
    Tactical decision-making agent for UAV cybersecurity
    
    This agent uses a trained Q-table to make decisions about:
    1. Which DDoS detection model to run (XGBOOST or TST)
    2. Which CPU frequency to operate at (POWERSAVE, BALANCED, PERFORMANCE, TURBO)
    3. How many cores to utilize (1, 2, or 4)
    """
    
    def __init__(self, q_table_path="output/tactical_q_table.npy"):
        """
        Initialize the tactical agent
        
        Args:
            q_table_path: Path to the trained Q-table
        """
        # State dimensions
        self.state_dims = [4, 4, 3, 3]  # Threat, Battery, CPU Load, Task Priority
        self.action_dim = 24  # 2 models * 4 frequencies * 3 core counts
        
        # Load Q-table
        self.q_table = self._load_q_table(q_table_path)
        
        # CPU frequency and core count mappings
        self.freq_map = ["POWERSAVE", "BALANCED", "PERFORMANCE", "TURBO"]
        self.cores_map = [1, 2, 4]
        self.model_map = ["XGBOOST", "TST"]
        
        # Decision history
        self.decisions = []
    
    def _load_q_table(self, filepath: str) -> np.ndarray:
        """Load the Q-table from a file"""
        if os.path.exists(filepath):
            q_table = np.load(filepath)
            print(f"Tactical agent loaded policy from {filepath}")
            return q_table
        else:
            print(f"Warning: Q-table file {filepath} not found. Using zeros.")
            return np.zeros(self.state_dims + [self.action_dim])
    
    def _state_to_index(self, state: List[int]) -> Tuple[int, ...]:
        """Convert state vector to index tuple for Q-table"""
        return tuple(state)
    
    def _decode_action(self, action: int) -> Tuple[int, int, int]:
        """
        Decode the action into its components
        
        Args:
            action: Integer from 0 to 23
            
        Returns:
            model_idx: 0 for XGBOOST, 1 for TST
            freq_idx: 0 for POWERSAVE, 1 for BALANCED, 2 for PERFORMANCE, 3 for TURBO
            cores_idx: 0 for 1 core, 1 for 2 cores, 2 for 4 cores
        """
        model_idx = action // 12  # 0 or 1
        remaining = action % 12
        freq_idx = remaining // 3  # 0, 1, 2, or 3
        cores_idx = remaining % 3  # 0, 1, or 2
        
        return model_idx, freq_idx, cores_idx
    
    def make_decision(self, threat_level: int, battery_percent: float, 
                     cpu_load: int, task_priority: int) -> Dict[str, Any]:
        """
        Make a tactical decision based on the current UAV state
        
        Args:
            threat_level: 0=NONE, 1=POTENTIAL, 2=CONFIRMING, 3=CRITICAL
            battery_percent: Battery percentage (0-100)
            cpu_load: 0=LOW, 1=NORMAL, 2=HIGH
            task_priority: 0=CRITICAL, 1=HIGH, 2=MEDIUM
            
        Returns:
            decision: Dictionary containing the decision details
        """
        # Map battery percentage to battery state
        if battery_percent < 20:
            battery_state = 0  # CRITICAL
        elif battery_percent < 50:
            battery_state = 1  # LOW
        elif battery_percent < 80:
            battery_state = 2  # MEDIUM
        else:
            battery_state = 3  # HIGH
        
        # Create state vector
        state = [threat_level, battery_state, cpu_load, task_priority]
        
        # Get best action
        state_index = self._state_to_index(state)
        action = np.argmax(self.q_table[state_index])
        q_value = self.q_table[state_index][action]
        
        # Decode action
        model_idx, freq_idx, cores_idx = self._decode_action(action)
        
        # Create decision dict
        decision = {
            "model": self.model_map[model_idx],
            "frequency": self.freq_map[freq_idx],
            "cores": self.cores_map[cores_idx],
            "q_value": q_value,
            "state": state,
            "action": action,
            "confidence": 1.0 if q_value > 0 else 0.5  # Higher confidence for positive Q-values
        }
        
        # Store decision
        self.decisions.append(decision)
        
        return decision
    
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Get the history of decisions made by the agent"""
        return self.decisions
    
    def get_explanation(self, decision: Dict[str, Any]) -> str:
        """
        Get a human-readable explanation for a decision
        
        Args:
            decision: Decision dictionary from make_decision
            
        Returns:
            explanation: Human-readable explanation
        """
        state = decision["state"]
        threat_levels = ["NONE", "POTENTIAL", "CONFIRMING", "CRITICAL"]
        battery_states = ["CRITICAL", "LOW", "MEDIUM", "HIGH"]
        cpu_loads = ["LOW", "NORMAL", "HIGH"]
        task_priorities = ["CRITICAL", "HIGH", "MEDIUM"]
        
        threat = threat_levels[state[0]]
        battery = battery_states[state[1]]
        cpu = cpu_loads[state[2]]
        priority = task_priorities[state[3]]
        
        model = decision["model"]
        frequency = decision["frequency"]
        cores = decision["cores"]
        
        # Build explanation
        explanation = f"Decision: {model}, {frequency} mode, {cores} cores\n"
        explanation += f"State: Threat={threat}, Battery={battery}, CPU Load={cpu}, Priority={priority}\n"
        
        # Add reasoning based on state
        if state[0] >= 2:  # CONFIRMING or CRITICAL threat
            if model == "TST":
                explanation += "- Using TST for high-security threat detection\n"
            else:
                explanation += "- Using XGBOOST for efficient threat detection\n"
        else:
            if model == "XGBOOST":
                explanation += "- Using lightweight XGBOOST for baseline protection\n"
        
        if state[1] <= 1:  # CRITICAL or LOW battery
            if frequency == "POWERSAVE" or frequency == "BALANCED":
                explanation += "- Conserving power due to low battery\n"
        
        if state[2] == 2:  # HIGH CPU load
            if cores > 1:
                explanation += "- Using multiple cores to handle high CPU load\n"
        
        if state[3] == 0:  # CRITICAL task priority
            if frequency in ["PERFORMANCE", "TURBO"]:
                explanation += "- Using high performance for critical task\n"
        
        return explanation
