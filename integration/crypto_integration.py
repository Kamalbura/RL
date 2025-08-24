"""
Crypto RL Integration Utilities

This module provides utilities to integrate the cryptographic RL agent with
the UAV and GCS schedulers.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.crypto_config import CRYPTO_ALGORITHMS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crypto_integration")

def map_system_state_to_rl_state(
    threat_level: int,
    battery_percent: float,
    cpu_usage: float,
    flight_mode: int,
    communication_intensity: Optional[int] = None,
) -> List[int]:
    """
    Maps UAV system state to the RL agent state space
    
    Args:
        threat_level: 0=NONE, 1=POTENTIAL, 2=CONFIRMING, 3=CONFIRMED
        battery_percent: Battery percentage (0-100)
        cpu_usage: CPU usage percentage (0-100)
        flight_mode: 0=LOITER, 1=MISSION, 2=RTL
        communication_intensity: Optional communication intensity override
        
    Returns:
        List of integers representing the RL state
    """
    # Map security risk
    security_risk = 0  # LOW by default
    if threat_level == 1:  # POTENTIAL
        security_risk = 1  # MEDIUM
    elif threat_level == 2:  # CONFIRMING
        security_risk = 2  # HIGH
    elif threat_level >= 3:  # CONFIRMED or higher
        security_risk = 3  # CRITICAL
        
    # Map battery level
    battery_state = 3  # HIGH by default
    if battery_percent < 20:
        battery_state = 0  # CRITICAL
    elif battery_percent < 50:
        battery_state = 1  # LOW
    elif battery_percent < 80:
        battery_state = 2  # MEDIUM
        
    # Map computation capacity
    computation = 1  # NORMAL by default
    if cpu_usage > 70:
        computation = 0  # CONSTRAINED
    elif cpu_usage < 30:
        computation = 2  # ABUNDANT
        
    # Map mission criticality
    mission_criticality = 1  # MEDIUM by default
    if flight_mode == 1:  # MISSION
        mission_criticality = 2  # HIGH
    elif flight_mode == 2:  # RTL
        mission_criticality = 3  # CRITICAL
        
    # Estimate communication intensity if not provided
    comm_intensity = 0  # LOW by default
    if communication_intensity is not None:
        comm_intensity = min(max(0, communication_intensity), 2)
        
    # Map threat context
    threat_context = 0  # BENIGN by default
    if threat_level == 1:  # POTENTIAL
        threat_context = 1  # SUSPICIOUS
    elif threat_level >= 2:  # CONFIRMING or CONFIRMED
        threat_context = 2  # HOSTILE
        
    return [
        security_risk,
        battery_state,
        computation,
        mission_criticality,
        comm_intensity,
        threat_context
    ]

def algorithm_name_to_code(algorithm_name: str, crypto_map: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """
    Maps an algorithm name to the corresponding code in the crypto map
    
    Args:
        algorithm_name: The name of the algorithm (e.g., "FALCON512")
        crypto_map: The crypto map from the config
        
    Returns:
        The code (e.g., "c4") or None if not found
    """
    for code, data in crypto_map.items():
        if data["name"] == algorithm_name:
            return code
            
    # Try partial matching
    for code, data in crypto_map.items():
        if algorithm_name in data["name"] or data["name"] in algorithm_name:
            return code
            
    return None

def crypto_rl_decision_explanation(state: List[int], algorithm: Dict[str, Any]) -> str:
    """
    Generates a human-readable explanation of the RL agent's decision
    
    Args:
        state: The RL state vector
        algorithm: The selected algorithm
        
    Returns:
        A string explaining the decision
    """
    # Map state components to human-readable values
    security_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    battery_states = ["CRITICAL", "LOW", "MEDIUM", "HIGH"]
    computation_levels = ["CONSTRAINED", "NORMAL", "ABUNDANT"]
    mission_criticality = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    communication_levels = ["LOW", "MEDIUM", "HIGH"]
    threat_contexts = ["BENIGN", "SUSPICIOUS", "HOSTILE"]
    
    security = security_levels[state[0]]
    battery = battery_states[state[1]]
    computation = computation_levels[state[2]]
    mission = mission_criticality[state[3]]
    communication = communication_levels[state[4]]
    threat = threat_contexts[state[5]]
    
    algo_name = algorithm["name"]
    security_rating = algorithm["security_rating"]
    latency = algorithm["latency_ms"]
    power = algorithm["power_multiplier"]
    
    explanation = f"Selected {algo_name} based on:\n"
    
    # Security reasoning
    if state[0] >= 2 and security_rating >= 7:
        explanation += f"- High security ({security_rating}/10) for {security} risk\n"
    elif state[0] <= 1 and security_rating <= 6:
        explanation += f"- Adequate security ({security_rating}/10) for {security} risk\n"
        
    # Battery reasoning
    if state[1] <= 1 and power < 1.2:
        explanation += f"- Power efficient (x{power}) for {battery} battery\n"
    
    # Latency reasoning
    if state[3] >= 2 and latency < 5.0:
        explanation += f"- Low latency ({latency}ms) for {mission} mission\n"
        
    # Add any specific considerations
    if state[1] == 0:  # CRITICAL battery
        explanation += "- Battery preservation is highest priority\n"
    if state[5] == 2:  # HOSTILE threat
        explanation += "- Enhanced security due to hostile threat context\n"
        
    return explanation

if __name__ == "__main__":
    # Simple test
    test_state = [2, 1, 1, 2, 0, 2]  # HIGH risk, LOW battery, NORMAL computation, HIGH mission
    test_algorithm = {
        "name": "FALCON512",
        "latency_ms": 0.7,
        "security_rating": 8,
        "power_multiplier": 1.10
    }
    
    print(crypto_rl_decision_explanation(test_state, test_algorithm))
