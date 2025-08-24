import numpy as np
import os
import sys

# Add parent directory to path and robustly load crypto_config to avoid name clash with root config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from config.crypto_config import CRYPTO_ALGORITHMS, CRYPTO_KPI  # type: ignore
except Exception:
    import runpy
    _cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'crypto_config.py'))
    _data = runpy.run_path(_cfg_path)
    CRYPTO_ALGORITHMS = _data.get('CRYPTO_ALGORITHMS', {})
    CRYPTO_KPI = _data.get('CRYPTO_KPI', {}).copy() if isinstance(_data.get('CRYPTO_KPI', {}), dict) else {}
from crypto_rl.rl_agent import QLearningAgent

class CryptoScheduler:
    """
    A scheduler for cryptographic algorithm selection in UAV applications.
    Uses a trained RL agent to select the optimal cryptographic algorithm
    based on current security needs and resource constraints.
    """
    
    def __init__(self, model_path="../output/crypto_q_table.npy"):
        # State space: [Security Risk, Battery State, Computation, Mission, Communication, Threat]
        self.state_dims = [4, 4, 3, 4, 3, 3]
        self.action_dim = 4  # 4 cryptographic algorithms
        self.agent = QLearningAgent(self.state_dims, self.action_dim)
        
        try:
            self.agent.load_policy(model_path)
            print(f"Loaded crypto policy from {model_path}")
        except FileNotFoundError:
            print(f"Warning: {model_path} not found. Using random policy.")
        
        # Initialize performance tracking
        self.crypto_kpi = CRYPTO_KPI.copy()
        self.algorithm_history = []
    
    def _get_simulated_state(self):
        """
        Manually edit this function to test different scenarios.
        Returns a discrete state: [security_risk, battery_state, computation, mission, communication, threat]
        """
        # Example: Medium security risk scenario
        # Security: MEDIUM (1), Battery: HIGH (3), Computation: NORMAL (1),
        # Mission: MEDIUM (1), Communication: LOW (0), Threat: BENIGN (0)
        return [1, 3, 1, 1, 0, 0]
    
    def select_crypto_algorithm(self, state=None):
        """
        Select the optimal cryptographic algorithm based on current state
        
        Args:
            state: Optional state vector. If None, will use simulated state.
        
        Returns:
            dict: Selected algorithm info including name, latency, etc.
        """
        if state is None:
            state = self._get_simulated_state()
        
        # Use the RL agent to select the best action (algorithm)
        action = self.agent.choose_action(state, training=False)
        
        # Get the selected algorithm
        algorithm = CRYPTO_ALGORITHMS[action]
        
        # Update algorithm history
        self.algorithm_history.append(action)
        if len(self.algorithm_history) > 100:
            self.algorithm_history.pop(0)
        
        # Update KPIs
        self._update_kpi(action, state)
        
        # Format the state for printing
        formatted_state = self._format_state(state)
        
        # Print the decision
        print(f"[Crypto RL] State: {formatted_state}")
        print(f"  → Selected Algorithm: {algorithm['name']}")
        print(f"  → Security Rating: {algorithm['security_rating']}/10")
        print(f"  → Latency: {algorithm['latency_ms']} ms")
        print(f"  → Power Multiplier: {algorithm['power_multiplier']}x")
        
        return algorithm
    
    def _format_state(self, state):
        """Format the state vector for human-readable output"""
        security_risk = ["LOW", "MEDIUM", "HIGH", "CRITICAL"][state[0]]
        battery_state = ["CRITICAL", "LOW", "MEDIUM", "HIGH"][state[1]]
        computation = ["CONSTRAINED", "NORMAL", "ABUNDANT"][state[2]]
        mission = ["LOW", "MEDIUM", "HIGH", "CRITICAL"][state[3]]
        communication = ["LOW", "MEDIUM", "HIGH"][state[4]]
        threat = ["BENIGN", "SUSPICIOUS", "HOSTILE"][state[5]]
        
        return f"Security={security_risk}, Battery={battery_state}, Computation={computation}, Mission={mission}, Comm={communication}, Threat={threat}"
    
    def _update_kpi(self, action, state):
        """Update Key Performance Indicators based on the selected algorithm and state"""
        # Track power saved (compared to SPHINCS which is most power-hungry)
        power_diff = CRYPTO_ALGORITHMS[2]["power_multiplier"] - CRYPTO_ALGORITHMS[action]["power_multiplier"]
        self.crypto_kpi["POWER_SAVED_WH"] += power_diff * 0.1  # Approximate Watt-hours saved
        
        # Track successful encryptions
        self.crypto_kpi["SUCCESSFUL_ENCRYPTIONS"] += 1
        
        # Track average latency
        current_avg = self.crypto_kpi["AVERAGE_LATENCY_MS"]
        n = self.crypto_kpi["SUCCESSFUL_ENCRYPTIONS"]
        new_latency = CRYPTO_ALGORITHMS[action]["latency_ms"]
        self.crypto_kpi["AVERAGE_LATENCY_MS"] = ((n-1) * current_avg + new_latency) / n
        
        # Track algorithm selection consistency (0-100%)
        if len(self.algorithm_history) >= 10:
            # Calculate consistency based on last 10 selections
            recent = self.algorithm_history[-10:]
            most_common = max(set(recent), key=recent.count)
            consistency = recent.count(most_common) / 10 * 100
            self.crypto_kpi["ALGORITHM_SELECTION_CONSISTENCY"] = consistency
        
        # Track security breaches prevented (simulated)
        security_risk = state[0]  # 0-3
        threat_context = state[5]  # 0-2
        algorithm_security = CRYPTO_ALGORITHMS[action]["security_rating"]  # 1-10
        
        # If security matches or exceeds the threat, count as prevented breach
        security_needed = (security_risk * 2 + threat_context) * 0.8  # Scaled to approximate 1-10
        if algorithm_security >= security_needed:
            self.crypto_kpi["SECURITY_BREACHES_PREVENTED"] += 1
        
        # Track adaptability score (how well algorithm matches the situation)
        security_match = min(algorithm_security, security_needed) / max(algorithm_security, security_needed)
        power_match = 1.0 / CRYPTO_ALGORITHMS[action]["power_multiplier"]
        latency_match = 1.0 / (CRYPTO_ALGORITHMS[action]["latency_ms"] / 5)  # Normalize
        
        # Weight factors based on state
        security_weight = 0.6 if threat_context > 0 else 0.3
        power_weight = 0.6 if state[1] < 2 else 0.3  # Higher weight for low battery
        latency_weight = 0.6 if state[3] > 2 else 0.3  # Higher weight for critical mission
        
        # Ensure weights sum to 1
        total_weight = security_weight + power_weight + latency_weight
        security_weight /= total_weight
        power_weight /= total_weight
        latency_weight /= total_weight
        
        adaptability = security_match * security_weight + power_match * power_weight + latency_match * latency_weight
        self.crypto_kpi["ADAPTABILITY_SCORE"] = adaptability * 100  # 0-100%
    
    def get_kpi_report(self):
        """Get a report of the Key Performance Indicators"""
        return self.crypto_kpi
    
    def run_demo(self, scenarios=None):
        """Run a demonstration with different scenarios"""
        print("=== Cryptographic Algorithm Selection RL Demonstration ===")
        
        if scenarios is None:
            # Default test scenarios
            scenarios = [
                # [Security, Battery, Computation, Mission, Communication, Threat]
                [0, 3, 1, 0, 0, 0],  # Low risk, high battery
                [2, 3, 1, 2, 1, 1],  # High risk, high battery, high mission
                [3, 3, 1, 3, 2, 2],  # Critical risk, high battery, critical mission
                [3, 0, 0, 3, 2, 2],  # Critical risk, critical battery
                [1, 1, 1, 0, 0, 0],  # Medium risk, low battery
            ]
        
        for i, state in enumerate(scenarios):
            print(f"\n--- Scenario {i+1} ---")
            self.select_crypto_algorithm(state)
        
        print("\n--- KPI Report ---")
        for key, value in self.get_kpi_report().items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")

if __name__ == "__main__":
    scheduler = CryptoScheduler()
    scheduler.run_demo()
