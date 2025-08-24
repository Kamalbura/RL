"""
Example usage of the cryptographic RL agent
"""

import os
import sys

# Add root directory to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from crypto_rl.crypto_scheduler import CryptoScheduler
from integration.crypto_integration import map_system_state_to_rl_state, crypto_rl_decision_explanation

def main():
    # Create the crypto scheduler
    scheduler = CryptoScheduler(model_path="output/crypto_q_table.npy")
    
    # Example usage: direct state
    print("\n--- Example 1: Direct State ---")
    state = [1, 3, 1, 1, 0, 0]  # Medium risk, high battery, normal computation, etc.
    algorithm = scheduler.select_crypto_algorithm(state)
    
    # Example usage: mapping from UAV system state
    print("\n--- Example 2: Mapping from UAV System State ---")
    # Map from UAV system state to RL state
    threat_level = 2  # CONFIRMING
    battery_percent = 45.0  # 45% battery
    cpu_usage = 65.0  # 65% CPU usage
    flight_mode = 1  # MISSION
    
    mapped_state = map_system_state_to_rl_state(
        threat_level=threat_level,
        battery_percent=battery_percent,
        cpu_usage=cpu_usage,
        flight_mode=flight_mode
    )
    
    print(f"UAV State: threat={threat_level}, battery={battery_percent}%, CPU={cpu_usage}%, flight_mode={flight_mode}")
    print(f"Mapped RL State: {mapped_state}")
    
    algorithm = scheduler.select_crypto_algorithm(mapped_state)
    
    # Example usage: get explanation for decision
    print("\n--- Example 3: Decision Explanation ---")
    explanation = crypto_rl_decision_explanation(mapped_state, algorithm)
    print(explanation)
    
    # Example usage: run built-in demo with scenarios
    print("\n--- Example 4: Built-in Demo ---")
    scheduler.run_demo()
    
    # Example usage: get KPI report
    print("\n--- Example 5: KPI Report ---")
    kpi_report = scheduler.get_kpi_report()
    for key, value in kpi_report.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()
