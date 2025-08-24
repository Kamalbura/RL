"""
GCS Strategic Agent Integration

This module provides the integration layer between the GCS UI and the
strategic RL agent for fleet-wide cryptographic policy management.
"""

import os
import numpy as np
import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, Optional

# Assuming rl_agent and other necessary components are in the project structure
from rl_agent import QLearningAgent

class GCSStrategicIntegration:
    """
    Integrates the strategic RL agent with the GCS scheduler UI.
    """
    
    def __init__(self, gcs_app, strategic_agent_path: str = "output/strategic_q_table.npy"):
        """
        Initialize integration between GCS UI and strategic agent.
        
        Args:
            gcs_app: The main GCS scheduler application instance.
            strategic_agent_path: Path to the trained strategic agent's Q-table.
        """
        self.gcs_app = gcs_app
        self.strategic_agent_path = strategic_agent_path
        
        # Initialize the strategic agent
        state_dims = [3, 3, 4]  # Swarm Threat, Fleet Battery, Mission Phase
        action_dim = 4  # 4 crypto policies
        self.strategic_agent = QLearningAgent(state_dims, action_dim)
        self.strategic_agent.load_policy(self.strategic_agent_path)
        
        self.algorithm_map = ["ASCON_128", "KYBER_CRYPTO", "SPHINCS", "FALCON512"]
        self.crypto_code_map = {
            "ASCON_128": "c1",
            "KYBER_CRYPTO": "c2",
            "SPHINCS": "c8", # Mapped to AES-256-GCM as a high-security option
            "FALCON512": "c4"
        }

    def setup_ui_controls(self, parent_frame):
        """
        Add RL controls to the GCS UI.
        
        Args:
            parent_frame: The parent tkinter frame to add controls to.
        """
        lf_rl = ttk.LabelFrame(parent_frame, text="Strategic RL Agent", padding=8)
        lf_rl.pack(fill=tk.X, padx=8, pady=6)
        
        self.gcs_app.use_strategic_rl = tk.BooleanVar(value=True)
        ttk.Checkbutton(lf_rl, text="Enable Strategic RL", variable=self.gcs_app.use_strategic_rl).pack(side=tk.LEFT, padx=4)
        
        ttk.Button(lf_rl, text="Get RL Recommendation", command=self.make_strategic_decision).pack(side=tk.LEFT, padx=6)

    def make_strategic_decision(self):
        """
        Use the strategic agent to select a crypto algorithm and update the UI.
        """
        if not self.gcs_app.use_strategic_rl.get():
            self.gcs_app._log("Strategic RL agent is disabled.")
            return

        try:
            # Build state from the fleet data available in the GCS app
            state = self._build_state_from_fleet()
            
            # Get the best action from the agent
            action = self.strategic_agent.choose_action(state, training=False)
            
            # Get algorithm name and code
            algorithm_name = self.algorithm_map[action]
            crypto_code = self.crypto_code_map.get(algorithm_name)
            
            if crypto_code:
                self.gcs_app._log(f"RL Agent recommends: {algorithm_name} (code {crypto_code})")
                
                # Find the corresponding item in the combobox and select it
                for i, item in enumerate(self.gcs_app.crypto_combo['values']):
                    if item.startswith(crypto_code):
                        self.gcs_app.crypto_combo.current(i)
                        break
                
                # Automatically apply the decision if the option is checked
                if self.gcs_app.auto_local_crypto.get():
                    self.gcs_app._apply_crypto()
            else:
                self.gcs_app._log(f"RL Agent recommended '{algorithm_name}', but no matching code found.")

        except Exception as e:
            self.gcs_app._log(f"Error getting RL recommendation: {e}")

    def _build_state_from_fleet(self) -> list[int]:
        """
        Build the state vector for the strategic agent from fleet data.
        
        Returns:
            A list of integers representing the state.
        """
        # 1. Swarm Threat Level
        threat_levels = [drone.get('threat_level', 0) for drone in self.gcs_app.drones.values()]
        max_threat = max(threat_levels) if threat_levels else 0
        
        swarm_threat_level_idx = 0
        if max_threat >= 3: # CONFIRMED
             swarm_threat_level_idx = 2 # CRITICAL
        elif max_threat >= 1: # POTENTIAL
            swarm_threat_level_idx = 1 # CAUTION

        # 2. Fleet Battery Status
        battery_levels = [drone.get('battery', 100) for drone in self.gcs_app.drones.values()]
        avg_battery = np.mean(battery_levels) if battery_levels else 100
        
        fleet_battery_status_idx = 0 # HEALTHY
        if avg_battery < 30:
            fleet_battery_status_idx = 2 # CRITICAL
        elif avg_battery < 60:
            fleet_battery_status_idx = 1 # DEGRADING

        # 3. Mission Phase (assuming a simple mapping for now)
        # This can be enhanced with more detailed mission state from the GCS
        mission_phase_idx = 0 # IDLE
        if any(d.get('flight_mode') == 'MISSION' for d in self.gcs_app.drones.values()):
            mission_phase_idx = 2 # LOITER_ON_TARGET
        elif any(d.get('flight_mode') == 'RTL' for d in self.gcs_app.drones.values()):
            mission_phase_idx = 3 # EGRESS

        state = [swarm_threat_level_idx, fleet_battery_status_idx, mission_phase_idx]
        self.gcs_app._log(f"Built strategic state: {state}")
        return state
