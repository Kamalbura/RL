"""
GCS Strategic Agent Integration

This module provides the integration layer between the GCS UI and the
strategic RL agent for fleet-wide cryptographic policy management.
"""

import os
import time
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any, Optional

# Import strategic RL components
from crypto_rl.strategic_agent import QLearningAgent
from shared.crypto_profiles import MissionPhase

class GCSStrategicIntegration:
    """
    Integrates the strategic RL agent with the GCS scheduler UI.
    """
    
    def __init__(self, mqtt_client, gcs_app):
        self.mqtt_client = mqtt_client
        self.gcs_app = gcs_app
        self.strategic_agent = None
        self.last_strategic_update = 0
        self.strategic_update_interval = 30  # seconds
        self.strategic_enabled = False
        
        # Load strategic RL agent
        self._load_strategic_agent()

    def _load_strategic_agent(self):
        state_dims = [3, 3, 4]  # Swarm Threat, Fleet Battery, Mission Phase
        action_dim = 4  # 4 crypto policies
        self.strategic_agent = QLearningAgent(state_dims, action_dim)
        
        # Load trained policy if available
        strategic_agent_path = "output/strategic_q_table.npy"
        if os.path.exists(strategic_agent_path):
            self.strategic_agent.load_policy(strategic_agent_path)
            print(f"Loaded strategic RL policy from {strategic_agent_path}")
        else:
            print(f"Strategic RL policy not found at {strategic_agent_path}, using random policy")
        
        self.algorithm_map = ["KYBER", "DILITHIUM", "SPHINCS", "FALCON"]
        self.crypto_code_map = {
            "KYBER": "c1",
            "DILITHIUM": "c2", 
            "SPHINCS": "c3",
            "FALCON": "c4"
        }
        
        # Human-in-the-loop recommendation tracking
        self.last_recommendation = None
        self.recommendation_accepted = False

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
                
                # Store recommendation for human review
                self.last_recommendation = {
                    'algorithm': algorithm_name,
                    'code': crypto_code,
                    'state': state,
                    'timestamp': time.time()
                }
                
                # Show recommendation dialog for human approval
                self._show_recommendation_dialog(algorithm_name, crypto_code)
            else:
                self.gcs_app._log(f"RL Agent recommended '{algorithm_name}', but no matching code found.")

        except Exception as e:
            self.gcs_app._log(f"Error getting RL recommendation: {e}")

    def _build_state_from_fleet(self) -> list:
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

    def _show_recommendation_dialog(self, algorithm_name: str, crypto_code: str):
        """
        Show human-in-the-loop recommendation dialog.
        
        Args:
            algorithm_name: Name of recommended algorithm
            crypto_code: Crypto code to apply
        """
        result = messagebox.askyesno(
            "Strategic RL Recommendation",
            f"RL Agent recommends switching to:\n\n"
            f"Algorithm: {algorithm_name}\n"
            f"Code: {crypto_code}\n\n"
            f"Apply this recommendation to the fleet?",
            icon="question"
        )
        
        if result:
            self.recommendation_accepted = True
            self.gcs_app._log(f"Human operator accepted RL recommendation: {algorithm_name}")
            # Apply the crypto change
            if hasattr(self.gcs_app, '_apply_crypto'):
                self.gcs_app._apply_crypto()
            elif hasattr(self.gcs_app, '_send_crypto'):
                self.gcs_app._send_crypto()
        else:
            self.recommendation_accepted = False
            self.gcs_app._log(f"Human operator rejected RL recommendation: {algorithm_name}")

    def periodic_strategic_update(self):
        """
        Periodic update function to be called by GCS main loop.
        Gathers state and provides recommendations every 30 seconds.
        """
        if not self.gcs_app.use_strategic_rl.get():
            return
            
        try:
            # Check if enough time has passed since last recommendation
            if (self.last_recommendation and 
                time.time() - self.last_recommendation['timestamp'] < 30):
                return
                
            # Build current state
            current_state = self._build_state_from_fleet()
            
            # Get recommendation from agent
            action = self.strategic_agent.choose_action(current_state, training=False)
            algorithm_name = self.algorithm_map[action]
            crypto_code = self.crypto_code_map.get(algorithm_name)
            
            # Only show recommendation if it's different from current
            current_crypto = getattr(self.gcs_app, 'current_crypto_selection', None)
            if crypto_code and crypto_code != current_crypto:
                self.gcs_app._log(f"Periodic RL check suggests: {algorithm_name}")
                # Store but don't auto-apply - wait for manual trigger
                self.last_recommendation = {
                    'algorithm': algorithm_name,
                    'code': crypto_code,
                    'state': current_state,
                    'timestamp': time.time()
                }
                
        except Exception as e:
            self.gcs_app._log(f"Error in periodic strategic update: {e}")
