"""
TacticalUAVEnv (moved from top-level tactical_simulator.py)
"""

import numpy as np
import subprocess
import psutil
from typing import Dict, List, Tuple
try:
	import gym
	from gym import spaces
except ImportError:
	import gymnasium as gym
	from gymnasium import spaces

# Import data-driven profiles
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.crypto_profiles import (
    DDOS_PROFILES, ThermalState, get_ddos_performance,
    is_algorithm_safe_for_thermal_state, get_optimal_frequency_for_thermal_state
)

# Battery specifications - CORRECTED to match context.txt exact hardware
# "Pro Range Lipo, 6 Cells, 22.2 V, 35C" from context.txt
BATTERY_SPECS = {
    "VOLTAGE": 22.2,          # 6S LiPo (6 cells)
    "CAPACITY_MAH": 5200,     # mAh (standard for Pro Range)
    "CAPACITY_WH": 115.44,    # 22.2V * 5.2Ah = 115.44 Wh
    "C_RATING": 35,           # 35C discharge rate
    "MAX_DISCHARGE_A": 182,   # 5.2Ah * 35C = 182A max discharge
    "CELL_COUNT": 6,          # 6S configuration
    "CELL_VOLTAGE_NOMINAL": 3.7,  # 3.7V per cell nominal
    "CELL_VOLTAGE_MAX": 4.2,      # 4.2V per cell fully charged
    "CELL_VOLTAGE_MIN": 3.0,      # 3.0V per cell minimum safe
}

# CPU frequency presets from empirical data
CPU_FREQUENCY_PRESETS = {
    "POWERSAVE": 600,
    "BALANCED": 1200,
    "PERFORMANCE": 1800,
    "TURBO": 1800  # Same as performance for thermal safety
}


class TacticalUAVEnv(gym.Env):
	"""
	A simulator for the tactical UAV environment.
	This environment simulates a single drone making decisions about DDoS detection
	and CPU management to optimize security, performance, and battery life.
	"""

	def __init__(self):
		super(TacticalUAVEnv, self).__init__()

		# Actions: 2 models x 4 CPU presets + 1 de-escalate = 9
		# Mapping:
		# 0..3: XGBOOST @ [POWERSAVE, BALANCED, PERFORMANCE, TURBO]
		# 4..7: TST     @ [POWERSAVE, BALANCED, PERFORMANCE, TURBO]
		# 8:    DE-ESCALATE (no DDoS scanning)
		self.action_space = spaces.Discrete(9)

		# State space: Threat(4), Battery(4), CPU Load(3), Task Priority(3), Thermal State(4)
		self.observation_space = spaces.MultiDiscrete([4, 4, 3, 3, 4])

		# Time and battery model
		self.time_step = 5  # seconds per step
		self.max_steps = 500
		self.capacity_Wh = BATTERY_SPECS["CAPACITY_WH"]

		# Initialize episode
		self.reset()

	def _get_state(self):
		return [
			self.threat_level_idx,
			self.battery_state_idx,
			self.cpu_load_idx,
			self.task_priority_idx,
			self.thermal_state_idx,
		]

	def reset(self):
		# Battery and env state
		self.battery_percentage = 100.0
		self.threat_level_idx = 0  # NONE
		self.battery_state_idx = 3  # HIGH
		self.cpu_load_idx = 1  # NORMAL
		self.task_priority_idx = 1  # MEDIUM
		self.thermal_state_idx = 0  # OPTIMAL
		self.steps = 0

		# Current configuration
		self._cpu_keys = list(CPU_FREQUENCY_PRESETS.keys())
		self.current_model_idx = 0  # XGBOOST
		self.current_freq_idx = 1  # BALANCED
		self.default_cores = 2     # Use 2 cores by default for profiles
		self.cpu_frequency = CPU_FREQUENCY_PRESETS[self._cpu_keys[self.current_freq_idx]]
		self.active_cores = self.default_cores
		return self._get_state()

	def _update_battery_state(self):
		p = self.battery_percentage
		if p < 20:
			self.battery_state_idx = 0  # CRITICAL
		elif p < 50:
			self.battery_state_idx = 1  # LOW
		elif p < 80:
			self.battery_state_idx = 2  # MEDIUM
		else:
			self.battery_state_idx = 3  # HIGH

	def _get_cpu_temperature(self) -> float:
		"""Get RPi CPU temperature with fallback."""
		try:
			result = subprocess.run(['vcgencmd', 'measure_temp'], 
			                      capture_output=True, text=True)
			temp_str = result.stdout.strip()
			temp = float(temp_str.split('=')[1].split("'")[0])
			return temp
		except:
			# Fallback: estimate from CPU usage and frequency
			cpu_percent = psutil.cpu_percent()
			base_temp = 45.0
			load_temp = cpu_percent * 0.4
			freq_temp = (self.cpu_frequency - 600) / 1200 * 15  # Frequency impact
			return base_temp + load_temp + freq_temp

	def _classify_thermal_state(self, temperature: float) -> int:
		"""Classify thermal state based on temperature."""
		if temperature < 60:
			return ThermalState.OPTIMAL.value
		elif temperature < 70:
			return ThermalState.WARM.value
		elif temperature < 80:
			return ThermalState.HOT.value
		else:
			return ThermalState.CRITICAL.value

	def _update_environment(self):
		# Threat drift
		if np.random.random() < 0.1:
			change = np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
			self.threat_level_idx = int(np.clip(self.threat_level_idx + change, 0, 3))
		
		# CPU load derived from preset intensity
		base_load = np.random.normal(0.5, 0.2)
		freq_factor = 1.0 - (self.current_freq_idx / 6.0)
		adjusted = base_load * freq_factor
		if adjusted < 0.3:
			self.cpu_load_idx = 0
		elif adjusted < 0.7:
			self.cpu_load_idx = 1
		else:
			self.cpu_load_idx = 2
		
		# Update thermal state based on current temperature
		current_temp = self._get_cpu_temperature()
		self.thermal_state_idx = self._classify_thermal_state(current_temp)
		
		# Task priority occasional change
		if np.random.random() < 0.05:
			self.task_priority_idx = int(np.random.choice([0, 1, 2]))

	def _calculate_reward(self, model_idx, freq_idx, deescalate: bool):
		reward = 0.0
		reward_breakdown = {}
		
		if deescalate:
			# No scanning: low power is good on low threat/low battery; risky on high threat
			threat_penalty = 0
			battery_bonus = 0
			thermal_bonus = 0
			
			if self.threat_level_idx >= 2:
				threat_penalty = -25  # under-monitoring under high threat
				reward += threat_penalty
			if self.battery_state_idx <= 1:
				battery_bonus = 8   # preserving battery when low
				reward += battery_bonus
			if self.thermal_state_idx >= 2:  # HOT or CRITICAL
				thermal_bonus = 15  # Good thermal management
				reward += thermal_bonus
			
			reward_breakdown = {
				"threat_penalty": threat_penalty,
				"battery_bonus": battery_bonus,
				"thermal_bonus": thermal_bonus,
				"total": reward
			}
			return reward, reward_breakdown

		# Use data-driven profiles
		model = ["XGBOOST", "TST"][model_idx]
		cpu_key = self._cpu_keys[freq_idx]
		cpu_frequency = CPU_FREQUENCY_PRESETS[cpu_key]
		active_cores = self.default_cores

		# Get empirical performance data
		perf_data = get_ddos_performance(model, active_cores, cpu_frequency)
		power_watts = perf_data["power_watts"]
		total_time_ms = perf_data["create_sequence_ms"] + perf_data["prediction_ms"]
		accuracy = perf_data["accuracy"]
		security_rating = perf_data["security_rating"]

		# Thermal safety constraints - CRITICAL PENALTY
		thermal_penalty = 0
		if self.thermal_state_idx == ThermalState.CRITICAL.value:
			if freq_idx >= 2:  # PERFORMANCE or TURBO
				thermal_penalty = -100  # Massive penalty for high freq when critical
			elif freq_idx == 1:  # BALANCED
				thermal_penalty = -50   # Moderate penalty
			else:  # POWERSAVE
				thermal_penalty = 10    # Reward for appropriate response
		elif self.thermal_state_idx == ThermalState.HOT.value:
			if freq_idx >= 2:
				thermal_penalty = -25   # Penalty for high freq when hot
			else:
				thermal_penalty = 5     # Small bonus for thermal awareness
		else:
			thermal_penalty = 0     # No penalty for optimal/warm
		reward += thermal_penalty

		# Energy-conscious term with thermal multiplier
		thermal_multiplier = 1.0 + (self.thermal_state_idx * 0.2)  # Increase penalty with heat
		energy_penalty = 0
		if self.battery_state_idx <= 1:
			energy_penalty = power_watts * 2.0 * thermal_multiplier
		else:
			energy_penalty = power_watts * 0.5 * thermal_multiplier
		reward -= energy_penalty

		# Performance term based on empirical latency
		execution_time_s = total_time_ms / 1000.0
		execution_penalty = np.clip(execution_time_s - 0.1, 0, 1) * 50  # Scale for ms timing
		reward -= execution_penalty
		performance_bonus = 10 if execution_time_s <= 0.1 else 0
		reward += performance_bonus

		# Security term based on empirical accuracy
		security_reward = 0
		if self.threat_level_idx >= 2:
			security_reward = 20 if accuracy >= 0.95 else -15
		else:
			if accuracy >= 0.90:
				security_reward = 5
		reward += security_reward

		# CPU load scaling
		cpu_bonus = 10 if (self.cpu_load_idx == 2 and freq_idx >= 2) else 0
		reward += cpu_bonus

		# Task priority scaling
		priority_bonus = 15 if (self.task_priority_idx == 0 and freq_idx >= 2) else 0
		reward += priority_bonus

		reward_breakdown = {
			"thermal_penalty": thermal_penalty,
			"energy_penalty": -energy_penalty,
			"execution_penalty": -execution_penalty,
			"performance_bonus": performance_bonus,
			"security_reward": security_reward,
			"cpu_bonus": cpu_bonus,
			"priority_bonus": priority_bonus,
			"total": reward
		}

		return reward, reward_breakdown

	def step(self, action):
		assert self.action_space.contains(action), f"Invalid action: {action}"
		# Decode: 0..7 are (model,freq), 8 is de-escalate
		if action == 8:
			deescalate = True
			model_idx = None
			freq_idx = 0  # use POWERSAVE for power computation
		else:
			deescalate = False
			model_idx = action // 4
			freq_idx = action % 4

		# Apply configuration
		self.current_model_idx = model_idx if model_idx is not None else self.current_model_idx
		self.current_freq_idx = freq_idx
		self._cpu_keys = list(CPU_FREQUENCY_PRESETS.keys())
		cpu_key = self._cpu_keys[freq_idx]
		self.cpu_frequency = CPU_FREQUENCY_PRESETS[cpu_key]
		self.active_cores = self.default_cores

		reward, reward_breakdown = self._calculate_reward(model_idx, freq_idx, deescalate)
		
		# Get power consumption from empirical data
		if deescalate:
			power_watts = 3.5  # Base RPi power
		else:
			model_name = ["XGBOOST", "TST"][model_idx]
			perf_data = get_ddos_performance(model_name, self.active_cores, self.cpu_frequency)
			power_watts = perf_data["power_watts"]

		# Battery drain via Wh model
		energy_Wh = power_watts * (self.time_step / 3600.0)
		pct_drain = (energy_Wh / self.capacity_Wh) * 100.0
		self.battery_percentage -= pct_drain

		self._update_battery_state()
		self._update_environment()
		self.steps += 1

		done = False
		if self.battery_percentage <= 0 or self.steps >= self.max_steps:
			done = True
			if self.battery_percentage <= 0:
				reward -= 50

		state = self._get_state()
		info = {
			"battery_percentage": self.battery_percentage,
			"model": (None if deescalate else ["XGBOOST", "TST"][self.current_model_idx]),
			"cpu_frequency": self.cpu_frequency,
			"active_cores": self.active_cores,
			"power_watts": power_watts,
			"energy_Wh": energy_Wh,
			"battery_capacity_Wh": self.capacity_Wh,
			"thermal_state": ThermalState(self.thermal_state_idx).name,
			"cpu_temperature": self._get_cpu_temperature(),
			"reward": reward,
			"reward_breakdown": reward_breakdown,
		}
		return state, reward, done, info

	def render(self, mode='human'):
		threat_levels = ["NONE", "POTENTIAL", "CONFIRMING", "CRITICAL"]
		battery_states = ["CRITICAL", "LOW", "MEDIUM", "HIGH"]
		cpu_loads = ["LOW", "NORMAL", "HIGH"]
		task_priorities = ["CRITICAL", "HIGH", "MEDIUM"]
		thermal_states = ["OPTIMAL", "WARM", "HOT", "CRITICAL"]
		models = ["XGBOOST", "TST"]
		frequencies = list(CPU_FREQUENCY_PRESETS.keys())

		print(f"Step: {self.steps}/{self.max_steps}")
		print(f"Battery: {self.battery_percentage:.1f}% ({battery_states[self.battery_state_idx]})")
		print(f"Threat Level: {threat_levels[self.threat_level_idx]}")
		print(f"CPU Load: {cpu_loads[self.cpu_load_idx]}")
		print(f"Task Priority: {task_priorities[self.task_priority_idx]}")
		print(f"Thermal State: {thermal_states[self.thermal_state_idx]} ({self._get_cpu_temperature():.1f}°C)")
		model_str = "DE-ESCALATED" if self.current_model_idx is None else models[self.current_model_idx]
		print(f"Current Model: {model_str}")
		print(f"CPU Frequency: {frequencies[self.current_freq_idx]} ({self.cpu_frequency} MHz)")
		print(f"Active Cores: {self.active_cores}")
		
		# Show power consumption from empirical data
		if self.current_model_idx is not None:
			model_name = models[self.current_model_idx]
			perf_data = get_ddos_performance(model_name, self.active_cores, self.cpu_frequency)
			print(f"Power Consumption: {perf_data['power_watts']:.2f}W")
			print(f"Detection Accuracy: {perf_data['accuracy']:.3f}")
		else:
			print(f"Power Consumption: 3.5W (Base)")
		print("-" * 30)

__all__ = ["TacticalUAVEnv"]
