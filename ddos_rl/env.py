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
	ThermalState, get_ddos_performance,
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

		# Factored action space: (model_choice, cpu_freq)
		# model_choice: 0=DE-ESCALATE, 1=XGBOOST, 2=TST
		# cpu_freq: 0=POWERSAVE, 1=BALANCED, 2=PERFORMANCE, 3=TURBO
		self.action_space = spaces.Tuple((spaces.Discrete(3), spaces.Discrete(4)))

		# Continuous normalized observation space Box[0,1]^5
		# [Threat, Battery, CPU_Load, Task_Priority, Temperature]
		low = np.zeros(5, dtype=np.float32)
		high = np.ones(5, dtype=np.float32)
		self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

		# Time and battery model
		self.time_step = 5  # seconds per step
		self.max_steps = 500
		self.capacity_Wh = BATTERY_SPECS["CAPACITY_WH"]

		# Initialize episode
		self.reset()

	def _get_state(self):
		# Normalize to [0,1]
		threat = self.threat_level_idx / 3.0
		battery = np.clip(self.battery_percentage, 0.0, 100.0) / 100.0
		cpu_load = [0.15, 0.5, 0.85][self.cpu_load_idx]
		task_priority = [1.0, 0.5, 0.2][self.task_priority_idx]  # CRITICAL=1.0 .. MEDIUM=0.2
		temp_c = self._last_temp_c
		temp_norm = np.clip((temp_c - 30.0) / (85.0 - 30.0), 0.0, 1.0)
		return np.array([threat, battery, cpu_load, task_priority, temp_norm], dtype=np.float32)

	def reset(self):
		# Battery and env state
		self.battery_percentage = 100.0
		self.threat_level_idx = 0  # NONE
		self.battery_state_idx = 3  # HIGH
		self.cpu_load_idx = 1  # NORMAL
		self.task_priority_idx = 1  # MEDIUM
		self.thermal_state_idx = 0  # OPTIMAL
		self._last_temp_c = 45.0
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
			result = subprocess.run(['vcgencmd', 'measure_temp'], capture_output=True, text=True)
			temp_str = result.stdout.strip()
			temp = float(temp_str.split('=')[1].split("'")[0])
			return temp
		except Exception:
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
		self._last_temp_c = current_temp
		self.thermal_state_idx = self._classify_thermal_state(current_temp)
		
		# Task priority occasional change
		if np.random.random() < 0.05:
			self.task_priority_idx = int(np.random.choice([0, 1, 2]))

	def _calculate_reward(self, model_idx: int, freq_idx: int, deescalate: bool):
		"""Normalized reward components in [-1,1], then sum.

		Components:
		- thermal_safety: penalize high freq when HOT/CRITICAL
		- energy: penalize watts normalized by band [3W..15W]
		- latency: penalize total inference time normalized to [0..500ms]
		- security: reward accuracy esp. under high threat
		- context: cpu load and task priority alignment
		"""
		# Defaults
		thermal_term = 0.0
		energy_term = 0.0
		latency_term = 0.0
		security_term = 0.0
		context_term = 0.0

		if deescalate:
			# Encourage de-escalation when cool battery is low/thermal high/threat low
			threat_low = 1.0 - (self.threat_level_idx / 3.0)
			low_batt = 1.0 if self.battery_percentage < 30 else 0.0
			hot = 1.0 if self.thermal_state_idx >= ThermalState.HOT.value else 0.0
			thermal_term = 0.5 * hot
			energy_term = 0.5 * low_batt
			security_term = -1.0 * (1.0 - threat_low)  # negative if threat high
			return float(np.clip(thermal_term + energy_term + security_term, -1.0, 1.0)), {
				"thermal": thermal_term,
				"energy": energy_term,
				"security": security_term,
				"latency": 0.0,
				"context": 0.0,
			}

		model = ["XGBOOST", "TST"][model_idx]
		cpu_key = self._cpu_keys[freq_idx]
		cpu_frequency = CPU_FREQUENCY_PRESETS[cpu_key]
		active_cores = self.default_cores
		perf = get_ddos_performance(model, active_cores, cpu_frequency)
		watts = float(perf["power_watts"])
		lat_ms = float(perf["create_sequence_ms"] + perf["prediction_ms"])
		acc = float(perf["accuracy"])  # in [0,1]

		# Thermal safety [-1,1]
		if self.thermal_state_idx == ThermalState.CRITICAL.value:
			thermal_term = -1.0 if freq_idx >= 1 else -0.5  # only POWERSAVE is less bad
		elif self.thermal_state_idx == ThermalState.HOT.value:
			thermal_term = -0.5 if freq_idx >= 2 else -0.2
		else:
			thermal_term = 0.2 if freq_idx <= 1 else 0.0

		# Energy cost normalized: assume 3W..15W band
		energy_term = -np.clip((watts - 3.0) / (15.0 - 3.0), 0.0, 1.0)
		# Make it harsher when battery low
		if self.battery_percentage < 30:
			energy_term *= 1.2

		# Latency normalized to 0..500ms
		latency_term = -np.clip(lat_ms / 500.0, 0.0, 1.0)

		# Security: scale with threat
		threat_w = [0.2, 0.5, 1.0, 1.2][self.threat_level_idx]
		security_term = np.clip((acc - 0.85) / (1.0 - 0.85), 0.0, 1.0) * threat_w
		security_term = np.clip(security_term, 0.0, 1.0)

		# Context: prefer higher freq when cpu load high or task priority critical
		cpu_pref = 1.0 if (self.cpu_load_idx == 2 and freq_idx >= 2) else 0.0
		task_pref = 1.0 if (self.task_priority_idx == 0 and freq_idx >= 2) else 0.0
		context_term = 0.5 * cpu_pref + 0.5 * task_pref

		# Sum and clip
		total = float(np.clip(thermal_term + energy_term + latency_term + security_term + context_term, -1.0, 1.0))
		return total, {
			"thermal": float(thermal_term),
			"energy": float(energy_term),
			"latency": float(latency_term),
			"security": float(security_term),
			"context": float(context_term),
		}

	def step(self, action):
		# action is a tuple (model_choice, freq_choice)
		assert self.action_space.contains(action), f"Invalid action: {action}"
		model_choice, freq_idx = action
		deescalate = (model_choice == 0)
		model_idx = None if deescalate else (model_choice - 1)  # 0->None, 1->0(XGB), 2->1(TST)

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
			"model": (None if deescalate else ["XGBOOST", "TST"][self.current_model_idx] if self.current_model_idx is not None else None),
			"cpu_frequency": self.cpu_frequency,
			"active_cores": self.active_cores,
			"power_watts": power_watts,
			"energy_Wh": energy_Wh,
			"battery_capacity_Wh": self.capacity_Wh,
			"thermal_state": ThermalState(self.thermal_state_idx).name,
			"cpu_temperature": self._last_temp_c,
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
