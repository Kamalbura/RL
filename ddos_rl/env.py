"""
TacticalUAVEnv (moved from top-level tactical_simulator.py)
"""

import numpy as np
try:
	import gym
	from gym import spaces
except ImportError:
	import gymnasium as gym
	from gymnasium import spaces
from . import config
from .profiles import (
	get_power_consumption, get_ddos_execution_time,
	get_security_rating, CPU_FREQUENCY_PRESETS,
)


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

		# State space: Threat(4), Battery(4), CPU Load(3), Task Priority(3)
		self.observation_space = spaces.MultiDiscrete([4, 4, 3, 3])

		# Time and battery model
		self.time_step = 5  # seconds per step
		self.max_steps = 500
		self.capacity_Wh = float(config.BATTERY_SPECS.get("CAPACITY_WH", 115.0)) or 115.0

		# Initialize episode
		self.reset()

	def _get_state(self):
		return [
			self.threat_level_idx,
			self.battery_state_idx,
			self.cpu_load_idx,
			self.task_priority_idx,
		]

	def reset(self):
		# Battery and env state
		self.battery_percentage = 100.0
		self.threat_level_idx = 0  # NONE
		self.battery_state_idx = 3  # HIGH
		self.cpu_load_idx = 1  # NORMAL
		self.task_priority_idx = 1  # MEDIUM
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
		# Task priority occasional change
		if np.random.random() < 0.05:
			self.task_priority_idx = int(np.random.choice([0, 1, 2]))

	def _calculate_reward(self, model_idx, freq_idx, deescalate: bool):
		reward = 0.0
		if deescalate:
			# No scanning: low power is good on low threat/low battery; risky on high threat
			if self.threat_level_idx >= 2:
				reward -= 25  # under-monitoring under high threat
			if self.battery_state_idx <= 1:
				reward += 8   # preserving battery when low
			return reward

		model = ["XGBOOST", "TST"][model_idx]
		cpu_key = self._cpu_keys[freq_idx]
		cpu_frequency = CPU_FREQUENCY_PRESETS[cpu_key]
		active_cores = self.default_cores

		power_watts = get_power_consumption(cpu_frequency, active_cores)
		execution_time = get_ddos_execution_time(model, cpu_frequency, active_cores)
		security = get_security_rating(model)

		# Energy-conscious term
		if self.battery_state_idx <= 1:
			reward -= power_watts * 2.0
		else:
			reward -= power_watts * 0.5

		# Performance term
		if execution_time > 5.0:
			reward -= 20
		elif execution_time > 2.0:
			reward -= 10
		else:
			reward += 5

		# Security term
		if self.threat_level_idx >= 2:
			reward += 20 if security >= 8 else -15
		else:
			if security >= 4:
				reward += 5

		# CPU load scaling
		if self.cpu_load_idx == 2 and freq_idx >= 2:
			reward += 10

		# Task priority scaling
		if self.task_priority_idx == 0 and freq_idx >= 2:
			reward += 15

		return reward

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

		reward = self._calculate_reward(model_idx, freq_idx, deescalate)
		power_watts = get_power_consumption(self.cpu_frequency, self.active_cores)

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
			"reward": reward,
		}
		return state, reward, done, info

	def render(self, mode='human'):
		threat_levels = ["NONE", "POTENTIAL", "CONFIRMING", "CRITICAL"]
		battery_states = ["CRITICAL", "LOW", "MEDIUM", "HIGH"]
		cpu_loads = ["LOW", "NORMAL", "HIGH"]
		task_priorities = ["CRITICAL", "HIGH", "MEDIUM"]
		models = ["XGBOOST", "TST"]
		frequencies = list(CPU_FREQUENCY_PRESETS.keys())

		print(f"Step: {self.steps}/{self.max_steps}")
		print(f"Battery: {self.battery_percentage:.1f}% ({battery_states[self.battery_state_idx]})")
		print(f"Threat Level: {threat_levels[self.threat_level_idx]}")
		print(f"CPU Load: {cpu_loads[self.cpu_load_idx]}")
		print(f"Task Priority: {task_priorities[self.task_priority_idx]}")
		model_str = "DE-ESCALATED" if self.current_model_idx is None else models[self.current_model_idx]
		print(f"Current Model: {model_str}")
		print(f"CPU Frequency: {frequencies[self.current_freq_idx]} ({self.cpu_frequency} MHz)")
		print(f"Active Cores: {self.active_cores}")
		print(f"Power Consumption: {get_power_consumption(self.cpu_frequency, self.active_cores):.2f}W")
		print("-" * 30)

__all__ = ["TacticalUAVEnv"]
