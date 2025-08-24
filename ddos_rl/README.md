# ddos_rl - Tactical UAV DDoS RL (self-contained)

This package contains the drone-side, power-aware DDoS detection decision logic using tabular Q-learning. It is fully self-contained and grounded in empirical performance tables.

## Overview
- Goal: choose a DDoS model and CPU preset (or de-escalate) to balance security, latency, and battery life on a UAV.
- State space: MultiDiscrete [4, 4, 3, 3]
  - Threat: NONE, POTENTIAL, CONFIRMING, CRITICAL
  - Battery state: CRITICAL, LOW, MEDIUM, HIGH
  - CPU load: LOW, NORMAL, HIGH
  - Task priority: CRITICAL, HIGH, MEDIUM
- Action space (9):
  - 0..3: XGBOOST @ [POWERSAVE, BALANCED, PERFORMANCE, TURBO]
  - 4..7: TST     @ [POWERSAVE, BALANCED, PERFORMANCE, TURBO]
  - 8:    DE-ESCALATE (no scanning)
- Reward: energy penalty (Wh-based), performance (execution time), security term, scaled by load/priority.
- Battery model: watt-hours drain per step using `BATTERY_SPECS.CAPACITY_WH`.

## Files
- `env.py` - TacticalUAVEnv (Gym/Gymnasium-compatible)
  - Uses local `config.py` (battery) and `profiles.py` (empirical tables).
  - Step returns `(state, reward, done, info)` with power/energy metrics.
- `agent.py` - QLearningAgent (tabular)
  - `choose_action`, `learn`, `save_policy`, `load_policy`, `get_best_action_for_state`, `get_policy_summary`.
- `train.py` - training and evaluation helpers
  - `train_tactical_agent(episodes, output_dir)`, `evaluate_agent(...)`, plotting.
- `profiles.py` - empirical power/latency/task/security tables and getters.
- `config.py` - local battery spec (e.g., `CAPACITY_WH`).
- `UAVScheduler.py` - demo that loads a trained Q-table and prints decisions.

## Quick start (Windows PowerShell)
```powershell
# From repo root with conda env active
python UAVScheduler.py  # runs ddos_rl.UAVScheduler demo

# Train tactical agent (saves to output/)
python -c "from ddos_rl.train import train_tactical_agent; train_tactical_agent(episodes=5000, output_dir='output')"

# Evaluate a saved policy
python -c "from ddos_rl.train import evaluate_agent; from ddos_rl.env import TacticalUAVEnv; from ddos_rl.agent import QLearningAgent; import numpy as np; env=TacticalUAVEnv(); agent=QLearningAgent([4,4,3,3], 9); agent.q_table = np.load('output/tactical_q_table_best.npy'); print(evaluate_agent(agent, env, 100))"
```

## API highlights
- `TacticalUAVEnv.reset() -> state`
- `TacticalUAVEnv.step(action) -> (state, reward, done, info)`
- `QLearningAgent.choose_action(state, training=True) -> int`
- `QLearningAgent.learn(state, action, reward, next_state, done)`

## Notes
- Gym import warnings are OK; the env falls back to gymnasium when gym is missing.
- Profiles and config are package-local to avoid root dependencies.
