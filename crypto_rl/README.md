# crypto_rl - Strategic GCS-side Crypto RL

This package contains the ground-control (GCS) strategic cryptographic algorithm selection components, a training environment, helpers for coordination and swarm consensus, and validation stubs.

## Overview
- Goal: select a crypto algorithm for the fleet that balances security vs. power and latency under mission constraints.
- Strategic state (minimal env): [Threat (0..2), AvgFleetBattery (0..2), MissionPhase (0..3)].
- Actions: indices 0..3 map to algorithms in `config/crypto_config.py` (ASCON_128, KYBER_CRYPTO, SPHINCS, FALCON512).
- Reward: `security_reward(threat) - power_penalty(battery) - latency_penalty(mission)` using `CRYPTO_ALGORITHMS` characteristics.
- Uses tabular Q-learning via `crypto_rl/rl_agent.py`.

## Files
- `strategic_agent.py` - StrategicCryptoEnv/Agent and train/evaluate helpers
  - `train_strategic_agent(episodes, eval_every, out_dir)` saves `strategic_crypto_q_table.npy`.
  - Robust loader for `config/crypto_config.py` to avoid name clashes with root `config.py`.
- `train_strategic.py` - tiny entrypoint that trains and saves a strategic policy under `output/`.
- `crypto_simulator.py` - richer crypto environment (6D state) used by existing `train_crypto.py`.
  - Gym/Gymnasium import fallback and robust config loader.
- `rl_agent.py` - generic `QLearningAgent` used by both strategic and crypto_simulator flows.
- `crypto_scheduler.py` - demo scheduler selecting crypto given a state, with KPI tracking.
- `coordination.py` - JSON message helpers for GCS<->drone and swarm topics.
- `swarm_consensus.py` - `SwarmConsensusManager` implementing byzantine-tolerant threat consensus.
- `validation.py` - baseline vs RL comparison runner using `CryptoEnv`.
- `tests/` - simple tests for strategic agent, coordination and consensus.

## Quick start (Windows PowerShell)
```powershell
# From repo root with conda env active
# Train strategic agent (minimal state env)
python crypto_rl/train_strategic.py

# Train in the richer crypto environment
python crypto_rl/train_crypto.py

# Run the crypto scheduler demo
python crypto_rl/crypto_scheduler.py

# Compare baseline vs RL (after training)
python -c "from crypto_rl.validation import compare_baseline_vs_rl; import pprint; print(pprint.pformat(compare_baseline_vs_rl(policy_index=3, rl_policy_path='output/crypto_q_table.npy')))"
```

## API highlights
- `StrategicCryptoEnv.reset() -> state`
- `StrategicCryptoEnv.step(action) -> (state, reward, done, info)`
- `StrategicCryptoAgent.choose_action(state, training=True) -> int`
- `StrategicCryptoAgent.learn(...)`, `save_policy(path)`, `load_policy(path)`
- `QLearningAgent` in `rl_agent.py` is shared and configurable.

## Coordination and consensus
- Build and parse messages with `coordination.py` for publishing on MQTT topics (GCS->drone crypto_policy; drone_status; ddos_alert; swarm_state).
- Compute swarm threat consensus with `SwarmConsensusManager`.

## Notes
- The package reads crypto algorithm characteristics from `config/crypto_config.py`. A robust loader is included to avoid conflicts with a top-level `config.py` file. Keep both files present as in this repo.
- Install `pytest` to run tests in `crypto_rl/tests`.
