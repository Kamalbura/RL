# Operational Guide: Training and Running the Hierarchical RL Agents

This guide explains how to train the tactical and strategic agents, then deploy their trained policies for live inference. Commands are PowerShell-ready for Windows.

## Prerequisites
- Activate your Conda env (example name: gcs-env) and install dependencies from `requirements.txt`.
- Ensure shared profiles are present: `shared/crypto_profiles.py` and `shared/system_profiles.py`.

## Phase 1: Train the Agents (Learning)

Goal: Produce optimized policy files: `tactical_dqn_best.pt` and `strategic_crypto_dqn_best.pt` written to `output/`.

### 1.1 Train Tactical (UAV-side)
- Script: `ddos_rl/train.py`
- What it does: creates `TacticalUAVEnv` (normalized Box state; Tuple(3,4) actions), wraps `DQNAgent`, logs CSV, checkpoints, and saves best.
- Outputs in `output/`:
  - `tactical_dqn_best.pt` (best checkpoint)
  - `tactical_dqn.pt` (last)
  - `tactical_training_curves.png`, `tactical_epsilon.png`, CSV and metadata

Run:

```powershell
conda activate gcs-env
python ddos_rl/train.py --episodes 10000 --eval-frequency 100 --checkpoint-every 500 --output-dir output
```

Optional shorter smoke run:

```powershell
python ddos_rl/train.py --episodes 300 --eval-frequency 50 --checkpoint-every 100 --output-dir output_smoke
```

### 1.2 Train Strategic (GCS-side)
- Script: `crypto_rl/train_strategic.py`
- What it does: trains `StrategicCryptoAgent` over `StrategicCryptoEnv` (normalized Box state, consensus threat); logs and saves best.
- Outputs in `output/`:
  - `strategic_crypto_dqn_best.pt` (best)
  - Curves/CSV depending on trainer

Run:

```powershell
conda activate gcs-env
python crypto_rl/train_strategic.py
```

Notes:
- Both trainers assume `torch` and `gymnasium` installed.
- Adjust episodes/eval cadence to your hardware. GPU is optional but accelerates training.

## Phase 2: Run for Live Inference (Thinking)

Two options: integrated coordinator (headless) or full apps (UAV and GCS schedulers).

### 2.1 Quick headless smoke test
- Script: `run_inference.py`
- Loads `.pt` policies if present and starts the integrated `SystemCoordinator` (no MQTT/UI dependencies).

```powershell
conda activate gcs-env
python run_inference.py --tactical-policy .\output\tactical_dqn_best.pt --strategic-policy .\output\strategic_crypto_dqn_best.pt
```

### 2.2 Intelligent Drone Scheduler (UAV)
- Script: `uav-scheduler.py`
- Behavior: loads tactical agent and uses `choose_action(state, training=False)` inside its monitor loop to drive decisions.

```powershell
conda activate gcs-env
python uav-scheduler.py --drone_id uavpi-drone-01
```

Tip: Ensure certificates and MQTT broker config in `uav-scheduler.py` match your environment if you use MQTT features.

### 2.3 Intelligent GCS (GUI + Advisor)
- Script: `gcs-scheduler.py`
- Behavior: starts GUI, connects to MQTT, and shows a live AI crypto recommendation label when the strategic agent is available. Operator remains in control.

```powershell
conda activate gcs-env
python gcs-scheduler.py
```

## Files and Artifacts
- Tactical: `ddos_rl/train.py`, outputs under `output/`
- Strategic: `crypto_rl/train_strategic.py`, outputs under `output/`
- Inference: `run_inference.py` (optional), `uav-scheduler.py`, `gcs-scheduler.py`

## Troubleshooting
- Missing torch/gymnasium: `pip install -r requirements.txt` inside your Conda env.
- Models not loading: verify paths; ensure `.pt` files exist in `output/`.
- MQTT/certs: see logs; adjust broker and certificate paths in the schedulers.

Ready for takeoff: train offline, then deploy the brains to your schedulers for real-time decisions.
