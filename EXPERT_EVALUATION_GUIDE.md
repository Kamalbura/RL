# RL Dual-Agent System: Expert Evaluation & Audit Guide

This document is intended for an experienced RL practitioner to evaluate the current dual-agent (Strategic + Tactical) reinforcement learning setup for UAV swarm cryptographic posture and DDoS detection scheduling. It enumerates architecture, assumptions, reward design, data provenance, reproducibility, risks, and recommended inspection hooks.

---
## 1. System Overview
- Dual-tabular Q-learning agents:
  - Strategic (GCS-side): Chooses cryptographic algorithm for fleet-wide policy.
  - Tactical (UAV-side): Chooses DDoS model + CPU frequency preset or de-escalates monitoring.
- Objectives:
  - Balance security vs. energy endurance vs. latency/throughput.
  - Provide adaptive escalation under evolving threat and resource constraints.
- Deployment integration:
  - `gcs-scheduler.py` (strategic decisions) and `uav-scheduler.py` (tactical decisions) load saved Q-tables.
  - Tactical runtime can fall back to heuristic state machine if policy file not present.

---
## 2. Environments & State/Action Spaces
### 2.1 StrategicCryptoEnv (`crypto_rl/strategic_agent.py`)
State: [Threat(0–2), AvgBattery(0–2), MissionPhase(0–3)]
Action: 4 cryptographic algorithms (from `config/crypto_config.py`).
Step stochasticity: random drift in threat & mission; battery decreases probabilistically with high-power algorithms.
Episode length: up to 200 steps (fixed horizon).

### 2.2 TacticalUAVEnv (`ddos_rl/env.py`)
State (MultiDiscrete): [Threat(4), Battery(4), CPU Load(3), Task Priority(3)].
Actions (9):
  - 0..3: XGBOOST @ {POWERSAVE,BALANCED,PERFORMANCE,TURBO}
  - 4..7: TST @ same 4 CPU presets
  - 8: De-escalate (no DDoS scanning)
Time step: 5 seconds simulated.
Battery model: Wh drain using power tables and cumulative depletion; episode cap 500 steps.

---
## 3. Reward Engineering
### Strategic
Reward = SecurityWeighted − PowerPenalty − LatencyPenalty.
- Threat-modulated security scaling (greater weight under higher threat).
- Battery-sensitive power penalty.
- Mission-phase latency penalty.
Potential expert checks:
- Confirm relative magnitudes keep Q-values bounded (observe training logs for divergence).
- Sensitivity analysis: vary algorithm latency/power multipliers ±20%.

### Tactical
Composite reward with terms:
- Energy penalty (scaled by current battery band).
- Performance penalty for long execution time tiers.
- Security bonus under elevated threat for high-security model.
- CPU load & task priority synergy bonuses for higher frequency when appropriate.
- High penalization of de-escalation under severe threat.
Potential risks:
- Reward discontinuities around frequency thresholds (step-like changes).
- Over-penalization may encourage frequent de-escalation if threat rarely escalates; validate empirical action distribution.

---
## 4. Data & Profiles Provenance
Canonical source: `ddos_rl/profiles.py` (import-reexport via `performance_profiles.py`). Contains:
- `POWER_PROFILES`, `LATENCY_PROFILES`, `DDOS_TASK_TIME_PROFILES`, `SECURITY_RATINGS`, `CPU_FREQUENCY_PRESETS`.
Assumptions:
- Independent factors; no cross-correlation modeling (e.g., power vs. latency covariance).
- CPU frequency presets discretized—no partial scaling.
Recommendation: consider uncertainty intervals per profile for robust training (domain randomization).

---
## 5. Training Configuration
### Strategic
- Episodes: default 5000 (suggest warm-up 1000).
- Exploration decay: 0.9995 (slow decay ensures late exploration).
- Logging: `strategic_training_log.csv` with per-episode reward & periodic eval.
- Checkpoints: best eval saved as `strategic_crypto_q_table_best.npy`.

### Tactical
- Episodes: default 10000 (recommend initial 1500–2000 smoke run).
- Exploration decay: 0.995 (epsilon approaches 0.01 near end).
- Logging: `tactical_training_log.csv` (episodic + evaluation slices).
- Checkpoints: periodic `tactical_q_table_ckpt_<N>.npy`, best model & final.

### Suggested Expert Verifications
1. Plot moving average (window=100) and confirm monotonic or plateau shape; watch for oscillation.
2. Inspect Q-table sparsity: `np.count_nonzero(q_table) / q_table.size`.
3. Check for negative reward drift (indicating mis-weighting of penalties).
4. Evaluate action distribution stability across last 10% episodes.

---
## 6. Reproducibility Status
Implemented:
- NumPy global RNG seeding in training functions.
Pending (recommended):
- Commit hash capture.
- Environment snapshot (`pip freeze`).
- Run metadata JSON: seed, episodes, hyperparams, wallclock, platform.
- Hash (SHA256) of saved Q-tables.

---
## 7. Integration Runtime Behavior
`uav-scheduler.py`:
- Loads tactical policy if `output/tactical_q_table_best.npy` or fallback final exists.
- Periodic (10s interval) tactical RL action selection; switches model tasks (process-based) accordingly.
- Current limitation: CPU frequency index not yet applied to real system scaling (TODO: tie to `cpufreq` or OS-level governor adjustment).
`gcs-scheduler.py` (not shown here) expected to load strategic table (ensure matching filename: `strategic_crypto_q_table*.npy`).

---
## 8. Validation & Baselines
Legacy validator (`validation.py`) references outdated action dimension (24) and filenames—requires refactor before authoritative comparison. Interim approach: create targeted scripts:
- Baseline fixed-policy vs RL: compute mean±std returns over 100 episodes.
- Policy action histogram sampling from environment resets.
Work needed: unify evaluation harness to avoid stale imports.

---
## 9. Risk & Gap Analysis
| Area | Current State | Risk | Recommendation |
|------|---------------|------|----------------|
| Reward scaling | Manual heuristic weights | Potential imbalance | Normalize each component; track reward term contributions. |
| Exploration schedule | Fixed epsilon decay | Over/under exploration | Consider adaptive decay (based on eval plateau). |
| State abstraction | Coarse discrete bins | Hidden confounders | Add engineered features (e.g., energy trend, threat persistence). |
| Runtime fidelity | Frequency choice unused | Policy mismatch | Implement CPU scaling or remove frequency action dimension. |
| Validation harness | Partially outdated | False confidence | Replace with modular evaluator referencing live env classes. |
| Reproducibility | Partial | Non-repeatability | Add metadata & artifact hashing. | 

---
## 10. Suggested Enhancement Roadmap (Priority Order)
1. Validation overhaul: implement unified evaluator for both agents (baseline vs RL, CI-friendly outputs).
2. Metadata & provenance layer (`run_metadata.json` per training run).
3. CPU frequency enforcement (tactical) or action space simplification if not feasible.
4. Reward diagnostics: log per-term contributions for first N episodes & periodic snapshots.
5. Domain randomization: perturb profile tables ±X% each episode to improve robustness.
6. Early stopping & learning curve slope detection.
7. Optional upgrade: Move to function approximation (e.g., small DQN) if state aliasing harms generalization.

---
## 11. Inspection Commands (Examples)
```python
# Q-table sparsity & stats
import numpy as np
qt = np.load('output/tactical_q_table_best.npy')
print(qt.shape, 'nonzero ratio', np.count_nonzero(qt)/qt.size)
print('mean', qt.mean(), 'std', qt.std(), 'max', qt.max(), 'min', qt.min())

# Action distribution sampling
from ddos_rl.env import TacticalUAVEnv
from ddos_rl.agent import QLearningAgent
env=TacticalUAVEnv(); agent=QLearningAgent([4,4,3,3],9); agent.load_policy('output/tactical_q_table_best.npy')
from collections import Counter
c=Counter()
for _ in range(500): s=env.reset(); a=agent.choose_action(s, training=False); c[a]+=1
print(c)
```

---
## 12. Key Questions for Expert Review
1. Are reward term magnitudes producing a well-conditioned Q-table (no saturation)?
2. Is the action space (esp. frequency dimension) justifiable without runtime effect?
3. Should frequency and model selection be factorized (separate decisions)?
4. Is tabular representation sufficient given stochastic drifts? (Assess state visitation coverage.)
5. Would prioritized experience replay benefit the tactical agent’s sparse severe-threat states? (A PER buffer exists but not integrated.)
6. Should strategic battery dynamics be modeled more continuously to avoid abrupt category shifts?

---
## 13. Artefacts To Review First
1. `output/strategic_training_log.csv` – eval improvement trend.
2. `output/tactical_training_log.csv` – reward smoothing & epsilon trajectory.
3. Best vs final Q-table diff statistics.
4. Sample action histograms (see script above).

---
## 14. Immediate Minimal Changes Before Scaling Up
- Add metadata JSON on next training run.
- Patch validation harness (remove outdated 24-action assumption).
- Confirm strategic loader path in runtime scheduler.

---
## 15. Glossary
- De-escalate: tactical choice to suspend DDoS scanning (action 8).
- TST: High-fidelity test scan model (heavier resource profile).
- XGBOOST: Baseline model (lighter power & CPU). 

---
## 16. Contact & Next Steps
Provide this document to the reviewing RL expert. Their prioritized feedback will drive the next implementation sprint (validation rewrite, reward diagnostic logging, runtime fidelity improvements).
