# EXPERT AUDIT REPORT — Hierarchical RL for Autonomous UAV Cybersecurity

Author: GitHub Copilot  
Date: 2025-08-28

---

## 1) Executive Summary

This repository implements an ambitious dual-agent, hierarchical RL system for UAV cybersecurity: a tactical UAV-side agent for DDoS detection and CPU scaling, and a strategic GCS-side agent for fleet-wide cryptographic policy. The project demonstrates strong systems thinking, extensive documentation, and significant empirical grounding from Raspberry Pi 4B measurements. It includes validation data, trained Q-tables, and integration scaffolding for deployment.

Core strengths
- Clear hierarchical split of concerns (crypto policy at GCS; real-time, power-aware scheduling on UAV).
- Empirical, hardware-grounded performance profiles and Wh-based battery modeling.
- Extensive docs, evaluation guides, and validation CSVs for transparency and reproducibility.
- Thermal awareness and Byzantine consensus modules show forward-looking safety and robustness.

Critical blockers to production readiness
1) Tactical observation mismatch and hidden-state aliasing: `ddos_rl/env.py` declares a 5D observation space (includes thermal) but returns a 4D state from `_get_state()` (omits thermal). The reward uses thermal, so the agent is trained without observing a critical variable. This is a fundamental POMDP/aliasing bug and must be fixed.  
   Evidence: `ddos_rl/env.py` observation_space MultiDiscrete([4,4,3,3,4]) vs `_get_state()` returning [threat, battery, cpu_load, task_priority].
2) Strategic environment duplication/inconsistency: Two strategic envs exist (`crypto_rl/strategic_agent.py` minimal vs `crypto_rl/crypto_simulator.py` enhanced) with differing states, reward scales, and latency units. Swarm consensus is not consistently integrated into the live strategic training loop. This creates policy ambiguity.  
   Evidence: differing state definitions and reward/latency handling between those files.
3) Reward scaling/variance and baseline behavior: Validation shows high variance for RL under some scenarios, and catastrophic negative returns for HighPerf baselines (e.g., low_battery mean ≈ −8600, critical_all ≈ −9717). This suggests over-penalization/imbalance and instability under tight resource regimes.  
   Evidence: `validation_results/tactical_validation_summary.csv`.
4) Single-source-of-truth drift: Both `shared/system_profiles.py` and legacy per-package profiles (e.g., `ddos_rl/profiles.py`, constants in `crypto_rl/crypto_simulator.py`) exist with overlapping/competing data. This risks sim-to-real gaps and inconsistent training vs deployment decisions.

---

## 2) Deep Dive: Tactical RL Agent (ddos_rl)

Files reviewed: `ddos_rl/env.py`, `ddos_rl/agent.py`, `ddos_rl/profiles.py`, `ddos_rl/config.py`, plus `shared/system_profiles.py` and `shared/crypto_profiles.py` used by ddos.

### Environment (env.py)
- State space: Declared as MultiDiscrete [Threat(4), Battery(4), CPU Load(3), Task Priority(3), Thermal(4)]. However, `_get_state()` returns only 4 entries `[threat, battery, cpu_load, task_priority]`, omitting thermal. Meanwhile, reward uses `self.thermal_state_idx` extensively (large penalties/bonuses).  
  Implication: The agent cannot observe thermal state yet is judged by it. This induces state aliasing and learned policies that appear erratic under temperature drift. This is the top-priority bug to fix.
- Sufficiency: Even with thermal added, discrete buckets are low-resolution (coarse quantization of battery, cpu_load, thermal). Missing signals likely helpful for stability: rate-of-change features (battery drain slope), temperature slope, threat persistence, and network QoS (packet loss/latency spikes) to capture DDoS context.
- Dangers of low-resolution buckets: aliasing similar but meaningfully different conditions; reward discontinuities at bin edges; brittle policies when real telemetry sits near boundaries.

### Action Space
- Current: Single Discrete(9) mixing model choice (XGBOOST/TST) × CPU frequency preset (4) plus a “de-escalate” action.
- Efficiency critique: Coupling “what to run” and “how fast to run” reduces compositionality, makes exploration harder, and obscures credit assignment. The de-escalate special-case mixes action semantics further.
- Factored action spaces: Prefer `Tuple((Discrete(models), Discrete(freqs)))` and represent de-escalation as a model=None or separate boolean. This enables independent learning of model-vs-frequency decisions and simplifies transfer to parameterized actions (continuous frequency) later.

### Reward Function
- Composition (high-level):  
  - Thermal penalties up to −100 (critical), −25 (hot); small bonuses when thermally prudent.  
  - Energy penalty proportional to Watts with multiplier up to 2.0 and thermal scaling.  
  - Execution penalty up to ~50 for slow steps; bonus +10 if ≤0.1s.  
  - Security term up to +20 (high threat) or −15 if inaccurate under threat.  
  - Context bonuses: CPU load and task priority when high frequency is warranted.
- Scaling concerns: Thermal penalties can dominate, and energy penalties scale directly with instantaneous Watts without normalization by per-step budget. Result is likely heavy-tailed reward distribution. The validation data shows high variance and negative means for low-battery scenarios for RL, indicating mis-weighting under scarce energy.  
  Evidence: `validation_results/tactical_validation_summary.csv` low_battery RL mean ≈ −309 with std ≈ 1491; many episodes negative.
- Recommendation: Normalize reward components (z-score or min-max to target bands), cap extreme penalties, and log per-term contributions to tune weights quantitatively.

### Agent (agent.py)
- Standard tabular Q-Learning; adequate for small discrete spaces. However, current effective observation (omitting thermal) is 4D while env declares 5D—this mismatch must be resolved before further training. For richer/continuous features, tabular will not scale; plan for function approximation (see DQN recommendation) or tile-coding as an interim.

### Profiles (profiles.py)
- There is a legacy `ddos_rl/profiles.py` with large, partly extrapolated tables and a newer centralized knowledge base in `shared/system_profiles.py` and `shared/crypto_profiles.py`. `ddos_rl/env.py` already imports from `shared.crypto_profiles` (good).  
- Risk: Multiple sources with different numbers and units (e.g., latencies ms vs s). This can create sim-to-real gaps, inconsistent training, and brittle policies.

---

## 3) Deep Dive: Strategic RL Agent (crypto_rl)

Files reviewed: `crypto_rl/strategic_agent.py` (minimal env and agent wrapper), `crypto_rl/crypto_simulator.py` (enhanced env), `crypto_rl/rl_agent.py`, `crypto_rl/consensus.py`, and shared profiles.

### Environment(s)
- Minimal env (`crypto_rl/strategic_agent.py`): State = [Threat(3), AvgBattery(3), MissionPhase(4)], Actions = 4 algorithms. Reward balances security, power multiplier, and latency with mission-modulated weights. This env is coherent and compact.
- Enhanced env (`crypto_rl/crypto_simulator.py`): Docstring claims inclusion of swarm consensus threat; internally it defines a different multi-factor state (security risk, computation capacity, mission criticality, communication intensity, threat context). Latency and power appear in different scales, and some comments suggest different units (“SPHINCS ~20ms” vs hundreds of ms elsewhere).  
- Concern: Two divergent envs with inconsistent units and states makes it unclear which policy is authoritative. The enhanced env’s consensus integration is not reflected in the minimal env used by the trainer.

### Reward Function
- Minimal env penalizes latency as `(latency_ms / 10) * phase_weight` and power as `power_mult * 10 * battery_weight`. Without global normalization, different algorithm dictionaries can swing rewards drastically. Real SPHINCS latency costs (hundreds of ms) must be properly weighted during critical phases to reflect mission risk; numbers across modules differ.  
- Recommendation: Centralize KPIs (latency, power, security) in `shared/system_profiles.py` and compute rewards from that single source for both env variants.

### Swarm Intelligence (consensus.py)
- Strength: `EnhancedByzantineConsensus` provides a signature-checked, thresholded agreement for threat reports. This is a substantial positive for robustness.
- Gap: There’s no explicit loop feeding consensus-derived threat into the strategic env used for training (`strategic_agent.py`). The enhanced simulator suggests it, but integration is incomplete. Consensus should set or modulate the “Threat” dimension every step, with appropriate lag and uncertainty modeling.

---

## 4) Analysis of Training and Validation (validation_results/)

Reviewed: `tactical_validation_results.csv`, `tactical_validation_summary.csv` (PNG graphs referenced but not needed to reach these findings).

Key observations
- RL vs Baselines (baseline scenario):  
  - RL mean ≈ 6797, std ≈ 1571 — solid performance but with high variability.
  - PowerSave baseline mean ≈ 3580, std ≈ 1652 — lower and also variable.
  - HighPerf baseline mean ≈ −325, std ≈ 2216 — oscillatory and often catastrophic.
- High-threat scenario:  
  - RL mean ≈ 7192, std ≈ 1819 — strong average with high variance.
  - PowerSave baseline ≈ 4834, std ≈ 1635 — acceptable but weaker than RL.
  - HighPerf baseline ≈ −3837, std ≈ 2388 — severe negatives indicate over-penalized high-power actions under stress.
- Low-battery scenario:  
  - RL mean ≈ −309, std ≈ 1491 — unstable policy under energy scarcity; many episodes negative.
  - PowerSave baseline ≈ 282, std ≈ 1843 — weak but at least non-catastrophic on average.
  - HighPerf baseline ≈ −8600, std ≈ 2610 — catastrophic penalties confirm power scaling dominance.
- Critical-all scenario:  
  - RL mean ≈ 1156, std ≈ 1595 — modest positive with high variance.
  - PowerSave baseline ≈ 1664, std ≈ 1482 — surprisingly competitive.
  - HighPerf baseline ≈ −9717, std ≈ 2301 — catastrophic scaling.

Interpretation
- High variance for the RL agent suggests reward imbalance and/or latent state issues (notably the unobserved thermal variable) causing non-stationarity from the agent’s perspective.  
- Catastrophic negative baselines confirm that penalty magnitudes for power/latency are large, but they may also overwhelm security/mission rewards, especially when state abstraction is coarse.
- The RL agent’s underperformance in low-battery suggests energy penalties and de-escalation incentives are not consistently aligned with threat/mission constraints; a safety layer and explicit energy budget terms would help.

Do the plots show convergence?
- CSV-level statistics indicate non-trivial variance and scenario sensitivity. Without normalized reward and consistent observables (include thermal), clean convergence is unlikely. Expect reward distributions to remain broad.

---

## 5) The Path to an Incredible System: A Concrete Action Plan

Priority is ordered; steps 1–3 are foundational and unblock stability.

1) Adopt and enforce a single Data-Driven Knowledge Base (Single Source of Truth)
- Action: Make `shared/system_profiles.py` (and `shared/crypto_profiles.py`) the only authoritative source for latency, power, accuracy, and thermal constraints.  
- Refactor: Remove or deprecate `ddos_rl/profiles.py` and any hard-coded constants in `crypto_rl/crypto_simulator.py`. Ensure all envs call shared accessors (e.g., `get_algorithm_performance`, `get_ddos_performance`).  
- Add: Confidence intervals/uncertainty and domain-randomization toggles per episode to harden policies.  
- Deliverables: Profiles unit tests; a provenance table mapping each metric to source measurements (from context.txt).

2) Evolve to continuous, normalized state spaces
- Rationale: Discrete, low-resolution buckets cause aliasing and brittleness. With thermal omitted from observed state, the tactical agent is effectively solving a POMDP.  
- Action: Move both envs to `gym.spaces.Box` with normalized features in [0,1], e.g., battery %, cpu load %, temperature normalized to [0, Tmax], threat level as probability/confidence, moving averages and slopes (battery drain rate, temp slope, threat persistence).  
- Tactical must include ThermalState (continuous temperature or one-hot thermal bins) as a first-class observable. Strategic must include Swarm_Consensus_Threat as a first-class observable.

3) Upgrade to a Deep Q-Network (DQN) for function approximation
- Rationale: Continuous features make Q-tables intractable; DQN handles generalization and smoothness.  
- Architecture: Small MLP (2–3 layers × 64–128), Double+Dueling DQN, Prioritized Experience Replay.  
- Edge deployment: Quantize (INT8), export via ONNX/TFLite, measure on RPi. Keep a simple rule-based safety shield and/or fall back to tabular under fault.  
- Process: Sim-first training with domain randomization, then hardware A/B tests vs tabular on power draw and latency.

4) Factor the tactical action space
- Action: Represent actions as `Tuple((Discrete(models=3), Discrete(freqs=4)))` where models = {XGBOOST, TST, DEESCALATE}.  
- Benefit: Clearer credit assignment; scalable to parameterized/continuous CPU frequency later.  
- Integration: Ensure `hardware/rpi_interface.py` actually enforces chosen frequency on RPi (governor/`cpufreq`). If hardware control is unavailable, remove frequency from action space to avoid training-runtime mismatch.

5) Integrate critical missing states into the learning loop
- Tactical: Include thermal (or raw temperature) in the observation; ensure no reward term depends on unobserved variables.  
- Strategic: Inject consensus-derived swarm threat from `crypto_rl/consensus.py` into the state each step (with lag/noise). Make it the primary “Threat” driver instead of an independent random drift.  
- Telemetry bus: Define a small schema for UAV→GCS messages so strategic state reflects fleet statistics and consensus.

Additional high-impact improvements
- Reward normalization and diagnostics: Log per-term contributions; target each term into a calibrated range (e.g., |term| ≤ 10).  
- Safety layer: Disallow high-frequency actions in HOT/CRITICAL thermal states by construction.  
- Evaluation harness: Unify validation scripts so they reference the live envs (no stale action counts, no duplicate constants).  
- Reproducibility: Save `run_metadata.json` (seeds, hyperparams, profile hashes, git commit) alongside Q-tables.

Success criteria
- Reduced reward variance and improved mean across all scenarios, especially low-battery.  
- No negative mean returns in any scenario for RL agent.  
- Verified CPU frequency enforcement on hardware or justified removal from action space.  
- Strategic policy’s threat signal sourced from consensus, validated via scenario tests.

---

## 6) Final Conclusion

This project shows excellent architectural judgment and thoroughness: a hierarchical split aligned with operational constraints; empirical profiles; and consideration of thermal safety and distributed consensus. The current prototype is close to a robust system, but four issues block production: tactical state omission (thermal), strategic env duplication/inconsistency, reward scaling/variance, and profile single-source drift. Addressing these—starting with state/observation fixes, profile unification, reward normalization, factored actions, and a compact DQN for continuous features—will bridge the gap from an impressive academic prototype to a robust, intelligent, deployable autonomous defense system.

---

### Appendix: Evidence Pointers
- Tactical state/action/reward: `ddos_rl/env.py` (observation_space vs `_get_state()`, `_calculate_reward`, `step()` info).  
- Tactical agent: `ddos_rl/agent.py` (tabular Q-Learning, stats helpers).  
- Strategic envs: `crypto_rl/strategic_agent.py` (minimal), `crypto_rl/crypto_simulator.py` (enhanced, inconsistent units/comments).  
- Consensus: `crypto_rl/consensus.py` (signature-checked thresholding; not yet wired into `strategic_agent.py`).  
- Shared profiles: `shared/system_profiles.py`, `shared/crypto_profiles.py` (single source to enforce).  
- Legacy/duplicate profiles: `ddos_rl/profiles.py` (to be deprecated).  
- Validation data: `validation_results/tactical_validation_results.csv`, `tactical_validation_summary.csv` (high variance and catastrophic baselines).  
- Docs: `EXPERT_EVALUATION_GUIDE.md`, `EXPERT_MODEL_SUMMARY.md`, `README.md` (helpful but need alignment with code paths).
