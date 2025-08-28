"""
Strategic (GCS-side) Crypto RL agent and minimal environment.

Design goals
- Continuous state: [Threat, AvgFleetBattery, MissionPhase, SwarmConsensusThreat] normalized to [0,1]
- Actions: 4 crypto algorithms from config.crypto_config.CRYPTO_ALGORITHMS
- Reward: Normalized components for security, power, and latency for stability

This module integrates with the existing crypto_rl package and config.
It reuses the generic QLearningAgent already present in crypto_rl.rl_agent.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import numpy as np

# Make sure we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from config.crypto_config import CRYPTO_ALGORITHMS, CRYPTO_RL  # type: ignore
except Exception:
    # Fallback loader to avoid top-level config.py name clash
    import runpy
    cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'crypto_config.py'))
    data = runpy.run_path(cfg_path)
    CRYPTO_ALGORITHMS = data.get('CRYPTO_ALGORITHMS', {})
    CRYPTO_RL = data.get('CRYPTO_RL', {})
from crypto_rl.rl_agent import CryptoDQNAgent
from utils.reproducibility import set_random_seeds
from utils.early_stopping import EarlyStopping, create_early_stopping_config
from utils.reward_monitor import RewardBalanceMonitor


try:
    import gym
    from gym import spaces
except ImportError:
    import gymnasium as gym
    from gymnasium import spaces

from shared.crypto_profiles import get_algorithm_performance, ThermalState
from shared.crypto_profiles import MissionPhase as SharedMissionPhase
from shared.crypto_profiles import SWARM_THREAT_LEVELS


class StrategicCryptoEnv(gym.Env):
    """
    Lightweight environment for GCS-side crypto selection.

    State: Continuous Box [0,1]^4 = [Threat, AvgFleetBattery, MissionPhase, SwarmConsensusThreat]
      - Threat: normalized 0..1
      - AvgBattery: normalized 0..1
      - MissionPhase: normalized 0..1 (IDLE=0.0 .. CRITICAL_TASK=1.0)
      - SwarmConsensusThreat: normalized 0..1

    Action: index into CRYPTO_ALGORITHMS (0..3) mapping to algorithms.

    Reward: normalized components for security(+), power(-), latency(-) each in [-1,1]
    """

    def __init__(self):
        super().__init__()
        self.action_dim = 4
        self.max_steps = 200
        self.steps = 0
        self._rng = np.random.default_rng()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.action_dim)
        self._consensus_noise = 0.1
        self.reset()

    def _get_state(self) -> np.ndarray:
        threat = np.clip(self.threat_level, 0.0, 1.0)
        batt = np.clip(self.avg_fleet_battery / 100.0, 0.0, 1.0)
        mission_norm = self.mission_phase / 3.0
        consensus = np.clip(self.swarm_consensus_threat, 0.0, 1.0)
        return np.array([threat, batt, mission_norm, consensus], dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.steps = 0
        self.threat_level = float(self._rng.uniform(0.0, 0.4))
        self.avg_fleet_battery = 100.0
        self.mission_phase = int(self._rng.choice([0, 1, 2, 3], p=[0.35, 0.3, 0.25, 0.1]))
        self.swarm_consensus_threat = float(self._rng.uniform(0.0, 0.3))
        return self._get_state()

    def _reward_components(self, security_rating: float, latency_ms: float, power_watts: float) -> Dict[str, float]:
        # Security term: scale to [0,1] and weight by threat/consensus
        sec_norm = np.clip(security_rating / 10.0, 0.0, 1.0)
        threat_w = 0.5 * self.threat_level + 0.5 * self.swarm_consensus_threat
        security = sec_norm * (0.5 + threat_w)  # 0.5..1.5 then clip to 1 below
        security = float(np.clip(security, 0.0, 1.0))

        # Power term: normalize in 3..20W band; harsher when battery low
        p = float(np.clip((power_watts - 3.0) / (20.0 - 3.0), 0.0, 1.0))
        batt_w = 1.2 if self.avg_fleet_battery < 40.0 else 1.0
        power = -p * batt_w

        # Latency term: normalize to 0..1000ms, harsher in critical mission
        l = float(np.clip(latency_ms / 1000.0, 0.0, 1.0))
        mission_w = [0.3, 0.6, 1.0, 1.3][self.mission_phase]
        latency = -l * mission_w

        return {"security": security, "power": power, "latency": latency}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        assert self.action_space.contains(action), f"Invalid action {action}"
        algo = CRYPTO_ALGORITHMS[action]
        algo_name = algo.get("name", str(action))

        # Pull performance from shared profiles at nominal GCS resources (4 cores, 1800MHz)
        perf = get_algorithm_performance(algo_name, cores=4, frequency=1800)
        latency_ms = float(perf["latency_ms"])
        power_watts = float(perf["power_watts"])
        security_rating = float(perf["security_rating"])

        comps = self._reward_components(security_rating, latency_ms, power_watts)
        # Sum and clip to [-1,1]
        reward = float(np.clip(comps["security"] + comps["power"] + comps["latency"], -1.0, 1.0))

        # Dynamics: threat and consensus drift; battery drain by power
        rnd = np.random.random()
        if rnd < 0.12:
            delta = float(self._rng.normal(0.0, 0.05))
            self.threat_level = float(np.clip(self.threat_level + delta, 0.0, 1.0))
        if np.random.random() < 0.15:
            # consensus approaches threat with noise
            self.swarm_consensus_threat = float(np.clip(
                0.8 * self.swarm_consensus_threat + 0.2 * self.threat_level + self._rng.normal(0, self._consensus_noise), 0.0, 1.0))

        # Battery drain simplified
        self.avg_fleet_battery = float(max(0.0, self.avg_fleet_battery - (power_watts / 20.0)))

        self.steps += 1
        done = self.steps >= self.max_steps or self.avg_fleet_battery <= 0.0

        state = self._get_state()
        info = {
            "algorithm": algo_name,
            "latency_ms": latency_ms,
            "power_watts": power_watts,
            "security_rating": security_rating,
            "reward_components": comps,
            "reward": reward,
        }
        return state, reward, done, info


class StrategicCryptoAgent:
    """
    Thin wrapper using the shared QLearningAgent with a fixed strategic state/action space.
    """

    def __init__(self,
                 learning_rate: float | None = None,
                 discount_factor: float | None = None,
                 exploration_rate: float | None = None,
                 exploration_decay: float | None = None,
                 min_exploration_rate: float | None = None):
        self.state_dim = 4
        self.action_dim = 4
        self.agent = CryptoDQNAgent(state_dim=self.state_dim, action_dim=self.action_dim)

    def choose_action(self, state: List[float], training: bool = True) -> int:
        return int(self.agent.choose_action(np.asarray(state, dtype=np.float32), training=training))

    def learn(self, state: List[float], action: int, reward: float, next_state: List[float], done: bool):
        self.agent.remember(np.asarray(state, dtype=np.float32), int(action), float(reward), np.asarray(next_state, dtype=np.float32), bool(done))
        self.agent.learn()

    def save_policy(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.agent.save_policy(path)

    def load_policy(self, path: str) -> bool:
        return self.agent.load_policy(path)


def train_strategic_agent(
    episodes: int = 3000,
    eval_every: int = 200,
    out_dir: str = "output",
    seed: int = 42,
    enable_early_stopping: bool = True,
    enable_domain_randomization: bool = False,
) -> StrategicCryptoAgent:
    """Train the strategic crypto agent.

    Adds:
      - Deterministic seeding
      - Periodic evaluation
      - CSV logging of episode + eval scores
      - Consistent filename expected by integration (strategic_crypto_q_table.npy)
    """
    # Setup reproducibility
    set_random_seeds(seed)
    
    np.random.seed(seed)
    env = StrategicCryptoEnv()
    agent = StrategicCryptoAgent(
        learning_rate=CRYPTO_RL.get("LEARNING_RATE", 0.1),
        discount_factor=CRYPTO_RL.get("DISCOUNT_FACTOR", 0.99),
        exploration_rate=CRYPTO_RL.get("EXPLORATION_RATE", 1.0),
        exploration_decay=CRYPTO_RL.get("EXPLORATION_DECAY", 0.9995),
        min_exploration_rate=CRYPTO_RL.get("MIN_EXPLORATION_RATE", 0.01),
    )

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Initialize monitoring systems
    early_stopping = None
    if enable_early_stopping:
        config = create_early_stopping_config("strategic")
        early_stopping = EarlyStopping(**config)
    
    reward_monitor = RewardBalanceMonitor()
    
    # CSV logging with enhanced columns
    csv_path = os.path.join(out_dir, "strategic_training_log.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("episode,reward,eval_score,epsilon\n")

    rewards: List[float] = []
    eval_scores: List[float] = []
    best_eval = -float("inf")
    
    for ep in tqdm(range(episodes), desc="Training Strategic Agent"):
        s = env.reset()
        done = False
        total = 0.0
        step_rewards = []
        
        while not done:
            a = agent.choose_action(s, training=True)
            ns, r, done, info = env.step(a)
            agent.learn(s, a, r, ns, done)
            s = ns
            total += r
            step_rewards.append(r)
            
        rewards.append(total)
        
        # Update monitoring systems
        if step_rewards:
            reward_breakdown = {"total": total, "mean_step": np.mean(step_rewards)}
            reward_monitor.update(reward_breakdown)
        
        # Check early stopping
        should_stop = False
        if early_stopping:
            stop_info = early_stopping.update(total)
            should_stop = stop_info["should_stop"]
            
        eval_mean = ""  # blank unless evaluation interval
        if (ep + 1) % eval_every == 0:
            score = evaluate(agent, env, 20)
            eval_scores.append(score)
            eval_mean = f"{score:.4f}"
            if score > best_eval:
                best_eval = score
                agent.save_policy(os.path.join(out_dir, "strategic_crypto_dqn_best.pt"))
                
        # Enhanced CSV logging
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(f"{ep+1},{total:.4f},{eval_mean},{agent.agent.epsilon:.6f}\n")
            
        # Early stopping check
        if should_stop:
            print(f"\nEarly stopping triggered at episode {ep+1}")
            if early_stopping.converged:
                print("Reason: Reward convergence detected")
            else:
                print("Reason: Patience exceeded without improvement")
            break

    # Final save and metadata
    agent.save_policy(os.path.join(out_dir, "strategic_crypto_dqn.pt"))
    
    # Save training metadata
    training_metadata = {
        "episodes_completed": len(rewards),
        "total_episodes_planned": episodes,
        "best_eval_score": best_eval,
        "final_reward": rewards[-1] if rewards else 0.0,
        "early_stopped": len(rewards) < episodes,
    "agent_epsilon": agent.agent.epsilon
    }
    
    # Caller may persist training_metadata externally if needed.
    
    # Generate monitoring reports
    reward_monitor.export_analysis_report(os.path.join(out_dir, "strategic_reward_analysis.txt"))
    
    try:
        fig = reward_monitor.plot_component_trends(os.path.join(out_dir, "strategic_reward_trends.png"))
        if fig:
            fig.close()
    except Exception as e:
        print(f"Warning: Could not generate reward trend plots: {e}")
    
    return agent


def evaluate(agent: StrategicCryptoAgent, env: StrategicCryptoEnv, episodes: int = 10) -> float:
    scores: List[float] = []
    for _ in range(episodes):
        s = env.reset()
        done = False
        total = 0.0
        while not done:
            a = agent.choose_action(s, training=False)
            ns, r, done, _ = env.step(a)
            s = ns
            total += r
        scores.append(total)
    return float(np.mean(scores))


__all__ = [
    "StrategicCryptoEnv",
    "StrategicCryptoAgent",
    "train_strategic_agent",
    "evaluate",
]
