"""
Strategic (GCS-side) Crypto RL agent and minimal environment.

Design goals
- Minimal state: [Threat, AvgFleetBattery, MissionPhase]
- Actions: 4 crypto algorithms from config.crypto_config.CRYPTO_ALGORITHMS
- Reward: Balance security strength vs. battery impact and latency under mission pressure

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
from crypto_rl.rl_agent import QLearningAgent
from utils.reproducibility import set_random_seeds
from utils.early_stopping import EarlyStopping, create_early_stopping_config
from utils.reward_monitor import RewardBalanceMonitor


class StrategicCryptoEnv:
    """
    Lightweight environment for GCS-side crypto selection.

    State: [Threat(0-2), AvgBattery(0-2), MissionPhase(0-3)]
      - Threat: 0=LOW, 1=ELEVATED, 2=CRITICAL
      - AvgBattery: 0=CRITICAL, 1=DEGRADING, 2=HEALTHY
      - MissionPhase: 0=IDLE, 1=TRANSIT, 2=TASK, 3=CRITICAL_TASK

    Action: index into CRYPTO_ALGORITHMS (0..3) mapping to algorithms.

    Reward: security_reward(threat) - power_penalty(battery) - latency_penalty(mission)
    """

    def __init__(self):
        self.state_dims = [3, 3, 4]
        self.action_dim = 4
        self.max_steps = 200
        self.steps = 0
        self._rng = np.random.default_rng()
        self.reset()

    def reset(self) -> List[int]:
        self.steps = 0
        # Start reasonably safe defaults
        self.threat = int(self._rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15]))
        self.battery = 2  # HEALTHY
        self.mission = int(self._rng.choice([0, 1, 2, 3], p=[0.35, 0.3, 0.25, 0.1]))
        return [self.threat, self.battery, self.mission]

    def _security_reward(self, algo_security: float) -> float:
        # Weight security more when threat increases
        weights = [0.6, 1.0, 1.6]
        return algo_security * weights[self.threat]

    def _power_penalty(self, power_mult: float) -> float:
        # Penalize power more when battery is low
        weights = [2.0, 1.0, 0.5]
        return power_mult * 10.0 * weights[self.battery]

    def _latency_penalty(self, latency_ms: float) -> float:
        # Penalize latency more in critical missions
        weights = [0.25, 0.5, 1.0, 1.5]
        return (latency_ms / 10.0) * weights[self.mission]

    def step(self, action: int) -> Tuple[List[int], float, bool, Dict]:
        assert 0 <= action < self.action_dim, f"Invalid action {action}"
        algo = CRYPTO_ALGORITHMS[action]
        security = float(algo.get("security_rating", 5))
        power_mult = float(algo.get("power_multiplier", 1.0))
        latency_ms = float(algo.get("latency_ms", 5.0))

        reward = self._security_reward(security)
        reward -= self._power_penalty(power_mult)
        reward -= self._latency_penalty(latency_ms)

        # Simple drift: threat and mission can change; battery degrades slowly if power heavy
        if np.random.random() < 0.1:
            self.threat = int(np.clip(self.threat + self._rng.choice([-1, 0, 1], p=[0.15, 0.7, 0.15]), 0, 2))
        if np.random.random() < 0.12:
            self.mission = int(np.clip(self.mission + self._rng.choice([-1, 0, 1], p=[0.25, 0.5, 0.25]), 0, 3))

        # Battery trend
        drain = 0.02 * power_mult  # abstract units per step
        if drain > 0.03 and np.random.random() < 0.7:
            self.battery = max(0, self.battery - 1) if np.random.random() < 0.3 else self.battery

        self.steps += 1
        done = self.steps >= self.max_steps
        state = [self.threat, self.battery, self.mission]
        info = {"algorithm": algo.get("name", str(action)), "security_rating": security,
                "power_multiplier": power_mult, "latency_ms": latency_ms, "reward": reward}
        return state, float(reward), done, info


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
        self.state_dims = [3, 3, 4]
        self.action_dim = 4
        self.agent = QLearningAgent(
            state_dims=self.state_dims,
            action_dim=self.action_dim,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_rate=exploration_rate,
            exploration_decay=exploration_decay,
            min_exploration_rate=min_exploration_rate,
        )

    def choose_action(self, state: List[int], training: bool = True) -> int:
        return int(self.agent.choose_action(state, training=training))

    def learn(self, state: List[int], action: int, reward: float, next_state: List[int], done: bool):
        self.agent.learn(state, action, reward, next_state, done)

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
        f.write("episode,reward,eval_score,epsilon,avg_q_value,state_coverage,action_entropy\n")

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
        
        # Get agent statistics
        policy_summary = agent.agent.get_policy_summary()
        avg_q = policy_summary.get("average_q_value", 0.0)
        state_coverage = policy_summary.get("state_coverage_percentage", 0.0)
        action_entropy = policy_summary.get("action_entropy", 0.0)
        
        # Check early stopping
        should_stop = False
        if early_stopping:
            stop_info = early_stopping.update(total, avg_q)
            should_stop = stop_info["should_stop"]
            
        eval_mean = ""  # blank unless evaluation interval
        if (ep + 1) % eval_every == 0:
            score = evaluate(agent, env, 20)
            eval_scores.append(score)
            eval_mean = f"{score:.4f}"
            if score > best_eval:
                best_eval = score
                agent.save_policy(os.path.join(out_dir, "strategic_crypto_q_table_best.npy"))
                
        # Enhanced CSV logging
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(f"{ep+1},{total:.4f},{eval_mean},{agent.agent.epsilon:.6f},{avg_q:.6f},{state_coverage:.2f},{action_entropy:.4f}\n")
            
        # Early stopping check
        if should_stop:
            print(f"\nEarly stopping triggered at episode {ep+1}")
            if early_stopping.converged:
                print("Reason: Reward convergence detected")
            else:
                print("Reason: Patience exceeded without improvement")
            break

    # Final save and metadata
    agent.save_policy(os.path.join(out_dir, "strategic_crypto_q_table.npy"))
    
    # Save training metadata
    training_metadata = {
        "episodes_completed": len(rewards),
        "total_episodes_planned": episodes,
        "best_eval_score": best_eval,
        "final_reward": rewards[-1] if rewards else 0.0,
        "early_stopped": len(rewards) < episodes,
        "agent_summary": agent.get_policy_summary()
    }
    
    save_run_metadata(
        metadata, 
        training_metadata, 
        os.path.join(out_dir, "strategic_crypto_q_table.npy"),
        out_dir
    )
    
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
