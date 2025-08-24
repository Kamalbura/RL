"""
Validation framework stubs for crypto RL vs. baseline.
Runs scenarios in the existing CryptoEnv and reports simple metrics.
"""

from __future__ import annotations

import os
import statistics
from typing import Dict, List

import numpy as np

from crypto_rl.crypto_simulator import CryptoEnv
from crypto_rl.rl_agent import QLearningAgent


def run_baseline(env: CryptoEnv, policy_index: int, episodes: int = 20) -> Dict[str, float]:
    rewards: List[float] = []
    for _ in range(episodes):
        s = env.reset()
        done = False
        total = 0.0
        while not done:
            a = policy_index  # fixed policy
            s, r, done, _ = env.step(a)
            total += r
        rewards.append(total)
    return {"avg_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards))}


def run_with_rl(env: CryptoEnv, agent: QLearningAgent, episodes: int = 20) -> Dict[str, float]:
    rewards: List[float] = []
    for _ in range(episodes):
        s = env.reset()
        done = False
        total = 0.0
        while not done:
            a = int(agent.choose_action(s, training=False))
            s, r, done, _ = env.step(a)
            total += r
        rewards.append(total)
    return {"avg_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards))}


def compare_baseline_vs_rl(policy_index: int = 0, rl_policy_path: str | None = None) -> Dict[str, Dict[str, float]]:
    env = CryptoEnv()
    # Prepare RL agent matching env state/action dims
    agent = QLearningAgent(state_dims=[4, 4, 3, 4, 3, 3], action_dim=4)
    if rl_policy_path and os.path.exists(rl_policy_path):
        agent.load_policy(rl_policy_path)
    baseline = run_baseline(env, policy_index)
    with_rl = run_with_rl(env, agent)
    return {"baseline": baseline, "rl": with_rl}


__all__ = ["run_baseline", "run_with_rl", "compare_baseline_vs_rl"]
