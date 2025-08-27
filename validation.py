"""
Comprehensive Validation Framework

This module provides a framework for systematically validating and comparing
the performance of different RL agents and baseline algorithms across a
standardized set of scenarios.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

# Import the correct environment and agent classes
from ddos_rl.env import TacticalUAVEnv
from crypto_rl.strategic_agent import StrategicCryptoEnv
from ddos_rl.agent import QLearningAgent

class ComprehensiveValidationFramework:
    """
    A framework for comprehensive validation of RL agents against baselines.
    """
    
    def __init__(self, output_dir: str = "validation_results"):
        """
        Initialize the validation framework.
        
        Args:
            output_dir: Directory to save validation results and visualizations.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define standard scenarios for tactical validation
        self.tactical_scenarios = {
            "baseline": {"threat_level": 0, "battery_percent": 90},
            "high_threat": {"threat_level": 3, "battery_percent": 90},
            "low_battery": {"threat_level": 1, "battery_percent": 25},
            "critical_all": {"threat_level": 3, "battery_percent": 15}
        }
        
        # Define standard scenarios for strategic validation
        self.strategic_scenarios = {
            "peace_time": {"swarm_threat_level": 0, "fleet_battery_status": 0},
            "under_attack": {"swarm_threat_level": 2, "fleet_battery_status": 0},
            "low_endurance": {"swarm_threat_level": 1, "fleet_battery_status": 2}
        }

    def run_validation(self, episodes_per_scenario: int = 30):
        """
        Run a comprehensive validation across all defined scenarios and algorithms.
        
        Args:
            episodes_per_scenario: The number of episodes to run for each validation.
        """
        print("--- Starting Comprehensive Validation ---")
        
        # --- Tactical Validation ---
        print("\n--- Tactical Agent Validation ---")
        tactical_results = self._run_tactical_validation(episodes_per_scenario)
        tactical_summary = self._summarize_results(tactical_results)
        self._save_results(tactical_results, "tactical_validation_results.csv")
        self._save_results(tactical_summary, "tactical_validation_summary.csv")
        self._generate_visualizations(tactical_summary, "Tactical Agent Performance")
        
        # --- Strategic Validation ---
        print("\n--- Strategic Agent Validation ---")
        strategic_results = self._run_strategic_validation(episodes_per_scenario)
        strategic_summary = self._summarize_results(strategic_results)
        self._save_results(strategic_results, "strategic_validation_results.csv")
        self._save_results(strategic_summary, "strategic_validation_summary.csv")
        self._generate_visualizations(strategic_summary, "Strategic Agent Performance")
        
        print("\n--- Validation Complete ---")

    def _run_tactical_validation(self, episodes: int) -> pd.DataFrame:
        """Run validation for the tactical agent."""
        results = []
        env = TacticalUAVEnv()
        
        # Load the trained RL agent
        rl_agent = QLearningAgent(state_dims=[4, 4, 3, 3], action_dim=9)
        rl_agent.load_policy("tactical_policy.npy")
        
        algorithms = {
            "RL_Agent": lambda state: rl_agent.choose_action(state, training=False),
            "Baseline_PowerSave": lambda state: 0, # XGBoost, Powersave (action 0)
            "Baseline_HighPerf": lambda state: 7 # TST, Turbo (action 7)
        }
        
        for scenario_name, scenario_params in self.tactical_scenarios.items():
            for algo_name, policy_fn in algorithms.items():
                for i in range(episodes):
                    state = env.reset()
                    # Set scenario conditions
                    env.threat_level_idx = scenario_params['threat_level']
                    env.battery_percentage = scenario_params['battery_percent']
                    env._update_battery_state()
                    
                    done = False
                    total_reward = 0
                    
                    while not done:
                        action = policy_fn(state)
                        state, reward, done, info = env.step(action)
                        total_reward += reward
                    
                    results.append({
                        "scenario": scenario_name,
                        "algorithm": algo_name,
                        "episode": i,
                        "total_reward": total_reward,
                        "final_battery": info['battery_percentage']
                    })
        return pd.DataFrame(results)

    def _run_strategic_validation(self, episodes: int) -> pd.DataFrame:
        """Run validation for the strategic agent."""
        results = []
        env = StrategicCryptoEnv()
        
        # Load the trained RL agent
        rl_agent = QLearningAgent(state_dims=[3, 3, 4], action_dim=4)
        rl_agent.load_policy("strategic_policy.npy")
        
        algorithms = {
            "RL_Agent": lambda state: rl_agent.choose_action(state, training=False),
            "Baseline_Lightweight": lambda state: 0, # ASCON_128
            "Baseline_MaxSecurity": lambda state: 2  # SPHINCS
        }
        
        for scenario_name, scenario_params in self.strategic_scenarios.items():
            for algo_name, policy_fn in algorithms.items():
                for i in range(episodes):
                    state = env.reset()
                    # Set scenario conditions
                    env.swarm_threat_level_idx = scenario_params['swarm_threat_level']
                    env.fleet_battery_status_idx = scenario_params['fleet_battery_status']
                    
                    done = False
                    total_reward = 0
                    
                    while not done:
                        action = policy_fn(state)
                        state, reward, done, info = env.step(action)
                        total_reward += reward
                    
                    results.append({
                        "scenario": scenario_name,
                        "algorithm": algo_name,
                        "episode": i,
                        "total_reward": total_reward,
                        "final_fleet_battery": info['fleet_battery_mean']
                    })
        return pd.DataFrame(results)

    def _summarize_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Summarize the validation results."""
        summary = results_df.groupby(['scenario', 'algorithm']).agg(
            mean_reward=('total_reward', 'mean'),
            std_reward=('total_reward', 'std'),
            min_reward=('total_reward', 'min'),
            max_reward=('total_reward', 'max')
        ).reset_index()
        return summary

    def _save_results(self, df: pd.DataFrame, filename: str):
        """Save results to a CSV file."""
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        print(f"Saved results to {path}")

    def _generate_visualizations(self, summary_df: pd.DataFrame, title_prefix: str):
        """Generate and save publication-quality visualizations."""
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Bar plot of mean rewards
        plt.figure(figsize=(12, 7))
        sns.barplot(data=summary_df, x='scenario', y='mean_reward', hue='algorithm')
        plt.title(f'{title_prefix}: Mean Reward Comparison')
        plt.ylabel('Mean Total Reward')
        plt.xlabel('Scenario')
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{title_prefix.lower().replace(' ', '_')}_rewards.png"))
        plt.close()
        
        # Box plot for reward distribution
        # This requires the raw results, so we'll assume it's passed or re-read
        raw_results_path = os.path.join(self.output_dir, f"{title_prefix.lower().replace(' ', '_').split('_')[0]}_validation_results.csv")
        if os.path.exists(raw_results_path):
            raw_df = pd.read_csv(raw_results_path)
            plt.figure(figsize=(14, 8))
            sns.boxplot(data=raw_df, x='scenario', y='total_reward', hue='algorithm')
            plt.title(f'{title_prefix}: Reward Distribution')
            plt.ylabel('Total Reward')
            plt.xlabel('Scenario')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{title_prefix.lower().replace(' ', '_')}_reward_distribution.png"))
            plt.close()

if __name__ == '__main__':
    validation_framework = ComprehensiveValidationFramework()
    validation_framework.run_validation(episodes_per_scenario=50)
