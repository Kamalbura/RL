#!/usr/bin/env python3
"""
Generate comprehensive validation graphs from CSV results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_validation_graphs():
    """Generate comprehensive validation visualizations"""
    
    # Check if validation results exist
    results_dir = "validation_results"
    if not os.path.exists(results_dir):
        print("No validation results found. Run 'python validation.py' first.")
        return
    
    # Load tactical results
    tactical_file = os.path.join(results_dir, "tactical_validation_results.csv")
    tactical_summary_file = os.path.join(results_dir, "tactical_validation_summary.csv")
    
    if os.path.exists(tactical_file):
        print("📊 Generating Tactical Agent Visualizations...")
        
        # Load data
        tactical_df = pd.read_csv(tactical_file)
        if os.path.exists(tactical_summary_file):
            tactical_summary = pd.read_csv(tactical_summary_file)
        
        # 1. Performance Comparison Bar Chart
        plt.figure(figsize=(14, 8))
        if os.path.exists(tactical_summary_file):
            sns.barplot(data=tactical_summary, x='scenario', y='mean_reward', hue='algorithm')
            plt.title('Tactical Agent: Performance Comparison vs Baselines', fontsize=16, fontweight='bold')
            plt.ylabel('Mean Total Reward', fontsize=12)
            plt.xlabel('Scenario', fontsize=12)
            plt.xticks(rotation=45)
            plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'tactical_performance_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Reward Distribution Box Plot
        plt.figure(figsize=(16, 10))
        sns.boxplot(data=tactical_df, x='scenario', y='total_reward', hue='algorithm')
        plt.title('Tactical Agent: Reward Distribution Analysis', fontsize=16, fontweight='bold')
        plt.ylabel('Total Reward', fontsize=12)
        plt.xlabel('Scenario', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'tactical_reward_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Performance by Scenario Heatmap
        if os.path.exists(tactical_summary_file):
            pivot_data = tactical_summary.pivot(index='algorithm', columns='scenario', values='mean_reward')
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', center=0, fmt='.0f')
            plt.title('Tactical Agent: Performance Heatmap by Scenario', fontsize=16, fontweight='bold')
            plt.ylabel('Algorithm', fontsize=12)
            plt.xlabel('Scenario', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'tactical_performance_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Statistical Analysis
        plt.figure(figsize=(14, 6))
        
        # Subplot 1: Mean rewards
        plt.subplot(1, 2, 1)
        if os.path.exists(tactical_summary_file):
            for algo in tactical_summary['algorithm'].unique():
                algo_data = tactical_summary[tactical_summary['algorithm'] == algo]
                plt.bar(algo_data['scenario'], algo_data['mean_reward'], alpha=0.7, label=algo)
            plt.title('Mean Rewards by Scenario', fontweight='bold')
            plt.ylabel('Mean Reward')
            plt.xlabel('Scenario')
            plt.xticks(rotation=45)
            plt.legend()
        
        # Subplot 2: Standard deviation
        plt.subplot(1, 2, 2)
        if os.path.exists(tactical_summary_file):
            for algo in tactical_summary['algorithm'].unique():
                algo_data = tactical_summary[tactical_summary['algorithm'] == algo]
                plt.bar(algo_data['scenario'], algo_data['std_reward'], alpha=0.7, label=algo)
            plt.title('Standard Deviation by Scenario', fontweight='bold')
            plt.ylabel('Std Reward')
            plt.xlabel('Scenario')
            plt.xticks(rotation=45)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'tactical_statistical_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Tactical visualizations saved:")
        print("  - tactical_performance_comparison.png")
        print("  - tactical_reward_distribution.png") 
        print("  - tactical_performance_heatmap.png")
        print("  - tactical_statistical_analysis.png")
    
    # Load strategic results if available
    strategic_file = os.path.join(results_dir, "strategic_validation_results.csv")
    strategic_summary_file = os.path.join(results_dir, "strategic_validation_summary.csv")
    
    if os.path.exists(strategic_file):
        print("\n📊 Generating Strategic Agent Visualizations...")
        
        strategic_df = pd.read_csv(strategic_file)
        if os.path.exists(strategic_summary_file):
            strategic_summary = pd.read_csv(strategic_summary_file)
        
        # Strategic agent visualizations (similar structure)
        plt.figure(figsize=(14, 8))
        if os.path.exists(strategic_summary_file):
            sns.barplot(data=strategic_summary, x='scenario', y='mean_reward', hue='algorithm')
            plt.title('Strategic Agent: Performance Comparison vs Baselines', fontsize=16, fontweight='bold')
            plt.ylabel('Mean Total Reward', fontsize=12)
            plt.xlabel('Scenario', fontsize=12)
            plt.xticks(rotation=45)
            plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'strategic_performance_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✅ Strategic visualizations saved:")
        print("  - strategic_performance_comparison.png")
    
    # Generate summary report
    print("\n📋 Generating Performance Summary Report...")
    
    summary_report = []
    summary_report.append("# Hierarchical RL UAV System - Validation Results\n")
    
    if os.path.exists(tactical_summary_file):
        tactical_summary = pd.read_csv(tactical_summary_file)
        summary_report.append("## Tactical Agent Performance\n")
        
        # Find best performing algorithm per scenario
        for scenario in tactical_summary['scenario'].unique():
            scenario_data = tactical_summary[tactical_summary['scenario'] == scenario]
            best_algo = scenario_data.loc[scenario_data['mean_reward'].idxmax()]
            summary_report.append(f"**{scenario.upper()}**: {best_algo['algorithm']} (Reward: {best_algo['mean_reward']:.1f})\n")
        
        # Overall performance
        rl_performance = tactical_summary[tactical_summary['algorithm'] == 'RL_Agent']['mean_reward'].mean()
        baseline_performance = tactical_summary[tactical_summary['algorithm'] != 'RL_Agent']['mean_reward'].mean()
        improvement = ((rl_performance - baseline_performance) / abs(baseline_performance)) * 100
        
        summary_report.append(f"\n**Overall RL Agent Performance**: {rl_performance:.1f} avg reward\n")
        summary_report.append(f"**Baseline Average**: {baseline_performance:.1f} avg reward\n")
        summary_report.append(f"**Improvement**: {improvement:.1f}%\n")
    
    # Save summary report
    with open(os.path.join(results_dir, 'performance_summary.md'), 'w') as f:
        f.writelines(summary_report)
    
    print("✅ Performance summary saved: performance_summary.md")
    print(f"\n🎯 All validation visualizations available in: {results_dir}/")

if __name__ == "__main__":
    create_validation_graphs()
