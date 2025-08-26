"""
Reward balance monitoring utilities for analyzing reward component contributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import seaborn as sns


class RewardBalanceMonitor:
    """Monitor and analyze reward component balance during training"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.component_history = defaultdict(lambda: deque(maxlen=window_size))
        self.total_rewards = deque(maxlen=window_size)
        self.episodes_tracked = 0
        
    def update(self, reward_breakdown: Dict[str, float]):
        """Update with new reward breakdown"""
        total = reward_breakdown.get('total', 0.0)
        self.total_rewards.append(total)
        
        for component, value in reward_breakdown.items():
            if component != 'total':
                self.component_history[component].append(value)
        
        self.episodes_tracked += 1
        
    def get_balance_analysis(self) -> Dict[str, any]:
        """Analyze reward component balance"""
        if self.episodes_tracked < 10:
            return {"insufficient_data": True}
            
        analysis = {
            "episodes_analyzed": min(self.episodes_tracked, self.window_size),
            "component_stats": {},
            "balance_issues": {},
            "recommendations": []
        }
        
        # Analyze each component
        total_magnitude = 0
        for component, values in self.component_history.items():
            if not values:
                continue
                
            values_list = list(values)
            mean_val = np.mean(values_list)
            std_val = np.std(values_list)
            magnitude = abs(mean_val)
            total_magnitude += magnitude
            
            analysis["component_stats"][component] = {
                "mean": mean_val,
                "std": std_val,
                "magnitude": magnitude,
                "contribution_ratio": 0.0,  # Will be calculated below
                "variance_coefficient": std_val / (abs(mean_val) + 1e-8)
            }
        
        # Calculate contribution ratios
        for component in analysis["component_stats"]:
            magnitude = analysis["component_stats"][component]["magnitude"]
            analysis["component_stats"][component]["contribution_ratio"] = magnitude / (total_magnitude + 1e-8)
            
        # Detect balance issues
        self._detect_balance_issues(analysis)
        
        return analysis
        
    def _detect_balance_issues(self, analysis: Dict):
        """Detect common reward balance issues"""
        issues = {}
        recommendations = []
        
        component_stats = analysis["component_stats"]
        
        # Check for dominant components (>70% of total magnitude)
        for component, stats in component_stats.items():
            if stats["contribution_ratio"] > 0.7:
                issues[f"{component}_dominant"] = True
                recommendations.append(f"Reduce {component} weight - it dominates reward signal")
                
        # Check for negligible components (<2% of total magnitude)  
        for component, stats in component_stats.items():
            if stats["contribution_ratio"] < 0.02:
                issues[f"{component}_negligible"] = True
                recommendations.append(f"Increase {component} weight - minimal impact on learning")
                
        # Check for high variance components (CV > 2.0)
        for component, stats in component_stats.items():
            if stats["variance_coefficient"] > 2.0:
                issues[f"{component}_high_variance"] = True
                recommendations.append(f"Stabilize {component} - high variance may hurt learning")
                
        # Check for opposing forces (negative correlation between components)
        self._check_component_correlations(analysis, issues, recommendations)
        
        analysis["balance_issues"] = issues
        analysis["recommendations"] = recommendations
        
    def _check_component_correlations(self, analysis: Dict, issues: Dict, recommendations: List[str]):
        """Check for problematic correlations between reward components"""
        components = list(self.component_history.keys())
        
        for i, comp1 in enumerate(components):
            for comp2 in components[i+1:]:
                if len(self.component_history[comp1]) < 20 or len(self.component_history[comp2]) < 20:
                    continue
                    
                values1 = list(self.component_history[comp1])[-20:]
                values2 = list(self.component_history[comp2])[-20:]
                
                correlation = np.corrcoef(values1, values2)[0, 1]
                
                if correlation < -0.7:
                    issues[f"{comp1}_{comp2}_opposing"] = True
                    recommendations.append(f"Strong negative correlation between {comp1} and {comp2} - may cause conflicting signals")
                    
    def plot_component_trends(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot reward component trends over time"""
        if not self.component_history:
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Reward Component Analysis', fontsize=16)
        
        # Plot 1: Component values over time
        ax1 = axes[0, 0]
        for component, values in self.component_history.items():
            if values:
                ax1.plot(list(values), label=component, alpha=0.7)
        ax1.set_title('Component Values Over Time')
        ax1.set_xlabel('Episode (Recent)')
        ax1.set_ylabel('Reward Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Component magnitudes (bar chart)
        ax2 = axes[0, 1]
        components = []
        magnitudes = []
        for component, values in self.component_history.items():
            if values:
                components.append(component)
                magnitudes.append(np.mean([abs(v) for v in values]))
        
        if components:
            bars = ax2.bar(components, magnitudes)
            ax2.set_title('Average Component Magnitudes')
            ax2.set_ylabel('Average |Value|')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            
            # Color bars by magnitude
            max_mag = max(magnitudes) if magnitudes else 1
            for bar, mag in zip(bars, magnitudes):
                bar.set_color(plt.cm.viridis(mag / max_mag))
        
        # Plot 3: Total reward trend
        ax3 = axes[1, 0]
        if self.total_rewards:
            rewards = list(self.total_rewards)
            ax3.plot(rewards, 'b-', alpha=0.7, label='Total Reward')
            
            # Add moving average
            if len(rewards) > 10:
                window = min(20, len(rewards) // 2)
                ma = pd.Series(rewards).rolling(window).mean()
                ax3.plot(ma, 'r-', linewidth=2, label=f'MA({window})')
                
        ax3.set_title('Total Reward Trend')
        ax3.set_xlabel('Episode (Recent)')
        ax3.set_ylabel('Total Reward')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Component contribution pie chart
        ax4 = axes[1, 1]
        if components and magnitudes:
            # Only show components with >1% contribution
            filtered_components = []
            filtered_magnitudes = []
            total_mag = sum(magnitudes)
            
            for comp, mag in zip(components, magnitudes):
                if mag / total_mag > 0.01:
                    filtered_components.append(comp)
                    filtered_magnitudes.append(mag)
                    
            if filtered_components:
                ax4.pie(filtered_magnitudes, labels=filtered_components, autopct='%1.1f%%')
                ax4.set_title('Component Contribution Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def export_analysis_report(self, filepath: str):
        """Export detailed analysis report to file"""
        analysis = self.get_balance_analysis()
        
        with open(filepath, 'w') as f:
            f.write("# Reward Balance Analysis Report\n\n")
            f.write(f"Episodes Analyzed: {analysis.get('episodes_analyzed', 0)}\n\n")
            
            if analysis.get("insufficient_data"):
                f.write("Insufficient data for analysis (need at least 10 episodes)\n")
                return
                
            f.write("## Component Statistics\n\n")
            for component, stats in analysis["component_stats"].items():
                f.write(f"### {component}\n")
                f.write(f"- Mean: {stats['mean']:.4f}\n")
                f.write(f"- Std Dev: {stats['std']:.4f}\n")
                f.write(f"- Contribution: {stats['contribution_ratio']*100:.1f}%\n")
                f.write(f"- Variance Coefficient: {stats['variance_coefficient']:.3f}\n\n")
                
            if analysis["balance_issues"]:
                f.write("## Balance Issues Detected\n\n")
                for issue in analysis["balance_issues"]:
                    f.write(f"- {issue}\n")
                f.write("\n")
                
            if analysis["recommendations"]:
                f.write("## Recommendations\n\n")
                for rec in analysis["recommendations"]:
                    f.write(f"- {rec}\n")


def analyze_reward_logs(csv_path: str) -> RewardBalanceMonitor:
    """Analyze reward balance from training CSV logs"""
    monitor = RewardBalanceMonitor()
    
    try:
        df = pd.read_csv(csv_path)
        
        # Look for reward breakdown columns
        reward_columns = [col for col in df.columns if 'reward' in col.lower()]
        
        for _, row in df.iterrows():
            breakdown = {}
            for col in reward_columns:
                if not pd.isna(row[col]):
                    breakdown[col] = row[col]
            
            if breakdown:
                monitor.update(breakdown)
                
    except Exception as e:
        print(f"Error analyzing reward logs: {e}")
        
    return monitor
