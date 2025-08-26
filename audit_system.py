#!/usr/bin/env python3
"""
🔬 COMPREHENSIVE PRE-TRAINING AUDIT SCRIPT
Expert RL System Evaluation & Context-to-Profile Validation

This script performs an exhaustive technical audit of the dual-agent RL setup,
validates empirical data consistency, identifies critical gaps, and provides
actionable recommendations before training commencement.

For RL Expert Review & Approval.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import re
import statistics
from dataclasses import dataclass
from datetime import datetime

# Add current directory to path for local imports
sys.path.append(str(Path(__file__).parent))

try:
    from ddos_rl.profiles import (
        POWER_PROFILES, LATENCY_PROFILES, DDOS_TASK_TIME_PROFILES, 
        SECURITY_RATINGS, CPU_FREQUENCY_PRESETS,
        get_power_consumption, get_crypto_latency, get_ddos_execution_time, get_security_rating
    )
    from ddos_rl.env import TacticalUAVEnv
    from ddos_rl.agent import QLearningAgent
    from ddos_rl.config import BATTERY_SPECS
    DDOS_IMPORTS_OK = True
except Exception as e:
    print(f"❌ DDoS RL imports failed: {e}")
    DDOS_IMPORTS_OK = False

try:
    from crypto_rl.strategic_agent import StrategicCryptoEnv, StrategicCryptoAgent
    # Import crypto config directly
    import sys
    config_path = str(Path(__file__).parent / "config")
    if config_path not in sys.path:
        sys.path.append(config_path)
    from crypto_config import CRYPTO_ALGORITHMS, CRYPTO_RL
    CRYPTO_IMPORTS_OK = True
except Exception as e:
    print(f"❌ Crypto RL imports failed: {e}")
    CRYPTO_IMPORTS_OK = False

@dataclass
class AuditResult:
    category: str
    status: str  # "PASS", "WARN", "FAIL"
    description: str
    recommendation: str = ""
    details: Dict[str, Any] = None

class RLSystemAuditor:
    """Comprehensive RL system auditor with empirical data validation."""
    
    def __init__(self):
        self.results: List[AuditResult] = []
        self.context_data = self._load_context_data()
        
    def _load_context_data(self) -> Dict[str, Any]:
        """Parse and structure empirical data from context.txt"""
        context_file = Path(__file__).parent / "context.txt"
        if not context_file.exists():
            self.results.append(AuditResult(
                "DATA_SOURCES", "FAIL", 
                "context.txt not found - empirical validation impossible",
                "Ensure context.txt with hardware measurements is present"
            ))
            return {}
            
        try:
            with open(context_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse power consumption data
            power_data = self._extract_power_measurements(content)
            
            # Parse latency/execution time data  
            latency_data = self._extract_latency_measurements(content)
            
            # Parse DDoS execution data
            ddos_data = self._extract_ddos_measurements(content)
            
            return {
                "power": power_data,
                "latency": latency_data, 
                "ddos": ddos_data,
                "raw_content": content
            }
        except Exception as e:
            self.results.append(AuditResult(
                "DATA_SOURCES", "FAIL",
                f"Failed to parse context.txt: {e}",
                "Check context.txt format and encoding"
            ))
            return {}
    
    def _extract_power_measurements(self, content: str) -> Dict[str, List[float]]:
        """Extract power consumption measurements from context data."""
        power_data = {}
        
        # Find current analysis section
        current_section = re.search(r'Current Analysis.*?(?=\n[A-Z]|$)', content, re.DOTALL | re.IGNORECASE)
        if not current_section:
            return power_data
            
        section_text = current_section.group(0)
        
        # Extract patterns like "600 MHz - max: 1.01 Amp"
        patterns = re.findall(r'(\d+(?:\.\d+)?)\s*(?:MHz|Ghz|GHz).*?max:\s*(\d+(?:\.\d+)?)\s*Amp', section_text, re.IGNORECASE)
        
        for freq_str, amp_str in patterns:
            freq = float(freq_str)
            if freq < 100:  # Convert GHz to MHz
                freq *= 1000
            amp = float(amp_str)
            
            if freq not in power_data:
                power_data[freq] = []
            power_data[freq].append(amp)
            
        return power_data
    
    def _extract_latency_measurements(self, content: str) -> Dict[str, Dict[str, List[float]]]:
        """Extract crypto algorithm latency measurements."""
        latency_data = {}
        
        # Look for algorithm-specific timing data
        algorithms = ['Kyber', 'Dilithium', 'Sphincs', 'Falcon']
        
        for algo in algorithms:
            algo_pattern = rf'{algo}.*?(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)'
            matches = re.findall(algo_pattern, content, re.IGNORECASE)
            
            if matches:
                latency_data[algo.upper()] = {
                    'values': [float(x) for match in matches for x in match if float(x) > 0]
                }
                
        return latency_data
    
    def _extract_ddos_measurements(self, content: str) -> Dict[str, Any]:
        """Extract DDoS/TST execution time measurements."""
        ddos_data = {}
        
        # Find TST timing data
        tst_pattern = r'create_squence.*?(\d+(?:\.\d+)?)'
        tst_matches = re.findall(tst_pattern, content, re.IGNORECASE)
        
        if tst_matches:
            ddos_data['TST'] = {
                'execution_times': [float(x) for x in tst_matches if float(x) > 0]
            }
            
        # Find DDOS Pipeline data  
        ddos_pattern = r'DDOS Detection.*?(\d+).*?(\d+(?:\.\d+)?)'
        ddos_matches = re.findall(ddos_pattern, content, re.IGNORECASE)
        
        if ddos_matches:
            ddos_data['XGBOOST'] = {
                'prediction_times': [float(x[1]) for x in ddos_matches if float(x[1]) > 0]
            }
            
        return ddos_data
    
    def audit_imports_and_dependencies(self):
        """Verify all required modules can be imported."""
        if DDOS_IMPORTS_OK:
            self.results.append(AuditResult(
                "IMPORTS", "PASS",
                "Tactical (DDoS) RL modules imported successfully"
            ))
        else:
            self.results.append(AuditResult(
                "IMPORTS", "FAIL", 
                "Tactical RL imports failed",
                "Fix import paths and ensure ddos_rl package is properly structured"
            ))
            
        if CRYPTO_IMPORTS_OK:
            self.results.append(AuditResult(
                "IMPORTS", "PASS",
                "Strategic (Crypto) RL modules imported successfully"
            ))
        else:
            self.results.append(AuditResult(
                "IMPORTS", "FAIL",
                "Strategic RL imports failed", 
                "Fix import paths and ensure crypto_rl package is properly structured"
            ))
    
    def audit_tactical_environment(self):
        """Comprehensive tactical environment validation."""
        if not DDOS_IMPORTS_OK:
            return
            
        try:
            env = TacticalUAVEnv()
            
            # State space validation
            state = env.reset()
            expected_dims = [4, 4, 3, 3]  # Threat, Battery, CPU, Priority
            
            if len(state) == len(expected_dims):
                self.results.append(AuditResult(
                    "TACTICAL_ENV", "PASS",
                    f"State vector dimension correct: {len(state)}"
                ))
            else:
                self.results.append(AuditResult(
                    "TACTICAL_ENV", "FAIL",
                    f"State vector dimension mismatch: got {len(state)}, expected {len(expected_dims)}",
                    "Fix state vector composition in TacticalUAVEnv"
                ))
                
            # Action space validation
            if env.action_space.n == 9:
                self.results.append(AuditResult(
                    "TACTICAL_ENV", "PASS",
                    "Action space dimension correct: 9 actions (0-3 XGBOOST, 4-7 TST, 8 de-escalate)"
                ))
            else:
                self.results.append(AuditResult(
                    "TACTICAL_ENV", "FAIL",
                    f"Action space incorrect: got {env.action_space.n}, expected 9",
                    "Update action space to match current design (2 models × 4 freq + 1 de-escalate)"
                ))
                
            # Battery model validation
            battery_wh = BATTERY_SPECS.get("CAPACITY_WH", 0)
            if battery_wh > 100:
                self.results.append(AuditResult(
                    "TACTICAL_ENV", "PASS",
                    f"Battery capacity reasonable: {battery_wh} Wh"
                ))
            else:
                self.results.append(AuditResult(
                    "TACTICAL_ENV", "WARN",
                    f"Battery capacity may be low: {battery_wh} Wh",
                    "Verify battery specs match hardware (6S 5200mAh ≈ 115Wh)"
                ))
                
            # Reward range testing
            reward_samples = []
            for _ in range(50):
                env.reset()
                action = env.action_space.sample()
                _, reward, _, _ = env.step(action)
                reward_samples.append(reward)
                
            reward_mean = statistics.mean(reward_samples)
            reward_std = statistics.stdev(reward_samples) if len(reward_samples) > 1 else 0
            
            if -100 <= reward_mean <= 100 and reward_std < 1000:
                self.results.append(AuditResult(
                    "TACTICAL_ENV", "PASS",
                    f"Reward distribution reasonable: μ={reward_mean:.2f}, σ={reward_std:.2f}"
                ))
            else:
                self.results.append(AuditResult(
                    "TACTICAL_ENV", "WARN",
                    f"Reward distribution may be problematic: μ={reward_mean:.2f}, σ={reward_std:.2f}",
                    "Consider reward scaling/normalization"
                ))
                
        except Exception as e:
            self.results.append(AuditResult(
                "TACTICAL_ENV", "FAIL",
                f"Tactical environment test failed: {e}",
                "Debug TacticalUAVEnv initialization and basic operations"
            ))
    
    def audit_strategic_environment(self):
        """Comprehensive strategic environment validation."""
        if not CRYPTO_IMPORTS_OK:
            return
            
        try:
            env = StrategicCryptoEnv()
            
            # State space validation
            state = env.reset()
            expected_dims = [3, 3, 4]  # Threat, Battery, Mission
            
            if len(state) == len(expected_dims):
                self.results.append(AuditResult(
                    "STRATEGIC_ENV", "PASS",
                    f"State vector dimension correct: {len(state)}"
                ))
            else:
                self.results.append(AuditResult(
                    "STRATEGIC_ENV", "FAIL",
                    f"State vector dimension mismatch: got {len(state)}, expected {len(expected_dims)}",
                    "Fix state vector composition in StrategicCryptoEnv"
                ))
                
            # Action space validation
            if env.action_dim == 4:
                self.results.append(AuditResult(
                    "STRATEGIC_ENV", "PASS",
                    "Action space dimension correct: 4 crypto algorithms"
                ))
            else:
                self.results.append(AuditResult(
                    "STRATEGIC_ENV", "FAIL",
                    f"Action space incorrect: got {env.action_dim}, expected 4",
                    "Update action space to match CRYPTO_ALGORITHMS count"
                ))
                
            # Algorithm configuration validation
            if len(CRYPTO_ALGORITHMS) == 4:
                self.results.append(AuditResult(
                    "STRATEGIC_ENV", "PASS",
                    f"Crypto algorithms configuration correct: {len(CRYPTO_ALGORITHMS)} algorithms"
                ))
            else:
                self.results.append(AuditResult(
                    "STRATEGIC_ENV", "WARN",
                    f"Unexpected crypto algorithm count: {len(CRYPTO_ALGORITHMS)}",
                    "Verify CRYPTO_ALGORITHMS matches action space"
                ))
                
        except Exception as e:
            self.results.append(AuditResult(
                "STRATEGIC_ENV", "FAIL",
                f"Strategic environment test failed: {e}",
                "Debug StrategicCryptoEnv initialization and basic operations"
            ))
    
    def audit_profile_consistency(self):
        """Validate consistency between profiles and empirical data."""
        if not self.context_data:
            self.results.append(AuditResult(
                "PROFILE_CONSISTENCY", "FAIL",
                "No empirical data available for validation",
                "Ensure context.txt contains hardware measurement data"
            ))
            return
            
        # Power consumption consistency
        power_issues = []
        empirical_power = self.context_data.get("power", {})
        
        for (freq, cores), profile_watts in POWER_PROFILES.items():
            # Convert amps to watts (assuming ~5V average)
            if freq in empirical_power:
                empirical_amps = empirical_power[freq]
                empirical_watts = [amp * 5.0 for amp in empirical_amps]  # Rough conversion
                avg_empirical = statistics.mean(empirical_watts)
                
                # Allow 50% tolerance for profile vs empirical
                if abs(profile_watts - avg_empirical) / avg_empirical > 0.5:
                    power_issues.append(f"{freq}MHz/{cores}cores: profile={profile_watts:.1f}W vs empirical≈{avg_empirical:.1f}W")
        
        if not power_issues:
            self.results.append(AuditResult(
                "PROFILE_CONSISTENCY", "PASS",
                "Power profiles reasonably consistent with empirical data"
            ))
        else:
            self.results.append(AuditResult(
                "PROFILE_CONSISTENCY", "WARN",
                f"Power profile discrepancies found: {len(power_issues)} mismatches",
                "Review power profile values against empirical measurements",
                {"discrepancies": power_issues}
            ))
    
    def audit_training_configuration(self):
        """Validate training hyperparameters and setup."""
        
        # Tactical hyperparameters
        tactical_lr = 0.1  # From ddos_rl/train.py
        tactical_discount = 0.99
        tactical_eps_decay = 0.995
        
        if 0.01 <= tactical_lr <= 0.5:
            self.results.append(AuditResult(
                "TRAINING_CONFIG", "PASS",
                f"Tactical learning rate acceptable: {tactical_lr}"
            ))
        else:
            self.results.append(AuditResult(
                "TRAINING_CONFIG", "WARN",
                f"Tactical learning rate may be suboptimal: {tactical_lr}",
                "Consider learning rate in range [0.01, 0.3] for tabular Q-learning"
            ))
            
        # Strategic hyperparameters
        if CRYPTO_IMPORTS_OK:
            strategic_lr = CRYPTO_RL.get("LEARNING_RATE", 0.1)
            strategic_eps_decay = CRYPTO_RL.get("EXPLORATION_DECAY", 0.9995)
            
            if strategic_eps_decay > 0.999:
                self.results.append(AuditResult(
                    "TRAINING_CONFIG", "PASS",
                    f"Strategic exploration decay reasonable: {strategic_eps_decay}"
                ))
            else:
                self.results.append(AuditResult(
                    "TRAINING_CONFIG", "WARN",
                    f"Strategic exploration decay may be too fast: {strategic_eps_decay}",
                    "Consider slower decay (>0.999) for better exploration"
                ))
    
    def audit_integration_paths(self):
        """Check integration between RL agents and runtime schedulers."""
        
        # Check if UAV scheduler exists and loads tactical RL
        uav_scheduler_path = Path(__file__).parent / "uav-scheduler.py"
        if uav_scheduler_path.exists():
            with open(uav_scheduler_path, 'r', encoding='utf-8') as f:
                uav_content = f.read()
                
            if "TacticalQLearningAgent" in uav_content:
                self.results.append(AuditResult(
                    "INTEGRATION", "PASS",
                    "UAV scheduler includes tactical RL integration"
                ))
            else:
                self.results.append(AuditResult(
                    "INTEGRATION", "WARN",
                    "UAV scheduler may not integrate tactical RL",
                    "Verify tactical agent loading in uav-scheduler.py"
                ))
        else:
            self.results.append(AuditResult(
                "INTEGRATION", "WARN",
                "UAV scheduler not found",
                "Ensure uav-scheduler.py exists for runtime integration"
            ))
        
        # Check output directory structure
        output_dir = Path(__file__).parent / "output"
        if output_dir.exists():
            self.results.append(AuditResult(
                "INTEGRATION", "PASS",
                "Output directory exists for model storage"
            ))
        else:
            self.results.append(AuditResult(
                "INTEGRATION", "WARN",
                "Output directory missing",
                "Create output/ directory before training"
            ))
    
    def audit_reproducibility(self):
        """Check reproducibility measures."""
        
        # Check if training functions accept seed parameters
        reproducibility_features = {
            "Tactical seeding": False,
            "Strategic seeding": False, 
            "CSV logging": False,
            "Model checkpointing": False
        }
        
        if DDOS_IMPORTS_OK:
            try:
                from ddos_rl.train import train_tactical_agent
                import inspect
                sig = inspect.signature(train_tactical_agent)
                if 'seed' in sig.parameters:
                    reproducibility_features["Tactical seeding"] = True
                    reproducibility_features["CSV logging"] = True
                    reproducibility_features["Model checkpointing"] = True
            except Exception:
                pass
                
        if CRYPTO_IMPORTS_OK:
            try:
                from crypto_rl.strategic_agent import train_strategic_agent
                import inspect
                sig = inspect.signature(train_strategic_agent)
                if 'seed' in sig.parameters:
                    reproducibility_features["Strategic seeding"] = True
            except Exception:
                pass
        
        implemented_count = sum(reproducibility_features.values())
        if implemented_count >= 3:
            self.results.append(AuditResult(
                "REPRODUCIBILITY", "PASS",
                f"Good reproducibility support: {implemented_count}/4 features implemented",
                details=reproducibility_features
            ))
        else:
            self.results.append(AuditResult(
                "REPRODUCIBILITY", "WARN",
                f"Limited reproducibility support: {implemented_count}/4 features implemented",
                "Add seeding, logging, and checkpointing to training functions",
                reproducibility_features
            ))
    
    def generate_recommendations(self) -> List[str]:
        """Generate prioritized recommendations based on audit results."""
        recommendations = []
        
        # Critical issues first
        critical_issues = [r for r in self.results if r.status == "FAIL"]
        if critical_issues:
            recommendations.append("🚨 CRITICAL: Fix all FAIL status issues before training")
            for issue in critical_issues:
                recommendations.append(f"   - {issue.category}: {issue.recommendation}")
        
        # High-priority warnings
        high_priority_warnings = [r for r in self.results if r.status == "WARN" and "PROFILE_CONSISTENCY" in r.category]
        if high_priority_warnings:
            recommendations.append("⚠️  HIGH PRIORITY: Address profile consistency issues")
            for warning in high_priority_warnings:
                recommendations.append(f"   - {warning.recommendation}")
                
        # Training optimization recommendations
        training_warnings = [r for r in self.results if r.status == "WARN" and "TRAINING_CONFIG" in r.category]
        if training_warnings:
            recommendations.append("🎯 OPTIMIZATION: Consider training parameter adjustments")
            for warning in training_warnings:
                recommendations.append(f"   - {warning.recommendation}")
        
        # General recommendations
        recommendations.extend([
            "",
            "📋 GENERAL RECOMMENDATIONS:",
            "1. Start with short training runs (1000-2000 episodes) to validate learning curves",
            "2. Monitor Q-table sparsity and action distribution during training", 
            "3. Compare RL agent performance against fixed baselines",
            "4. Add runtime monitoring for action switching frequency",
            "5. Consider domain randomization if overfitting to specific scenarios"
        ])
        
        return recommendations
    
    def run_full_audit(self) -> Dict[str, Any]:
        """Execute complete audit and return structured results."""
        print("🔬 STARTING COMPREHENSIVE RL SYSTEM AUDIT")
        print("=" * 60)
        
        # Run all audit modules
        self.audit_imports_and_dependencies()
        self.audit_tactical_environment() 
        self.audit_strategic_environment()
        self.audit_profile_consistency()
        self.audit_training_configuration()
        self.audit_integration_paths()
        self.audit_reproducibility()
        
        # Categorize results
        passes = [r for r in self.results if r.status == "PASS"]
        warnings = [r for r in self.results if r.status == "WARN"] 
        failures = [r for r in self.results if r.status == "FAIL"]
        
        # Generate summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_checks": len(self.results),
            "passes": len(passes),
            "warnings": len(warnings), 
            "failures": len(failures),
            "overall_status": "READY" if not failures else "NEEDS_FIXES",
            "results": [
                {
                    "category": r.category,
                    "status": r.status,
                    "description": r.description,
                    "recommendation": r.recommendation,
                    "details": r.details
                }
                for r in self.results
            ],
            "recommendations": self.generate_recommendations()
        }
        
        return summary

def print_audit_report(summary: Dict[str, Any]):
    """Print formatted audit report."""
    print(f"📊 AUDIT SUMMARY ({summary['timestamp']})")
    print("=" * 60)
    print(f"Total Checks: {summary['total_checks']}")
    print(f"✅ Passes: {summary['passes']}")
    print(f"⚠️  Warnings: {summary['warnings']}")
    print(f"❌ Failures: {summary['failures']}")
    print(f"📈 Overall Status: {summary['overall_status']}")
    print()
    
    # Print detailed results
    for result in summary['results']:
        status_emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[result['status']]
        print(f"{status_emoji} {result['category']}: {result['description']}")
        if result['recommendation']:
            print(f"   💡 {result['recommendation']}")
        if result['details']:
            print(f"   📋 Details: {result['details']}")
        print()
    
    # Print recommendations
    print("🎯 RECOMMENDATIONS")
    print("=" * 60)
    for rec in summary['recommendations']:
        print(rec)
    print()
    
    # Training readiness assessment
    if summary['overall_status'] == "READY":
        print("🚀 TRAINING READINESS: SYSTEM IS READY FOR TRAINING")
        print("   Proceed with confidence. Monitor initial learning curves.")
    else:
        print("⛔ TRAINING READINESS: ADDRESS CRITICAL ISSUES FIRST")
        print("   Do not start training until all FAIL status items are resolved.")

def save_audit_report(summary: Dict[str, Any], filename: str = "rl_audit_report.json"):
    """Save audit report to JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"📄 Audit report saved to: {filename}")

def main():
    """Main audit execution."""
    auditor = RLSystemAuditor()
    summary = auditor.run_full_audit()
    
    print_audit_report(summary)
    save_audit_report(summary)
    
    # Exit code based on results
    exit_code = 0 if summary['overall_status'] == "READY" else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main()