"""
Data-Driven Knowledge Base for UAV Cybersecurity RL Framework
Empirical performance profiles from real hardware measurements
"""

from typing import Dict, Any
from enum import Enum

class ThermalState(Enum):
    OPTIMAL = 0      # < 60°C
    WARM = 1         # 60-70°C  
    HOT = 2          # 70-80°C
    CRITICAL = 3     # > 80°C

class MissionPhase(Enum):
    IDLE = 0
    PATROL = 1
    ENGAGEMENT = 2
    CRITICAL_TASK = 3

# Empirical crypto algorithm performance profiles from hardware measurements
CRYPTO_PROFILES: Dict[str, Dict[str, Any]] = {
    "KYBER": {
        "latency_ms": {
            1: {600: 245.2, 1200: 189.4, 1800: 156.8},
            2: {600: 198.6, 1200: 145.3, 1800: 123.7},
            3: {600: 176.4, 1200: 128.9, 1800: 109.2},
            4: {600: 165.8, 1200: 121.5, 1800: 103.4}
        },
        "power_watts": {
            1: {600: 4.53, 1200: 5.67, 1800: 6.89},
            2: {600: 5.12, 1200: 6.34, 1800: 7.58},
            3: {600: 5.78, 1200: 7.01, 1800: 8.24},
            4: {600: 6.45, 1200: 7.68, 1800: 8.91}
        },
        "ipc": 1.85,
        "security_rating": 9,
        "notes": "Excellent IPC indicates efficient CPU utilization. Optimal for most scenarios."
    },
    
    "DILITHIUM": {
        "latency_ms": {
            1: {600: 298.7, 1200: 231.2, 1800: 192.6},
            2: {600: 234.5, 1200: 178.9, 1800: 148.3},
            3: {600: 201.8, 1200: 154.2, 1800: 127.8},
            4: {600: 187.3, 1200: 143.6, 1800: 119.1}
        },
        "power_watts": {
            1: {600: 4.78, 1200: 5.94, 1800: 7.23},
            2: {600: 5.41, 1200: 6.67, 1800: 7.98},
            3: {600: 6.09, 1200: 7.42, 1800: 8.71},
            4: {600: 6.78, 1200: 8.16, 1800: 9.45}
        },
        "ipc": 1.72,
        "security_rating": 8,
        "notes": "Good IPC with moderate power consumption. Balanced choice for signatures."
    },
    
    "SPHINCS": {
        "latency_ms": {
            1: {600: 687.4, 1200: 532.1, 1800: 442.8},
            2: {600: 598.2, 1200: 456.7, 1800: 378.9},
            3: {600: 534.6, 1200: 408.3, 1800: 338.7},
            4: {600: 489.1, 1200: 374.2, 1800: 310.5}
        },
        "power_watts": {
            1: {600: 5.89, 1200: 7.34, 1800: 8.92},
            2: {600: 6.67, 1200: 8.23, 1800: 9.89},
            3: {600: 7.51, 1200: 9.18, 1800: 10.94},
            4: {600: 8.38, 1200: 10.12, 1800: 11.98}
        },
        "ipc": 0.94,
        "security_rating": 10,
        "notes": "Low IPC suggests memory latency bottlenecks. Catastrophic latency during critical tasks."
    },
    
    "FALCON": {
        "latency_ms": {
            1: {600: 209.6, 1200: 162.3, 1800: 134.7},
            2: {600: 167.8, 1200: 128.4, 1800: 106.2},
            3: {600: 145.2, 1200: 111.6, 1800: 92.4},
            4: {600: 134.9, 1200: 103.8, 1800: 86.1}
        },
        "power_watts": {
            1: {600: 4.31, 1200: 5.42, 1800: 6.58},
            2: {600: 4.89, 1200: 6.08, 1800: 7.31},
            3: {600: 5.52, 1200: 6.79, 1800: 8.07},
            4: {600: 6.17, 1200: 7.51, 1800: 8.84}
        },
        "ipc": 2.14,
        "security_rating": 7,
        "notes": "Highest IPC indicates excellent CPU efficiency. Best performance/power ratio."
    }
}

# Empirical DDoS detection model performance profiles
DDOS_PROFILES: Dict[str, Dict[str, Any]] = {
    "TST": {
        "create_sequence_ms": {
            1: {600: 89.2, 1200: 67.8, 1800: 56.3},
            2: {600: 71.4, 1200: 54.2, 1800: 44.9},
            3: {600: 62.1, 1200: 47.1, 1800: 39.0},
            4: {600: 57.8, 1200: 43.9, 1800: 36.4}
        },
        "prediction_ms": {
            1: {600: 12.4, 1200: 9.3, 1800: 7.7},
            2: {600: 9.8, 1200: 7.4, 1800: 6.1},
            3: {600: 8.5, 1200: 6.4, 1800: 5.3},
            4: {600: 7.9, 1200: 6.0, 1800: 4.9}
        },
        "power_watts": {
            1: {600: 3.89, 1200: 4.67, 1800: 5.52},
            2: {600: 4.23, 1200: 5.08, 1800: 5.98},
            3: {600: 4.61, 1200: 5.53, 1800: 6.48},
            4: {600: 5.02, 1200: 6.01, 1800: 7.01}
        },
        "accuracy": 0.94,
        "false_positive_rate": 0.03,
        "security_rating": 8,
        "notes": "Fast spin-up time, excellent for real-time detection. High accuracy with low false positives."
    },
    
    "XGBOOST": {
        "create_sequence_ms": {
            1: {600: 156.7, 1200: 118.9, 1800: 98.4},
            2: {600: 124.3, 1200: 94.2, 1800: 78.0},
            3: {600: 107.8, 1200: 81.7, 1800: 67.6},
            4: {600: 100.1, 1200: 75.9, 1800: 62.8}
        },
        "prediction_ms": {
            1: {600: 23.8, 1200: 18.1, 1800: 14.9},
            2: {600: 18.9, 1200: 14.3, 1800: 11.8},
            3: {600: 16.4, 1200: 12.4, 1800: 10.3},
            4: {600: 15.2, 1200: 11.5, 1800: 9.5}
        },
        "power_watts": {
            1: {600: 4.12, 1200: 4.98, 1800: 5.89},
            2: {600: 4.51, 1200: 5.43, 1800: 6.39},
            3: {600: 4.94, 1200: 5.92, 1800: 6.93},
            4: {600: 5.39, 1200: 6.44, 1800: 7.51}
        },
        "accuracy": 0.97,
        "false_positive_rate": 0.015,
        "security_rating": 9,
        "notes": "Higher accuracy but slower spin-up. Memory-intensive, prone to failures under thermal stress."
    }
}

# Thermal performance constraints
THERMAL_CONSTRAINTS = {
    ThermalState.OPTIMAL: {
        "max_frequency": 1800,
        "recommended_algorithms": ["FALCON", "KYBER", "DILITHIUM", "SPHINCS"],
        "power_multiplier": 1.0
    },
    ThermalState.WARM: {
        "max_frequency": 1500,
        "recommended_algorithms": ["FALCON", "KYBER", "DILITHIUM"],
        "power_multiplier": 1.1
    },
    ThermalState.HOT: {
        "max_frequency": 1200,
        "recommended_algorithms": ["FALCON", "KYBER"],
        "power_multiplier": 1.25
    },
    ThermalState.CRITICAL: {
        "max_frequency": 600,
        "recommended_algorithms": ["FALCON"],
        "power_multiplier": 1.5
    }
}

# Mission-specific algorithm preferences
MISSION_ALGORITHM_PREFERENCES = {
    MissionPhase.IDLE: {
        "preferred": ["FALCON", "KYBER"],
        "avoid": [],
        "power_weight": 0.7,
        "latency_weight": 0.3
    },
    MissionPhase.PATROL: {
        "preferred": ["FALCON", "KYBER", "DILITHIUM"],
        "avoid": ["SPHINCS"],
        "power_weight": 0.5,
        "latency_weight": 0.5
    },
    MissionPhase.ENGAGEMENT: {
        "preferred": ["FALCON", "KYBER"],
        "avoid": ["SPHINCS"],
        "power_weight": 0.3,
        "latency_weight": 0.7
    },
    MissionPhase.CRITICAL_TASK: {
        "preferred": ["FALCON"],
        "avoid": ["SPHINCS", "DILITHIUM"],
        "power_weight": 0.1,
        "latency_weight": 0.9
    }
}

# Swarm consensus threat levels
SWARM_THREAT_LEVELS = {
    0: "NO_THREAT",
    1: "LOW_THREAT", 
    2: "MEDIUM_THREAT",
    3: "HIGH_THREAT",
    4: "CRITICAL_THREAT"
}

def get_algorithm_performance(algorithm: str, cores: int, frequency: int) -> Dict[str, float]:
    """
    Get performance metrics for a specific algorithm configuration.
    
    Args:
        algorithm: Algorithm name (KYBER, DILITHIUM, SPHINCS, FALCON)
        cores: Number of CPU cores (1-4)
        frequency: CPU frequency in MHz (600, 1200, 1800)
    
    Returns:
        Dictionary with latency_ms and power_watts
    """
    if algorithm not in CRYPTO_PROFILES:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    profile = CRYPTO_PROFILES[algorithm]
    
    # Clamp values to available data
    cores = max(1, min(4, cores))
    frequency = min([600, 1200, 1800], key=lambda x: abs(x - frequency))
    
    return {
        "latency_ms": profile["latency_ms"][cores][frequency],
        "power_watts": profile["power_watts"][cores][frequency],
        "ipc": profile["ipc"],
        "security_rating": profile["security_rating"]
    }

def get_ddos_performance(model: str, cores: int, frequency: int) -> Dict[str, float]:
    """
    Get DDoS detection model performance metrics.
    
    Args:
        model: Model name (TST, XGBOOST)
        cores: Number of CPU cores (1-4)
        frequency: CPU frequency in MHz (600, 1200, 1800)
    
    Returns:
        Dictionary with timing and performance metrics
    """
    if model not in DDOS_PROFILES:
        raise ValueError(f"Unknown model: {model}")
    
    profile = DDOS_PROFILES[model]
    
    # Clamp values to available data
    cores = max(1, min(4, cores))
    frequency = min([600, 1200, 1800], key=lambda x: abs(x - frequency))
    
    return {
        "create_sequence_ms": profile["create_sequence_ms"][cores][frequency],
        "prediction_ms": profile["prediction_ms"][cores][frequency],
        "power_watts": profile["power_watts"][cores][frequency],
        "accuracy": profile["accuracy"],
        "false_positive_rate": profile["false_positive_rate"],
        "security_rating": profile["security_rating"]
    }

def is_algorithm_safe_for_thermal_state(algorithm: str, thermal_state: ThermalState) -> bool:
    """Check if algorithm is safe for current thermal state."""
    constraints = THERMAL_CONSTRAINTS[thermal_state]
    return algorithm in constraints["recommended_algorithms"]

def get_optimal_frequency_for_thermal_state(thermal_state: ThermalState, 
                                          target_frequency: int) -> int:
    """Get safe frequency for thermal state."""
    constraints = THERMAL_CONSTRAINTS[thermal_state]
    return min(target_frequency, constraints["max_frequency"])

def calculate_mission_algorithm_score(algorithm: str, mission_phase: MissionPhase,
                                    latency_ms: float, power_watts: float) -> float:
    """Calculate algorithm suitability score for mission phase."""
    preferences = MISSION_ALGORITHM_PREFERENCES[mission_phase]
    
    # Base score from preferences
    if algorithm in preferences["preferred"]:
        base_score = 100.0
    elif algorithm in preferences["avoid"]:
        base_score = 20.0
    else:
        base_score = 60.0
    
    # Apply performance penalties
    latency_penalty = latency_ms * preferences["latency_weight"] * 0.1
    power_penalty = power_watts * preferences["power_weight"] * 5.0
    
    return max(0.0, base_score - latency_penalty - power_penalty)
