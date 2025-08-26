"""
Configuration for Cryptographic RL Agent
"""

# Cryptographic algorithms and their characteristics
# Updated to match context.txt empirical measurements and post-quantum standards
CRYPTO_ALGORITHMS = {
    0: {"name": "KYBER", "latency_ms": 209.6, "security_rating": 8.5, "power_multiplier": 1.10},
    1: {"name": "DILITHIUM", "latency_ms": 264.0, "security_rating": 9.0, "power_multiplier": 1.15},
    2: {"name": "SPHINCS", "latency_ms": 687.4, "security_rating": 9.5, "power_multiplier": 1.40},
    3: {"name": "FALCON", "latency_ms": 445.4, "security_rating": 8.8, "power_multiplier": 1.20}
}

# Key Performance Indicators for the crypto agent
CRYPTO_KPI = {
    "POWER_SAVED_WH": 0.0,
    "SUCCESSFUL_ENCRYPTIONS": 0,
    "AVERAGE_LATENCY_MS": 0.0,
    "ALGORITHM_SELECTION_CONSISTENCY": 0.0,
    "SECURITY_BREACHES_PREVENTED": 0,
    "ADAPTABILITY_SCORE": 0.0
}

# RL Agent parameters
CRYPTO_RL = {
    "LEARNING_RATE": 0.1,
    "DISCOUNT_FACTOR": 0.99,
    "EXPLORATION_RATE": 1.0,
    "EXPLORATION_DECAY": 0.9995,
    "MIN_EXPLORATION_RATE": 0.01,
    "REWARDS": {
        "SECURITY_MATCH_BONUS": 10.0,
        "UNDERKILL_PENALTY": -15.0,
        "OVERKILL_PENALTY": -5.0,
        "POWER_EFFICIENCY_FACTOR": 5.0,
        "LATENCY_PENALTY_FACTOR": 5.0,
        "BATTERY_PRESERVATION_BONUS": 8.0
    }
}
