"""
Configuration for Cryptographic RL Agent
"""

# Cryptographic algorithms and their characteristics
CRYPTO_ALGORITHMS = {
    0: {"name": "ASCON_128", "latency_ms": 5.0, "security_rating": 3, "power_multiplier": 1.05},
    1: {"name": "KYBER_CRYPTO", "latency_ms": 2.8, "security_rating": 6, "power_multiplier": 1.15},
    2: {"name": "SPHINCS", "latency_ms": 19.6, "security_rating": 10, "power_multiplier": 1.40},
    3: {"name": "FALCON512", "latency_ms": 0.7, "security_rating": 8, "power_multiplier": 1.10}
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
