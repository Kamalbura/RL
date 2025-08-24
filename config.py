# Enhanced Configuration file with IEEE Paper findings
# Paper: "Attention Meets UAVs: A Comprehensive Evaluation of DDoS Detection in Low-Cost UAVs"
# DOI: 10.1109/CASE59546.2024.10711508

# Component Power Draw (Watts)
POWER_DRAW = {
    "PI_IDLE": 2.5,
    "MAVLINK_PROXY": 1.2,
    "XGBOOST_TASK": 2.0,
    "TST_TASK": 4.5,
    "CAMERA": 1.8,
    "GPS": 0.5,
    "COMMUNICATION": 1.2,
    "SENSORS": 0.8
}

# IEEE Paper: Detection Performance Metrics (Empirical Data)
DETECTION_PERFORMANCE = {
    "XGBOOST": {
        "F1_TCP": 0.82,
        "F1_ICMP": 0.82,
        "F1_MIXED": 0.67,
        "INFERENCE_TIME_MS": 15,
        "MEMORY_MB": 45,
        "ACCURACY": 0.82,
        "PRECISION": 0.82,
        "RECALL": 0.82
    },
    "TST": {
        "F1_TCP": 0.999,
        "F1_ICMP": 0.997,
        "F1_MIXED": 0.943,
        "INFERENCE_TIME_MS": 100,  # ~0.1 seconds as per paper
        "MEMORY_MB": 85,
        "ACCURACY": 0.975,
        "PRECISION": 0.985,
        "RECALL": 0.965
    }
}

# Battery Specifications
BATTERY_SPECS = {
    "VOLTAGE": 22.2,  # 6S LiPo
    "CAPACITY_MAH": 5200,  # mAh
    "CAPACITY_WH": (22.2 * 5.2),  # Watt-hours
    "MAX_FLIGHT_TIME": 720,  # seconds
    "MIN_FLIGHT_TIME": 659   # seconds
}

# Total Power Budget
MAX_POWER_CONSUMPTION = 8.4  # Watts

# State Space Discretization
BATTERY_LEVELS = {
    "CRITICAL": (0, 20),
    "LOW": (20, 60),
    "MEDIUM": (60, 90),
    "HIGH": (90, 101)
}

THREAT_LEVELS = ["NONE", "POTENTIAL", "CONFIRMED"]
FLIGHT_MODES = ["LOITER", "MISSION", "RTL"]
CPU_LOADS = {
    "LOW": (0, 30),
    "MEDIUM": (30, 70),
    "HIGH": (70, 101)
}

# Communication Parameters
COMMUNICATION = {
    "BANDWIDTH": [1, 5, 10, 20],  # Mbps
    "PACKET_LOSS": [0, 0.05, 0.1, 0.2],  # percentage
    "LATENCY": [10, 50, 100, 200],  # ms
    "SIGNAL_STRENGTH": ["STRONG", "MODERATE", "WEAK", "CRITICAL"]
}

# Environmental Factors
ENVIRONMENT = {
    "WIND_SPEED": [0, 5, 10, 15, 20],  # m/s
    "TEMPERATURE": [-10, 0, 10, 20, 30, 40],  # Celsius
    "VISIBILITY": ["CLEAR", "MODERATE", "POOR", "CRITICAL"],
    "RAIN": ["NONE", "LIGHT", "MODERATE", "HEAVY"]
}

# Power Consumption Modifiers
POWER_MODIFIERS = {
    "WIND_FACTOR": 0.1,  # Power increase per m/s of wind
    "TEMP_HIGH_FACTOR": 0.05,  # Power increase per degree above 25C
    "TEMP_LOW_FACTOR": 0.08,  # Power increase per degree below 5C
    "PAYLOAD_FACTOR": 0.2  # Power increase per 100g of additional payload
}

# Action Space
ACTIONS = {
    0: "RUN_XGBOOST",  # Baseline monitoring
    1: "RUN_TST",      # Threat confirmation
    2: "DE_ESCALATE"   # Stop all DDoS scanning
}

# Reward Parameters
REWARDS = {
    "STAYING_OPERATIONAL": 1.0,
    "BATTERY_DEPLETION": -1000.0,
    "POWER_PENALTY_FACTOR": 0.5,
    "TST_GOOD_CONTEXT": 10.0,
    "TST_BAD_CONTEXT": -50.0,
    "DE_ESCALATE_WITH_THREAT": -15.0,
    "DE_ESCALATE_WITH_THREAT_LOW_BATTERY": -5.0
}

# Strategic Agent Configuration
SWARM_THREAT_LEVELS = ["NONE", "CAUTION", "CRITICAL"]
FLEET_BATTERY_STATUS = ["HEALTHY", "DEGRADING", "CRITICAL"]
MISSION_PHASES = ["IDLE", "INGRESS", "LOITER_ON_TARGET", "EGRESS"]

# Cryptographic Algorithm Performance Data
CRYPTO_ALGORITHMS = {
    0: {
        "name": "ASCON_128",
        "latency_ms": 5.0,
        "security_rating": 3,  # 1-10 scale
        "power_multiplier": 1.05
    },
    1: {
        "name": "KYBER_CRYPTO",
        "latency_ms": 2.8,
        "security_rating": 6,
        "power_multiplier": 1.15
    },
    2: {
        "name": "SPHINCS",
        "latency_ms": 19.6,
        "security_rating": 10,
        "power_multiplier": 1.40
    },
    3: {
        "name": "FALCON512",
        "latency_ms": 0.7,
        "security_rating": 8,
        "power_multiplier": 1.10
    }
}

# Additional Cryptographic Context
CRYPTO_CONTEXTS = {
    "LOW_RISK": {
        "recommended_algorithm": 0,  # ASCON_128
        "key_rotation_frequency": "LOW"
    },
    "MEDIUM_RISK": {
        "recommended_algorithm": 1,  # KYBER_CRYPTO
        "key_rotation_frequency": "MEDIUM"
    },
    "HIGH_RISK": {
        "recommended_algorithm": 2,  # SPHINCS
        "key_rotation_frequency": "HIGH"
    },
    "URGENT_PERFORMANCE": {
        "recommended_algorithm": 3,  # FALCON512
        "key_rotation_frequency": "MEDIUM"
    }
}

# Mission Parameters
MISSION_PARAMETERS = {
    "PRIORITY": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    "DURATION": ["SHORT", "MEDIUM", "LONG"],
    "COMPLEXITY": ["SIMPLE", "MODERATE", "COMPLEX"],
    "STEALTH_REQUIRED": [False, True]
}

# Strategic Reward Parameters
STRATEGIC_REWARDS = {
    "FLEET_CRITICAL_PENALTY": -10.0,
    "HIGH_LATENCY_CRITICAL_PHASE_PENALTY": -20.0,
    "SECURITY_THREAT_BONUS": 2.0,
    "EFFICIENCY_SAFE_PHASE_BONUS": 5.0,
    "STRATEGIC_COHERENCE_BONUS": 10.0,
    "SUCCESSFUL_COMMUNICATION": 5.0,
    "STEALTH_MAINTENANCE": 8.0,
    "MISSION_COMPLETION_BONUS": 25.0,
    "ADAPTIVE_SECURITY_BONUS": 15.0,
    "ENVIRONMENTAL_ADAPTATION": 7.0
}

# Cryptographic Algorithm Selection RL Agent Configuration
CRYPTO_RL = {
    # State Space Components
    "STATE_SPACE": {
        "SECURITY_RISK_LEVELS": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        "BATTERY_STATE": ["CRITICAL", "LOW", "MEDIUM", "HIGH"],
        "COMPUTATION_CAPACITY": ["CONSTRAINED", "NORMAL", "ABUNDANT"],
        "MISSION_CRITICALITY": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        "COMMUNICATION_INTENSITY": ["LOW", "MEDIUM", "HIGH"],
        "THREAT_CONTEXT": ["BENIGN", "SUSPICIOUS", "HOSTILE"]
    },
    
    # Action Space (corresponds to CRYPTO_ALGORITHMS keys)
    "ACTIONS": [0, 1, 2, 3],  # Select from available crypto algorithms
    
    # Reward Function Parameters
    "REWARDS": {
        "SECURITY_MATCH_BONUS": 10.0,  # Reward for matching security level to threat
        "POWER_EFFICIENCY_FACTOR": 5.0,  # Reward multiplier for power-efficient choices
        "LATENCY_PENALTY_FACTOR": 0.5,  # Penalty multiplier for high latency in critical situations
        "BATTERY_PRESERVATION_BONUS": 7.0,  # Bonus for preserving battery when low
        "OVERKILL_PENALTY": -8.0,  # Penalty for using excessive security in benign contexts
        "UNDERKILL_PENALTY": -15.0,  # Penalty for insufficient security in hostile contexts
        "CONSISTENT_CHOICE_BONUS": 2.0,  # Small bonus for consistent algorithm selection
        "SUCCESSFUL_KEY_EXCHANGE": 12.0  # Reward for completed key exchange
    },
    
    # Algorithm Performance in Different Contexts
    "ALGORITHM_CONTEXTS": {
        "LOW_BATTERY": {
            0: 0.8,  # ASCON_128 is reasonably efficient
            1: 0.5,  # KYBER_CRYPTO is less efficient
            2: 0.1,  # SPHINCS is very power-hungry
            3: 0.9   # FALCON512 is most efficient
        },
        "HIGH_THREAT": {
            0: 0.3,  # ASCON_128 provides minimal security
            1: 0.6,  # KYBER_CRYPTO provides moderate security
            2: 1.0,  # SPHINCS provides maximum security
            3: 0.8   # FALCON512 provides good security
        },
        "LOW_LATENCY_NEEDED": {
            0: 0.7,  # ASCON_128 has moderate latency
            1: 0.8,  # KYBER_CRYPTO has good latency
            2: 0.1,  # SPHINCS has high latency
            3: 1.0   # FALCON512 has lowest latency
        }
    },
    
    # Learning Parameters
    "LEARNING": {
        "DISCOUNT_FACTOR": 0.95,
        "LEARNING_RATE": 0.001,
        "EXPLORATION_RATE_INITIAL": 0.9,
        "EXPLORATION_RATE_DECAY": 0.995,
        "EXPLORATION_RATE_MIN": 0.05,
        "BATCH_SIZE": 64,
        "MEMORY_SIZE": 10000,
        "TARGET_UPDATE_FREQUENCY": 10
    }
}

# Key Performance Indicators for Crypto RL Agent
CRYPTO_KPI = {
    "SECURITY_BREACHES_PREVENTED": 0,
    "POWER_SAVED_WH": 0,
    "SUCCESSFUL_ENCRYPTIONS": 0,
    "AVERAGE_LATENCY_MS": 0,
    "ALGORITHM_SELECTION_CONSISTENCY": 0,
    "ADAPTABILITY_SCORE": 0
}