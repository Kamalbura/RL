"""
Performance Profiles Database

This file contains lookup tables for drone performance characteristics
including power usage, crypto latency, and task execution times
across different hardware configurations.
"""

# CRITICAL: Separate power profiles for RPi (5V) vs Drone Motors (22.2V)
# RPi power consumption (Watts) - Based on empirical measurements from context.txt @ 5V
RPI_POWER_PROFILES = {
    # (frequency_mhz, active_cores): rpi_power_watts (Current * 5V)
    # 600 MHz measurements - RPi only
    (600, 1): 4.50,   # avg 0.90A * 5V = 4.5W
    (600, 2): 4.50,   # avg 0.90A * 5V = 4.5W  
    (600, 3): 4.75,   # avg 0.95A * 5V = 4.75W
    (600, 4): 5.00,   # avg 1.00A * 5V = 5.0W
    
    # 1200 MHz measurements - RPi only
    (1200, 1): 5.38,  # avg 1.075A * 5V = 5.375W
    (1200, 2): 5.63,  # avg 1.125A * 5V = 5.625W
    (1200, 3): 6.28,  # avg 1.255A * 5V = 6.275W
    (1200, 4): 6.83,  # avg 1.365A * 5V = 6.825W
    
    # 1800 MHz measurements - RPi only
    (1800, 1): 6.00,  # avg 1.20A * 5V = 6.0W
    (1800, 2): 6.30,  # avg 1.26A * 5V = 6.3W
    (1800, 3): 7.23,  # avg 1.445A * 5V = 7.225W
    (1800, 4): 8.38,  # avg 1.675A * 5V = 8.375W
    
    # 2000 MHz (extrapolated) - RPi only
    (2000, 1): 7.2,
    (2000, 2): 7.8,
    (2000, 4): 9.5,
}

# Drone motor power consumption (Watts) - Estimated hover power @ 22.2V
# Motors consume significantly more power than RPi
DRONE_MOTOR_POWER = {
    "hover_base": 180.0,      # Base hover power ~8A * 22.2V = 177.6W
    "hover_variation": 20.0,  # ±20W variation based on conditions
    "climb_power": 250.0,     # Climbing power
    "descent_power": 120.0,   # Descent power (less than hover)
}

# Combined power profiles (RPi + Motors) for total drone power consumption
POWER_PROFILES = {
    # (frequency_mhz, active_cores): total_drone_power_watts
    # Total = RPi Power + Motor Power (hover mode)
    (600, 1): 184.5,   # 4.5W (RPi) + 180W (motors) = 184.5W
    (600, 2): 184.5,   # 4.5W (RPi) + 180W (motors) = 184.5W
    (600, 3): 184.75,  # 4.75W (RPi) + 180W (motors) = 184.75W
    (600, 4): 185.0,   # 5.0W (RPi) + 180W (motors) = 185.0W
    
    (1200, 1): 185.38, # 5.38W (RPi) + 180W (motors) = 185.38W
    (1200, 2): 185.63, # 5.63W (RPi) + 180W (motors) = 185.63W
    (1200, 3): 186.28, # 6.28W (RPi) + 180W (motors) = 186.28W
    (1200, 4): 186.83, # 6.83W (RPi) + 180W (motors) = 186.83W
    
    (1800, 1): 186.0,  # 6.0W (RPi) + 180W (motors) = 186.0W
    (1800, 2): 186.3,  # 6.3W (RPi) + 180W (motors) = 186.3W
    (1800, 3): 187.23, # 7.23W (RPi) + 180W (motors) = 187.23W
    (1800, 4): 188.38, # 8.38W (RPi) + 180W (motors) = 188.38W
    
    (2000, 1): 187.2,  # 7.2W (RPi) + 180W (motors) = 187.2W
    (2000, 2): 187.8,  # 7.8W (RPi) + 180W (motors) = 187.8W
    (2000, 4): 189.5,  # 9.5W (RPi) + 180W (motors) = 189.5W
}

# Crypto algorithm latency profiles: Maps (algorithm, frequency_mhz, active_cores) to latency in milliseconds
# CORRECTED: Based on EXACT measurements from context.txt research data
LATENCY_PROFILES = {
    # Format: (algorithm, frequency_mhz, active_cores): latency_ms
    # KYBER measurements from context.txt (converted from seconds to ms)
    ("KYBER", 600, 1): 3138.4,   # avg(1.602,1.776,3.638,3.307,3.937,2.593,4.575,4.667) * 1000
    ("KYBER", 600, 2): 210.2,    # avg(0.494,0.195,0.116,0.095,0.151) * 1000
    ("KYBER", 600, 3): 207.6,    # avg(0.131,0.094,0.454,0.236,0.123) * 1000
    ("KYBER", 600, 4): 205.6,    # avg(0.159,0.239,0.118,0.302,0.21) * 1000
    ("KYBER", 1200, 1): 209.6,   # avg(0.147,0.178,0.109,0.485,0.129) * 1000
    ("KYBER", 1200, 2): 115.8,   # avg(0.135,0.09,0.094,0.121,0.139) * 1000
    ("KYBER", 1200, 3): 101.0,   # avg(0.101,0.11,0.089,0.094,0.111) * 1000
    ("KYBER", 1200, 4): 301.0,   # avg(0.098,0.306,0.135,0.769,0.197) * 1000
    ("KYBER", 1800, 1): 117.6,   # avg(0.094,0.108,0.122,0.128,0.136) * 1000
    ("KYBER", 1800, 2): 172.8,   # avg(0.105,0.372,0.091,0.125,0.171) * 1000
    ("KYBER", 1800, 3): 97.2,    # avg(0.104,0.102,0.094,0.093,0.093) * 1000
    ("KYBER", 1800, 4): 144.6,   # avg(0.107,0.134,0.224,0.168,0.09) * 1000
    
    # DILITHIUM measurements from context.txt - CORRECTED
    ("DILITHIUM", 600, 1): 4411.2,  # avg(4.219,3.508,4.174,5.149,5.006) * 1000
    ("DILITHIUM", 600, 2): 530.4,   # avg(1.783,0.465,0.231,0.536,0.461) * 1000 (take 5)
    ("DILITHIUM", 600, 3): 223.2,   # avg(0.249,0.135,0.146,0.464,0.122) * 1000
    ("DILITHIUM", 600, 4): 1192.2,  # avg(1.083,0.767,0.878,1.373,1.86) * 1000
    ("DILITHIUM", 1200, 1): 268.0,  # avg(0.248,0.221,0.386,0.257,0.228) * 1000
    ("DILITHIUM", 1200, 2): 194.6,  # avg(0.266,0.16,0.118,0.235,0.194) * 1000
    ("DILITHIUM", 1200, 3): 227.0,  # avg(0.112,0.233,0.259,0.224,0.307) * 1000
    ("DILITHIUM", 1200, 4): 340.4,  # avg(0.155,0.22,0.388,0.424,0.515) * 1000
    ("DILITHIUM", 1800, 1): 306.0,  # avg(0.441,0.354,0.283,0.157,0.295) * 1000
    ("DILITHIUM", 1800, 2): 189.4,  # avg(0.12,0.289,0.11,0.255,0.173) * 1000
    ("DILITHIUM", 1800, 3): 143.0,  # avg(0.147,0.245,0.111,0.109,0.103) * 1000
    ("DILITHIUM", 1800, 4): 317.6,  # avg(0.322,0.237,0.376,0.536,0.117) * 1000
    
    # SPHINCS measurements from context.txt - CORRECTED
    ("SPHINCS", 600, 1): 4230.6,   # avg(4.128,3.961,5.236,3.724,4.104) * 1000
    ("SPHINCS", 600, 2): 1583.0,   # avg(1.417,1.277,1.318,1.166,2.047) * 1000 (take 5)
    ("SPHINCS", 600, 3): 318.8,    # avg(0.326,0.263,0.439,0.267,0.299) * 1000
    ("SPHINCS", 600, 4): 1220.0,   # avg(1.717,1.455,1.408,0.563,0.957) * 1000
    ("SPHINCS", 1200, 1): 718.2,   # avg(0.549,0.86,0.763,0.643,0.776) * 1000
    ("SPHINCS", 1200, 2): 315.0,   # avg(0.292,0.36,0.195,0.408,0.32) * 1000
    ("SPHINCS", 1200, 3): 176.6,   # avg(0.25,0.197,0.156,0.144,0.136) * 1000
    ("SPHINCS", 1200, 4): 205.2,   # avg(0.289,0.156,0.222,0.139,0.22) * 1000
    ("SPHINCS", 1800, 1): 360.4,   # avg(0.657,0.26,0.291,0.364,0.23) * 1000
    ("SPHINCS", 1800, 2): 232.4,   # avg(0.308,0.146,0.246,0.295,0.167) * 1000
    ("SPHINCS", 1800, 3): 149.2,   # avg(0.167,0.122,0.125,0.114,0.117) * 1000 (take 5)
    ("SPHINCS", 1800, 4): 262.4,   # avg(0.132,0.749,0.273,0.154,0.324) * 1000 (take 5)
    
    # FALCON measurements from context.txt - CORRECTED
    ("FALCON", 600, 1): 4756.8,    # avg(4.896,4.242,3.555,5.794,5.297) * 1000
    ("FALCON", 600, 2): 441.8,     # avg(0.29,0.469,0.222,0.313,0.905) * 1000
    ("FALCON", 600, 3): 347.2,     # avg(0.226,0.717,0.339,0.096,0.358) * 1000
    ("FALCON", 600, 4): 1078.0,    # avg(0.948,1.161,0.958,1.081,1.242) * 1000
    ("FALCON", 1200, 1): 385.2,    # avg(0.245,0.783,0.489,0.187,0.222) * 1000
    ("FALCON", 1200, 2): 178.2,    # avg(0.115,0.222,0.181,0.189,0.184) * 1000
    ("FALCON", 1200, 3): 218.0,    # avg(0.377,0.146,0.216,0.188,0.163) * 1000
    ("FALCON", 1200, 4): 366.2,    # avg(0.411,0.389,0.155,0.359,0.517) * 1000
    ("FALCON", 1800, 1): 250.2,    # avg(0.236,0.322,0.294,0.203,0.196) * 1000
    ("FALCON", 1800, 2): 335.0,    # avg(0.583,0.154,0.195,0.124,0.619) * 1000
    ("FALCON", 1800, 3): 228.0,    # avg(0.155,0.308,0.237,0.134,0.306) * 1000
    ("FALCON", 1800, 4): 170.0,    # avg(0.229,0.112,0.116,0.205,0.188) * 1000
}

# DDoS detection task time profiles: Maps (model, frequency_mhz, active_cores) to execution time in seconds
# CORRECTED: Based on EXACT TST measurements from context.txt
DDOS_TASK_TIME_PROFILES = {
    # (model, frequency_mhz, active_cores): execution_time_seconds
    # XGBoost performance (estimated from research context)
    ("XGBOOST", 600, 1): 0.25,
    ("XGBOOST", 600, 2): 0.18,
    ("XGBOOST", 600, 4): 0.12,
    ("XGBOOST", 1200, 1): 0.15,
    ("XGBOOST", 1200, 2): 0.11,
    ("XGBOOST", 1200, 4): 0.08,
    ("XGBOOST", 1800, 1): 0.10,
    ("XGBOOST", 1800, 2): 0.07,
    ("XGBOOST", 1800, 4): 0.05,
    ("XGBOOST", 2000, 1): 0.08,
    ("XGBOOST", 2000, 2): 0.06,
    ("XGBOOST", 2000, 4): 0.04,
    
    # TST performance - EXACT measurements from context.txt
    # Based on create_sequence times (most computationally intensive)
    ("TST", 600, 1): 13.5459,  # Single core @ 600MHz
    ("TST", 600, 2): 8.7863,   # Double core @ 600MHz
    ("TST", 600, 3): 8.8293,   # Triple core @ 600MHz
    ("TST", 600, 4): 8.8759,   # Quad core @ 600MHz
    ("TST", 1200, 1): 5.0813,  # Single core @ 1.2GHz
    ("TST", 1200, 2): 4.1946,  # Double core @ 1.2GHz
    ("TST", 1200, 3): 4.2098,  # Triple core @ 1.2GHz
    ("TST", 1200, 4): 4.1985,  # Quad core @ 1.2GHz
    ("TST", 1800, 1): 3.2256,  # Single core @ 1.8GHz
    ("TST", 1800, 2): 2.7212,  # Double core @ 1.8GHz
    ("TST", 1800, 3): 2.7831,  # Triple core @ 1.8GHz
    ("TST", 1800, 4): 2.8066,  # Quad core @ 1.8GHz
    ("TST", 2000, 1): 2.5,     # Extrapolated
    ("TST", 2000, 2): 2.2,     # Extrapolated
    ("TST", 2000, 4): 2.0,     # Extrapolated
}

# TST Model Loading Times (seconds) - From context.txt
TST_LOAD_TIMES = {
    # (frequency_mhz, active_cores): load_time_seconds
    (600, 1): 0.0977,   # load_model time @ 600MHz single
    (600, 2): 0.0554,   # load_model time @ 600MHz double
    (600, 3): 0.0553,   # load_model time @ 600MHz triple
    (600, 4): 0.0540,   # load_model time @ 600MHz quad
    (1200, 1): 0.0271, # load_model time @ 1.2GHz single
    (1200, 2): 0.0321, # load_model time @ 1.2GHz double
    (1200, 3): 0.0258, # load_model time @ 1.2GHz triple
    (1200, 4): 0.0262, # load_model time @ 1.2GHz quad
    (1800, 1): 0.0452, # load_model time @ 1.8GHz single
    (1800, 2): 0.0190, # load_model time @ 1.8GHz double
    (1800, 3): 0.0196, # load_model time @ 1.8GHz triple
    (1800, 4): 0.0208, # load_model time @ 1.8GHz quad
}

# Decryption timing profiles from context.txt empirical data (seconds)
DECRYPTION_PROFILES = {
    # Format: (algorithm, frequency_mhz, active_cores): decryption_time_seconds
    # 600MHz Decryption Times
    ("KYBER", 600, 1): 0.0064,    # avg(0.0011,0.0011,0.0170)
    ("KYBER", 600, 2): 0.0014,    # avg(0.0022,0.0009,0.0011)
    ("KYBER", 600, 3): 0.0018,    # avg(0.0008,0.0013,0.0032)
    ("KYBER", 600, 4): 0.0014,    # avg(0.0015,0.0012,0.0015)
    
    ("DILITHIUM", 600, 1): 0.0015,  # avg(0.0019,0.0013,0.0013)
    ("DILITHIUM", 600, 2): 0.0220,  # avg(0.0633,0.0013,0.0013)
    ("DILITHIUM", 600, 3): 0.0036,  # avg(0.0046,0.0049,0.0012)
    ("DILITHIUM", 600, 4): 0.0019,  # avg(0.0022,0.0020,0.0015)
    
    ("SPHINCS", 600, 1): 0.2040,   # avg(0.2302,0.0854,0.2963)
    ("SPHINCS", 600, 2): 0.0686,   # avg(0.0478,0.0988,0.0592)
    ("SPHINCS", 600, 3): 0.0766,   # avg(0.0810,0.0903,0.0584)
    ("SPHINCS", 600, 4): 0.1466,   # avg(0.3560,0.0189,0.0650)
    
    ("FALCON", 600, 1): 0.0011,    # avg(0.0015,0.0010,0.0007)
    ("FALCON", 600, 2): 0.0012,    # avg(0.0012,0.0012,0.0011)
    ("FALCON", 600, 3): 0.0276,    # avg(0.0807,0.0010,0.0011)
    ("FALCON", 600, 4): 0.0022,    # avg(0.0047,0.0011,0.0008)
    
    # 1200MHz Decryption Times
    ("KYBER", 1200, 1): 0.0005,    # avg(0.0005,0.0006,0.0003)
    ("KYBER", 1200, 2): 0.0005,    # avg(0.0006,0.0005,0.0005)
    ("KYBER", 1200, 3): 0.0005,    # avg(0.0006,0.0006,0.0004)
    ("KYBER", 1200, 4): 0.0006,    # avg(0.0005,0.0005,0.0008)
    
    ("DILITHIUM", 1200, 1): 0.0009, # avg(0.0012,0.0007,0.0007)
    ("DILITHIUM", 1200, 2): 0.0008, # avg(0.0010,0.0007,0.0007)
    ("DILITHIUM", 1200, 3): 0.0008, # avg(0.0011,0.0007,0.0007)
    ("DILITHIUM", 1200, 4): 0.0014, # avg(0.0013,0.0009,0.0020)
    
    ("SPHINCS", 1200, 1): 0.1103,  # avg(0.1381,0.1203,0.0725)
    ("SPHINCS", 1200, 2): 0.0257,  # avg(0.0235,0.0254,0.0282)
    ("SPHINCS", 1200, 3): 0.0207,  # avg(0.0249,0.0133,0.0238)
    ("SPHINCS", 1200, 4): 0.0420,  # avg(0.0364,0.0657,0.0240)
    
    ("FALCON", 1200, 1): 0.0029,   # avg(0.0009,0.0040,0.0039)
    ("FALCON", 1200, 2): 0.0006,   # avg(0.0008,0.0004,0.0006)
    ("FALCON", 1200, 3): 0.0022,   # avg(0.0008,0.0005,0.0054)
    ("FALCON", 1200, 4): 0.0007,   # avg(0.0010,0.0006,0.0006)
    
    # 1800MHz Decryption Times
    ("KYBER", 1800, 1): 0.0008,    # avg(0.0004,0.0005,0.0015)
    ("KYBER", 1800, 2): 0.0004,    # avg(0.0004,0.0004,0.0004)
    ("KYBER", 1800, 3): 0.0005,    # avg(0.0009,0.0004,0.0003)
    ("KYBER", 1800, 4): 0.0028,    # avg(0.0046,0.0015,0.0024)
    
    ("DILITHIUM", 1800, 1): 0.0031, # avg(0.0054,0.0033,0.0005)
    ("DILITHIUM", 1800, 2): 0.0006, # avg(0.0008,0.0005,0.0005)
    ("DILITHIUM", 1800, 3): 0.0006, # avg(0.0008,0.0006,0.0005)
    ("DILITHIUM", 1800, 4): 0.0016, # avg(0.0033,0.0007,0.0007)
    
    ("SPHINCS", 1800, 1): 0.0641,  # avg(0.0581,0.0738,0.0603)
    ("SPHINCS", 1800, 2): 0.0116,  # avg(0.0186,0.0084,0.0079)
    ("SPHINCS", 1800, 3): 0.0215,  # avg(0.0084,0.0415,0.0145)
    ("SPHINCS", 1800, 4): 0.0196,  # avg(0.0340,0.0084,0.0163)
    
    ("FALCON", 1800, 1): 0.0005,   # avg(0.0007,0.0004,0.0003)
    ("FALCON", 1800, 2): 0.0004,   # avg(0.0006,0.0004,0.0003)
    ("FALCON", 1800, 3): 0.0005,   # avg(0.0006,0.0004,0.0004)
    ("FALCON", 1800, 4): 0.0007,   # avg(0.0009,0.0005,0.0006)
}

# Security ratings (1-10 scale, higher is more secure)
# Based on post-quantum cryptography standards and research performance
SECURITY_RATINGS = {
    "KYBER": 8.5,     # NIST selected algorithm for key encapsulation
    "DILITHIUM": 9.0, # NIST selected algorithm for digital signatures
    "SPHINCS": 9.5,   # Hash-based signatures, very secure but slow
    "FALCON": 8.8,    # Lattice-based, good security-performance balance
    "XGBOOST": 7.8,   # Good traditional ML performance
    "TST": 9.2,       # Superior performance with attention mechanism
}

# CPU Frequency Presets (MHz) - Based on RPi 4B Testing
CPU_FREQUENCY_PRESETS = {
    "POWERSAVE": 600,
    "BALANCED": 1200,
    "PERFORMANCE": 1800,
    "TURBO": 2000  # Extended beyond tested range
}

# Helper functions to query the profiles
def get_power_consumption(frequency_mhz, active_cores, noise_std=0.0):
    """Get power consumption in Watts for a given CPU configuration
    
    Args:
        frequency_mhz: CPU frequency in MHz
        active_cores: Number of active cores
        noise_std: Standard deviation for Gaussian noise (0.0 = no noise, 0.1 = 10% noise)
    """
    base_power = POWER_PROFILES.get((frequency_mhz, active_cores), 0.0)
    if noise_std > 0 and base_power > 0:
        import numpy as np
        noise_factor = np.random.normal(1.0, noise_std)
        # Clip to reasonable bounds (50% to 150% of base)
        noise_factor = np.clip(noise_factor, 0.5, 1.5)
        return base_power * noise_factor
    return base_power

def get_crypto_latency(algorithm, frequency_mhz, active_cores, noise_std=0.0):
    """Get cryptographic algorithm latency in milliseconds for a given configuration
    
    Args:
        algorithm: Crypto algorithm name
        frequency_mhz: CPU frequency in MHz  
        active_cores: Number of active cores
        noise_std: Standard deviation for Gaussian noise
    """
    base_latency = LATENCY_PROFILES.get((algorithm, frequency_mhz, active_cores), 0.0)
    if noise_std > 0 and base_latency > 0:
        import numpy as np
        noise_factor = np.random.normal(1.0, noise_std)
        noise_factor = np.clip(noise_factor, 0.5, 2.0)  # Latency can vary more
        return base_latency * noise_factor
    return base_latency

def get_ddos_execution_time(model, frequency_mhz, active_cores, noise_std=0.0):
    """Get DDoS detection execution time in seconds for a given configuration
    
    Args:
        model: DDoS model name (XGBOOST or TST)
        frequency_mhz: CPU frequency in MHz
        active_cores: Number of active cores  
        noise_std: Standard deviation for Gaussian noise
    """
    base_time = DDOS_TASK_TIME_PROFILES.get((model, frequency_mhz, active_cores), 0.0)
    if noise_std > 0 and base_time > 0:
        import numpy as np
        noise_factor = np.random.normal(1.0, noise_std)
        noise_factor = np.clip(noise_factor, 0.3, 3.0)  # Execution time can vary significantly
        return base_time * noise_factor
    return base_time

def get_security_rating(algorithm):
    """Get security rating (1-10) for a given algorithm"""
    return SECURITY_RATINGS.get(algorithm, 0)

def get_decryption_time(algorithm, frequency_mhz, active_cores, noise_std=0.0):
    """Get decryption time in seconds for a given configuration
    
    Args:
        algorithm: Crypto algorithm name
        frequency_mhz: CPU frequency in MHz
        active_cores: Number of active cores
        noise_std: Standard deviation for Gaussian noise
    """
    base_time = DECRYPTION_PROFILES.get((algorithm, frequency_mhz, active_cores), 0.0)
    if noise_std > 0 and base_time > 0:
        import numpy as np
        noise_factor = np.random.normal(1.0, noise_std)
        noise_factor = np.clip(noise_factor, 0.3, 3.0)  # Decryption time can vary significantly
        return base_time * noise_factor
    return base_time

def get_tst_load_time(frequency_mhz, active_cores):
    """Get TST model loading time in seconds"""
    return TST_LOAD_TIMES.get((frequency_mhz, active_cores), 0.0)

def get_rpi_power_only(frequency_mhz, active_cores):
    """Get RPi power consumption only (excluding motors)"""
    return RPI_POWER_PROFILES.get((frequency_mhz, active_cores), 0.0)

def get_motor_power(flight_mode="hover"):
    """Get drone motor power consumption based on flight mode"""
    if flight_mode == "hover":
        return DRONE_MOTOR_POWER["hover_base"]
    elif flight_mode == "climb":
        return DRONE_MOTOR_POWER["climb_power"]
    elif flight_mode == "descent":
        return DRONE_MOTOR_POWER["descent_power"]
    else:
        return DRONE_MOTOR_POWER["hover_base"]  # Default to hover

# Domain randomization control
_DOMAIN_RANDOMIZATION_ENABLED = False
_NOISE_STD = 0.1

def enable_domain_randomization(noise_std=0.1):
    """Enable domain randomization with specified noise level"""
    global _DOMAIN_RANDOMIZATION_ENABLED, _NOISE_STD
    _DOMAIN_RANDOMIZATION_ENABLED = True
    _NOISE_STD = noise_std

def disable_domain_randomization():
    """Disable domain randomization"""
    global _DOMAIN_RANDOMIZATION_ENABLED
    _DOMAIN_RANDOMIZATION_ENABLED = False

def get_power_consumption_robust(frequency_mhz, active_cores):
    """Get power consumption with optional domain randomization"""
    noise = _NOISE_STD if _DOMAIN_RANDOMIZATION_ENABLED else 0.0
    return get_power_consumption(frequency_mhz, active_cores, noise)

def get_ddos_execution_time_robust(model, frequency_mhz, active_cores):
    """Get execution time with optional domain randomization"""
    noise = _NOISE_STD if _DOMAIN_RANDOMIZATION_ENABLED else 0.0
    return get_ddos_execution_time(model, frequency_mhz, active_cores, noise)

def get_crypto_latency_robust(algorithm, frequency_mhz, active_cores):
    """Get crypto latency with optional domain randomization"""
    noise = _NOISE_STD if _DOMAIN_RANDOMIZATION_ENABLED else 0.0
    return get_crypto_latency(algorithm, frequency_mhz, active_cores, noise)
