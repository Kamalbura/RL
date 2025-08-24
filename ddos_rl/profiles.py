"""
Performance Profiles Database

This file contains lookup tables for drone performance characteristics
including power usage, crypto latency, and task execution times
across different hardware configurations.
"""

# Power profiles: Maps (frequency_mhz, active_cores) to power consumption in Watts
POWER_PROFILES = {
    # Format: (frequency_mhz, active_cores): watts
    (1200, 1): 3.2,
    (1200, 2): 4.5,
    (1200, 4): 7.8,
    (1500, 1): 4.1,
    (1500, 2): 5.8,
    (1500, 4): 9.7,
    (1800, 1): 5.3,
    (1800, 2): 7.9,
    (1800, 4): 12.6,
    (2000, 1): 6.5,
    (2000, 2): 9.8,
    (2000, 4): 15.4
}

# Crypto algorithm latency profiles: Maps (algorithm, frequency_mhz, active_cores) to latency in milliseconds
LATENCY_PROFILES = {
    # Format: (algorithm, frequency_mhz, active_cores): latency_ms
    ("ASCON_128", 1200, 1): 8.5,
    ("ASCON_128", 1500, 1): 6.8,
    ("ASCON_128", 1800, 1): 5.7,
    ("ASCON_128", 2000, 1): 5.0,
    ("ASCON_128", 1200, 2): 6.2,
    ("ASCON_128", 1500, 2): 5.0,
    ("ASCON_128", 1800, 2): 4.1,
    ("ASCON_128", 2000, 2): 3.7,
    ("ASCON_128", 1200, 4): 5.1,
    ("ASCON_128", 1500, 4): 4.1,
    ("ASCON_128", 1800, 4): 3.4,
    ("ASCON_128", 2000, 4): 3.0,
    
    ("KYBER_CRYPTO", 1200, 1): 5.6,
    ("KYBER_CRYPTO", 1500, 1): 4.4,
    ("KYBER_CRYPTO", 1800, 1): 3.7,
    ("KYBER_CRYPTO", 2000, 1): 3.3,
    ("KYBER_CRYPTO", 1200, 2): 3.5,
    ("KYBER_CRYPTO", 1500, 2): 2.8,
    ("KYBER_CRYPTO", 1800, 2): 2.3,
    ("KYBER_CRYPTO", 2000, 2): 2.1,
    ("KYBER_CRYPTO", 1200, 4): 2.9,
    ("KYBER_CRYPTO", 1500, 4): 2.3,
    ("KYBER_CRYPTO", 1800, 4): 1.9,
    ("KYBER_CRYPTO", 2000, 4): 1.7,
    
    ("SPHINCS", 1200, 1): 39.2,
    ("SPHINCS", 1500, 1): 31.4,
    ("SPHINCS", 1800, 1): 26.1,
    ("SPHINCS", 2000, 1): 23.5,
    ("SPHINCS", 1200, 2): 24.5,
    ("SPHINCS", 1500, 2): 19.6,
    ("SPHINCS", 1800, 2): 16.3,
    ("SPHINCS", 2000, 2): 14.7,
    ("SPHINCS", 1200, 4): 19.6,
    ("SPHINCS", 1500, 4): 15.7,
    ("SPHINCS", 1800, 4): 13.1,
    ("SPHINCS", 2000, 4): 11.8,
    
    ("FALCON512", 1200, 1): 1.4,
    ("FALCON512", 1500, 1): 1.1,
    ("FALCON512", 1800, 1): 0.9,
    ("FALCON512", 2000, 1): 0.8,
    ("FALCON512", 1200, 2): 0.9,
    ("FALCON512", 1500, 2): 0.7,
    ("FALCON512", 1800, 2): 0.6,
    ("FALCON512", 2000, 2): 0.5,
    ("FALCON512", 1200, 4): 0.7,
    ("FALCON512", 1500, 4): 0.6,
    ("FALCON512", 1800, 4): 0.5,
    ("FALCON512", 2000, 4): 0.4,
}

# DDoS detection task time profiles: Maps (model, frequency_mhz, active_cores) to execution time in seconds
DDOS_TASK_TIME_PROFILES = {
    # Format: (model, frequency_mhz, active_cores): execution_time_seconds
    ("XGBOOST", 1200, 1): 3.2,
    ("XGBOOST", 1500, 1): 2.6,
    ("XGBOOST", 1800, 1): 2.1,
    ("XGBOOST", 2000, 1): 1.9,
    ("XGBOOST", 1200, 2): 1.9,
    ("XGBOOST", 1500, 2): 1.5,
    ("XGBOOST", 1800, 2): 1.3,
    ("XGBOOST", 2000, 2): 1.1,
    ("XGBOOST", 1200, 4): 1.4,
    ("XGBOOST", 1500, 4): 1.1,
    ("XGBOOST", 1800, 4): 0.9,
    ("XGBOOST", 2000, 4): 0.8,
    
    ("TST", 1200, 1): 12.5,
    ("TST", 1500, 1): 10.0,
    ("TST", 1800, 1): 8.3,
    ("TST", 2000, 1): 7.5,
    ("TST", 1200, 2): 7.5,
    ("TST", 1500, 2): 6.0,
    ("TST", 1800, 2): 5.0,
    ("TST", 2000, 2): 4.5,
    ("TST", 1200, 4): 5.0,
    ("TST", 1500, 4): 4.0,
    ("TST", 1800, 4): 3.3,
    ("TST", 2000, 4): 3.0,
}

# Simplified Security Ratings (1-10 scale, higher is more secure)
SECURITY_RATINGS = {
    "ASCON_128": 3,
    "KYBER_CRYPTO": 6,
    "SPHINCS": 10,
    "FALCON512": 8,
    "XGBOOST": 4,
    "TST": 9
}

# CPU Frequency Presets (MHz)
CPU_FREQUENCY_PRESETS = {
    "POWERSAVE": 1200,
    "BALANCED": 1500,
    "PERFORMANCE": 1800,
    "TURBO": 2000
}

# Helper functions to query the profiles
def get_power_consumption(frequency_mhz, active_cores):
    """Get power consumption in Watts for a given CPU configuration"""
    return POWER_PROFILES.get((frequency_mhz, active_cores), 0.0)

def get_crypto_latency(algorithm, frequency_mhz, active_cores):
    """Get cryptographic algorithm latency in milliseconds for a given configuration"""
    return LATENCY_PROFILES.get((algorithm, frequency_mhz, active_cores), 0.0)

def get_ddos_execution_time(model, frequency_mhz, active_cores):
    """Get DDoS detection execution time in seconds for a given configuration"""
    return DDOS_TASK_TIME_PROFILES.get((model, frequency_mhz, active_cores), 0.0)

def get_security_rating(algorithm):
    """Get security rating (1-10) for a given algorithm"""
    return SECURITY_RATINGS.get(algorithm, 0)
