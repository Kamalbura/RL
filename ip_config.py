"""
Network Configuration - Single Source of Truth
All IP addresses set to 127.0.0.1 for local testing as per requirements.
"""

# Network Configuration
NETWORK_CONFIG = {
    # GCS (Ground Control Station) Configuration
    "GCS_IP": "127.0.0.1",
    "GCS_TELEMETRY_PORT": 14550,
    "GCS_COMMAND_PORT": 14551,
    "GCS_CRYPTO_SERVICE_PORT": 8080,
    "GCS_RL_AGENT_PORT": 8081,
    
    # Drone Configuration
    "DRONE_IP": "127.0.0.1", 
    "DRONE_TELEMETRY_PORT": 14552,
    "DRONE_COMMAND_PORT": 14553,
    "DRONE_CRYPTO_SERVICE_PORT": 8082,
    "DRONE_RL_AGENT_PORT": 8083,
    
    # MQTT Configuration for Crypto Switching
    "MQTT_BROKER_IP": "127.0.0.1",
    "MQTT_BROKER_PORT": 1883,
    "MQTT_CRYPTO_TOPIC": "uav/crypto/switch",
    "MQTT_STATUS_TOPIC": "uav/status",
    
    # MAVLink Configuration
    "MAVLINK_CONNECTION_STRING": "udp:127.0.0.1:14550",
    "MAVLINK_BAUD_RATE": 57600,
    
    # Timeouts and Intervals
    "CONNECTION_TIMEOUT": 30,  # seconds
    "HEARTBEAT_INTERVAL": 1,   # seconds
    "CRYPTO_SWITCH_TIMEOUT": 5, # seconds
}

# Algorithm Identifiers for Network Commands
CRYPTO_ALGORITHM_IDS = {
    "ASCON_128": 0,
    "KYBER_HYBRID": 1, 
    "DILITHIUM_SIGNATURE": 2,
    "SPHINCS_SIGNATURE": 3,
    "FALCON_SIGNATURE": 4,
    "CHACHA20_POLY1305": 5,
    "AES_256_GCM": 6,
    "RSA_4096": 7
}

# Reverse mapping for ID to algorithm name
ALGORITHM_ID_TO_NAME = {v: k for k, v in CRYPTO_ALGORITHM_IDS.items()}

def get_gcs_address():
    """Get GCS network address tuple"""
    return (NETWORK_CONFIG["GCS_IP"], NETWORK_CONFIG["GCS_CRYPTO_SERVICE_PORT"])

def get_drone_address():
    """Get Drone network address tuple"""
    return (NETWORK_CONFIG["DRONE_IP"], NETWORK_CONFIG["DRONE_CRYPTO_SERVICE_PORT"])

def get_mqtt_config():
    """Get MQTT broker configuration"""
    return {
        "host": NETWORK_CONFIG["MQTT_BROKER_IP"],
        "port": NETWORK_CONFIG["MQTT_BROKER_PORT"],
        "crypto_topic": NETWORK_CONFIG["MQTT_CRYPTO_TOPIC"],
        "status_topic": NETWORK_CONFIG["MQTT_STATUS_TOPIC"]
    }
