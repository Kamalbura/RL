#!/usr/bin/env python3
"""
UAV Scheduler (v14.0 - Enhanced MQTT Security & Reliability)

- Implements modular, thread-safe MQTT communication architecture
- Provides robust certificate management and reconnection strategies
- Incorporates proper error handling and resource management
- Maintains secure TLS configuration with proper certificate validation
- Separates MQTT message handling from business logic using queue-based processing
"""

import os
import sys
import time
import signal
import logging
import subprocess
import psutil
import csv
import threading
import argparse
import json
import socket
import queue
import random
import ssl
from enum import IntEnum, Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
import paho.mqtt.client as mqtt

# Tactical RL imports
try:
    from ddos_rl.agent import QLearningAgent as TacticalQLearningAgent
    from ddos_rl.env import TacticalUAVEnv
    from shared.crypto_profiles import DDOS_PROFILES, ThermalState
except Exception:
    TacticalQLearningAgent = None
    TacticalUAVEnv = None

# Import the crypto scheduler
from crypto_scheduler import CryptoScheduler

# --- CONFIGURATION & SETUP ---
LOG_FILE = f'/tmp/uav_scheduler_v14_{os.getuid()}.log'
METRICS_CSV_FILE = f'/tmp/uav_metrics_{os.getuid()}.csv'
THREAT_FLAG_FILE = '/tmp/uav_threat.flag'
CRYPTO_FLAG_FILE = '/tmp/crypto.flag'

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=LOG_FILE
)
logger = logging.getLogger('UAVScheduler')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s: %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# --- CORE DATA STRUCTURES & ENUMS ---
class TaskPriority(IntEnum):
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2

class AlgorithmType(str, Enum):
    TST = "tst"
    XGBOOST = "xgboost"
    MAVLINK = "mavlink"
    ASCON_128 = "ascon_128"
    KYBER_CRYPTO = "kyber_crypto"
    DILITHIUM2 = "dilithium2"
    FALCON512 = "falcon512"
    CAMELIA = "camellia"
    SPECK = "speck"
    HIGHT = "hight"

CRYPTO_OVERRIDE_MAP: Dict[int, AlgorithmType] = {
    1: AlgorithmType.ASCON_128,
    2: AlgorithmType.KYBER_CRYPTO,
    3: AlgorithmType.DILITHIUM2,
    4: AlgorithmType.FALCON512,
}

class ThreatLevel(IntEnum):
    NONE = 0
    POTENTIAL = 1
    CONFIRMING = 2
    CONFIRMED = 3

# --- Alert codes for MQTT communication ---
class AlertCode(str, Enum):
    CRITICAL = "awb-cri"
    CAUTION = "awb-cau"
    
class SwarmMessage(str, Enum):
    DDOS_DETECTED = "ddos_detected"
    SECURITY_CONCERN = "security_concern"
    THERMAL_EMERGENCY = "thermal_emergency"
    CRITICAL_BATTERY = "critical_battery"

# --- MQTT Configuration from standalone scripts ---
MQTT_BROKER_HOST = "192.168.0.103" # Your GCS Local IP
MQTT_BROKER_PORT = 8883

# Standard paths for certificates
STANDARD_CERT_PATHS = [
    "/home/dev/src/client/certs",
    "/home/dev/src/client",
    "/home/dev/swarm/certs",  # Corrected typo "swam" to "swarm"
    "/home/dev/certs"
]

# --- VENV PATHS ---
CRYPTO_ENV_PATH = "/home/dev/cenv"
DDOS_ENV_PATH = "/home/dev/nenv"
MAVLINK_ENV_PATH = "/home/dev/nenv"

@dataclass
class ResourceProfile:
    power_watts: float

@dataclass
class SystemState:
    timestamp: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    battery_percent: float = 100.0
    temperature: float = 45.0
    power_draw_watts: float = 0.0
    threat_level: ThreatLevel = ThreatLevel.NONE

@dataclass
class Task:
    id: str
    name: str
    command: List[str]
    priority: TaskPriority
    algorithm: Optional[AlgorithmType] = None
    resource_profile: Optional[ResourceProfile] = None
    status: str = "CREATED"
    start_time: Optional[float] = None
    process: Optional[subprocess.Popen] = None
    pid: Optional[int] = None
    auto_restart: bool = False
    capture_output: bool = False  # NEW: stream stdout if True
    output_thread: Optional[threading.Thread] = None  # NEW: holder for streaming thread

# --- NEW: MQTT Message for thread-safe processing ---
@dataclass
class MQTTMessage:
    topic: str
    payload: bytes
    qos: int
    retain: bool = False

# --- NEW: Certificate Manager class ---
class CertificateManager:
    def __init__(self, drone_id: str):
        self.drone_id = drone_id
        self.ca_cert = None
        self.client_cert = None
        self.client_key = None
        self.cert_error = None
    
    def resolve_certificates(self) -> bool:
        """Resolve certificate paths using standard locations with proper validation"""
        for base_path in STANDARD_CERT_PATHS:
            if not os.path.isdir(base_path):
                continue
                
            # Try standard naming convention
            ca_path = os.path.join(base_path, "ca-cert.pem")
            cert_path = os.path.join(base_path, f"{self.drone_id}-cert.pem")
            key_path = os.path.join(base_path, f"{self.drone_id}-key.pem")
            
            # If not found, check clients subdirectory
            if not all(os.path.isfile(p) for p in [ca_path, cert_path, key_path]):
                clients_dir = os.path.join(base_path, "clients")
                if os.path.isdir(clients_dir):
                    ca_path = os.path.join(base_path, "ca-cert.pem")  # CA usually in parent dir
                    cert_path = os.path.join(clients_dir, f"{self.drone_id}-cert.pem")
                    key_path = os.path.join(clients_dir, f"{self.drone_id}-key.pem")
            
            # Validate all files exist
            if all(os.path.isfile(p) for p in [ca_path, cert_path, key_path]):
                # Verify certificate format
                try:
                    self._verify_cert_format(cert_path)
                    self._verify_cert_format(ca_path)
                    
                    self.ca_cert = ca_path
                    self.client_cert = cert_path
                    self.client_key = key_path
                    
                    logger.info(f"Valid certificates found in: {base_path}")
                    logger.info(f"Using CA: {ca_path}")
                    logger.info(f"Using cert: {cert_path}")
                    logger.info(f"Using key: {key_path}")
                    return True
                except Exception as e:
                    logger.warning(f"Invalid certificate format in {base_path}: {e}")
                    self.cert_error = str(e)
                    continue
        
        # No valid certificates found
        missing_detail = "No valid certificate paths found"
        logger.error(f"Certificate resolution failed: {missing_detail}")
        self.cert_error = missing_detail
        return False
    
    def _verify_cert_format(self, cert_path: str):
        """Basic verification that the file is a valid certificate"""
        try:
            with open(cert_path, 'r') as f:
                content = f.read()
                if "-----BEGIN CERTIFICATE-----" not in content:
                    raise ValueError(f"File does not contain a certificate header")
                if "-----END CERTIFICATE-----" not in content:
                    raise ValueError(f"File does not contain a certificate footer")
        except Exception as e:
            raise ValueError(f"Certificate verification failed: {str(e)}")

# --- NEW: MQTT Client with robust error handling and reconnection ---
class MQTTClient:
    # Reconnection parameters
    BASE_RECONNECT_WAIT = 1.0  # seconds
    MAX_RECONNECT_WAIT = 60.0  # max backoff of 1 minute
    
    def __init__(self, drone_id: str, message_callback: Callable[[MQTTMessage], None]):
        self.drone_id = drone_id
        self.client = None
        self.connected = False
        self.reconnect_count = 0
        self.message_callback = message_callback
        self.cert_manager = CertificateManager(drone_id)
        
        # Thread-safe message queue
        self.stop_event = threading.Event()
        self.reconnect_timer = None
        
        # Metrics
        self.metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "connect_attempts": 0,
            "reconnect_attempts": 0,
            "connection_errors": 0
        }
        
        # Protocol version
        self.protocol_version = mqtt.MQTTv5
    
    def initialize(self) -> bool:
        """Initialize the MQTT client with certificate resolution"""
        if not self.cert_manager.resolve_certificates():
            logger.error(f"Failed to resolve certificates: {self.cert_manager.cert_error}")
            return False
            
        return self._setup_mqtt_client()
    
    def _setup_mqtt_client(self) -> bool:
        """Set up MQTT client with proper error handling and protocol negotiation"""
        try:
            # Try MQTT v5 first, then fall back as needed
            self._create_client_with_protocol()
            
            # Set up callbacks
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
            self.client.on_publish = self._on_publish
            
            # Set up TLS with proper verification
            self._configure_tls_for_ip()
            
            logger.info(f"MQTT client setup completed.")
            return True
        except Exception as e:
            logger.error(f"Failed to setup MQTT client: {str(e)}", exc_info=True)
            self.client = None
            return False
    
    def _create_client_with_protocol(self):
        """Create client with protocol version negotiation"""
        try:
            # Try MQTT v5 first
            self.client = mqtt.Client(protocol=self.protocol_version, client_id=self.drone_id)
            logger.info(f"Using MQTT v5 protocol")
        except Exception as e:
            logger.warning(f"MQTT v5 not supported, falling back to v3.1.1: {e}")
            try:
                self.protocol_version = mqtt.MQTTv311
                self.client = mqtt.Client(protocol=self.protocol_version, client_id=self.drone_id)
                logger.info(f"Using MQTT v3.1.1 protocol")
            except Exception as e2:
                logger.warning(f"MQTT v3.1.1 not supported, using default client: {e2}")
                self.protocol_version = mqtt.MQTTv31
                self.client = mqtt.Client(client_id=self.drone_id)
                logger.info(f"Using default MQTT client")
    
    def _configure_tls_for_ip(self):
        """Configure TLS with CA validation; hostname disabled (IP broker). Optional pinning hook stub."""
        try:
            ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=self.cert_manager.ca_cert)
            ctx.check_hostname = False  # IP broker so no hostname match
            ctx.verify_mode = ssl.CERT_REQUIRED
            ctx.load_cert_chain(certfile=self.cert_manager.client_cert, keyfile=self.cert_manager.client_key)
            # Certificate pinning placeholder (uncomment & set expected_fp to enforce)
            # expected_fp = "SHA256:..."
            # def _pin(sock):
            #     cert = sock.getpeercert(binary_form=True)
            #     import hashlib
            #     fp = "SHA256:" + hashlib.sha256(cert).hexdigest().upper()
            #     if fp != expected_fp:
            #         raise ssl.SSLError("Server certificate fingerprint mismatch")
            self.client.tls_set_context(ctx)
            logger.info("TLS context configured (CA validated, hostname disabled for IP broker)")
        except ssl.SSLError as e:
            logger.error(f"SSL configuration error: {e}")
            raise ConnectionError(f"TLS setup failed: {e}")
        except FileNotFoundError as e:
            logger.error(f"Certificate file not found: {e}")
            raise ConnectionError(f"Missing certificate: {e}")
    
    def connect(self):
        """Connect to the MQTT broker with proper error handling"""
        if not self.client:
            logger.error("Cannot connect: MQTT client not initialized")
            return False
            
        try:
            self.metrics["connect_attempts"] += 1
            logger.info(f"Connecting to MQTT broker at {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}...")
            self.client.connect_async(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
            self.client.loop_start()
            return True
        except Exception as e:
            self.metrics["connection_errors"] += 1
            logger.error(f"Error connecting to MQTT broker: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker and cleanup resources"""
        self.stop_event.set()
        if self.reconnect_timer:
            self.reconnect_timer.cancel()
            
        if self.client and self.connected:
            try:
                self.client.disconnect()
                self.client.loop_stop()
                logger.info("MQTT client disconnected")
            except Exception as e:
                logger.error(f"Error during MQTT disconnect: {e}")
    
    def publish(self, topic: str, payload: dict, qos: int = 1) -> bool:
        """Publish a message to the MQTT broker with error handling"""
        if not self.client or not self.connected:
            logger.warning(f"Cannot publish to {topic}: MQTT client not connected")
            return False
            
        try:
            json_payload = json.dumps(payload)
            result = self.client.publish(topic, json_payload, qos=qos)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.metrics["messages_sent"] += 1
                self.metrics["bytes_sent"] += len(json_payload)
                logger.info(f"Message published to {topic}")
                return True
            else:
                logger.warning(f"Failed to publish message to {topic}: {mqtt.error_string(result.rc)}")
                return False
        except Exception as e:
            logger.error(f"Error publishing message to {topic}: {str(e)}")
            return False
    
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Handle connection to MQTT broker"""
        if rc == 0:
            self.connected = True
            self.reconnect_count = 0  # Reset reconnect counter on successful connection
            logger.info("Connected to MQTT broker")
            try:
                client.subscribe("swarm/broadcast/#", qos=1)
                client.subscribe(f"swarm/config/#", qos=2)
                # Subscribe to individual command topic
                client.subscribe(f"swarm/commands/individual/{self.drone_id}", qos=2)
                logger.info("Subscribed to swarm broadcast, config, and individual command channels")
            except Exception as e:
                logger.error(f"Failed to subscribe to topics: {e}")
        else:
            reason_str = {
                1: "incorrect protocol version",
                2: "invalid client identifier",
                3: "server unavailable",
                4: "bad username or password",
                5: "not authorised",
            }.get(rc, f"unknown error code {rc}")
            
            logger.error(f"MQTT connection failed: {reason_str}")
            logger.error(f"Connection details: broker={MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}, client_id={self.drone_id}")
            self.connected = False
            self._schedule_reconnection()
    
    def _on_disconnect(self, client, userdata, rc, properties=None):
        """Handle disconnection from MQTT broker with exponential backoff reconnection"""
        self.connected = False
        reason_str = "clean disconnect" if rc == 0 else f"unexpected disconnect with code {rc}"
        logger.warning(f"Disconnected from MQTT broker. Reason: {reason_str}")
        
        # Try to reconnect with exponential backoff if not a clean disconnect
        if rc != 0 and not self.stop_event.is_set():
            self._schedule_reconnection()
    
    def _schedule_reconnection(self):
        """Schedule reconnection attempt with exponential backoff"""
        if self.stop_event.is_set():
            return
            
        self.metrics["reconnect_attempts"] += 1
        backoff_time = min(self.MAX_RECONNECT_WAIT, 
                          self.BASE_RECONNECT_WAIT * (2 ** min(self.reconnect_count, 6)))
        jitter = random.uniform(0, 0.5)  # Add jitter to avoid thundering herd
        wait_time = backoff_time + jitter
        
        logger.info(f"Scheduling reconnection in {wait_time:.2f} seconds (attempt {self.reconnect_count + 1})")
        
        # Cancel any existing timer
        if self.reconnect_timer:
            self.reconnect_timer.cancel()
            
        # Create new timer for reconnection
        self.reconnect_timer = threading.Timer(wait_time, self._reconnect)
        self.reconnect_timer.daemon = True
        self.reconnect_timer.start()
        self.reconnect_count += 1
    
    def _reconnect(self):
        """Attempt to reconnect to the MQTT broker"""
        if self.stop_event.is_set():
            return
            
        logger.info(f"Attempting to reconnect to MQTT broker...")
        try:
            if self.client:
                self.client.loop_stop()
                self.client.reconnect()
                self.client.loop_start()
        except Exception as e:
            logger.error(f"Reconnection attempt failed: {str(e)}")
            # Schedule next reconnection
            self._schedule_reconnection()
    
    def _on_message(self, client, userdata, msg):
        """Handle incoming MQTT messages with queue for thread safety"""
        try:
            # Record metrics
            self.metrics["messages_received"] += 1
            self.metrics["bytes_received"] += len(msg.payload)
            
            # Forward to callback for processing in main thread context
            if self.message_callback and not self.stop_event.is_set():
                mqtt_msg = MQTTMessage(
                    topic=msg.topic,
                    payload=msg.payload,
                    qos=msg.qos,
                    retain=msg.retain
                )
                self.message_callback(mqtt_msg)
        except Exception as e:
            logger.error(f"Error in MQTT message handler: {str(e)}")
    
    def _on_publish(self, client, userdata, mid):
        """Handle message publish acknowledgments"""
        logger.debug(f"Message {mid} published successfully")

class TheveninBatteryModel:
    def __init__(self, initial_soc=100.0):
        self.soc = initial_soc
        self.capacity = 5200.0
        self.nominal_voltage = 11.1
        self.last_update_time = time.time()

    def update(self, current_draw_amps: float) -> float:
        now = time.time()
        elapsed_seconds = now - self.last_update_time
        self.last_update_time = now
        if elapsed_seconds <= 0: return self.soc
        energy_drawn_Ah = (current_draw_amps * elapsed_seconds) / 3600.0
        soc_change = (energy_drawn_Ah / (self.capacity / 1000.0)) * 100
        self.soc = max(0.0, self.soc - soc_change)
        return self.soc

class DataLogger:
    def __init__(self, filename: str):
        self.filename = filename
        self.header = ['timestamp', 'cpu_usage', 'battery_percent', 'temperature', 'power_draw_watts', 'threat_level', 'active_ddos_model', 'active_crypto_model']
        try:
            with open(self.filename, 'w', newline='') as f:
                csv.writer(f).writerow(self.header)
            logger.info(f"Metrics will be logged to {self.filename}")
        except IOError as e:
            logger.error(f"Failed to create metrics log file: {e}")
            self.filename = None

    def log_state(self, state: SystemState, active_ddos_model: Optional[str], active_crypto_model: Optional[str]):
        if not self.filename: return
        try:
            log_data = asdict(state)
            log_data['active_ddos_model'] = active_ddos_model or "None"
            log_data['active_crypto_model'] = active_crypto_model or "None"
            log_data['threat_level'] = state.threat_level.name
            with open(self.filename, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.header)
                writer.writerow({k: log_data[k] for k in self.header if k in log_data})
        except Exception as e:
            logger.warning(f"Could not write to metrics file: {e}")


class UAVScheduler:
    def __init__(self, config: Dict = None):
        self.config = self._load_config(config)
        self.drone_id = self.config.get("drone_id", "uav-drone-001")
        self.state = SystemState()
        self.battery_model = TheveninBatteryModel(self.config.get("initial_battery", 100.0))
        self.data_logger = DataLogger(METRICS_CSV_FILE)
        self.task_queue: List[Task] = []
        self.running_tasks: Dict[str, Task] = {}
        self.is_running = False
        self.lock = threading.RLock()
        self.monitor_thread = None
        
        self.current_ddos_model: Optional[AlgorithmType] = None
        self.ddos_task_id: Optional[str] = None
        self.tst_start_time: Optional[float] = None
        
        self.current_crypto_model: Optional[AlgorithmType] = None
        self.crypto_task_id: Optional[str] = None
        
        # Initialize the crypto RL agent scheduler
        self.crypto_rl_scheduler = CryptoScheduler()

        # Tactical RL agent (loaded if available)
        self.tactical_agent: Optional[TacticalQLearningAgent] = None
        self.tactical_env: Optional[TacticalUAVEnv] = None
        self.tactical_agent_loaded: bool = False
        self.tactical_action_interval_sec: float = 5.0  # More frequent decisions
        self._last_tactical_decision: float = 0.0
        
        self._init_tactical_agent()

        # --- NEW: Message queue for thread-safe processing ---
        self.message_queue = queue.Queue()
        self.message_processor_thread = None
        
        # --- NEW: Improved MQTT Client ---
        self.mqtt_client = MQTTClient(self.drone_id, self._queue_message)
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_config(self, user_config: Optional[Dict]) -> Dict:
        default_config = {
            "initial_battery": 100.0, "battery_critical_threshold": 20.0,
            "battery_low_threshold": 60.0, "tst_max_runtime_sec": 60.0,
            "thermal_emergency_threshold_c": 78.0, "drone_id": "uav-drone-001"
        }
        if user_config: default_config.update(user_config)
        return default_config

    # --- NEW: Thread-safe message handling ---
    def _queue_message(self, message: MQTTMessage):
        """Queue message for processing in main thread context"""
        self.message_queue.put(message)
    
    def _process_messages(self):
        """Process queued messages in a thread-safe manner"""
        while self.is_running:
            try:
                # Get message from queue with timeout to allow clean shutdown
                message = self.message_queue.get(timeout=0.5)
                
                # Process message with lock to avoid race conditions
                with self.lock:
                    self._handle_mqtt_message(message)
                    
                self.message_queue.task_done()
            except queue.Empty:
                # No messages in queue, continue
                continue
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")

    def _handle_mqtt_message(self, message: MQTTMessage):
        """Process MQTT message with proper validation and individual command support"""
        try:
            logger.info(f"Message received on topic '{message.topic}'")
            # Handle direct commands to this drone
            if message.topic == f"swarm/commands/individual/{self.drone_id}":
                try:
                    cmd = message.payload.decode().strip().lower()
                    logger.info(f"Received individual command: {cmd}")
                    if cmd == "status_update":
                        self._publish_heartbeat(force_full=True)
                    elif cmd == "rtl":
                        logger.warning("Received RTL command (Return to Launch) - implement as needed.")
                        # Implement RTL logic here
                    elif cmd == "hover":
                        logger.warning("Received HOVER command - implement as needed.")
                        # Implement hover logic here
                    else:
                        logger.warning(f"Unknown individual command: {cmd}")
                except Exception as e:
                    logger.error(f"Error handling individual command: {e}")
                return
            # Special handling for crypto messages which may be plain text
            if message.topic == 'swarm/broadcast/crypto':
                try:
                    crypto_text = message.payload.decode().strip()
                    logger.info(f"Received crypto command: {crypto_text}")
                    self._handle_crypto_command(crypto_text)
                    return
                except Exception as e:
                    logger.error(f"Error processing crypto command: {e}")
            # Special handling for alert messages which may be plain text
            if message.topic == 'swarm/broadcast/alert':
                try:
                    alert_code = message.payload.decode().strip()
                    logger.info(f"Received alert command: {alert_code}")
                    self._handle_alert_command(alert_code)
                    return
                except Exception as e:
                    logger.error(f"Error processing alert command: {e}")
            # Validate and decode payload for JSON messages
            try:
                payload = json.loads(message.payload.decode())
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON payload on topic {message.topic}")
                return
            # Process message based on topic
            if message.topic.startswith('swarm/config/'):
                self._handle_config_message(payload)
            elif message.topic.startswith('swarm/broadcast/'):
                self._handle_broadcast_message(payload)
        except Exception as e:
            logger.error(f"Error processing MQTT message: {str(e)}")
    def _publish_heartbeat(self, force_full: bool = False):
        if not self.mqtt_client or not self.mqtt_client.connected:
            return
        # Heartbeat documentation:
        # Topic: swarm/status/<drone_id>
        # Payload schema:
        # {
        #   "type": "heartbeat",
        #   "drone_id": "uav-drone-001",
        #   "timestamp": <epoch_float>,
        #   "threat_level": "NONE|POTENTIAL|CONFIRMING|CONFIRMED",
        #   "data": {
        #       "cpu_usage": <float percent>,
        #       "memory_usage": <float percent>,
        #       "temperature": <float C>,
        #       "power_draw_watts": <float>,
        #       "battery_percent": <float>,
        #       "threat_level": <repeat threat level>,
        #       "crypto_algorithm": <string or null>,
        #       "ddos_model": <string or null>
        #   }
        # }
        data = {
            "cpu_usage": self.state.cpu_usage,
            "memory_usage": self.state.memory_usage,
            "temperature": self.state.temperature,
            "power_draw_watts": self.state.power_draw_watts,
            "battery_percent": float(self.state.battery_percent),
            "threat_level": self.state.threat_level.name,
            "crypto_algorithm": self.current_crypto_model.value if self.current_crypto_model else None,
            "ddos_model": self.current_ddos_model.value if self.current_ddos_model else None,
        }
        payload = {
            "type": "heartbeat",
            "drone_id": self.drone_id,
            "timestamp": time.time(),
            "threat_level": self.state.threat_level.name,
            "data": data
        }
        self.mqtt_client.publish(f"swarm/status/{self.drone_id}", payload, qos=0)
    
    def _handle_config_message(self, payload: dict):
        """Handle configuration messages from GCS"""
        new_algo_str = payload.get("algorithm")
        if not new_algo_str:
            logger.warning("Config message missing algorithm field")
            return
            
        try:
            target_algorithm = AlgorithmType(new_algo_str)
            if self.current_crypto_model != target_algorithm:
                logger.critical(f"GCS DIRECTIVE: Switching crypto to {target_algorithm.value}")
                
                # Create task resources
                crypto_task = create_crypto_task(target_algorithm)
                
                # Stop old task with proper resource cleanup
                if self.crypto_task_id:
                    self._stop_task(self.crypto_task_id)
                    
                # Submit new task
                self.submit_task(crypto_task)
        except ValueError:
            logger.error(f"Invalid algorithm type: {new_algo_str}")
    
    def _handle_broadcast_message(self, payload: dict):
        """Handle broadcast messages from other drones"""
        original_sender = payload.get("original_sender")
        if not original_sender:
            logger.warning("Broadcast message missing original_sender field")
            return
        if original_sender != self.drone_id:
            alert = payload.get("alert", "No details")
            # Removed emoji for plain log
            logger.warning(f"SWARM ALERT relayed from {original_sender}: {alert}")
            if self.state.threat_level == ThreatLevel.NONE:
                self.state.threat_level = ThreatLevel.POTENTIAL
                logger.info("Elevating local threat level due to swarm alert.")
                
    def _handle_crypto_command(self, crypto_text: str):
        """Handle plain text crypto command from GCS"""
        try:
            # Check if it's a crypto command like c1, c2, etc.
            if crypto_text.startswith('c') and len(crypto_text) == 2 and crypto_text[1].isdigit():
                crypto_num = int(crypto_text[1])
                if 1 <= crypto_num <= 8:  # Validate range
                    logger.info(f"GCS DIRECTIVE: Received crypto command {crypto_text}")
                    
                    # Map command to algorithm
                    algorithm_map = {
                        1: AlgorithmType.ASCON_128,
                        2: AlgorithmType.KYBER_CRYPTO, 
                        3: AlgorithmType.DILITHIUM2,
                        4: AlgorithmType.FALCON512,
                        5: AlgorithmType.CAMELIA,
                        6: AlgorithmType.SPECK,
                        7: AlgorithmType.HIGHT,
                        8: AlgorithmType.ASCON_128  # Default backup
                    }
                    
                    target_algorithm = algorithm_map.get(crypto_num)
                    if target_algorithm and self.current_crypto_model != target_algorithm:
                        logger.critical(f"GCS DIRECTIVE: Switching crypto to {target_algorithm.value}")
                        
                        # Stop old task with proper resource cleanup
                        if self.crypto_task_id:
                            self._stop_task(self.crypto_task_id)
                            
                        # Submit new task
                        self.submit_task(create_crypto_task(target_algorithm))
                    return
            
            # If we get here, it's not a valid crypto command format
            logger.warning(f"Unrecognized crypto command format: {crypto_text}")
        except Exception as e:
            logger.error(f"Error processing crypto command {crypto_text}: {e}")
            
    def _handle_alert_command(self, alert_code: str):
        """Handle plain text alert command from GCS"""
        try:
            # Convert GCS alert codes to drone alert codes
            alert_code_map = {
                # GCS alert codes to drone alert codes
                "alb-cri": AlertCode.CRITICAL,
                "alb-cau": AlertCode.CAUTION,
            }
            # Process the alert code
            if alert_code in alert_code_map:
                received_alert = alert_code_map[alert_code]
                logger.critical(f"ALERT RECEIVED FROM GCS: {alert_code}")
                # Take appropriate action based on alert level
                if received_alert == AlertCode.CRITICAL:
                    logger.critical("IMPLEMENTING CRITICAL SECURITY MEASURES")
                    # Increase threat level
                    if self.state.threat_level < ThreatLevel.CONFIRMING:
                        self.state.threat_level = ThreatLevel.CONFIRMING
                    # Switch to high-security crypto
                    if self.current_crypto_model != AlgorithmType.FALCON512:
                        logger.info("Switching to FALCON512 crypto due to critical alert")
                        if self.crypto_task_id:
                            self._stop_task(self.crypto_task_id)
                        self.submit_task(create_crypto_task(AlgorithmType.FALCON512))
                    # Start TST scanning if not already running
                    if self.current_ddos_model != AlgorithmType.TST:
                        if self.ddos_task_id:
                            self._stop_task(self.ddos_task_id)
                        self.submit_task(create_ddos_task(AlgorithmType.TST))
                elif received_alert == AlertCode.CAUTION:
                    logger.warning("IMPLEMENTING HEIGHTENED SECURITY MEASURES")
                    # Increase threat level if currently none
                    if self.state.threat_level == ThreatLevel.NONE:
                        self.state.threat_level = ThreatLevel.POTENTIAL
            else:
                logger.warning(f"Unknown alert code received: {alert_code}")
        except Exception as e:
            logger.error(f"Error processing alert command {alert_code}: {e}")

    def _publish_threat_alert(self, alert_code: AlertCode, threat_data: Dict):
        """Publish threat alert with proper error handling"""
        if not self.mqtt_client:
            logger.warning("Cannot publish alert - MQTT client not initialized")
            return False
            
        # Create alert payload
        alert_payload = {
            "drone_id": self.drone_id,
            "timestamp": time.time(),
            "alert_code": alert_code.value,
            "data": threat_data
        }
        
        # Publish to alert topic
        alert_topic = f"swarm/alert/{self.drone_id}"
        result = self.mqtt_client.publish(alert_topic, alert_payload, qos=1)
        if result:
            logger.critical(f"ALERT PUBLISHED to GCS: {alert_code.value}")
        return result

    def _priority_to_nice(self, priority: TaskPriority) -> int:
        return {TaskPriority.CRITICAL: -20, TaskPriority.HIGH: -10, TaskPriority.MEDIUM: 0}.get(priority, 0)

    def submit_task(self, task: Task):
        with self.lock:
            self.task_queue.append(task)
            logger.info(f"Task '{task.name}' has been queued for execution.")

    def _start_task(self, task: Task) -> bool:
        if task.id in self.running_tasks:
            return False
        resources_allocated = []
        try:
            logger.info(f"Attempting to start task '{task.name}'...")
            if task.capture_output:
                process = subprocess.Popen(
                    task.command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    preexec_fn=os.setsid
                )
            else:
                process = subprocess.Popen(
                    task.command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid
                )
            resources_allocated.append(('process', process))
            task.pid = process.pid
            task.start_time = time.time()
            task.status = "RUNNING"
            self.running_tasks[task.id] = task
            resources_allocated.append(('task_registry', task.id))
            if task.pid:
                try:
                    psutil.Process(task.pid).nice(self._priority_to_nice(task.priority))
                except Exception as e:
                    logger.warning(f"Could not set priority for '{task.name}': {e}")
            if task.algorithm == AlgorithmType.TST:
                self.tst_start_time = task.start_time
                resources_allocated.append(('tst_time', True))
            if task.name.startswith("DDOS"):
                self.ddos_task_id = task.id
                self.current_ddos_model = task.algorithm
                resources_allocated.append(('ddos_model', task.algorithm))
            if task.name.startswith("Crypto"):
                self.crypto_task_id = task.id
                self.current_crypto_model = task.algorithm
                resources_allocated.append(('crypto_model', task.algorithm))
            logger.info(f"Task '{task.name}' started successfully (PID: {task.pid}).")
            if task.capture_output and process.stdout:
                task.output_thread = threading.Thread(target=self._stream_task_output, args=(task,), daemon=True)
                task.output_thread.start()
            return True
        except Exception as e:
            logger.error(f"A fatal error occurred while starting task '{task.name}': {e}")
            for resource_type, resource in reversed(resources_allocated):
                self._cleanup_resource(resource_type, resource, task)
            return False

    def _cleanup_resource(self, resource_type: str, resource, task: Optional[Task] = None):
        """Clean up allocated resources on failure"""
        try:
            if resource_type == 'process' and resource:
                try:
                    os.killpg(os.getpgid(resource.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                    
            elif resource_type == 'task_registry':
                self.running_tasks.pop(resource, None)
                
            elif resource_type == 'tst_time':
                self.tst_start_time = None
                
            elif resource_type == 'ddos_model':
                self.ddos_task_id = None
                self.current_ddos_model = None
                
            elif resource_type == 'crypto_model':
                self.crypto_task_id = None
                self.current_crypto_model = None
                
        except Exception as e:
            logger.error(f"Error during resource cleanup ({resource_type}): {e}")

    def _stop_task(self, task_id: str):
        with self.lock:
            task = self.running_tasks.pop(task_id, None)
            if not task:
                return
            logger.info(f"Stopping task '{task.name}' (PID: {task.pid}).")
            if task.algorithm == AlgorithmType.TST:
                self.tst_start_time = None
            if task.id == self.ddos_task_id:
                self.ddos_task_id, self.current_ddos_model = None, None
            if task.id == self.crypto_task_id:
                self.crypto_task_id, self.current_crypto_model = None, None
            try:
                if task.pid:
                    os.killpg(os.getpgid(task.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            if task.output_thread and task.output_thread.is_alive():
                try:
                    task.output_thread.join(timeout=1.5)
                except Exception:
                    pass

    def _update_system_state(self):
        self.state.timestamp = time.time()
        self.state.cpu_usage, self.state.memory_usage = psutil.cpu_percent(), psutil.virtual_memory().percent
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f: self.state.temperature = float(f.read().strip()) / 1000.0
        except Exception: self.state.temperature = 60.0
        base_power = 2.5 + (self.state.cpu_usage / 100.0 * 5.0)
        task_power = sum(t.resource_profile.power_watts for t in self.running_tasks.values())
        self.state.power_draw_watts = base_power + task_power
        current_draw = self.state.power_draw_watts / self.battery_model.nominal_voltage
        self.state.battery_percent = self.battery_model.update(current_draw)

    def _check_for_threat_signals(self):
        """Detect local threat indicators and raise to POTENTIAL only (state machine governs escalation)."""
        if os.path.exists(THREAT_FLAG_FILE):
            if self.state.threat_level == ThreatLevel.NONE:
                logger.warning("Threat flag detected: escalating to POTENTIAL")
                self.state.threat_level = ThreatLevel.POTENTIAL
            try:
                os.remove(THREAT_FLAG_FILE)
            except OSError:
                pass

    def _check_for_crypto_override(self):
        if not os.path.exists(CRYPTO_FLAG_FILE): return
        try:
            with open(CRYPTO_FLAG_FILE, 'r') as f: content = f.read().strip()
            
            target_algorithm = None
            if not content: # Empty file case
                target_algorithm = AlgorithmType.CAMELIA
            else: # Numbered case
                override_val = int(content)
                target_algorithm = CRYPTO_OVERRIDE_MAP.get(override_val)
            
            if target_algorithm and self.current_crypto_model != target_algorithm:
                logger.info(f"MANUAL OVERRIDE: Switching crypto to {target_algorithm.value}")
                if self.crypto_task_id: self._stop_task(self.crypto_task_id)
                self.submit_task(create_crypto_task(target_algorithm))
            
            os.remove(CRYPTO_FLAG_FILE)
        except Exception as e:
            logger.error(f"Could not process crypto override flag: {e}")

    def _manage_all_rounder_policy(self):
        """State machine for threat handling and resource governance."""
        # If tactical RL agent is active, delegate DDoS model management to it
        if self.tactical_agent_loaded:
            self._manage_tactical_rl_policy()
            return
        s, c = self.state, self.config
        # Survival overrides
        if s.battery_percent < c["battery_critical_threshold"] or s.temperature > c["thermal_emergency_threshold_c"]:
            if self.ddos_task_id:
                self._stop_task(self.ddos_task_id)
            if s.threat_level != ThreatLevel.NONE:
                logger.info("Resetting threat level due to survival override")
            s.threat_level = ThreatLevel.NONE
            if not self.ddos_task_id and self.is_running:
                self.submit_task(create_ddos_task(AlgorithmType.XGBOOST))
            return

        # Ensure baseline XGBOOST in NONE state
        if s.threat_level == ThreatLevel.NONE:
            if not self.ddos_task_id and self.is_running:
                logger.debug("Starting baseline XGBOOST monitor")
                self.submit_task(create_ddos_task(AlgorithmType.XGBOOST))
            return

        # POTENTIAL -> start TST if resources allow and not already running
        if s.threat_level == ThreatLevel.POTENTIAL:
            if s.battery_percent < c["battery_low_threshold"]:
                logger.info("Deferring escalation due to low battery")
                return
            if self.current_ddos_model != AlgorithmType.TST:
                logger.info("Escalating to TST (CONFIRMING phase)")
                if self.ddos_task_id:
                    self._stop_task(self.ddos_task_id)
                self.submit_task(create_ddos_task(AlgorithmType.TST))
                s.threat_level = ThreatLevel.CONFIRMING
            return

        # CONFIRMING -> wait for TST completion or timeout
        if s.threat_level == ThreatLevel.CONFIRMING:
            task = self.running_tasks.get(self.ddos_task_id)
            runtime = 0
            if task and task.start_time:
                runtime = time.time() - task.start_time
            if runtime > c["tst_max_runtime_sec"]:
                logger.info("TST window elapsed; confirming threat")
                self._publish_threat_alert(AlertCode.CRITICAL, {"model": "TST", "finding": "DDoS Confirmed"})
                if self.ddos_task_id:
                    self._stop_task(self.ddos_task_id)
                s.threat_level = ThreatLevel.CONFIRMED
            return

        # CONFIRMED -> maintain posture (optionally could re-start lightweight monitor)
        if s.threat_level == ThreatLevel.CONFIRMED:
            if not self.ddos_task_id and self.is_running:
                # Optionally keep a monitoring process
                self.submit_task(create_ddos_task(AlgorithmType.XGBOOST))
            return

    def _monitor_loop(self):
        while self.is_running:
            with self.lock:
                self._update_system_state()
                self._check_for_crypto_override()
                self._check_for_threat_signals()
                
                # Use crypto RL agent to select algorithm based on current state
                self._manage_crypto_rl_policy()
                
                # Use tactical RL agent if available, otherwise fallback to heuristic
                if self.tactical_agent_loaded:
                    self._manage_tactical_rl_policy()
                else:
                    self._manage_all_rounder_policy()
                if self.task_queue: self._start_task(self.task_queue.pop(0))
                self.data_logger.log_state(self.state,
                    self.current_ddos_model.value if self.current_ddos_model else None,
                    self.current_crypto_model.value if self.current_crypto_model else None)
                # Heartbeat after state update
                self._publish_heartbeat()
            time.sleep(2.0)
    
    def _manage_crypto_rl_policy(self):
        """Use the RL-based crypto scheduler to select optimal algorithm"""
        # Map system state to crypto RL state space
        security_risk = 0  # LOW by default
        if self.state.threat_level == ThreatLevel.POTENTIAL:
            security_risk = 1  # MEDIUM
        elif self.state.threat_level == ThreatLevel.CONFIRMING:
            security_risk = 2  # HIGH
        elif self.state.threat_level == ThreatLevel.CONFIRMED:
            security_risk = 3  # CRITICAL
            
        # Map battery level to RL state space
        battery_state = 3  # HIGH by default
        if self.state.battery_percent < 20:
            battery_state = 0  # CRITICAL
        elif self.state.battery_percent < 50:
            battery_state = 1  # LOW
        elif self.state.battery_percent < 80:
            battery_state = 2  # MEDIUM
            
        # Estimate computation capacity based on CPU load
        computation = 1  # NORMAL by default
        if self.state.cpu_usage > 70:
            computation = 0  # CONSTRAINED
        elif self.state.cpu_usage < 30:
            computation = 2  # ABUNDANT
            
        # Map mission criticality
        flight_mode_idx = 0  # Default to LOITER
        if hasattr(self, 'flight_mode_idx'):
            flight_mode_idx = self.flight_mode_idx
        mission_criticality = 1  # MEDIUM by default
        if flight_mode_idx == 1:  # MISSION
            mission_criticality = 2  # HIGH
        elif flight_mode_idx == 2:  # RTL
            mission_criticality = 3  # CRITICAL
            
        # Estimate communication intensity (can be refined with actual metrics)
        communication_intensity = 0  # LOW by default
        
        # Map threat context
        threat_context = 0  # BENIGN by default
        if self.state.threat_level == ThreatLevel.POTENTIAL:
            threat_context = 1  # SUSPICIOUS
        elif self.state.threat_level in [ThreatLevel.CONFIRMING, ThreatLevel.CONFIRMED]:
            threat_context = 2  # HOSTILE
            
        # Create RL state vector
        crypto_state = [
            security_risk,
            battery_state,
            computation,
            mission_criticality,
            communication_intensity,
            threat_context
        ]
        
        # Get recommendation from RL agent
        algorithm = self.crypto_rl_scheduler.select_crypto_algorithm(crypto_state)
        
        # Convert to AlgorithmType
        algorithm_name = algorithm["name"]
        target_algorithm = None
        
        # Map algorithm name to AlgorithmType
        if algorithm_name == "ASCON_128":
            target_algorithm = AlgorithmType.ASCON_128
        elif algorithm_name == "KYBER_CRYPTO":
            target_algorithm = AlgorithmType.KYBER_CRYPTO
        elif algorithm_name == "SPHINCS":
            # Map SPHINCS to DILITHIUM2 as closest available alternative
            target_algorithm = AlgorithmType.DILITHIUM2
        elif algorithm_name == "FALCON512":
            target_algorithm = AlgorithmType.FALCON512
            
        # Apply the selected algorithm if different from current
        if target_algorithm and self.current_crypto_model != target_algorithm:
            logger.info(f"RL AGENT: Switching crypto to {target_algorithm.value}")
            if self.crypto_task_id:
                self._stop_task(self.crypto_task_id)
            self.submit_task(create_crypto_task(target_algorithm))

    # --- Tactical RL integration ---
    def _init_tactical_agent(self):
        """Attempt to load tactical Q-table if available."""
        if TacticalQLearningAgent is None or TacticalUAVEnv is None:
            logger.warning("Tactical RL modules not available; running heuristic state machine.")
            return
        try:
            # Initialize tactical environment with thermal monitoring
            self.tactical_env = TacticalUAVEnv()
            
            # Initialize agent with updated state dimensions including thermal state
            state_dims = [4, 4, 3, 3, 3]  # Added ThermalState dimension
            action_dim = 9
            self.tactical_agent = TacticalQLearningAgent(state_dims=state_dims, action_dim=action_dim)
            
            # Try to load trained policy
            model_paths = [
                os.path.join("output", "tactical_q_table_best.npy"),
                os.path.join("output", "tactical_q_table.npy"),
                os.path.join("output_smoke", "tactical_q_table_best.npy"),
                os.path.join("output_smoke", "tactical_q_table.npy")
            ]
            
            loaded = False
            for model_path in model_paths:
                if os.path.exists(model_path):
                    try:
                        loaded = self.tactical_agent.load_policy(model_path)
                        if loaded:
                            logger.info(f"Tactical RL policy loaded from {model_path}")
                            break
                    except Exception as e:
                        logger.warning(f"Failed to load policy from {model_path}: {e}")
                        continue
            
            if loaded:
                self.tactical_agent_loaded = True
                logger.info("Tactical RL agent initialized with thermal monitoring enabled")
            else:
                logger.warning("No tactical RL policy found; using random policy for exploration")
                self.tactical_agent_loaded = True  # Still use agent for exploration
                
        except Exception as e:
            logger.error(f"Failed to initialize tactical RL agent: {e}")

    def _compose_tactical_state(self) -> Optional[list]:
        """Map current UAV runtime metrics to tactical RL state vector with thermal awareness."""
        if not self.tactical_agent_loaded:
            return None
            
        # Threat level (0-3)
        threat_idx = int(getattr(self.state.threat_level, 'value', 0))
        
        # Battery level mapping (0-3)
        bat = self.state.battery_percent
        if bat < 20: bat_band = 0      # CRITICAL
        elif bat < 50: bat_band = 1    # LOW  
        elif bat < 80: bat_band = 2    # MEDIUM
        else: bat_band = 3             # HIGH
        
        # CPU load mapping (0-2)
        cpu = self.state.cpu_usage
        if cpu < 30: cpu_band = 0      # LOW
        elif cpu < 70: cpu_band = 1    # MEDIUM
        else: cpu_band = 2             # HIGH
        
        # Task priority mapping (0-2)
        task_priority = 1  # MEDIUM default
        if self.current_ddos_model == AlgorithmType.TST:
            task_priority = 0  # CRITICAL (resource intensive)
        elif self.current_ddos_model == AlgorithmType.XGBOOST:
            task_priority = 2  # LOW (lightweight)
            
        # Thermal state mapping (0-2)
        temp = self.state.temperature
        if temp < 60: thermal_state = 0      # NORMAL
        elif temp < 75: thermal_state = 1    # ELEVATED  
        else: thermal_state = 2              # CRITICAL
        
        return [threat_idx, bat_band, cpu_band, task_priority, thermal_state]

    def _manage_tactical_rl_policy(self):
        """Use tactical RL agent with thermal awareness for DDoS model selection and CPU management."""
        if not self.tactical_agent_loaded:
            return
            
        now = time.time()
        if (now - self._last_tactical_decision) < self.tactical_action_interval_sec:
            return
        self._last_tactical_decision = now
        
        state_vec = self._compose_tactical_state()
        if state_vec is None:
            return
            
        try:
            action = self.tactical_agent.choose_action(state_vec, training=False)
        except Exception as e:
            logger.error(f"Tactical RL action selection failed: {e}")
            return
            
        # Decode action: 0..3 XGBOOST freq, 4..7 TST freq, 8 de-escalate
        if action == 8:
            # De-escalate => stop active DDoS task
            if self.ddos_task_id:
                logger.info("TACTICAL RL: De-escalating threat response (stopping DDoS detection)")
                self._stop_task(self.ddos_task_id)
                # Reset threat level if no external threats
                if self.state.threat_level > ThreatLevel.NONE:
                    self.state.threat_level = ThreatLevel.NONE
            return
            
        # Decode model and frequency
        model_idx = 0 if action < 4 else 1
        freq_idx = action % 4
        target_model = AlgorithmType.XGBOOST if model_idx == 0 else AlgorithmType.TST
        
        # Apply thermal safety constraints
        if self.state.temperature > 75 and target_model == AlgorithmType.TST:
            logger.warning("TACTICAL RL: Thermal protection - avoiding TST due to high temperature")
            target_model = AlgorithmType.XGBOOST
            
        # Switch DDoS model if different
        if self.current_ddos_model != target_model:
            if self.ddos_task_id:
                self._stop_task(self.ddos_task_id)
            self.submit_task(create_ddos_task(target_model))
            logger.info(f"TACTICAL RL: Switched to {target_model.value} (freq preset {freq_idx})")
            
        # Apply CPU frequency scaling based on freq_idx
        self._apply_cpu_frequency_scaling(freq_idx)
        
        # Publish enhanced tactical state for swarm coordination
        self._publish_tactical_state_update()

    def _apply_cpu_frequency_scaling(self, freq_idx: int):
        """Apply CPU frequency scaling based on tactical RL decision."""
        try:
            # Map frequency index to actual CPU frequencies (in MHz)
            freq_map = {
                0: 600,   # Low power mode
                1: 1000,  # Balanced mode
                2: 1500,  # Performance mode
                3: 1800   # Maximum performance
            }
            
            target_freq = freq_map.get(freq_idx, 1000)
            
            # Apply CPU frequency scaling using cpufreq-set if available
            cmd = f"sudo cpufreq-set -f {target_freq}MHz"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"TACTICAL RL: Set CPU frequency to {target_freq}MHz")
            else:
                logger.warning(f"Failed to set CPU frequency: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Error applying CPU frequency scaling: {e}")

    def _publish_tactical_state_update(self):
        """Publish enhanced tactical state including thermal info for swarm coordination."""
        if not self.mqtt_client or not self.mqtt_client.connected:
            return
            
        try:
            # Enhanced tactical state for swarm awareness
            tactical_data = {
                "drone_id": self.drone_id,
                "timestamp": time.time(),
                "tactical_state": {
                    "threat_level": self.state.threat_level.name,
                    "battery_percent": self.state.battery_percent,
                    "cpu_usage": self.state.cpu_usage,
                    "temperature": self.state.temperature,
                    "thermal_state": "CRITICAL" if self.state.temperature > 75 else "ELEVATED" if self.state.temperature > 60 else "NORMAL",
                    "active_ddos_model": self.current_ddos_model.value if self.current_ddos_model else None,
                    "power_draw_watts": self.state.power_draw_watts,
                    "rl_agent_active": self.tactical_agent_loaded
                }
            }
            
            # Publish to tactical coordination topic
            topic = f"swarm/tactical/{self.drone_id}"
            self.mqtt_client.publish(topic, tactical_data, qos=1)
            
        except Exception as e:
            logger.error(f"Error publishing tactical state update: {e}")

    def start(self):
        self.is_running = True
        
        # Initialize and connect MQTT client
        if self.mqtt_client.initialize():
            self.mqtt_client.connect()
        else:
            logger.error("Failed to initialize MQTT client - continuing without connectivity")
            
        # Start message processor thread
        self.message_processor_thread = threading.Thread(target=self._process_messages, daemon=True)
        self.message_processor_thread.start()
        
        # Start monitor thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Scheduler v14.0 (Enhanced MQTT Security & Reliability) Started.")

    def stop(self):
        logger.info("Shutdown sequence initiated.")
        self.is_running = False
        
        # Disconnect MQTT client
        if self.mqtt_client:
            self.mqtt_client.disconnect()
            
        # Stop all tasks
        for task_id in list(self.running_tasks.keys()):
            self._stop_task(task_id)
            
        # Wait for threads to exit
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
            
        if self.message_processor_thread and self.message_processor_thread.is_alive():
            self.message_processor_thread.join(timeout=2)
            
        logger.info("Scheduler stopped.")
        
    def _signal_handler(self, sig, frame):
        if not self.is_running: sys.exit(0)
        print("\nCtrl+C detected. Shutting down gracefully...")
        self.stop()

# --- TASK CREATION FUNCTIONS ---
def create_ddos_task(model: AlgorithmType) -> Task:
    venv_path = DDOS_ENV_PATH
    base_path = "/home/dev/src"
    script, pwr = ("ddos/testing/tst2.py", 4.5) if model == AlgorithmType.TST else ("ddos/xgboost/ddos_pipeline.py", 2.0)
    cmd = [f"{venv_path}/bin/python", f"{base_path}/{script}"]
    return Task(f"ddos-{model.value}-{int(time.time())}", f"DDOS {model.value}", cmd, TaskPriority.MEDIUM, model, ResourceProfile(pwr))

def create_crypto_task(algorithm: AlgorithmType) -> Task:
    venv_path = CRYPTO_ENV_PATH
    base_path = "/home/dev/src"
    
    script_map = {
        AlgorithmType.KYBER_CRYPTO: ("crypto/utils/drone_kyber_proxy.py", 2.5),
        AlgorithmType.CAMELIA: ("crypto/utils/custom_camellia/drone_camellia.py", 2.2),
        AlgorithmType.SPECK: ("crypto/pre-quantum/speck/drone_speck_proxy_final.py", 2.8),
        AlgorithmType.HIGHT: ("custom_hight/HIGHT-Python/drone_hight_final.py", 2.8),
        AlgorithmType.ASCON_128: ("crypto/utils/drone_ascon_proxy_final.py", 1.5),
        AlgorithmType.DILITHIUM2: ("crypto/utils/drone_dilithium_proxy.py", 2.6),
        AlgorithmType.FALCON512: ("crypto/utils/drone_falcon_proxy.py", 2.7)
    }
    script, pwr = script_map.get(algorithm, script_map[AlgorithmType.ASCON_128])
    
    cmd = [f"{venv_path}/bin/python", f"{base_path}/{script}"]
    return Task(f"crypto-{algorithm.value}-{int(time.time())}", f"Crypto {algorithm.value}", cmd, TaskPriority.HIGH, algorithm, ResourceProfile(pwr), auto_restart=True)

def create_mavlink_task() -> Task:
    venv_path = MAVLINK_ENV_PATH
    cmd = [f"{venv_path}/bin/mavproxy.py", "--master=/dev/ttyACM0", "--baudrate=921600", "--out=udp:127.0.0.1:5010"]
    return Task(
        f"mavlink-{int(time.time())}",
        "MAVLink Comms",
        cmd,
        TaskPriority.CRITICAL,
        AlgorithmType.MAVLINK,
        ResourceProfile(1.2),
        auto_restart=True,
        capture_output=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UAV Scheduler v14.0 - Enhanced MQTT Security & Reliability")
    parser.add_argument("--battery", type=float, default=100.0, help="Set initial battery percentage.")
    parser.add_argument("--drone_id", type=str, default="uav-drone-001", help="Set the unique ID for this drone.")
    args = parser.parse_args()

    if os.geteuid() != 0:
        logger.warning("Run with sudo to enable task priority setting.")

    # Clean up old flag files
    if os.path.exists(THREAT_FLAG_FILE): os.remove(THREAT_FLAG_FILE)
    if os.path.exists(CRYPTO_FLAG_FILE): os.remove(CRYPTO_FLAG_FILE)

    scheduler = UAVScheduler(config={"initial_battery": args.battery, "drone_id": args.drone_id})
    scheduler.start()

    # Submit baseline tasks (MAVLink, Crypto, XGBOOST monitor)
    scheduler.submit_task(create_mavlink_task())
    scheduler.submit_task(create_crypto_task(AlgorithmType.ASCON_128))
    scheduler.submit_task(create_ddos_task(AlgorithmType.XGBOOST))
    
    try:
        logger.info(f"Scheduler is running for drone: {args.drone_id}")
        logger.info(f"To simulate a local threat, run: touch {THREAT_FLAG_FILE}")
        logger.info(f"To manually rotate crypto, run: echo <1-4> > {CRYPTO_FLAG_FILE}")
        
        while scheduler.is_running:
            time.sleep(15)
            with scheduler.lock:
                s = scheduler.state
                ddos_model = scheduler.current_ddos_model.value if scheduler.current_ddos_model else "None"
                crypto_model = scheduler.current_crypto_model.value if scheduler.current_crypto_model else "None"
                print(f"\n--- [STATUS] Bat:{s.battery_percent:.1f}%|Temp:{s.temperature:.1f}°C|CPU:{s.cpu_usage:.1f}%|Threat:{s.threat_level.name}|DDoS:{ddos_model}|Crypto:{crypto_model} ---")
    except KeyboardInterrupt:
        pass
    finally:
        scheduler.stop()