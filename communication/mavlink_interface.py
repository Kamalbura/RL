"""
MAVLink Communication Interface for UAV-GCS Coordination
Handles message passing between tactical UAV agent and strategic GCS agent
"""

import time
import json
import threading
import queue
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging

class MessageType(Enum):
    """Message types for UAV-GCS communication"""
    TACTICAL_STATUS = "tactical_status"
    STRATEGIC_COMMAND = "strategic_command"
    THREAT_ALERT = "threat_alert"
    BATTERY_STATUS = "battery_status"
    CRYPTO_UPDATE = "crypto_update"
    SYSTEM_HEALTH = "system_health"
    COORDINATION_REQUEST = "coordination_request"
    HEARTBEAT = "heartbeat"

@dataclass
class TacticalStatus:
    """Status message from tactical UAV agent"""
    timestamp: float
    uav_id: str
    threat_level: int
    battery_percentage: float
    cpu_load: float
    current_model: str
    current_frequency: int
    detection_count: int
    power_consumption: float
    temperature: float

@dataclass
class StrategicCommand:
    """Command message from strategic GCS agent"""
    timestamp: float
    target_uav: str
    crypto_algorithm: str
    mission_phase: str
    coordination_mode: str
    priority_level: int

@dataclass
class ThreatAlert:
    """Threat detection alert"""
    timestamp: float
    source_uav: str
    threat_type: str
    severity: int
    location: Optional[Dict[str, float]]
    confidence: float

@dataclass
class SystemHealth:
    """System health status"""
    timestamp: float
    uav_id: str
    overall_status: str
    alerts: list
    metrics: Dict[str, float]

class MAVLinkInterface:
    """MAVLink communication interface for UAV-GCS coordination"""
    
    def __init__(self, node_id: str, node_type: str = "UAV"):
        self.node_id = node_id
        self.node_type = node_type  # "UAV" or "GCS"
        self.logger = logging.getLogger(f"{__name__}.{node_id}")
        
        # Message queues
        self.incoming_queue = queue.Queue(maxsize=100)
        self.outgoing_queue = queue.Queue(maxsize=100)
        
        # Message handlers
        self.message_handlers: Dict[MessageType, Callable] = {}
        
        # Connection status
        self.connected = False
        self.last_heartbeat = 0
        self.heartbeat_interval = 5.0  # seconds
        
        # Threading
        self.running = False
        self.comm_thread = None
        self.heartbeat_thread = None
        
    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register a message handler for specific message type"""
        self.message_handlers[message_type] = handler
        self.logger.info(f"Registered handler for {message_type.value}")
        
    def start(self):
        """Start communication threads"""
        if self.running:
            return
            
        self.running = True
        self.comm_thread = threading.Thread(target=self._communication_loop, daemon=True)
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        
        self.comm_thread.start()
        self.heartbeat_thread.start()
        
        self.logger.info(f"MAVLink interface started for {self.node_id}")
        
    def stop(self):
        """Stop communication threads"""
        self.running = False
        if self.comm_thread:
            self.comm_thread.join(timeout=2.0)
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=2.0)
            
        self.logger.info(f"MAVLink interface stopped for {self.node_id}")
        
    def send_message(self, message_type: MessageType, data: Any) -> bool:
        """Send a message to the communication queue"""
        try:
            message = {
                'type': message_type.value,
                'timestamp': time.time(),
                'source': self.node_id,
                'data': asdict(data) if hasattr(data, '__dict__') else data
            }
            
            self.outgoing_queue.put(message, timeout=1.0)
            return True
            
        except queue.Full:
            self.logger.warning("Outgoing queue full, message dropped")
            return False
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            return False
            
    def send_tactical_status(self, status: TacticalStatus) -> bool:
        """Send tactical status update"""
        return self.send_message(MessageType.TACTICAL_STATUS, status)
        
    def send_strategic_command(self, command: StrategicCommand) -> bool:
        """Send strategic command"""
        return self.send_message(MessageType.STRATEGIC_COMMAND, command)
        
    def send_threat_alert(self, alert: ThreatAlert) -> bool:
        """Send threat detection alert"""
        return self.send_message(MessageType.THREAT_ALERT, alert)
        
    def send_system_health(self, health: SystemHealth) -> bool:
        """Send system health status"""
        return self.send_message(MessageType.SYSTEM_HEALTH, health)
        
    def _communication_loop(self):
        """Main communication loop"""
        while self.running:
            try:
                # Process incoming messages
                self._process_incoming_messages()
                
                # Process outgoing messages
                self._process_outgoing_messages()
                
                time.sleep(0.1)  # 10Hz communication loop
                
            except Exception as e:
                self.logger.error(f"Communication loop error: {e}")
                time.sleep(1.0)
                
    def _process_incoming_messages(self):
        """Process incoming messages from queue"""
        try:
            while not self.incoming_queue.empty():
                message = self.incoming_queue.get_nowait()
                self._handle_message(message)
                
        except queue.Empty:
            pass
        except Exception as e:
            self.logger.error(f"Error processing incoming messages: {e}")
            
    def _process_outgoing_messages(self):
        """Process outgoing messages from queue"""
        try:
            while not self.outgoing_queue.empty():
                message = self.outgoing_queue.get_nowait()
                self._transmit_message(message)
                
        except queue.Empty:
            pass
        except Exception as e:
            self.logger.error(f"Error processing outgoing messages: {e}")
            
    def _handle_message(self, message: Dict):
        """Handle received message"""
        try:
            message_type = MessageType(message['type'])
            
            if message_type in self.message_handlers:
                handler = self.message_handlers[message_type]
                handler(message)
            else:
                self.logger.warning(f"No handler for message type: {message_type.value}")
                
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            
    def _transmit_message(self, message: Dict):
        """Transmit message (placeholder for actual MAVLink implementation)"""
        # In a real implementation, this would use pymavlink to send messages
        # For now, we'll simulate transmission with logging
        self.logger.debug(f"Transmitting: {message['type']} to network")
        
        # Simulate network transmission
        # In practice, this would be:
        # self.mavlink_connection.send(message)
        
    def _heartbeat_loop(self):
        """Send periodic heartbeat messages"""
        while self.running:
            try:
                heartbeat_data = {
                    'node_id': self.node_id,
                    'node_type': self.node_type,
                    'timestamp': time.time(),
                    'status': 'active'
                }
                
                self.send_message(MessageType.HEARTBEAT, heartbeat_data)
                self.last_heartbeat = time.time()
                
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                time.sleep(self.heartbeat_interval)
                
    def is_connected(self) -> bool:
        """Check if connection is active"""
        return time.time() - self.last_heartbeat < self.heartbeat_interval * 3
        
    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status"""
        return {
            'connected': self.is_connected(),
            'last_heartbeat': self.last_heartbeat,
            'incoming_queue_size': self.incoming_queue.qsize(),
            'outgoing_queue_size': self.outgoing_queue.qsize(),
            'node_id': self.node_id,
            'node_type': self.node_type
        }


class UAVCoordinator:
    """Coordinates multiple UAV agents through MAVLink"""
    
    def __init__(self, gcs_id: str = "GCS_MAIN"):
        self.gcs_id = gcs_id
        self.mavlink = MAVLinkInterface(gcs_id, "GCS")
        self.logger = logging.getLogger(__name__)
        
        # UAV tracking
        self.active_uavs: Dict[str, Dict] = {}
        self.fleet_status = {
            'total_uavs': 0,
            'active_uavs': 0,
            'average_battery': 0.0,
            'threat_level': 0,
            'mission_phase': 'IDLE'
        }
        
        # Register message handlers
        self.mavlink.register_handler(MessageType.TACTICAL_STATUS, self._handle_tactical_status)
        self.mavlink.register_handler(MessageType.THREAT_ALERT, self._handle_threat_alert)
        self.mavlink.register_handler(MessageType.SYSTEM_HEALTH, self._handle_system_health)
        
    def start(self):
        """Start coordination system"""
        self.mavlink.start()
        self.logger.info("UAV Coordinator started")
        
    def stop(self):
        """Stop coordination system"""
        self.mavlink.stop()
        self.logger.info("UAV Coordinator stopped")
        
    def send_strategic_command(self, uav_id: str, crypto_algorithm: str, 
                             mission_phase: str = "NORMAL", priority: int = 1) -> bool:
        """Send strategic command to specific UAV"""
        command = StrategicCommand(
            timestamp=time.time(),
            target_uav=uav_id,
            crypto_algorithm=crypto_algorithm,
            mission_phase=mission_phase,
            coordination_mode="FLEET",
            priority_level=priority
        )
        
        return self.mavlink.send_strategic_command(command)
        
    def broadcast_crypto_update(self, crypto_algorithm: str) -> int:
        """Broadcast crypto algorithm update to all UAVs"""
        success_count = 0
        
        for uav_id in self.active_uavs.keys():
            if self.send_strategic_command(uav_id, crypto_algorithm):
                success_count += 1
                
        self.logger.info(f"Crypto update sent to {success_count}/{len(self.active_uavs)} UAVs")
        return success_count
        
    def get_fleet_status(self) -> Dict[str, Any]:
        """Get current fleet status"""
        # Update fleet statistics
        if self.active_uavs:
            batteries = [uav['battery_percentage'] for uav in self.active_uavs.values() 
                        if 'battery_percentage' in uav]
            self.fleet_status['average_battery'] = sum(batteries) / len(batteries) if batteries else 0
            
            threat_levels = [uav['threat_level'] for uav in self.active_uavs.values() 
                           if 'threat_level' in uav]
            self.fleet_status['threat_level'] = max(threat_levels) if threat_levels else 0
            
        self.fleet_status['total_uavs'] = len(self.active_uavs)
        self.fleet_status['active_uavs'] = len([uav for uav in self.active_uavs.values() 
                                              if time.time() - uav.get('last_update', 0) < 30])
        
        return self.fleet_status.copy()
        
    def _handle_tactical_status(self, message: Dict):
        """Handle tactical status update from UAV"""
        data = message['data']
        uav_id = data['uav_id']
        
        # Update UAV status
        self.active_uavs[uav_id] = {
            **data,
            'last_update': time.time(),
            'source': message['source']
        }
        
        self.logger.debug(f"Updated status for UAV {uav_id}")
        
    def _handle_threat_alert(self, message: Dict):
        """Handle threat alert from UAV"""
        data = message['data']
        self.logger.warning(f"Threat alert from {data['source_uav']}: {data['threat_type']}")
        
        # Could trigger fleet-wide response
        if data['severity'] >= 3:
            self.broadcast_crypto_update("SPHINCS")  # High security mode
            
    def _handle_system_health(self, message: Dict):
        """Handle system health update from UAV"""
        data = message['data']
        uav_id = data['uav_id']
        
        if data['overall_status'] == 'CRITICAL':
            self.logger.error(f"Critical health status from UAV {uav_id}")
            # Could trigger emergency protocols


# Global coordinator instance
_coordinator = None

def get_coordinator() -> UAVCoordinator:
    """Get singleton coordinator instance"""
    global _coordinator
    if _coordinator is None:
        _coordinator = UAVCoordinator()
    return _coordinator
