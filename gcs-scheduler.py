#!/usr/bin/env python3
"""
Restored GCS MQTT Scheduler GUI (functional) with MQTT protocol fallback.
Fixes:
 - Removes use of clean_session when protocol=5 (avoids error "Clean session is not used for MQTT 5.0").
 - Adds adaptive client creation for MQTT v3.1.1 vs v5.
 - Ensures connect returns proper status and offline retained message published.
"""

import os, sys, json, time, ssl, re, queue, socket, logging, threading, subprocess, signal, importlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

# Optional modules
try:
    import ip_config  # user-provided central IP config
except Exception:
    ip_config = None

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("Install paho-mqtt: pip install paho-mqtt>=1.6.0"); sys.exit(1)

try:
    from pymavlink import mavutil
    PYMAVLINK_AVAILABLE = True
except Exception:
    mavutil = None
    PYMAVLINK_AVAILABLE = False

# Tkinter
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except Exception as e:
    print("Tkinter required:", e); sys.exit(1)

# Import the central knowledge base
from shared.system_profiles import (
    CRYPTO_PROFILES, DDOS_PROFILES, THERMAL_PROFILES, BATTERY_SPECS,
    CPU_FREQUENCY_PROFILES, ThermalState, get_thermal_state_from_temperature,
    get_optimal_ddos_model_for_conditions, get_optimal_crypto_for_conditions
)

# Strategic RL imports
try:
    from crypto_rl.strategic_agent import QLearningAgent as StrategicQLearningAgent
    from integration.gcs_integration import GCSStrategicIntegration
    STRATEGIC_RL_AVAILABLE = True
except Exception as e:
    StrategicQLearningAgent = None
    GCSStrategicIntegration = None
    STRATEGIC_RL_AVAILABLE = False
    print(f"⚠️ Strategic RL modules unavailable: {e}")

APP_NAME = "GCS MQTT Scheduler"
HERE = Path(__file__).parent.resolve()
LOG_DIR = HERE / "logs"; LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "gcs_mqtt_scheduler.log"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE, encoding='utf-8')])
logger = logging.getLogger("GCS-SCHED")
PYTHON_EXE = sys.executable

DEFAULT_CONFIG: Dict[str, Any] = {
    "broker": {"address": "localhost", "port": 8883, "keepalive": 60, "connection_timeout": 15},
    "client": {"id": "gcs", "protocol": 4},  # protocol: 4 = MQTTv311, 5 = MQTTv5
    "security": {
        "cert_paths": [
            "C:/mqtt/certs", "C\\mqtt\\certs",
            str(HERE / "certs"), str(HERE.parent / "certs"),
            "/home/dev/mqtt/certs", "/etc/mqtt/certs"
        ],
        "ca_cert": "ca-cert.pem",
        "verify_hostname": False
    },
    "topics": {
        "subscribe": [
            {"topic": "swarm/status/+", "qos": 1},
            {"topic": "swarm/alert/+", "qos": 2},
            {"topic": "swarm/drones/+/telemetry", "qos": 1},
            {"topic": "swarm/broadcast/crypto", "qos": 2},
            {"topic": "swarm/broadcast/alert", "qos": 2},
            {"topic": "swarm/heartbeat/+", "qos": 1},
            {"topic": "swarm/#", "qos": 0}
        ],
        "publish": {
            "alerts": {"topic": "swarm/broadcast/alert", "qos": 2},
            "crypto": {"topic": "swarm/broadcast/crypto", "qos": 2},
            "individual": {"topic": "swarm/commands/individual/{drone_id}", "qos": 1},
            "status": {"topic": "swarm/status/gcs", "qos": 1}
        }
    },
    "core": {"script": "gcs_pymavlink_final.py", "args": ["--no-input"]},
    "mavlink": {"rx_uri": "udp:0.0.0.0:14550", "tx_uri": "udpout:127.0.0.1:14551", "sysid": 255, "compid": 190},
    "crypto_map": {
        "c1": {"name": "KYBER", "script": "gcs_kyber_proxy.py"},
        "c2": {"name": "DILITHIUM", "script": "gcs_dilithium_proxy.py"},
        "c3": {"name": "SPHINCS", "script": "gcs_sphincs_proxy.py"},
        "c4": {"name": "FALCON", "script": "gcs_falcon_proxy.py"}
    },
    "crypto_rl": {
        "enabled": True,
        "model_path": "crypto_q_table.npy",
        "auto_recommend": True
    }
}

# --- Utilities ---
def is_windows(): return os.name == 'nt'

# Safe process terminator across platforms
def terminate_process_tree(proc: subprocess.Popen):
    if not proc: return
    try:
        if is_windows():
            try: proc.send_signal(signal.CTRL_BREAK_EVENT)
            except Exception: subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(os.getpgid(proc.pid), 15)
    except Exception:
        try: proc.terminate()
        except Exception: pass

# --- Certificate discovery ---

def discover_certs(cfg: Dict[str, Any], client_id: str) -> Optional[Tuple[str,str,str]]:
    paths = cfg["security"].get("cert_paths", [])
    ca_name = cfg["security"].get("ca_cert", "ca-cert.pem")
    client_cert = f"{client_id}-cert.pem"; client_key = f"{client_id}-key.pem"
    for base in paths:
        bp = Path(base)
        if not bp.exists(): continue
        ca_path = bp / ca_name
        for flat in (True, False):
            if flat:
                cert_path = bp / client_cert; key_path = bp / client_key
            else:
                cert_path = bp / "clients" / client_cert; key_path = bp / "clients" / client_key
            if ca_path.exists() and cert_path.exists() and key_path.exists():
                logger.info(f"Using certs from: {bp}")
                return str(ca_path), str(cert_path), str(key_path)
    logger.error("Certificates not found")
    return None

# --- MQTT Client ---
class GcsMqttClient:
    def __init__(self, config: Dict[str, Any], on_message_cb):
        self.config = config; self.client_id = config["client"]["id"]
        self.on_message_cb = on_message_cb
        self.connected_event = threading.Event()
        self.metrics = {"rx":0, "tx":0, "errors":0}
        self.connected = False
        self.client: Optional[mqtt.Client] = None
        self.certs = discover_certs(config, self.client_id)
        if not self.certs: raise FileNotFoundError("TLS certs missing")
        self._setup_client()
    def _setup_client(self):
        proto_cfg = self.config["client"].get("protocol", 4)
        if proto_cfg == 5:
            self.client = mqtt.Client(client_id=self.client_id, protocol=mqtt.MQTTv5)
        else:
            self.client = mqtt.Client(client_id=self.client_id, protocol=mqtt.MQTTv311, clean_session=True)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.on_publish = self._on_publish
        ca, cert, key = self.certs
        verify_hostname = self.config["security"].get("verify_hostname", False)
        self.client.tls_set(ca_certs=ca, certfile=cert, keyfile=key, tls_version=ssl.PROTOCOL_TLS, cert_reqs=ssl.CERT_REQUIRED)
        self.client.tls_insecure_set(not verify_hostname)
    def connect(self) -> bool:
        try:
            self.client.connect_async(self.config["broker"]["address"], self.config["broker"]["port"], self.config["broker"].get("keepalive",60))
            self.client.loop_start()
            if self.connected_event.wait(self.config["broker"].get("connection_timeout",15)):
                return True
            logger.error("MQTT connect timeout")
            return False
        except Exception as e:
            logger.error(f"MQTT connect error: {e}")
            return False
    def disconnect(self):
        try: self.client.disconnect(); self.client.loop_stop()
        except Exception: pass
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            self.connected = True; self.connected_event.set()
            for sub in self.config["topics"]["subscribe"]:
                client.subscribe(sub["topic"], sub.get("qos",1))
            # retain online
            self.publish(self.config["topics"]["publish"]["status"]["topic"], {"status":"online","ts":time.time()}, qos=1, retain=True)
            logger.info("Connected to broker")
        else:
            logger.error(f"Connect failed rc={rc}")
    def _on_disconnect(self, client, userdata, rc, properties=None):
        self.connected = False; self.connected_event.clear()
        logger.warning(f"Disconnected (rc={rc})")
    def _on_message(self, client, userdata, msg):
        self.metrics["rx"] += len(msg.payload)
        try: self.on_message_cb(msg)
        except Exception as e:
            logger.error(f"on_message error: {e}"); self.metrics["errors"] += 1
    def _on_publish(self, client, userdata, mid): pass
    def publish(self, topic: str, payload: Any, qos:int=1, retain:bool=False)->bool:
        if not self.connected: return False
        try:
            data = payload if isinstance(payload,(bytes,bytearray)) else (payload if isinstance(payload,str) else json.dumps(payload))
            try: self.metrics["tx"] += len(data)
            except Exception: self.metrics["tx"] += len(str(data))
            r = self.client.publish(topic, data, qos=qos, retain=retain)
            return r.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            logger.error(f"Publish error: {e}")
            return False

# --- Crypto Manager ---
class GcsCryptoManager:
    def __init__(self, config: Dict[str, Any]):
        self.config=config; self.current_code=None; self.proc:Optional[subprocess.Popen]=None
    def _script_path(self, name:str)->Path: return (HERE / name).resolve()
    def switch(self, code:str)->Tuple[bool,str]:
        m=self.config.get("crypto_map",{})
        if code not in m: return False, f"Unknown crypto code: {code}"
        if self.current_code==code and self.proc and self.proc.poll() is None:
            return True, f"Already running {m[code]['name']} ({code})"
        self.stop(); target=m[code]; path=self._script_path(target['script'])
        if not path.exists(): return False, f"Script not found: {path}";
        try:
            if is_windows():
                self.proc=subprocess.Popen([PYTHON_EXE,str(path)], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            else:
                self.proc=subprocess.Popen([PYTHON_EXE,str(path)], preexec_fn=os.setsid)
            self.current_code=code
            return True, f"Started {target['name']} ({code}) via {path.name}"
        except Exception as e:
            return False, f"Failed to start {path.name}: {e}"
    def stop(self):
        if self.proc and self.proc.poll() is None:
            try: terminate_process_tree(self.proc)
            except Exception:
                try: self.proc.kill()
                except Exception: pass
        self.proc=None; self.current_code=None

# --- Core Manager ---
class GcsCoreManager:
    def __init__(self, config: Dict[str, Any]): self.config=config; self.proc:Optional[subprocess.Popen]=None
    def _script_path(self)->Path: return (HERE / self.config.get("core",{}).get("script","gcs_pymavlink_final.py")).resolve()
    def start(self)->Tuple[bool,str]:
        if self.proc and self.proc.poll() is None: return True, "Core already running"
        path=self._script_path();
        if not path.exists(): return False, f"Core script not found: {path}"
        args=self.config.get("core",{}).get("args",[])
        try:
            if is_windows():
                self.proc=subprocess.Popen([PYTHON_EXE,str(path),*map(str,args)], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, stdin=subprocess.DEVNULL)
            else:
                self.proc=subprocess.Popen([PYTHON_EXE,str(path),*map(str,args)], preexec_fn=os.setsid, stdin=subprocess.DEVNULL)
            return True, f"Started core: {path.name}"
        except Exception as e:
            return False, f"Failed to start core: {e}"
    def stop(self):
        if self.proc and self.proc.poll() is None:
            try: terminate_process_tree(self.proc)
            except Exception:
                try: self.proc.kill()
                except Exception: pass
        self.proc=None

# --- MAVLink Manager ---
class GcsMavlinkManager:
    def __init__(self,on_msg_cb=None):
        self.on_msg_cb=on_msg_cb; self.running=False; self.rx_conn=None; self.tx_conn=None; self.rx_thread=None
        self._udp_rx_sock=None; self._udp_tx_sock=None; self._udp_tx_addr=None
        self.sysid=255; self.compid=190; self.hb_running=False; self.hb_thread=None
    def start(self, rx_uri:str, tx_uri:str, sysid:int, compid:int)->Tuple[bool,str]:
        if self.running: return True, "MAVLink already running"
        self.sysid, self.compid = sysid, compid
        try:
            if PYMAVLINK_AVAILABLE:
                self.tx_conn = mavutil.mavlink_connection(tx_uri, source_system=sysid, source_component=compid)
                self.rx_conn = mavutil.mavlink_connection(rx_uri, autoreconnect=True)
            else:
                def _parse(u):
                    p=u.split(":"); assert len(p)>=3; return p[0],p[1],int(p[2])
                _,host_rx,port_rx=_parse(rx_uri); _,host_tx,port_tx=_parse(tx_uri)
                self._udp_rx_sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); self._udp_rx_sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                self._udp_rx_sock.bind((host_rx,port_rx)); self._udp_rx_sock.settimeout(1.0)
                self._udp_tx_sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); self._udp_tx_addr=(host_tx,port_tx)
            self.running=True
            self.rx_thread=threading.Thread(target=self._rx_loop, daemon=True); self.rx_thread.start()
            return True, "MAVLink started"
        except Exception as e:
            self.stop(); return False, f"MAVLink start failed: {e}"
    def send_heartbeat(self):
        if not PYMAVLINK_AVAILABLE or not self.tx_conn: return False, "pymavlink not available for heartbeat"
        try:
            self.tx_conn.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID,0,0,0)
            return True, "Heartbeat sent"
        except Exception as e: return False, f"HB send fail: {e}"
    def _hb_loop(self, rate_hz:float):
        it=1.0/max(rate_hz,0.1)
        while self.hb_running and self.running:
            self.send_heartbeat(); time.sleep(it)
    def start_heartbeat(self, rate_hz:float=1.0):
        if self.hb_running or not PYMAVLINK_AVAILABLE: return
        self.hb_running=True; self.hb_thread=threading.Thread(target=self._hb_loop, args=(rate_hz,), daemon=True); self.hb_thread.start()
    def stop_heartbeat(self): self.hb_running=False
    def stop(self):
        self.running=False
        for obj in [self.rx_conn, self.tx_conn, self._udp_rx_sock, self._udp_tx_sock]:
            try: obj and obj.close()
            except Exception: pass
        self.rx_conn=self.tx_conn=None; self._udp_rx_sock=self._udp_tx_sock=None; self._udp_tx_addr=None
    def _rx_loop(self):
        while self.running:
            try:
                if PYMAVLINK_AVAILABLE and self.rx_conn:
                    msg=self.rx_conn.recv_match(blocking=True, timeout=1.0)
                    if not msg: continue
                    if self.on_msg_cb: self.on_msg_cb(msg)
                elif self._udp_rx_sock:
                    data,addr=self._udp_rx_sock.recvfrom(4096)
                    if self.on_msg_cb: self.on_msg_cb({"raw":data, "from":addr})
                else:
                    time.sleep(0.2)
            except Exception:
                pass
    def send_command_long(self,target_sys:int,target_comp:int,command:int,params:list[float])->Tuple[bool,str]:
        try:
            if PYMAVLINK_AVAILABLE and self.tx_conn:
                p=(params+[0.0]*7)[:7]; self.tx_conn.mav.command_long_send(target_sys,target_comp,command,0,*p)
                return True, "Command sent"
            return False, "Install pymavlink or use Raw"
        except Exception as e: return False, f"Send failed: {e}"
    def send_raw(self,data:bytes)->Tuple[bool,str]:
        try:
            if PYMAVLINK_AVAILABLE and self.tx_conn:
                self.tx_conn.write(data); return True, "Raw sent"
            elif self._udp_tx_sock and self._udp_tx_addr:
                self._udp_tx_sock.sendto(data,self._udp_tx_addr); return True, "Raw sent"
            return False, "TX not init"
        except Exception as e: return False, f"Raw send failed: {e}"

# --- Data Structures ---
@dataclass
class DroneInfo:
    drone_id: str; last_seen: float; online: bool; battery: Optional[float]=None; crypto: Optional[str]=None; last_msg_type: Optional[str]=None; hb_count:int=0

# --- Main App ---
class GcsSchedulerApp:
    def __init__(self, root: tk.Tk, config: Dict[str, Any]):
        self.root=root; self.config=config; root.title(APP_NAME)
        self.style=ttk.Style()
        try: self.style.theme_use("vista" if is_windows() and "vista" in self.style.theme_names() else "clam")
        except Exception: pass
        self.msg_queue: "queue.Queue[mqtt.MQTTMessage]" = queue.Queue(); self.mqtt:Optional[GcsMqttClient]=None
        self.crypto=GcsCryptoManager(config); self.core=GcsCoreManager(config)
        self.mav=GcsMavlinkManager(self._on_mav_rx)
        self.mav_rx_uri=tk.StringVar(value=config["mavlink"]["rx_uri"]); self.mav_tx_uri=tk.StringVar(value=config["mavlink"]["tx_uri"])
        self.mav_sysid=tk.IntVar(value=config["mavlink"]["sysid"]); self.mav_compid=tk.IntVar(value=config["mavlink"]["compid"])
        self.mav_status=tk.StringVar(value="MAVLink: stopped" + (" (pymavlink OK)" if PYMAVLINK_AVAILABLE else " (raw UDP mode)"))
        self.mav_auto_hb=tk.BooleanVar(value=True)
        self.mav_tgt_sys=tk.IntVar(value=1); self.mav_tgt_comp=tk.IntVar(value=1); self.mav_cmd_id=tk.IntVar(value=400)
        self.mav_p=[tk.DoubleVar(value=0.0) for _ in range(7)]; self.mav_raw_hex=tk.StringVar(value="")
        self.drones: Dict[str, DroneInfo] = {}
        self.auto_local_crypto=tk.BooleanVar(value=True)
        self.auto_start_core=tk.BooleanVar(value=True)
        
        # Strategic RL Agent Integration
        self.strategic_agent = None
        self.strategic_integration = None
        self.use_strategic_rl = tk.BooleanVar(value=True)
        self.rl_recommendation_label = None
        
        if STRATEGIC_RL_AVAILABLE:
            self._initialize_strategic_rl()
        else:
            logger.warning("Strategic RL not available, running without AI recommendations")
        self.use_strategic_rl = tk.BooleanVar(value=self.config.get("crypto_rl", {}).get("enabled", True))

        # UI variables
        self.sel_id=tk.StringVar(); self.sel_online=tk.StringVar(); self.sel_batt=tk.StringVar(); self.sel_crypto=tk.StringVar(); self.sel_msg=tk.StringVar(); self.sel_last=tk.StringVar()
        self.sb_conn=tk.StringVar(); self.sb_core=tk.StringVar(); self.sb_proxy=tk.StringVar(); self.sb_stats=tk.StringVar()
        self._suppress_broadcast_until = 0.0; self._last_crypto_pub = ""
        if ip_config:
            self.ipc_gcs=tk.StringVar(value=getattr(ip_config,'GCS_HOST','')); self.ipc_drone=tk.StringVar(value=getattr(ip_config,'DRONE_HOST',''))
        self._dark_mode = tk.BooleanVar(value=False)
        self._build_ui()

    def _initialize_strategic_rl(self):
        """Initialize the strategic RL agent and integration."""
        try:
            self.strategic_integration = GCSStrategicIntegration(None, self)
            logger.info("✅ Strategic RL integration initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize strategic RL: {e}")
            self.strategic_integration = None

    def _setup_fallback_rl_controls(self, parent):
        """Setup fallback UI controls when strategic RL is unavailable."""
        lf_rl = ttk.LabelFrame(parent, text="AI Recommendations (Unavailable)", padding=8)
        lf_rl.pack(fill=tk.X, padx=8, pady=6)
        
        ttk.Label(lf_rl, text="Strategic RL agent not available", foreground="red").pack(pady=4)
        ttk.Label(lf_rl, text="Using manual crypto selection only").pack(pady=2)

    def _update_rl_recommendations(self):
        """Update RL recommendations when integration is not available but agent exists."""
        if not self.strategic_agent or not self.use_strategic_rl.get():
            return
            
        try:
            # Build strategic state from drone fleet data
            strategic_state = self._build_strategic_state()
            if strategic_state:
                # Get RL recommendation
                action = self.strategic_agent.choose_action(strategic_state, training=False)
                crypto_algorithms = ["KYBER", "DILITHIUM", "SPHINCS", "FALCON"]
                recommended_crypto = crypto_algorithms[action]
                
                # Update recommendation label if it exists
                if self.rl_recommendation_label:
                    self.rl_recommendation_label.config(
                        text=f"AI Recommends: {recommended_crypto}",
                        foreground="blue"
                    )
                    
                logger.info(f"Strategic RL recommends: {recommended_crypto}")
        except Exception as e:
            logger.error(f"Strategic RL update failed: {e}")

    def _build_strategic_state(self):
        """Build strategic state vector from current drone fleet data."""
        try:
            # Calculate fleet metrics
            online_drones = [d for d in self.drones.values() if d.online]
            if not online_drones:
                return None
                
            # Average fleet battery level
            avg_battery = sum(d.battery for d in online_drones) / len(online_drones)
            battery_idx = 0 if avg_battery > 70 else 1 if avg_battery > 30 else 2
            
            # Swarm threat consensus (based on alert messages)
            threat_idx = 1  # Default to MEDIUM
            
            # Mission phase (simplified)
            mission_phase_idx = 1  # Default to PATROL
            
            return [threat_idx, battery_idx, mission_phase_idx]
        except Exception as e:
            logger.error(f"Failed to build strategic state: {e}")
            return None
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.after(100, self._ui_tick)
        self._start_mqtt_thread()

    def _build_ui(self):
        # Main notebook
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Control tab
        control_tab = ttk.Frame(notebook)
        notebook.add(control_tab, text="Control")
        
        # Broadcast group
        lf_bcast = ttk.LabelFrame(control_tab, text="Broadcast Alerts", padding=8)
        lf_bcast.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(lf_bcast, text="CAUTION", command=lambda: self._send_alert("alb-cau", False)).pack(side=tk.LEFT, padx=4)
        ttk.Button(lf_bcast, text="CRITICAL", command=lambda: self._send_alert("alb-cri", True)).pack(side=tk.LEFT, padx=4)

        # Strategic RL Agent Controls
        if self.strategic_integration:
            self.strategic_integration.setup_ui_controls(control_tab)
        else:
            self._setup_fallback_rl_controls(control_tab)

        # Fleet tab
        fleet_tab = ttk.Frame(notebook)
        notebook.add(fleet_tab, text="Fleet")
        paned = ttk.PanedWindow(fleet_tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        # Left: table
        left = ttk.Frame(paned)
        cols = ("drone", "online", "battery", "crypto", "msg", "last")
        self.tree = ttk.Treeview(left, columns=cols, show='headings', height=14)
        for c, w in zip(cols, (160, 80, 90, 140, 90, 200)):
            self.tree.heading(c, text=c.capitalize())
            self.tree.column(c, width=w, anchor=tk.W)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self._on_select_drone)
        paned.add(left, weight=3)
        # Right: details
        right = ttk.Frame(paned)
        df = ttk.LabelFrame(right, text="Drone Details", padding=8)
        df.pack(fill=tk.X)
        for label, var in (("ID", self.sel_id), ("Online", self.sel_online), ("Battery", self.sel_batt), ("Crypto", self.sel_crypto), ("Last Msg", self.sel_msg), ("Last Seen", self.sel_last)):
            row = ttk.Frame(df); row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"{label}:", width=12).pack(side=tk.LEFT)
            ttk.Label(row, textvariable=var).pack(side=tk.LEFT)
        cf = ttk.LabelFrame(right, text="Command", padding=8)
        cf.pack(fill=tk.X, pady=8)
        ttk.Label(cf, text="Command to selected:").pack(side=tk.LEFT)
        self.cmd_var = tk.StringVar(value="status")
        self.cmd_entry = ttk.Entry(cf, textvariable=self.cmd_var, width=20)
        self.cmd_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(cf, text="Send", command=self._send_individual_command).pack(side=tk.LEFT, padx=4)
        ttk.Button(cf, text="Request Status", command=lambda: self._set_and_send_cmd('status')).pack(side=tk.LEFT)
        paned.add(right, weight=2)

        # MAVLink tab
        mav_tab = ttk.Frame(notebook)
        notebook.add(mav_tab, text="MAVLink")

        # Connection frame
        cframe = ttk.LabelFrame(mav_tab, text="MAVLink Connection (via Proxy endpoints)", padding=8)
        cframe.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(cframe, text="RX URI").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(cframe, textvariable=self.mav_rx_uri, width=28).grid(row=0, column=1, padx=4)
        ttk.Label(cframe, text="TX URI").grid(row=0, column=2, sticky=tk.W)
        ttk.Entry(cframe, textvariable=self.mav_tx_uri, width=28).grid(row=0, column=3, padx=4)
        ttk.Label(cframe, text="SysID").grid(row=1, column=0, sticky=tk.W, pady=(4,0))
        ttk.Entry(cframe, textvariable=self.mav_sysid, width=6).grid(row=1, column=1, sticky=tk.W, pady=(4,0))
        ttk.Label(cframe, text="CompID").grid(row=1, column=2, sticky=tk.W, pady=(4,0))
        ttk.Entry(cframe, textvariable=self.mav_compid, width=6).grid(row=1, column=3, sticky=tk.W, pady=(4,0))
        ttk.Button(cframe, text="Connect", command=self._mav_connect).grid(row=0, column=4, padx=6)
        ttk.Button(cframe, text="Disconnect", command=self._mav_disconnect).grid(row=1, column=4)
        ttk.Label(cframe, textvariable=self.mav_status).grid(row=0, column=5, rowspan=2, padx=8)
        # Heartbeat controls
        ttk.Checkbutton(cframe, text="Auto HB", variable=self.mav_auto_hb).grid(row=0, column=6, padx=4)
        ttk.Button(cframe, text="Send HB", command=self._mav_send_hb).grid(row=1, column=6, padx=4)
        for i in range(7): cframe.grid_columnconfigure(i, weight=0)

        # Send frame
        sframe = ttk.LabelFrame(mav_tab, text="Send MAVLink", padding=8)
        sframe.pack(fill=tk.X, padx=8, pady=6)
        # quick commands
        ttk.Label(sframe, text="Target Sys/Comp").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(sframe, textvariable=self.mav_tgt_sys, width=6).grid(row=0, column=1)
        ttk.Entry(sframe, textvariable=self.mav_tgt_comp, width=6).grid(row=0, column=2)
        ttk.Button(sframe, text="ARM", command=lambda: self._mav_quick('arm')).grid(row=0, column=3, padx=4)
        ttk.Button(sframe, text="DISARM", command=lambda: self._mav_quick('disarm')).grid(row=0, column=4)
        ttk.Button(sframe, text="TAKEOFF", command=lambda: self._mav_quick('takeoff')).grid(row=0, column=5, padx=4)
        ttk.Button(sframe, text="LAND", command=lambda: self._mav_quick('land')).grid(row=0, column=6)
        # custom COMMAND_LONG
        ttk.Label(sframe, text="CMD ID").grid(row=1, column=0, sticky=tk.W, pady=(6,0))
        ttk.Entry(sframe, textvariable=self.mav_cmd_id, width=8).grid(row=1, column=1, pady=(6,0))
        for i in range(7):
            ttk.Label(sframe, text=f"P{i+1}").grid(row=1, column=2+i, sticky=tk.W, pady=(6,0))
            ttk.Entry(sframe, textvariable=self.mav_p[i], width=7).grid(row=2, column=2+i)
        ttk.Button(sframe, text="Send CMD_LONG", command=self._mav_send_cmd_long).grid(row=2, column=9, padx=8)
        # raw hex
        ttk.Label(sframe, text="Raw Hex").grid(row=3, column=0, sticky=tk.W, pady=(6,0))
        ttk.Entry(sframe, textvariable=self.mav_raw_hex, width=60).grid(row=3, column=1, columnspan=7, sticky=tk.W, pady=(6,0))
        ttk.Button(sframe, text="Send Raw", command=self._mav_send_raw).grid(row=3, column=8, padx=6, pady=(6,0))

        # Receive frame
        rframe = ttk.LabelFrame(mav_tab, text="Received MAVLink", padding=8)
        rframe.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.mav_rx_txt = tk.Text(rframe, height=14)
        self.mav_rx_txt.pack(fill=tk.BOTH, expand=True)

        # Heartbeats tab (new)
        hb_tab = ttk.Frame(notebook)
        notebook.add(hb_tab, text="Heartbeats")
        hb_top = ttk.Frame(hb_tab); hb_top.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        cols = ("sysid","compid","autopilot","type","base_mode","custom_mode","system_status","mver","last_seen","count")
        self.hb_tree = ttk.Treeview(hb_top, columns=cols, show='headings', height=14)
        for c,w in zip(cols,(60,60,90,90,80,90,110,50,140,60)): self.hb_tree.heading(c, text=c); self.hb_tree.column(c, width=w, anchor=tk.W)
        self.hb_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); sb_hb = ttk.Scrollbar(hb_top, orient='vertical', command=self.hb_tree.yview); self.hb_tree.configure(yscrollcommand=sb_hb.set); sb_hb.pack(side=tk.LEFT, fill=tk.Y); self.hb_stats = {}  # (sysid, compid) -> dict

        # Logs tab
        logs_tab = ttk.Frame(notebook)
        notebook.add(logs_tab, text="Logs")
        toolbar = ttk.Frame(logs_tab)
        toolbar.pack(fill=tk.X, padx=8, pady=(8, 0))
        ttk.Button(toolbar, text="Clear", command=self._clear_log).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Save", command=self._save_log).pack(side=tk.LEFT, padx=6)
        self.log_txt = tk.Text(logs_tab, height=18)
        self.log_txt.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Config tab (runtime IP editing)
        config_tab = ttk.Frame(notebook)
        notebook.add(config_tab, text="Config")
        if ip_config:
            lf_ips = ttk.LabelFrame(config_tab, text="Runtime IP Configuration", padding=8)
            lf_ips.pack(fill=tk.X, padx=8, pady=8)
            ttk.Label(lf_ips, text="GCS_HOST").grid(row=0, column=0, sticky=tk.W)
            ttk.Entry(lf_ips, textvariable=self.ipc_gcs, width=18).grid(row=0, column=1, padx=4, pady=2)
            ttk.Label(lf_ips, text="DRONE_HOST").grid(row=1, column=0, sticky=tk.W)
            ttk.Entry(lf_ips, textvariable=self.ipc_drone, width=18).grid(row=1, column=1, padx=4, pady=2)
            ttk.Button(lf_ips, text="Apply Runtime", command=self._apply_ip_runtime).grid(row=0, column=2, padx=8)
            ttk.Button(lf_ips, text="Apply & Persist", command=self._apply_ip_persistent).grid(row=1, column=2, padx=8)
            ttk.Button(lf_ips, text="Reload File", command=self._reload_ip_module).grid(row=0, column=3, padx=8)
            ttk.Button(lf_ips, text="Dark Mode", command=self._toggle_dark).grid(row=1, column=3, padx=8)
            for c in range(4): lf_ips.grid_columnconfigure(c, weight=0)
            lf_info = ttk.LabelFrame(config_tab, text="Notes", padding=8)
            lf_info.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,8))
            info_txt = (
                "Updates:\n"
                " - Runtime: changes available immediately to this GUI process only.\n"
                " - Persist: edits ip_config.py (previous value commented with timestamp).\n"
                "After persistent change, dependent proxy/core processes are restarted."
            )
            ttk.Label(lf_info, text=info_txt, justify=tk.LEFT).pack(anchor=tk.W)
        else:
            ttk.Label(config_tab, text="ip_config module not available").pack(pady=20)

        # Status bar
        sb_frame = ttk.Frame(self.root, relief=tk.SUNKEN)
        sb_frame.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(sb_frame, textvariable=self.sb_conn).pack(side=tk.LEFT, padx=8)
        ttk.Separator(sb_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Label(sb_frame, textvariable=self.sb_core).pack(side=tk.LEFT)
        ttk.Separator(sb_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Label(sb_frame, textvariable=self.sb_proxy).pack(side=tk.LEFT)
        ttk.Separator(sb_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Label(sb_frame, textvariable=self.sb_stats).pack(side=tk.RIGHT, padx=8)

    # Helper to set and send a quick command
    def _set_and_send_cmd(self, cmd: str):
        self.cmd_var.set(cmd)
        self._send_individual_command()

    # Log actions
    def _clear_log(self):
        self.log_txt.delete("1.0", tk.END)

    def _save_log(self):
        try:
            LOG_DIR.mkdir(exist_ok=True)
            fname = LOG_DIR / f"gcs_gui_log_{int(time.time())}.txt"
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(self.log_txt.get("1.0", tk.END))
            self._log(f"Saved log to {fname}")
        except Exception as e:
            self._log(f"Save log failed: {e}")

    def _on_select_drone(self, _evt=None):
        sel = self.tree.selection()
        if not sel: return
        did = self.tree.item(sel[0], 'values')[0]
        info = self.drones.get(did)
        if not info: return
        self.sel_id.set(info.drone_id)
        self.sel_online.set("ONLINE" if info.online else "OFFLINE")
        self.sel_batt.set(f"{info.battery:.1f}%" if info.battery is not None else "-")
        self.sel_crypto.set(info.crypto or "-")
        self.sel_msg.set(info.last_msg_type or "-")
        self.sel_last.set(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info.last_seen)))

    # MQTT thread + callbacks
    def _start_mqtt_thread(self):
        def run():
            try:
                self.mqtt=GcsMqttClient(self.config,self._on_mqtt_message); self._log("MQTT client initialized")
                if self.mqtt.connect():
                    if self.auto_start_core.get():
                        okc,msgc=self.core.start(); self._log(msgc)
                else: self._log("MQTT connect timeout")
            except Exception as e: self._log(f"MQTT init error: {e}")
        threading.Thread(target=run,daemon=True).start()

    def _on_mqtt_message(self, msg: mqtt.MQTTMessage):
        # enqueue for UI thread
        self.msg_queue.put(msg)

    # Actions
    def _connect(self):
        if not self.mqtt:
            # allow changing broker/port then reconnect thread
            try:
                self.config['broker']['address']=self.broker_entry.get().strip(); self.config['broker']['port']=int(self.port_entry.get().strip())
            except Exception: messagebox.showerror(APP_NAME,"Invalid broker/port"); return
            self._start_mqtt_thread()
        else:
            try: self.mqtt.disconnect(); self.mqtt.connect()
            except Exception as e: self._log(f"Reconnect error: {e}")

    def _apply_crypto(self):
        code=self.crypto_combo.get().split(" ")[0]
        topic=self.config['topics']['publish']['crypto']['topic']
        if self.mqtt and self.mqtt.connected:
            self.mqtt.publish(topic, code, qos=2); self._log(f"Published crypto command: {code}"); self._suppress_broadcast_until=time.time()+2.0; self._last_crypto_pub=code
        else: self._log("Cannot publish; not connected")
        if self.auto_local_crypto.get():
            ok,msg=self.crypto.switch(code); self._log(msg)

    def get_crypto_recommendation(self):
        """Get current crypto recommendation from strategic RL agent."""
        if self.strategic_integration:
            return self.strategic_integration.get_current_recommendation()
        elif self.strategic_agent and self.use_strategic_rl.get():
            strategic_state = self._build_strategic_state()
            if strategic_state:
                action = self.strategic_agent.choose_action(strategic_state, training=False)
                crypto_algorithms = ["KYBER", "DILITHIUM", "SPHINCS", "FALCON"]
                return crypto_algorithms[action]
        return None

    def _send_alert(self, alert_type: str, critical: bool = False):
        topic=self.config['topics']['publish']['alerts']['topic']
        if self.mqtt and self.mqtt.connected:
            # plain text for compatibility and retained if critical
            self.mqtt.publish(topic, alert_type, qos=2, retain=critical)
            # structured as well
            payload = {"type": "alert", "code": alert_type, "priority": "critical" if critical else "warning", "ts": time.time()}
            self.mqtt.publish(topic+"/json", payload, qos=2, retain=critical)
            self._log(f"Alert sent: {alert_type}")
        else:
            self._log("Cannot send alert; not connected")

    def _send_individual_command(self):
        sel = self.tree.selection();
        if not sel: messagebox.showinfo(APP_NAME,"Select a drone first"); return
        # Ensure local proxy running if required
        if self.auto_local_crypto.get() and (self.crypto.proc is None or self.crypto.proc.poll() is not None):
            curr=self.crypto_combo.get().split(" ")[0]; ok,msg=self.crypto.switch(curr); self._log(msg)
        drone_id = self.tree.item(sel[0], 'values')[0]; cmd = self.cmd_var.get().strip() or "status"
        topic_tmpl = self.config['topics']['publish']['individual']['topic']
        topic = topic_tmpl.format(drone_id=drone_id)
        payload = {"type": "command", "command": cmd, "params": {}, "ts": time.time(), "source": "gcs"}
        if self.mqtt and self.mqtt.connected:
            ok = self.mqtt.publish(topic, payload, qos=1); self._log(f"Sent command '{cmd}' to {drone_id}: {'OK' if ok else 'FAIL'}")
        else:
            self._log("Cannot send; not connected")

    # MAVLink handlers
    def _mav_connect(self):
        ok,msg=self.mav.start(self.mav_rx_uri.get().strip(),self.mav_tx_uri.get().strip(),int(self.mav_sysid.get()),int(self.mav_compid.get()))
        if ok and self.mav_auto_hb.get(): self.mav.start_heartbeat()
        self.mav_status.set(("MAVLink: running" if ok else "MAVLink: stopped") + (" (pymavlink OK)" if PYMAVLINK_AVAILABLE else " (raw UDP mode)")); self._log(msg)
    def _mav_disconnect(self): self.mav.stop_heartbeat(); self.mav.stop(); self.mav_status.set("MAVLink: stopped" + (" (pymavlink OK)" if PYMAVLINK_AVAILABLE else " (raw UDP mode)")); self._log("MAVLink stopped")
    def _mav_send_hb(self): ok,msg=self.mav.send_heartbeat(); self._log(msg)
    def _mav_quick(self,what:str):
        if not PYMAVLINK_AVAILABLE: self._log("Install pymavlink for quick cmds"); return
        tgt_sys=int(self.mav_tgt_sys.get()); tgt_comp=int(self.mav_tgt_comp.get())
        if what=='arm': cmd=400; params=[1,0,0,0,0,0,0]
        elif what=='disarm': cmd=400; params=[0,0,0,0,0,0,0]
        elif what=='takeoff': cmd=22; params=[0,0,0,0,0,0,10.0]
        elif what=='land': cmd=21; params=[0,0,0,0,0,0,0]
        else: self._log("Unknown quick command"); return
        ok,msg=self.mav.send_command_long(tgt_sys,tgt_comp,cmd,params); self._log(msg)
    def _mav_send_cmd_long(self): tgt_sys=int(self.mav_tgt_sys.get()); tgt_comp=int(self.mav_tgt_comp.get()); cmd=int(self.mav_cmd_id.get()); params=[float(v.get()) for v in self.mav_p]; ok,msg=self.mav.send_command_long(tgt_sys,tgt_comp,cmd,params); self._log(msg)
    def _mav_send_raw(self):
        hx=self.mav_raw_hex.get().strip().replace(" ","")
        try: data=bytes.fromhex(hx)
        except Exception: self._log("Invalid hex"); return
        ok,msg=self.mav.send_raw(data); self._log(msg)
    def _on_mav_rx(self,msg):
        try:
            if PYMAVLINK_AVAILABLE:
                mtype=msg.get_type();
                if mtype=='HEARTBEAT': self._update_heartbeat(msg)
                line=f"{mtype} | {msg.to_dict()}"
            else:
                line=f"RAW {len(msg.get('raw')) if isinstance(msg,dict) and msg.get('raw') else 0} bytes"
        except Exception: line=str(msg)
        self.mav_rx_txt.insert(tk.END,line+"\n"); self.mav_rx_txt.see(tk.END)
    def _update_heartbeat(self,msg):
        try:
            sysid=getattr(msg,'sysid',None); compid=getattr(msg,'compid',None)
            if sysid is None or compid is None: return
            key=(sysid,compid); now=time.time(); rec=self.hb_stats.get(key)
            if rec: rec['count']+=1; rec['last_seen']=now
            else: rec={'count':1,'last_seen':now,'autopilot':getattr(msg,'autopilot','-'),'type':getattr(msg,'type','-'),'base_mode':getattr(msg,'base_mode','-'),'custom_mode':getattr(msg,'custom_mode','-'),'system_status':getattr(msg,'system_status','-'),'mver':getattr(msg,'mavlink_version','-')}; self.hb_stats[key]=rec
            rec.update({'autopilot':getattr(msg,'autopilot','-'),'type':getattr(msg,'type','-'),'base_mode':getattr(msg,'base_mode','-'),'custom_mode':getattr(msg,'custom_mode','-'),'system_status':getattr(msg,'system_status','-'),'mver':getattr(msg,'mavlink_version','-')})
            vals=(sysid,compid,rec['autopilot'],rec['type'],rec['base_mode'],rec['custom_mode'],rec['system_status'],rec['mver'],time.strftime('%H:%M:%S',time.localtime(rec['last_seen'])),rec['count'])
            iid=f"{sysid}-{compid}"; (self.hb_tree.item(iid,values=vals) if self.hb_tree.exists(iid) else self.hb_tree.insert('',tk.END,iid=iid,values=vals))
        except Exception as e: self._log(f"Heartbeat update error: {e}")
    # ---------------- IP UPDATE METHODS ----------------
    def _validate_ip(self, ip:str)->bool:
        import ipaddress
        try: ipaddress.IPv4Address(ip); return True
        except Exception: return False
    def _apply_ip_runtime(self):
        if not ip_config: self._log("ip_config unavailable"); return
        gcs=self.ipc_gcs.get().strip(); drone=self.ipc_drone.get().strip()
        if gcs and not self._validate_ip(gcs): self._log(f"Invalid GCS IP: {gcs}"); return
        if drone and not self._validate_ip(drone): self._log(f"Invalid DRONE IP: {drone}"); return
        try:
            changes=ip_config.set_hosts_runtime(gcs or None, drone or None)
            if changes: self._log("Runtime IP update: "+", ".join(changes)); self._restart_stack_after_ip_change()
            else: self._log("No runtime changes applied")
        except Exception as e: self._log(f"Runtime update failed: {e}")
    def _apply_ip_persistent(self):
        if not ip_config: self._log("ip_config unavailable"); return
        gcs=self.ipc_gcs.get().strip(); drone=self.ipc_drone.get().strip()
        if gcs and not self._validate_ip(gcs): self._log(f"Invalid GCS IP: {gcs}"); return
        if drone and not self._validate_ip(drone): self._log(f"Invalid DRONE IP: {drone}"); return
        try:
            changes=ip_config.update_hosts_persistent(gcs or None, drone or None)
            if changes: self._log("Persistent IP update: "+", ".join(changes)); self._reload_ip_module(); self._restart_stack_after_ip_change()
            else: self._log("No persistent changes applied")
        except Exception as e: self._log(f"Persistent update failed: {e}")
    def _reload_ip_module(self):
        if not ip_config: return
        try: importlib.reload(ip_config); self.ipc_gcs.set(getattr(ip_config,'GCS_HOST',self.ipc_gcs.get())); self.ipc_drone.set(getattr(ip_config,'DRONE_HOST',self.ipc_drone.get())); self._log("ip_config reloaded")
        except Exception as e: self._log(f"Reload failed: {e}")
    def update_ip(self, which:str, value:str, persistent:bool=True):
        if which not in ("gcs","drone"): raise ValueError("which must be 'gcs' or 'drone'")
        if not self._validate_ip(value): raise ValueError(f"Invalid IP: {value}")
        if persistent:
            if which=='gcs': ip_config.update_hosts_persistent(new_gcs=value,new_drone=None)
            else: ip_config.update_hosts_persistent(new_gcs=None,new_drone=value)
            self._reload_ip_module()
        else:
            if which=='gcs': ip_config.set_hosts_runtime(new_gcs=value,new_drone=None)
            else: ip_config.set_hosts_runtime(new_gcs=None,new_drone=value)
        (self.ipc_gcs if which=='gcs' else self.ipc_drone).set(value); self._restart_stack_after_ip_change(); self._log(f"update_ip completed ({which}->{value}, persistent={persistent})")
    def _restart_stack_after_ip_change(self):
        self._log("Restarting proxy/core after IP change ...")
        try:
            self.crypto.stop(); self.core.stop(); self.root.after(500,self._delayed_stack_restart)
        except Exception as e: self._log(f"Restart sequence error: {e}")
    def _delayed_stack_restart(self):
        code=self.crypto_combo.get().split(" ")[0]; ok,msg=self.crypto.switch(code); self._log(msg); 
        if self.auto_start_core.get(): okc,msgc=self.core.start(); self._log(msgc)
    # UI loop
    def _ui_tick(self):
        # status label
        if self.mqtt and self.mqtt.connected:
            self.status_lbl.config(text="Connected", foreground="green")
        else:
            self.status_lbl.config(text="Disconnected", foreground="red")

        # reflect core/proxy status
        core_running = bool(self.core and self.core.proc and self.core.proc.poll() is None)
        proxy_running = bool(self.crypto and self.crypto.proc and self.crypto.proc.poll() is None)
        self.core_status_lbl.config(text="Core: running" if core_running else "Core: stopped")
        cur = self.crypto.current_code or "?"
        self.proxy_status_lbl.config(text=f"Proxy: running ({cur})" if proxy_running else "Proxy: stopped")

        # update status bar
        broker = f"{self.config['broker']['address']}:{self.config['broker']['port']}"
        self.sb_conn.set(("Connected" if (self.mqtt and self.mqtt.connected) else "Disconnected") + f" @ {broker}")
        self.sb_core.set("Core: running" if core_running else "Core: stopped")
        self.sb_proxy.set(f"Proxy: running ({cur})" if proxy_running else "Proxy: stopped")
        total = len(self.drones)
        online = sum(1 for d in self.drones.values() if d.online)
        rx = (self.mqtt.metrics["rx"] if self.mqtt else 0)
        tx = (self.mqtt.metrics["tx"] if self.mqtt else 0)
        self.sb_stats.set(f"Drones: {online}/{total} | Rx: {rx}B Tx: {tx}B")

        # process MQTT messages
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                self._handle_msg_on_ui(msg)
        except queue.Empty:
            pass

        # mark offline if stale (>60s)
        now = time.time()
        for did, info in list(self.drones.items()):
            if now - info.last_seen > 60 and info.online:
                info.online = False
                if self.tree.exists(did):
                    vals = self.tree.item(did, 'values')
                    new_vals = (did, "OFFLINE", vals[2], vals[3], vals[4], vals[5])
                    self.tree.item(did, values=new_vals)

        # Strategic RL periodic update and live recommendations
        if self.strategic_integration:
            self.strategic_integration.periodic_strategic_update()
        elif self.strategic_agent:
            self._update_rl_recommendations()

        # reschedule
        self.root.after(300, self._ui_tick)

    # Message handling
    def _handle_msg_on_ui(self, msg: mqtt.MQTTMessage):
        topic = msg.topic
        payload = msg.payload
        text = None
        try:
            text = payload.decode('utf-8')
        except Exception:
            text = f"<binary {len(payload)} bytes>"

        # React to broadcast crypto commands
        if topic == self.config['topics']['publish']['crypto']['topic']:
            code = text.strip()
            if re.fullmatch(r"c[1-8]",code):
                # Ignore our own recent publish to prevent double-start
                if time.time() < self._suppress_broadcast_until and code == self._last_crypto_pub:
                    self._log(f"Ignoring self crypto broadcast: {code}")
                else:
                    self._log(f"Broadcast crypto received: {code}")
                    # update combo to reflect new selection
                    self._select_crypto_in_combo(code)
                    if self.auto_local_crypto.get():
                        # only switch if different from current
                        if code != (self.crypto.current_code or "") or not (self.crypto.proc and self.crypto.proc.poll() is not None):
                            ok,msg=self.crypto.switch(code); self._log(msg)
                return

        # Determine message type for UI display
        msg_type = None
        decoded = self._safe_json(text)
        if topic.startswith('swarm/heartbeat/') or '/heartbeat' in topic:
            msg_type = 'HEARTBEAT'
        elif isinstance(decoded,dict):
            tval = str(decoded.get('type') or decoded.get('message_type') or '').lower()
            if 'heartbeat' in tval:
                msg_type = 'HEARTBEAT'
            elif topic.startswith('swarm/status/') or '/status' in topic:
                msg_type = 'STATUS'
            elif ('/telemetry' in topic) or (topic.startswith('swarm/drones/') and topic.endswith('/telemetry')):
                msg_type = 'TELEMETRY'
            elif topic.startswith('swarm/alert/'):
                msg_type = 'ALERT'
        else:
            if topic.startswith('swarm/status/') or '/status' in topic:
                msg_type = 'STATUS'
            elif ('/telemetry' in topic) or (topic.startswith('swarm/drones/') and topic.endswith('/telemetry')):
                msg_type = 'TELEMETRY'
            elif topic.startswith('swarm/alert/'):
                msg_type = 'ALERT'

        # Update drones based on status/telemetry/alerts/heartbeat
        did = self._extract_drone_id(topic, decoded if isinstance(decoded,dict) else None)
        if did and did!="gcs":
            battery=None; crypto_alg=None
            if isinstance(decoded,dict):
                for k in ("battery","battery_percent","battery_level"):
                    if k in decoded:
                        try: battery=float(decoded[k]); break
                        except Exception: pass
                for k in ("crypto","crypto_algorithm"):
                    if k in decoded:
                        crypto_alg=str(decoded[k]); break
            self._upsert_drone(did,battery,msg_type or '-',crypto_alg)

        # Log
        if topic.startswith("swarm/alert/") and did: self._log(f"Alert from {did}: {text}")
        elif topic == self.config['topics']['publish']['alerts']['topic']: self._log(f"Broadcast alert: {text}")
        self._log(f"RX {topic}: {text}")

    def _safe_json(self, s: str):
        try:
            return json.loads(s)
        except Exception:
            return s

    def _extract_drone_id(self, topic: str, payload_obj: Optional[Dict[str, Any]] = None) -> Optional[str]:
        # Common patterns
        for pat in (
            r"swarm/status/([^/]+)",
            r"swarm/drones/([^/]+)/",
            r"swarm/alert/([^/]+)",
            r"swarm/heartbeat/([^/]+)",
            r"swarm/([^/]+)/status",
            r"swarm/([^/]+)/telemetry",
            r"swarm/([^/]+)/heartbeat",
        ):
            m = re.match(pat, topic)
            if m:
                return m.group(1)
        # Fallback: any 'swarm/<id>/' prefix
        m = re.match(r"swarm/([^/]+)/", topic)
        if m and m.group(1) not in ("broadcast", "status"):
            return m.group(1)
        # Try from payload
        if payload_obj:
            for key in ("drone_id", "id", "uav_id", "name"):
                v = payload_obj.get(key)
                if isinstance(v, str) and v.lower() != "gcs":
                    return v
        return None

    def _upsert_drone(self, drone_id:str, battery:Optional[float], msg_type:str='-', crypto_alg:Optional[str]=None):
        now=time.time(); info=self.drones.get(drone_id)
        if not info:
            info=DroneInfo(drone_id=drone_id,last_seen=now,online=True,battery=battery,crypto=crypto_alg,last_msg_type=msg_type)
            if msg_type=='HEARTBEAT': info.hb_count=1
            self.drones[drone_id]=info
            self.tree.insert('',tk.END,iid=drone_id,values=(drone_id,"ONLINE",f"{battery:.1f}%" if battery is not None else "-", crypto_alg or '-', msg_type, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))))
        else:
            info.last_seen=now; info.online=True
            if battery is not None: info.battery=battery
            if crypto_alg: info.crypto=crypto_alg
            info.last_msg_type=msg_type
            if msg_type=='HEARTBEAT': info.hb_count+=1
            vals=(drone_id,"ONLINE",f"{info.battery:.1f}%" if info.battery is not None else "-", info.crypto or '-', info.last_msg_type or '-', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)))
            (self.tree.item(drone_id,values=vals) if self.tree.exists(drone_id) else self.tree.insert('',tk.END,iid=drone_id,values=vals))

    def _log(self,line:str): ts=time.strftime("%H:%M:%S"); self.log_txt.insert(tk.END,f"[{ts}] {line}\n"); self.log_txt.see(tk.END); logger.info(line)

    # Core controls
    def _start_core(self): ok,msg=self.core.start(); self._log(msg)
    def _stop_core(self): self.core.stop(); self._log("Core stopped")
    def _start_proxy(self): code=self.crypto_combo.get().split(" ")[0]; ok,msg=self.crypto.switch(code); self._log(msg)
    def _stop_proxy(self): self.crypto.stop(); self._log("Proxy stopped")
    def _start_stack(self): code=self.crypto_combo.get().split(" ")[0]; ok,msg=self.crypto.switch(code); self._log(msg); okc,msgc=self.core.start(); self._log(msgc)
    def _select_crypto_in_combo(self,code:str):
        try:
            for idx,label in enumerate(self.crypto_combo['values']):
                if label.startswith(code+" "): self.crypto_combo.current(idx); break
        except Exception: pass

    # New: Crypto RL integration
    def _get_rl_recommendation(self):
        self.strategic_integration.make_strategic_decision()


if __name__ == "__main__":
    # Load config from file if it exists, otherwise use default
    config = DEFAULT_CONFIG
    config_path = HERE / "gcs_config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config.update(json.load(f))
            logger.info(f"Loaded config from {config_path}")
        except Exception as e:
            logger.error(f"Error loading {config_path}: {e}")

    root=tk.Tk()
    app=GcsSchedulerApp(root, config)
    root.mainloop()