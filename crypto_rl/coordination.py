"""
Agent Coordination Protocol: JSON message formats and helpers

This module defines lightweight message schemas for:
- GCS -> Drone: crypto policy directives with constraints
- Drone -> GCS: status updates and DDoS alerts
- Swarm P2P: state sharing

It does not depend on an MQTT client; callers publish the produced JSON strings.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional


def now_ts() -> int:
    return int(time.time())


def make_crypto_policy_directive(algo_code: str,
                                 max_cpu_overhead: Optional[float] = None,
                                 min_security: Optional[int] = None,
                                 extra: Optional[Dict[str, Any]] = None) -> str:
    msg: Dict[str, Any] = {
        "type": "crypto_policy",
        "algo_code": algo_code,
        "constraints": {},
        "ts": now_ts(),
    }
    if max_cpu_overhead is not None:
        msg["constraints"]["max_cpu_overhead"] = float(max_cpu_overhead)
    if min_security is not None:
        msg["constraints"]["min_security"] = int(min_security)
    if extra:
        msg.update(extra)
    return json.dumps(msg, separators=(",", ":"))


def make_drone_status(drone_id: str,
                      battery: float,
                      threat_level: int,
                      cpu_load: int,
                      temperature_c: float,
                      active_crypto: Optional[str] = None,
                      extra: Optional[Dict[str, Any]] = None) -> str:
    msg: Dict[str, Any] = {
        "type": "drone_status",
        "drone_id": drone_id,
        "battery": float(battery),
        "threat_level": int(threat_level),
        "cpu_load": int(cpu_load),
        "temperature": float(temperature_c),
        "active_crypto": active_crypto,
        "ts": now_ts(),
    }
    if extra:
        msg.update(extra)
    return json.dumps(msg, separators=(",", ":"))


def make_ddos_alert(drone_id: str, level: int, confidence: float,
                    extra: Optional[Dict[str, Any]] = None) -> str:
    msg: Dict[str, Any] = {
        "type": "ddos_alert",
        "drone_id": drone_id,
        "level": int(level),
        "confidence": float(confidence),
        "ts": now_ts(),
    }
    if extra:
        msg.update(extra)
    return json.dumps(msg, separators=(",", ":"))


def make_swarm_state(drone_id: str, threat_level: int, battery: float,
                     extra: Optional[Dict[str, Any]] = None) -> str:
    msg: Dict[str, Any] = {
        "type": "swarm_state",
        "drone_id": drone_id,
        "threat_level": int(threat_level),
        "battery": float(battery),
        "ts": now_ts(),
    }
    if extra:
        msg.update(extra)
    return json.dumps(msg, separators=(",", ":"))


def parse_message(payload: str) -> Dict[str, Any]:
    """Parse and minimally validate a coordination message."""
    obj = json.loads(payload)
    if "type" not in obj:
        raise ValueError("message missing type")
    if obj["type"] not in {"crypto_policy", "drone_status", "ddos_alert", "swarm_state"}:
        raise ValueError(f"unknown message type {obj['type']}")
    return obj


__all__ = [
    "make_crypto_policy_directive",
    "make_drone_status",
    "make_ddos_alert",
    "make_swarm_state",
    "parse_message",
]
