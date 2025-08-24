"""
Swarm consensus utilities with simple Byzantine tolerance for threat levels.
"""

from __future__ import annotations

import math
import time
from typing import Dict


class SwarmConsensusManager:
    def __init__(self, drone_ids: list[str], consensus_threshold: float = 0.7):
        self.drone_ids = drone_ids
        self.consensus_threshold = consensus_threshold
        self.threat_reports: Dict[str, Dict[str, float]] = {
            did: {"level": 0.0, "timestamp": 0.0} for did in drone_ids
        }

    def process_threat_report(self, drone_id: str, threat_level: int, timestamp: float | None = None):
        if drone_id not in self.threat_reports:
            self.threat_reports[drone_id] = {"level": 0.0, "timestamp": 0.0}
        self.threat_reports[drone_id]["level"] = float(threat_level)
        self.threat_reports[drone_id]["timestamp"] = float(timestamp if timestamp is not None else time.time())

    def get_swarm_threat_consensus(self, max_report_age_sec: float = 30.0) -> int:
        now = time.time()
        fresh = [
            rep["level"]
            for rep in self.threat_reports.values()
            if now - rep["timestamp"] < max_report_age_sec
        ]
        if not fresh:
            return 0
        fresh.sort()
        n = len(fresh)
        if n > 4:
            f = (n - 1) // 3
            trimmed = fresh[f:-f] if f > 0 else fresh
            avg = sum(trimmed) / len(trimmed)
            consensus_pct = len([v for v in trimmed if v > 0]) / len(trimmed)
            if consensus_pct >= self.consensus_threshold:
                return int(math.ceil(avg))
        return int(max(fresh))


__all__ = ["SwarmConsensusManager"]
