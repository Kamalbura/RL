import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from crypto_rl.coordination import (
    make_crypto_policy_directive,
    make_drone_status,
    make_ddos_alert,
    make_swarm_state,
    parse_message,
)
from crypto_rl.swarm_consensus import SwarmConsensusManager


def test_coordination_messages_roundtrip():
    p = make_crypto_policy_directive("FALCON512", max_cpu_overhead=0.3, min_security=7)
    obj = parse_message(p)
    assert obj["type"] == "crypto_policy"
    s = make_drone_status("dr-1", 77.5, 1, 2, 45.0, active_crypto="ASCON_128")
    assert parse_message(s)["type"] == "drone_status"
    a = make_ddos_alert("dr-1", 2, 0.92)
    assert parse_message(a)["type"] == "ddos_alert"
    w = make_swarm_state("dr-1", 2, 55.0)
    assert parse_message(w)["type"] == "swarm_state"


def test_swarm_consensus():
    mgr = SwarmConsensusManager(["a", "b", "c", "d", "e", "f"])
    for did in ["a", "b", "c", "d", "e", "f"]:
        mgr.process_threat_report(did, 2)
    level = mgr.get_swarm_threat_consensus()
    assert level in (1, 2)
