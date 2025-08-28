"""
Simple runner that loads trained tactical and strategic policies and starts the integrated SystemCoordinator.
Use this for a quick smoke test of live inference (no MQTT/UI dependencies).
"""
from __future__ import annotations
import argparse
import os
import logging

from integration.system_coordinator import get_system_coordinator


def main():
    parser = argparse.ArgumentParser(description="Run live inference with trained policies")
    parser.add_argument("--tactical-policy", type=str, default=os.path.join("output", "tactical_dqn_best.pt"), help="Path to tactical .pt policy")
    parser.add_argument("--strategic-policy", type=str, default=os.path.join("output", "strategic_crypto_dqn_best.pt"), help="Path to strategic .pt policy")
    parser.add_argument("--uav-id", type=str, default="UAV_001", help="UAV ID")
    parser.add_argument("--gcs-id", type=str, default="GCS_MAIN", help="GCS ID")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    coord = get_system_coordinator()
    # Start both controllers; missing policies are tolerated by controllers
    tac_path = args.tactical_policy if os.path.exists(args.tactical_policy) else None
    strat_path = args.strategic_policy if os.path.exists(args.strategic_policy) else None
    coord.start(tactical_policy=tac_path, strategic_policy=strat_path)

    try:
        # Idle loop
        while True:
            status = coord.get_system_status()
            print(f"[status] running={status['running']} tact_run={status['tactical_status']['running']} strat_run={status['strategic_status']['running']}")
            import time
            time.sleep(10)
    except KeyboardInterrupt:
        pass
    finally:
        coord.stop()


if __name__ == "__main__":
    main()
