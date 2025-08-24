"""Minimal config for ddos_rl to be self-contained."""

BATTERY_SPECS = {
    "VOLTAGE": 22.2,          # 6S LiPo
    "CAPACITY_MAH": 5200,     # mAh
    "CAPACITY_WH": 22.2 * 5.2 # Watt-hours (~115.44 Wh)
}

__all__ = ["BATTERY_SPECS"]