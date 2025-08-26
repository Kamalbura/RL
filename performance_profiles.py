"""Unified performance profile interface.

This module now delegates to the canonical implementation in
`ddos_rl.profiles` to avoid data drift between duplicated copies.
All existing import sites that used:

    from performance_profiles import get_power_consumption

will continue to work. Update logic only in `ddos_rl/profiles.py`.
"""

from ddos_rl.profiles import (
    POWER_PROFILES,
    LATENCY_PROFILES,
    DDOS_TASK_TIME_PROFILES,
    SECURITY_RATINGS,
    CPU_FREQUENCY_PRESETS,
    get_power_consumption,
    get_crypto_latency,
    get_ddos_execution_time,
    get_security_rating,
)

__all__ = [
    "POWER_PROFILES",
    "LATENCY_PROFILES",
    "DDOS_TASK_TIME_PROFILES",
    "SECURITY_RATINGS",
    "CPU_FREQUENCY_PRESETS",
    "get_power_consumption",
    "get_crypto_latency",
    "get_ddos_execution_time",
    "get_security_rating",
]
