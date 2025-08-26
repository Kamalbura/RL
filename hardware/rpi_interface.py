"""
Hardware Abstraction Layer for Raspberry Pi 4B
Provides interfaces for CPU frequency control, thermal monitoring, and system metrics
"""

import os
import subprocess
import psutil
import time
from typing import Dict, Optional, Tuple
import logging

class RPiHardwareInterface:
    """Hardware abstraction layer for Raspberry Pi 4B control"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.available_frequencies = [600000, 1200000, 1800000, 2000000]  # kHz
        self.thermal_throttle_temp = 80.0  # Celsius
        self.critical_temp = 85.0
        
    def set_cpu_frequency(self, frequency_mhz: int) -> bool:
        """Set CPU frequency using cpufreq-set"""
        frequency_khz = frequency_mhz * 1000
        
        if frequency_khz not in self.available_frequencies:
            self.logger.error(f"Invalid frequency: {frequency_mhz}MHz")
            return False
            
        try:
            # Set frequency for all cores
            for cpu in range(psutil.cpu_count()):
                cmd = f"sudo cpufreq-set -c {cpu} -f {frequency_khz}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    self.logger.error(f"Failed to set CPU {cpu} frequency: {result.stderr}")
                    return False
            
            self.logger.info(f"CPU frequency set to {frequency_mhz}MHz")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting CPU frequency: {e}")
            return False
    
    def get_cpu_frequency(self) -> Optional[int]:
        """Get current CPU frequency in MHz"""
        try:
            with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "r") as f:
                freq_khz = int(f.read().strip())
                return freq_khz // 1000
        except Exception as e:
            self.logger.error(f"Error reading CPU frequency: {e}")
            return None
    
    def get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature in Celsius"""
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp_millicelsius = int(f.read().strip())
                return temp_millicelsius / 1000.0
        except Exception as e:
            self.logger.error(f"Error reading CPU temperature: {e}")
            return None
    
    def is_thermal_throttling(self) -> bool:
        """Check if CPU is thermal throttling"""
        temp = self.get_cpu_temperature()
        if temp is None:
            return False
        return temp > self.thermal_throttle_temp
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get comprehensive system metrics"""
        metrics = {}
        
        # CPU metrics
        metrics['cpu_percent'] = psutil.cpu_percent(interval=1)
        metrics['cpu_frequency_mhz'] = self.get_cpu_frequency() or 0
        metrics['cpu_temperature_c'] = self.get_cpu_temperature() or 0
        metrics['cpu_count'] = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics['memory_percent'] = memory.percent
        metrics['memory_available_mb'] = memory.available / (1024 * 1024)
        
        # Power/thermal status
        metrics['thermal_throttling'] = self.is_thermal_throttling()
        
        return metrics
    
    def set_cpu_cores(self, core_count: int) -> bool:
        """Enable/disable CPU cores (simplified implementation)"""
        if core_count < 1 or core_count > psutil.cpu_count():
            self.logger.error(f"Invalid core count: {core_count}")
            return False
            
        try:
            # This is a simplified implementation
            # In practice, you'd use CPU hotplug or taskset for process affinity
            total_cores = psutil.cpu_count()
            
            for cpu in range(core_count, total_cores):
                # Disable cores beyond core_count (requires root privileges)
                cmd = f"echo 0 | sudo tee /sys/devices/system/cpu/cpu{cpu}/online"
                subprocess.run(cmd, shell=True, capture_output=True)
            
            for cpu in range(core_count):
                # Enable cores up to core_count
                cmd = f"echo 1 | sudo tee /sys/devices/system/cpu/cpu{cpu}/online"
                subprocess.run(cmd, shell=True, capture_output=True)
                
            self.logger.info(f"CPU cores set to {core_count}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting CPU cores: {e}")
            return False
    
    def get_power_consumption_estimate(self) -> Optional[float]:
        """Estimate power consumption based on CPU frequency and load"""
        freq = self.get_cpu_frequency()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if freq is None:
            return None
            
        # Simple power model based on frequency and utilization
        # These coefficients would be calibrated from actual measurements
        base_power = 2.5  # Watts (idle)
        freq_factor = (freq / 1000.0) ** 2 * 0.001  # Quadratic frequency scaling
        load_factor = (cpu_percent / 100.0) * 2.0  # Linear load scaling
        
        estimated_power = base_power + freq_factor + load_factor
        return estimated_power


class BatteryMonitor:
    """Battery monitoring interface for UAV power management"""
    
    def __init__(self, voltage_nominal: float = 22.2, capacity_mah: float = 5200):
        self.voltage_nominal = voltage_nominal
        self.capacity_mah = capacity_mah
        self.capacity_wh = voltage_nominal * capacity_mah / 1000.0
        self.logger = logging.getLogger(__name__)
        
    def get_battery_voltage(self) -> Optional[float]:
        """Get battery voltage (would interface with actual battery monitor)"""
        # Placeholder - would interface with actual hardware
        # For simulation, return nominal voltage with some variation
        import random
        return self.voltage_nominal + random.uniform(-1.0, 0.5)
    
    def get_battery_current(self) -> Optional[float]:
        """Get battery current draw in Amperes"""
        # Placeholder - would interface with actual current sensor
        # For simulation, estimate based on system load
        rpi = RPiHardwareInterface()
        power_estimate = rpi.get_power_consumption_estimate()
        if power_estimate and self.voltage_nominal:
            return power_estimate / self.voltage_nominal
        return None
    
    def get_battery_percentage(self) -> Optional[float]:
        """Get battery charge percentage"""
        # Placeholder - would interface with battery management system
        # For simulation, calculate based on voltage
        voltage = self.get_battery_voltage()
        if voltage is None:
            return None
            
        # Simple voltage-based estimation (not accurate for LiPo)
        min_voltage = 18.0  # 3.0V per cell * 6 cells
        max_voltage = 25.2  # 4.2V per cell * 6 cells
        
        percentage = ((voltage - min_voltage) / (max_voltage - min_voltage)) * 100
        return max(0, min(100, percentage))
    
    def get_flight_time_remaining(self) -> Optional[float]:
        """Estimate remaining flight time in minutes"""
        current = self.get_battery_current()
        percentage = self.get_battery_percentage()
        
        if current is None or percentage is None or current <= 0:
            return None
            
        remaining_capacity_mah = (percentage / 100.0) * self.capacity_mah
        flight_time_hours = remaining_capacity_mah / (current * 1000)  # Convert A to mA
        
        return flight_time_hours * 60  # Convert to minutes


class SystemHealthMonitor:
    """System health monitoring and alerting"""
    
    def __init__(self):
        self.rpi = RPiHardwareInterface()
        self.battery = BatteryMonitor()
        self.logger = logging.getLogger(__name__)
        
    def get_health_status(self) -> Dict[str, any]:
        """Get comprehensive system health status"""
        status = {
            'timestamp': time.time(),
            'overall_health': 'UNKNOWN',
            'alerts': [],
            'metrics': {}
        }
        
        # Get system metrics
        metrics = self.rpi.get_system_metrics()
        battery_pct = self.battery.get_battery_percentage()
        flight_time = self.battery.get_flight_time_remaining()
        
        status['metrics'] = {
            **metrics,
            'battery_percentage': battery_pct,
            'flight_time_remaining_min': flight_time
        }
        
        # Health checks
        alerts = []
        health_score = 100
        
        # Temperature check
        if metrics.get('cpu_temperature_c', 0) > 80:
            alerts.append('HIGH_TEMPERATURE')
            health_score -= 20
            
        # Battery check
        if battery_pct and battery_pct < 20:
            alerts.append('LOW_BATTERY')
            health_score -= 30
            
        # Memory check
        if metrics.get('memory_percent', 0) > 90:
            alerts.append('HIGH_MEMORY_USAGE')
            health_score -= 10
            
        # CPU check
        if metrics.get('cpu_percent', 0) > 95:
            alerts.append('HIGH_CPU_USAGE')
            health_score -= 10
            
        # Thermal throttling check
        if metrics.get('thermal_throttling', False):
            alerts.append('THERMAL_THROTTLING')
            health_score -= 25
            
        status['alerts'] = alerts
        
        if health_score >= 80:
            status['overall_health'] = 'GOOD'
        elif health_score >= 60:
            status['overall_health'] = 'WARNING'
        else:
            status['overall_health'] = 'CRITICAL'
            
        return status


# Hardware interface singleton
_hardware_interface = None

def get_hardware_interface() -> RPiHardwareInterface:
    """Get singleton hardware interface"""
    global _hardware_interface
    if _hardware_interface is None:
        _hardware_interface = RPiHardwareInterface()
    return _hardware_interface
