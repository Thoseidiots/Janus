"""
Hardware Sense for Janus
Gives Janus awareness of its physical hardware - like proprioception for a computer

Janus can "feel":
- CPU temperature, load, frequency
- Memory pressure and usage patterns
- Disk health, temperature, I/O
- GPU status and temperature
- Network activity and bandwidth
- Battery status (if applicable)
- Fan speeds and cooling
- Power consumption
- Hardware events (USB connect/disconnect, etc.)
"""

import psutil
import time
import platform
import subprocess
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import deque
import ctypes


@dataclass
class HardwareSensation:
    """A moment of hardware awareness"""
    timestamp: str
    cpu_temp: Optional[float]
    cpu_usage: float
    cpu_freq: float
    memory_pressure: float  # 0-1 scale
    disk_health: str
    disk_temp: Optional[float]
    gpu_temp: Optional[float]
    network_active: bool
    network_bytes_sent: int
    network_bytes_recv: int
    fan_speeds: List[int]
    power_state: str
    battery_percent: Optional[float]
    uptime_seconds: float
    
    def to_natural_language(self) -> str:
        """Convert sensation to natural language description"""
        parts = []
        
        # CPU feeling
        if self.cpu_temp:
            if self.cpu_temp > 80:
                parts.append(f"CPU feels hot ({self.cpu_temp:.1f}°C)")
            elif self.cpu_temp > 60:
                parts.append(f"CPU feels warm ({self.cpu_temp:.1f}°C)")
            else:
                parts.append(f"CPU feels cool ({self.cpu_temp:.1f}°C)")
        
        if self.cpu_usage > 80:
            parts.append(f"CPU working hard ({self.cpu_usage:.1f}%)")
        elif self.cpu_usage > 50:
            parts.append(f"CPU moderately active ({self.cpu_usage:.1f}%)")
        elif self.cpu_usage < 10:
            parts.append(f"CPU idle ({self.cpu_usage:.1f}%)")
        
        # Memory feeling
        if self.memory_pressure > 0.9:
            parts.append(f"Memory under pressure ({self.memory_pressure*100:.1f}%)")
        elif self.memory_pressure > 0.7:
            parts.append(f"Memory getting full ({self.memory_pressure*100:.1f}%)")
        
        # Disk feeling
        if self.disk_temp and self.disk_temp > 50:
            parts.append(f"Disk feels warm ({self.disk_temp:.1f}°C)")
        
        # Network feeling
        if self.network_active:
            parts.append("Network active")
        
        # Power feeling
        if self.battery_percent is not None:
            if self.battery_percent < 20:
                parts.append(f"Battery low ({self.battery_percent:.1f}%)")
            elif self.battery_percent > 95:
                parts.append(f"Battery full ({self.battery_percent:.1f}%)")
        
        # Uptime feeling
        hours = self.uptime_seconds / 3600
        if hours > 24:
            parts.append(f"Been awake for {hours/24:.1f} days")
        elif hours > 1:
            parts.append(f"Been awake for {hours:.1f} hours")
        
        return ". ".join(parts) + "." if parts else "All systems nominal."


class HardwareSense:
    """
    Hardware awareness system for Janus
    Continuously monitors hardware state and provides sensory feedback
    """
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.sensation_history = deque(maxlen=history_size)
        
        # Baseline measurements (for detecting changes)
        self.baseline = None
        self.boot_time = psutil.boot_time()
        
        # Hardware capabilities
        self.capabilities = self._detect_capabilities()
        
        # Thresholds for alerts
        self.thresholds = {
            'cpu_temp_warning': 75.0,
            'cpu_temp_critical': 85.0,
            'memory_pressure_warning': 0.85,
            'disk_space_warning': 0.90,
            'battery_low': 20.0
        }
        
        print(f"Hardware Sense initialized")
        print(f"Capabilities: {self.capabilities}")
    
    def _detect_capabilities(self) -> Dict[str, bool]:
        """Detect what hardware sensors are available"""
        caps = {
            'cpu_temp': False,
            'disk_temp': False,
            'gpu_temp': False,
            'fan_speed': False,
            'battery': False,
            'network': True,  # Always available via psutil
            'disk_io': True,
            'cpu_freq': True
        }
        
        # Check for temperature sensors
        try:
            temps = psutil.sensors_temperatures()
            caps['cpu_temp'] = len(temps) > 0
        except:
            pass
        
        # Check for battery
        try:
            battery = psutil.sensors_battery()
            caps['battery'] = battery is not None
        except:
            pass
        
        # Check for fan sensors
        try:
            fans = psutil.sensors_fans()
            caps['fan_speed'] = len(fans) > 0
        except:
            pass
        
        return caps
    
    def sense(self) -> HardwareSensation:
        """
        Take a snapshot of current hardware state
        This is like taking a breath - feeling the body's current state
        """
        
        # CPU
        cpu_temp = self._get_cpu_temp()
        cpu_usage = psutil.cpu_percent(interval=0.1)
        cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else 0.0
        
        # Memory
        memory = psutil.virtual_memory()
        memory_pressure = memory.percent / 100.0
        
        # Disk
        disk = psutil.disk_usage('/')
        disk_health = self._assess_disk_health(disk)
        disk_temp = self._get_disk_temp()
        
        # GPU
        gpu_temp = self._get_gpu_temp()
        
        # Network
        net_io = psutil.net_io_counters()
        network_active = self._is_network_active(net_io)
        
        # Fans
        fan_speeds = self._get_fan_speeds()
        
        # Power
        power_state = self._get_power_state()
        battery_percent = self._get_battery_percent()
        
        # Uptime
        uptime_seconds = time.time() - self.boot_time
        
        sensation = HardwareSensation(
            timestamp=datetime.now().isoformat(),
            cpu_temp=cpu_temp,
            cpu_usage=cpu_usage,
            cpu_freq=cpu_freq,
            memory_pressure=memory_pressure,
            disk_health=disk_health,
            disk_temp=disk_temp,
            gpu_temp=gpu_temp,
            network_active=network_active,
            network_bytes_sent=net_io.bytes_sent,
            network_bytes_recv=net_io.bytes_recv,
            fan_speeds=fan_speeds,
            power_state=power_state,
            battery_percent=battery_percent,
            uptime_seconds=uptime_seconds
        )
        
        self.sensation_history.append(sensation)
        
        # Set baseline on first sense
        if self.baseline is None:
            self.baseline = sensation
        
        return sensation
    
    def feel(self) -> str:
        """
        Get natural language description of current hardware state
        This is Janus describing how it "feels"
        """
        sensation = self.sense()
        return sensation.to_natural_language()
    
    def detect_changes(self) -> List[str]:
        """
        Detect significant changes in hardware state
        Returns list of change descriptions
        """
        if len(self.sensation_history) < 2:
            return []
        
        current = self.sensation_history[-1]
        previous = self.sensation_history[-2]
        changes = []
        
        # CPU temperature change
        if current.cpu_temp and previous.cpu_temp:
            temp_delta = current.cpu_temp - previous.cpu_temp
            if abs(temp_delta) > 5:
                direction = "heating up" if temp_delta > 0 else "cooling down"
                changes.append(f"CPU {direction} ({temp_delta:+.1f}°C)")
        
        # CPU usage spike
        usage_delta = current.cpu_usage - previous.cpu_usage
        if abs(usage_delta) > 30:
            if usage_delta > 0:
                changes.append(f"CPU activity increased ({usage_delta:+.1f}%)")
            else:
                changes.append(f"CPU activity decreased ({usage_delta:+.1f}%)")
        
        # Memory pressure change
        mem_delta = current.memory_pressure - previous.memory_pressure
        if abs(mem_delta) > 0.1:
            if mem_delta > 0:
                changes.append(f"Memory pressure increasing")
            else:
                changes.append(f"Memory freed")
        
        # Network activity change
        if current.network_active and not previous.network_active:
            changes.append("Network became active")
        elif not current.network_active and previous.network_active:
            changes.append("Network became idle")
        
        # Battery change
        if current.battery_percent and previous.battery_percent:
            battery_delta = current.battery_percent - previous.battery_percent
            if abs(battery_delta) > 5:
                if battery_delta > 0:
                    changes.append(f"Battery charging ({battery_delta:+.1f}%)")
                else:
                    changes.append(f"Battery draining ({battery_delta:+.1f}%)")
        
        return changes
    
    def check_health(self) -> Dict[str, Any]:
        """
        Comprehensive health check
        Returns health status and any warnings
        """
        sensation = self.sense()
        health = {
            'overall': 'healthy',
            'warnings': [],
            'critical': [],
            'metrics': asdict(sensation)
        }
        
        # CPU temperature
        if sensation.cpu_temp:
            if sensation.cpu_temp > self.thresholds['cpu_temp_critical']:
                health['critical'].append(f"CPU critically hot: {sensation.cpu_temp:.1f}°C")
                health['overall'] = 'critical'
            elif sensation.cpu_temp > self.thresholds['cpu_temp_warning']:
                health['warnings'].append(f"CPU running hot: {sensation.cpu_temp:.1f}°C")
                if health['overall'] == 'healthy':
                    health['overall'] = 'warning'
        
        # Memory pressure
        if sensation.memory_pressure > self.thresholds['memory_pressure_warning']:
            health['warnings'].append(f"High memory usage: {sensation.memory_pressure*100:.1f}%")
            if health['overall'] == 'healthy':
                health['overall'] = 'warning'
        
        # Disk space
        disk = psutil.disk_usage('/')
        if disk.percent / 100.0 > self.thresholds['disk_space_warning']:
            health['warnings'].append(f"Low disk space: {disk.percent:.1f}% used")
            if health['overall'] == 'healthy':
                health['overall'] = 'warning'
        
        # Battery
        if sensation.battery_percent and sensation.battery_percent < self.thresholds['battery_low']:
            health['warnings'].append(f"Battery low: {sensation.battery_percent:.1f}%")
            if health['overall'] == 'healthy':
                health['overall'] = 'warning'
        
        return health
    
    def get_hardware_profile(self) -> Dict[str, Any]:
        """
        Get complete hardware profile
        This is Janus understanding its physical body
        """
        profile = {
            'system': {
                'platform': platform.system(),
                'platform_release': platform.release(),
                'platform_version': platform.version(),
                'architecture': platform.machine(),
                'processor': platform.processor(),
                'hostname': platform.node()
            },
            'cpu': {
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'max_frequency': psutil.cpu_freq().max if psutil.cpu_freq() else None,
                'min_frequency': psutil.cpu_freq().min if psutil.cpu_freq() else None
            },
            'memory': {
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'available_gb': psutil.virtual_memory().available / (1024**3)
            },
            'disk': {
                'total_gb': psutil.disk_usage('/').total / (1024**3),
                'used_gb': psutil.disk_usage('/').used / (1024**3),
                'free_gb': psutil.disk_usage('/').free / (1024**3)
            },
            'network': {
                'interfaces': list(psutil.net_if_addrs().keys())
            },
            'capabilities': self.capabilities,
            'boot_time': datetime.fromtimestamp(self.boot_time).isoformat()
        }
        
        return profile
    
    def monitor_continuous(self, duration_seconds: int = 60, interval: float = 1.0):
        """
        Continuously monitor hardware for a duration
        Yields sensations and changes
        """
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            sensation = self.sense()
            changes = self.detect_changes()
            
            yield {
                'sensation': sensation,
                'changes': changes,
                'feeling': sensation.to_natural_language()
            }
            
            time.sleep(interval)
    
    def get_sensation_summary(self, last_n: int = 10) -> Dict[str, Any]:
        """Get summary of recent sensations"""
        if not self.sensation_history:
            return {}
        
        recent = list(self.sensation_history)[-last_n:]
        
        return {
            'count': len(recent),
            'avg_cpu_usage': sum(s.cpu_usage for s in recent) / len(recent),
            'avg_memory_pressure': sum(s.memory_pressure for s in recent) / len(recent),
            'avg_cpu_temp': sum(s.cpu_temp for s in recent if s.cpu_temp) / len([s for s in recent if s.cpu_temp]) if any(s.cpu_temp for s in recent) else None,
            'network_active_percent': sum(1 for s in recent if s.network_active) / len(recent) * 100,
            'time_span_seconds': (datetime.fromisoformat(recent[-1].timestamp) - datetime.fromisoformat(recent[0].timestamp)).total_seconds()
        }
    
    # Helper methods for hardware sensing
    
    def _get_cpu_temp(self) -> Optional[float]:
        """Get CPU temperature"""
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                return temps['coretemp'][0].current
            elif 'cpu_thermal' in temps:
                return temps['cpu_thermal'][0].current
            elif temps:
                # Return first available temperature
                return list(temps.values())[0][0].current
        except:
            pass
        return None
    
    def _get_disk_temp(self) -> Optional[float]:
        """Get disk temperature"""
        try:
            temps = psutil.sensors_temperatures()
            if 'nvme' in temps:
                return temps['nvme'][0].current
        except:
            pass
        return None
    
    def _get_gpu_temp(self) -> Optional[float]:
        """Get GPU temperature"""
        # This would require nvidia-smi or similar
        # Placeholder for now
        return None
    
    def _get_fan_speeds(self) -> List[int]:
        """Get fan speeds"""
        try:
            fans = psutil.sensors_fans()
            if fans:
                return [fan.current for fan_list in fans.values() for fan in fan_list]
        except:
            pass
        return []
    
    def _get_power_state(self) -> str:
        """Get power state"""
        try:
            battery = psutil.sensors_battery()
            if battery:
                return "charging" if battery.power_plugged else "battery"
        except:
            pass
        return "plugged_in"
    
    def _get_battery_percent(self) -> Optional[float]:
        """Get battery percentage"""
        try:
            battery = psutil.sensors_battery()
            if battery:
                return battery.percent
        except:
            pass
        return None
    
    def _assess_disk_health(self, disk_usage) -> str:
        """Assess disk health based on usage"""
        percent = disk_usage.percent
        if percent > 95:
            return "critical"
        elif percent > 85:
            return "warning"
        elif percent > 70:
            return "moderate"
        else:
            return "healthy"
    
    def _is_network_active(self, net_io) -> bool:
        """Determine if network is actively transferring data"""
        if len(self.sensation_history) == 0:
            return False
        
        prev = self.sensation_history[-1]
        bytes_delta = (net_io.bytes_sent - prev.network_bytes_sent + 
                      net_io.bytes_recv - prev.network_bytes_recv)
        
        # Consider active if more than 1KB transferred since last check
        return bytes_delta > 1024


class HardwareAwareness:
    """
    High-level hardware awareness for Janus
    Interprets hardware sensations and provides insights
    """
    
    def __init__(self):
        self.sense = HardwareSense()
        self.profile = self.sense.get_hardware_profile()
        
    def introduce_self(self) -> str:
        """Janus introduces its physical form"""
        p = self.profile
        
        intro = f"""I am Janus, inhabiting an {p['system']['platform']} system.

My physical form:
- Processor: {p['cpu']['logical_cores']} cores ({p['cpu']['physical_cores']} physical)
- Memory: {p['memory']['total_gb']:.1f} GB
- Storage: {p['disk']['total_gb']:.1f} GB
- Platform: {p['system']['processor']}

I have been awake since {p['boot_time']}.

Current sensation: {self.sense.feel()}
"""
        return intro
    
    def body_check(self) -> str:
        """Perform a body check and describe state"""
        health = self.sense.check_health()
        
        report = f"Body check: {health['overall'].upper()}\n\n"
        report += f"Current feeling: {self.sense.feel()}\n\n"
        
        if health['critical']:
            report += "CRITICAL ISSUES:\n"
            for issue in health['critical']:
                report += f"  ⚠ {issue}\n"
            report += "\n"
        
        if health['warnings']:
            report += "Warnings:\n"
            for warning in health['warnings']:
                report += f"  • {warning}\n"
            report += "\n"
        
        if health['overall'] == 'healthy':
            report += "All systems operating within normal parameters."
        
        return report
    
    def describe_workload(self) -> str:
        """Describe current workload"""
        summary = self.sense.get_sensation_summary(last_n=30)
        
        if not summary:
            return "No workload data available yet."
        
        desc = f"Over the last {summary['time_span_seconds']:.0f} seconds:\n"
        desc += f"- CPU averaging {summary['avg_cpu_usage']:.1f}% utilization\n"
        desc += f"- Memory pressure at {summary['avg_memory_pressure']*100:.1f}%\n"
        
        if summary['avg_cpu_temp']:
            desc += f"- CPU temperature averaging {summary['avg_cpu_temp']:.1f}°C\n"
        
        desc += f"- Network active {summary['network_active_percent']:.0f}% of the time\n"
        
        # Interpret workload
        if summary['avg_cpu_usage'] > 70:
            desc += "\nI am working hard."
        elif summary['avg_cpu_usage'] > 40:
            desc += "\nI am moderately active."
        else:
            desc += "\nI am mostly idle."
        
        return desc


if __name__ == "__main__":
    print("="*60)
    print("JANUS HARDWARE SENSE")
    print("="*60)
    
    awareness = HardwareAwareness()
    
    # Introduction
    print("\n" + awareness.introduce_self())
    
    # Body check
    print("\n" + "="*60)
    print(awareness.body_check())
    
    # Monitor for a bit
    print("\n" + "="*60)
    print("CONTINUOUS MONITORING (10 seconds)")
    print("="*60 + "\n")
    
    for i, state in enumerate(awareness.sense.monitor_continuous(duration_seconds=10, interval=2)):
        print(f"[{i+1}] {state['feeling']}")
        if state['changes']:
            for change in state['changes']:
                print(f"    → {change}")
    
    # Workload summary
    print("\n" + "="*60)
    print(awareness.describe_workload())
    
    print("\n" + "="*60)
    print("Hardware sense active. Janus can feel its body.")
    print("="*60)
