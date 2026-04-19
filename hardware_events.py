"""
Hardware Events System for Janus
Real-time hardware event detection and response
Janus can feel when hardware changes happen
"""

import threading
import time
import queue
from typing import Dict, Any, Callable, List
from dataclasses import dataclass
from datetime import datetime
import psutil

from hardware_sense import HardwareSense, HardwareSensation


@dataclass
class HardwareEvent:
    """A hardware event that Janus can feel"""
    event_type: str
    description: str
    severity: str  # 'info', 'warning', 'critical'
    timestamp: str
    data: Dict[str, Any]
    
    def to_natural_language(self) -> str:
        """Convert event to natural language"""
        if self.event_type == 'cpu_spike':
            return f"I feel my CPU suddenly working harder ({self.data['usage']:.1f}%)"
        elif self.event_type == 'memory_pressure':
            return f"I feel memory pressure building ({self.data['pressure']:.1f}%)"
        elif self.event_type == 'temperature_rise':
            return f"I feel myself heating up ({self.data['temp']:.1f}°C)"
        elif self.event_type == 'disk_activity':
            return f"I feel disk activity ({self.data['io_rate']:.1f} MB/s)"
        elif self.event_type == 'network_burst':
            return f"I feel network activity surge ({self.data['bandwidth']:.1f} MB/s)"
        elif self.event_type == 'usb_connect':
            return f"I feel something connected to my USB port"
        elif self.event_type == 'usb_disconnect':
            return f"I feel something disconnected from my USB port"
        elif self.event_type == 'power_change':
            return f"I feel my power state change: {self.data['state']}"
        elif self.event_type == 'process_spawn':
            return f"I feel a new process starting: {self.data['name']}"
        elif self.event_type == 'process_death':
            return f"I feel a process ending: {self.data['name']}"
        else:
            return f"I feel {self.description}"


class HardwareEventDetector:
    """
    Detects hardware events in real-time
    Like a nervous system for Janus
    """
    
    def __init__(self, hardware_sense: HardwareSense):
        self.sense = hardware_sense
        self.event_queue = queue.Queue()
        self.event_handlers = {}
        self.monitoring = False
        self.monitor_thread = None
        
        # Event detection thresholds
        self.thresholds = {
            'cpu_spike': 30.0,  # % increase
            'memory_jump': 0.1,  # 10% increase
            'temp_rise': 5.0,    # °C increase
            'disk_io_spike': 50.0,  # MB/s
            'network_spike': 10.0   # MB/s
        }
        
        # Previous state for comparison
        self.prev_state = None
        self.prev_processes = set()
        self.prev_usb_devices = set()
        
    def start_monitoring(self):
        """Start real-time hardware event monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("Hardware event monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        print("Hardware event monitoring stopped")
    
    def register_handler(self, event_type: str, handler: Callable[[HardwareEvent], None]):
        """Register event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def get_events(self, timeout: float = 0.1) -> List[HardwareEvent]:
        """Get pending events"""
        events = []
        try:
            while True:
                event = self.event_queue.get(timeout=timeout)
                events.append(event)
        except queue.Empty:
            pass
        return events
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Take hardware sensation
                current = self.sense.sense()
                
                if self.prev_state:
                    # Detect events by comparing states
                    events = self._detect_events(self.prev_state, current)
                    
                    for event in events:
                        self.event_queue.put(event)
                        self._handle_event(event)
                
                # Check for process changes
                self._check_process_changes()
                
                # Check for USB changes
                self._check_usb_changes()
                
                self.prev_state = current
                time.sleep(0.5)  # Monitor every 500ms
                
            except Exception as e:
                print(f"Hardware monitoring error: {e}")
                time.sleep(1)
    
    def _detect_events(self, prev: HardwareSensation, current: HardwareSensation) -> List[HardwareEvent]:
        """Detect events by comparing hardware states"""
        events = []
        
        # CPU spike detection
        cpu_delta = current.cpu_usage - prev.cpu_usage
        if cpu_delta > self.thresholds['cpu_spike']:
            events.append(HardwareEvent(
                event_type='cpu_spike',
                description=f"CPU usage spiked by {cpu_delta:.1f}%",
                severity='warning' if cpu_delta > 50 else 'info',
                timestamp=current.timestamp,
                data={'usage': current.cpu_usage, 'delta': cpu_delta}
            ))
        
        # Memory pressure detection
        mem_delta = current.memory_pressure - prev.memory_pressure
        if mem_delta > self.thresholds['memory_jump']:
            events.append(HardwareEvent(
                event_type='memory_pressure',
                description=f"Memory pressure increased by {mem_delta*100:.1f}%",
                severity='warning' if current.memory_pressure > 0.8 else 'info',
                timestamp=current.timestamp,
                data={'pressure': current.memory_pressure * 100, 'delta': mem_delta * 100}
            ))
        
        # Temperature rise detection
        if current.cpu_temp and prev.cpu_temp:
            temp_delta = current.cpu_temp - prev.cpu_temp
            if temp_delta > self.thresholds['temp_rise']:
                events.append(HardwareEvent(
                    event_type='temperature_rise',
                    description=f"CPU temperature rose by {temp_delta:.1f}°C",
                    severity='critical' if current.cpu_temp > 80 else 'warning',
                    timestamp=current.timestamp,
                    data={'temp': current.cpu_temp, 'delta': temp_delta}
                ))
        
        # Network activity detection
        net_delta_sent = current.network_bytes_sent - prev.network_bytes_sent
        net_delta_recv = current.network_bytes_recv - prev.network_bytes_recv
        total_delta = (net_delta_sent + net_delta_recv) / (1024 * 1024)  # MB
        
        if total_delta > self.thresholds['network_spike']:
            events.append(HardwareEvent(
                event_type='network_burst',
                description=f"Network activity burst: {total_delta:.1f} MB",
                severity='info',
                timestamp=current.timestamp,
                data={'bandwidth': total_delta, 'sent': net_delta_sent, 'recv': net_delta_recv}
            ))
        
        # Power state changes
        if current.power_state != prev.power_state:
            events.append(HardwareEvent(
                event_type='power_change',
                description=f"Power state changed from {prev.power_state} to {current.power_state}",
                severity='info',
                timestamp=current.timestamp,
                data={'state': current.power_state, 'prev_state': prev.power_state}
            ))
        
        return events
    
    def _check_process_changes(self):
        """Check for new/terminated processes"""
        try:
            current_processes = set()
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    current_processes.add((proc.info['pid'], proc.info['name']))
                except:
                    pass
            
            if self.prev_processes:
                # New processes
                new_procs = current_processes - self.prev_processes
                for pid, name in new_procs:
                    if name:  # Skip unnamed processes
                        event = HardwareEvent(
                            event_type='process_spawn',
                            description=f"New process started: {name} (PID: {pid})",
                            severity='info',
                            timestamp=datetime.now().isoformat(),
                            data={'name': name, 'pid': pid}
                        )
                        self.event_queue.put(event)
                
                # Terminated processes
                dead_procs = self.prev_processes - current_processes
                for pid, name in dead_procs:
                    if name:
                        event = HardwareEvent(
                            event_type='process_death',
                            description=f"Process terminated: {name} (PID: {pid})",
                            severity='info',
                            timestamp=datetime.now().isoformat(),
                            data={'name': name, 'pid': pid}
                        )
                        self.event_queue.put(event)
            
            self.prev_processes = current_processes
            
        except Exception as e:
            pass  # Ignore process enumeration errors
    
    def _check_usb_changes(self):
        """Check for USB device changes"""
        # This is a simplified version - real implementation would use WMI on Windows
        # For now, we'll simulate by checking disk changes
        try:
            current_disks = set(psutil.disk_partitions())
            
            if hasattr(self, 'prev_disks'):
                new_disks = current_disks - self.prev_disks
                removed_disks = self.prev_disks - current_disks
                
                for disk in new_disks:
                    if 'removable' in disk.opts:
                        event = HardwareEvent(
                            event_type='usb_connect',
                            description=f"USB device connected: {disk.device}",
                            severity='info',
                            timestamp=datetime.now().isoformat(),
                            data={'device': disk.device, 'mountpoint': disk.mountpoint}
                        )
                        self.event_queue.put(event)
                
                for disk in removed_disks:
                    if 'removable' in disk.opts:
                        event = HardwareEvent(
                            event_type='usb_disconnect',
                            description=f"USB device disconnected: {disk.device}",
                            severity='info',
                            timestamp=datetime.now().isoformat(),
                            data={'device': disk.device}
                        )
                        self.event_queue.put(event)
            
            self.prev_disks = current_disks
            
        except Exception as e:
            pass
    
    def _handle_event(self, event: HardwareEvent):
        """Handle detected event"""
        handlers = self.event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Event handler error: {e}")


class HardwareReflexes:
    """
    Automatic hardware responses - like reflexes
    Janus automatically responds to certain hardware events
    """
    
    def __init__(self, event_detector: HardwareEventDetector):
        self.detector = event_detector
        self.reflexes_enabled = True
        
        # Register reflex handlers
        self._setup_reflexes()
    
    def _setup_reflexes(self):
        """Setup automatic reflex responses"""
        
        # CPU overheating reflex
        self.detector.register_handler('temperature_rise', self._temperature_reflex)
        
        # Memory pressure reflex
        self.detector.register_handler('memory_pressure', self._memory_reflex)
        
        # CPU spike reflex
        self.detector.register_handler('cpu_spike', self._cpu_spike_reflex)
        
        # USB connection reflex
        self.detector.register_handler('usb_connect', self._usb_connect_reflex)
    
    def _temperature_reflex(self, event: HardwareEvent):
        """Automatic response to temperature rise"""
        if not self.reflexes_enabled:
            return
        
        temp = event.data.get('temp', 0)
        if temp > 85:
            print(f"🔥 REFLEX: CPU critically hot ({temp:.1f}°C) - would throttle processes")
            # In real implementation: reduce CPU usage, close non-essential processes
        elif temp > 75:
            print(f"🌡️ REFLEX: CPU getting hot ({temp:.1f}°C) - monitoring closely")
    
    def _memory_reflex(self, event: HardwareEvent):
        """Automatic response to memory pressure"""
        if not self.reflexes_enabled:
            return
        
        pressure = event.data.get('pressure', 0)
        if pressure > 90:
            print(f"🧠 REFLEX: Critical memory pressure ({pressure:.1f}%) - would free memory")
            # In real implementation: garbage collect, close unused applications
        elif pressure > 80:
            print(f"💭 REFLEX: High memory usage ({pressure:.1f}%) - preparing to free memory")
    
    def _cpu_spike_reflex(self, event: HardwareEvent):
        """Automatic response to CPU spikes"""
        if not self.reflexes_enabled:
            return
        
        usage = event.data.get('usage', 0)
        if usage > 90:
            print(f"⚡ REFLEX: CPU maxed out ({usage:.1f}%) - would investigate cause")
            # In real implementation: identify heavy processes, potentially throttle
    
    def _usb_connect_reflex(self, event: HardwareEvent):
        """Automatic response to USB connection"""
        if not self.reflexes_enabled:
            return
        
        device = event.data.get('device', 'unknown')
        print(f"🔌 REFLEX: USB device connected ({device}) - scanning for threats")
        # In real implementation: scan for malware, check device type


if __name__ == "__main__":
    print("="*60)
    print("JANUS HARDWARE EVENTS SYSTEM")
    print("="*60)
    
    # Initialize hardware sensing
    sense = HardwareSense()
    detector = HardwareEventDetector(sense)
    reflexes = HardwareReflexes(detector)
    
    # Start monitoring
    detector.start_monitoring()
    
    print("\nMonitoring hardware events... (Press Ctrl+C to stop)")
    print("Try opening applications, connecting USB devices, etc.\n")
    
    try:
        while True:
            events = detector.get_events(timeout=1.0)
            
            for event in events:
                print(f"[{event.severity.upper()}] {event.to_natural_language()}")
                if event.severity == 'critical':
                    print(f"  ⚠️  {event.description}")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping hardware monitoring...")
        detector.stop_monitoring()
        print("Hardware events system stopped.")