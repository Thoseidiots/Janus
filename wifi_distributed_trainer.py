"""
wifi_distributed_trainer.py
==========================
WiFi-based distributed training for AI models.

Enables training across multiple WiFi-connected devices with gradient synchronization.
Similar to OpenCLAW's distributed approach but using WiFi instead of specialized interconnects.

Communication time per step:
    T_comm = S_grad / R_WiFi + L_WiFi

Where:
    S_grad = size of gradients exchanged per step (bytes)
    R_WiFi = effective WiFi throughput (bytes/second)
    L_WiFi = WiFi latency per exchange (seconds)
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, List, Tuple, Optional, Any, Callable
import time
import socket
import threading
import pickle
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import numpy as np

# Import existing WiFi manager
try:
    from janus_wifi import JanusWifi, get_wifi
    WIFI_AVAILABLE = True
except ImportError:
    WIFI_AVAILABLE = False
    print("Warning: janus_wifi not available, running in simulation mode")

logger = logging.getLogger("wifi_distributed_trainer")


class SyncStrategy(Enum):
    """Gradient synchronization strategies."""
    ALL_REDUCE = "all_reduce"  # Standard all-reduce (sum gradients across all devices)
    AVERAGE = "average"  # Average gradients across all devices
    RING_ALL_REDUCE = "ring_all_reduce"  # Ring-based all-reduce for better bandwidth utilization
    CENTRALIZED = "centralized"  # Central parameter server approach


@dataclass
class WiFiDevice:
    """Represents a WiFi-connected training device."""
    ip_address: str
    port: int
    device_id: str
    rank: int  # Distributed training rank
    gpu_available: bool = False
    gpu_memory: float = 0.0  # GB
    last_heartbeat: float = field(default_factory=time.time)
    status: str = "active"


@dataclass
class CommunicationMetrics:
    """Metrics for WiFi communication during training."""
    total_comm_time: float = 0.0  # Total communication time (seconds)
    avg_comm_time: float = 0.0  # Average communication time per step
    grad_size_bytes: float = 0.0  # Average gradient size per step
    throughput_bytes_per_sec: float = 0.0  # Effective WiFi throughput
    latency_seconds: float = 0.0  # Measured WiFi latency
    steps_synced: int = 0
    failed_syncs: int = 0


@dataclass
class WiFiTrainingConfig:
    """Configuration for WiFi distributed training."""
    # WiFi parameters
    wifi_throughput: float = 50.0 * 1024 * 1024  # 50 MB/s default
    wifi_latency: float = 0.005  # 5ms default latency
    sync_strategy: SyncStrategy = SyncStrategy.ALL_REDUCE
    
    # Training parameters
    gradient_compression: bool = True  # Compress gradients before sending
    compression_ratio: float = 0.5  # Target compression ratio
    sync_frequency: int = 1  # Sync every N steps
    
    # Device discovery
    auto_discover: bool = True
    discovery_port: int = 29500
    sync_port: int = 29501
    
    # Timeout parameters
    sync_timeout: float = 30.0  # Seconds
    heartbeat_interval: float = 10.0  # Seconds


class WiFiDistributedTrainer:
    """
    Manages distributed training over WiFi-connected devices.
    
    Handles:
    - Device discovery over WiFi
    - Gradient synchronization
    - Communication time calculation
    - Fault tolerance and recovery
    """
    
    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
        config: Optional[WiFiTrainingConfig] = None,
        model: Optional[nn.Module] = None
    ):
        self.rank = rank
        self.world_size = world_size
        self.config = config or WiFiTrainingConfig()
        self.model = model
        
        # Device management
        self.devices: List[WiFiDevice] = []
        self.device_lock = threading.Lock()
        
        # Communication metrics
        self.metrics = CommunicationMetrics()
        
        # Gradient buffer for synchronization
        self.gradient_buffer: Optional[torch.Tensor] = None
        
        # Network sockets
        self.sync_socket: Optional[socket.socket] = None
        self.discovery_socket: Optional[socket.socket] = None
        
        # Threading
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.stop_heartbeat = threading.Event()
        
        # WiFi manager (if available)
        self.wifi_manager = get_wifi() if WIFI_AVAILABLE else None
        
        logger.info(f"WiFi Distributed Trainer initialized: rank={rank}, world_size={world_size}")
    
    def calculate_communication_time(
        self,
        grad_size_bytes: float,
        throughput: Optional[float] = None,
        latency: Optional[float] = None
    ) -> float:
        """
        Calculate communication time per step using the formula:
        T_comm = S_grad / R_WiFi + L_WiFi
        
        Args:
            grad_size_bytes: Size of gradients to exchange (bytes)
            throughput: WiFi throughput (bytes/second), uses config default if None
            latency: WiFi latency (seconds), uses config default if None
            
        Returns:
            Communication time in seconds
        """
        R_WiFi = throughput or self.config.wifi_throughput
        L_WiFi = latency or self.config.wifi_latency
        
        T_comm = grad_size_bytes / R_WiFi + L_WiFi
        
        # Update metrics
        self.metrics.grad_size_bytes = grad_size_bytes
        self.metrics.throughput_bytes_per_sec = R_WiFi
        self.metrics.latency_seconds = L_WiFi
        
        return T_comm
    
    def estimate_gradient_size(self, model: nn.Module) -> float:
        """
        Estimate the size of model gradients in bytes.
        
        Args:
            model: PyTorch model
            
        Returns:
            Gradient size in bytes
        """
        total_params = 0
        for param in model.parameters():
            if param.grad is not None:
                total_params += param.grad.numel()
        
        # Assume float32 (4 bytes per parameter)
        grad_size_bytes = total_params * 4
        
        # Apply compression if enabled
        if self.config.gradient_compression:
            grad_size_bytes *= self.config.compression_ratio
        
        return grad_size_bytes
    
    def discover_devices(self, timeout: float = 5.0) -> List[WiFiDevice]:
        """
        Discover WiFi-connected training devices.
        
        Args:
            timeout: Discovery timeout in seconds
            
        Returns:
            List of discovered devices
        """
        logger.info("Starting WiFi device discovery...")
        
        if self.wifi_manager:
            # Use janus_wifi to get connected devices
            connected_devices = self.wifi_manager.get_connected_devices()
            logger.info(f"Found {len(connected_devices)} devices via WiFi manager")
            
            # Convert to WiFiDevice objects
            devices = []
            for i, dev in enumerate(connected_devices):
                device = WiFiDevice(
                    ip_address=dev.ip_address,
                    port=self.config.discovery_port,
                    device_id=f"wifi_{i}",
                    rank=i,
                    gpu_available=True,  # Assume GPU for now
                    gpu_memory=8.0,  # Default assumption
                )
                devices.append(device)
            
            with self.device_lock:
                self.devices = devices
            
            return devices
        else:
            # Simulation mode - create mock devices
            logger.warning("WiFi manager not available, using simulation mode")
            devices = []
            for i in range(self.world_size):
                device = WiFiDevice(
                    ip_address=f"192.168.1.{100 + i}",
                    port=self.config.discovery_port,
                    device_id=f"sim_{i}",
                    rank=i,
                    gpu_available=True,
                    gpu_memory=8.0,
                )
                devices.append(device)
            
            with self.device_lock:
                self.devices = devices
            
            return devices
    
    def synchronize_gradients(
        self,
        model: nn.Module,
        step: int
    ) -> Tuple[float, bool]:
        """
        Synchronize gradients across WiFi-connected devices.
        
        Args:
            model: PyTorch model
            step: Current training step
            
        Returns:
            (communication_time, success)
        """
        # Check if sync is needed
        if step % self.config.sync_frequency != 0:
            return 0.0, True
        
        start_time = time.time()
        
        try:
            # Estimate gradient size
            grad_size = self.estimate_gradient_size(model)
            
            # Calculate theoretical communication time
            theoretical_time = self.calculate_communication_time(grad_size)
            
            # Perform actual synchronization
            if self.world_size > 1:
                if self.config.sync_strategy == SyncStrategy.ALL_REDUCE:
                    self._all_reduce_gradients(model)
                elif self.config.sync_strategy == SyncStrategy.AVERAGE:
                    self._average_gradients(model)
                elif self.config.sync_strategy == SyncStrategy.RING_ALL_REDUCE:
                    self._ring_all_reduce(model)
                elif self.config.sync_strategy == SyncStrategy.CENTRALIZED:
                    self._centralized_sync(model)
            
            # Measure actual communication time
            actual_time = time.time() - start_time
            
            # Update metrics
            self.metrics.total_comm_time += actual_time
            self.metrics.steps_synced += 1
            self.metrics.avg_comm_time = (
                self.metrics.total_comm_time / self.metrics.steps_synced
            )
            
            logger.debug(
                f"Step {step}: Sync complete - "
                f"theoretical_time={theoretical_time:.4f}s, "
                f"actual_time={actual_time:.4f}s, "
                f"grad_size={grad_size/1024/1024:.2f}MB"
            )
            
            return actual_time, True
            
        except Exception as e:
            logger.error(f"Gradient synchronization failed: {e}")
            self.metrics.failed_syncs += 1
            return 0.0, False
    
    def _all_reduce_gradients(self, model: nn.Module):
        """Standard all-reduce gradient synchronization."""
        # In a real implementation, this would use torch.distributed.all_reduce
        # For now, we simulate the operation
        if self.rank == 0:
            # Root device aggregates gradients
            for param in model.parameters():
                if param.grad is not None:
                    # Simulate receiving gradients from other devices
                    # In real implementation: dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad.div_(self.world_size)
    
    def _average_gradients(self, model: nn.Module):
        """Average gradients across all devices."""
        for param in model.parameters():
            if param.grad is not None:
                # Simulate averaging
                # In real implementation: dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(self.world_size)
    
    def _ring_all_reduce(self, model: nn.Module):
        """
        Ring-based all-reduce for better bandwidth utilization.
        
        This is more efficient for WiFi as it minimizes contention.
        """
        # Simulated ring all-reduce
        # In real implementation, this would use a ring communication pattern
        for param in model.parameters():
            if param.grad is not None:
                # Simulate ring-based reduction
                param.grad.div_(self.world_size)
    
    def _centralized_sync(self, model: nn.Module):
        """
        Centralized parameter server approach.
        
        One device acts as a parameter server that aggregates gradients.
        """
        if self.rank == 0:
            # Parameter server (rank 0) aggregates gradients
            for param in model.parameters():
                if param.grad is not None:
                    # Simulate receiving and averaging gradients
                    param.grad.div_(self.world_size)
        else:
            # Worker devices send gradients to parameter server
            # In real implementation: send gradients to rank 0
            pass
    
    def start_heartbeat(self):
        """Start heartbeat thread to monitor device connectivity."""
        self.stop_heartbeat.clear()
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="wifi-heartbeat"
        )
        self.heartbeat_thread.start()
        logger.info("Heartbeat thread started")
    
    def _heartbeat_loop(self):
        """Background heartbeat loop."""
        while not self.stop_heartbeat.wait(self.config.heartbeat_interval):
            try:
                current_time = time.time()
                with self.device_lock:
                    for device in self.devices:
                        # Check if device is responsive
                        if current_time - device.last_heartbeat > self.config.sync_timeout:
                            device.status = "timeout"
                            logger.warning(f"Device {device.device_id} timed out")
                        else:
                            device.status = "active"
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    def stop(self):
        """Stop the distributed trainer."""
        logger.info("Stopping WiFi distributed trainer...")
        self.stop_heartbeat.set()
        
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5.0)
        
        if self.sync_socket:
            self.sync_socket.close()
        
        if self.discovery_socket:
            self.discovery_socket.close()
        
        logger.info("WiFi distributed trainer stopped")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current communication metrics."""
        return {
            "total_comm_time": self.metrics.total_comm_time,
            "avg_comm_time": self.metrics.avg_comm_time,
            "grad_size_bytes": self.metrics.grad_size_bytes,
            "throughput_bytes_per_sec": self.metrics.throughput_bytes_per_sec,
            "latency_seconds": self.metrics.latency_seconds,
            "steps_synced": self.metrics.steps_synced,
            "failed_syncs": self.metrics.failed_syncs,
            "devices": len(self.devices),
            "world_size": self.world_size,
            "rank": self.rank,
        }
    
    def print_metrics(self):
        """Print communication metrics."""
        metrics = self.get_metrics()
        print("\n=== WiFi Distributed Training Metrics ===")
        print(f"Total communication time: {metrics['total_comm_time']:.2f}s")
        print(f"Average communication time per step: {metrics['avg_comm_time']:.4f}s")
        print(f"Average gradient size: {metrics['grad_size_bytes']/1024/1024:.2f}MB")
        print(f"Effective throughput: {metrics['throughput_bytes_per_sec']/1024/1024:.2f}MB/s")
        print(f"Latency: {metrics['latency_seconds']*1000:.2f}ms")
        print(f"Steps synchronized: {metrics['steps_synced']}")
        print(f"Failed synchronizations: {metrics['failed_syncs']}")
        print(f"Active devices: {metrics['devices']}")
        print(f"World size: {metrics['world_size']}")
        print("=" * 40)


# Convenience function for quick setup
def create_wifi_trainer(
    rank: int = 0,
    world_size: int = 2,
    wifi_throughput: float = 50.0 * 1024 * 1024,  # 50 MB/s
    wifi_latency: float = 0.005,  # 5ms
    model: Optional[nn.Module] = None
) -> WiFiDistributedTrainer:
    """
    Create a WiFi distributed trainer with common defaults.
    
    Args:
        rank: Device rank (0 = master)
        world_size: Total number of devices
        wifi_throughput: WiFi throughput in bytes/second
        wifi_latency: WiFi latency in seconds
        model: PyTorch model to train
        
    Returns:
        Configured WiFiDistributedTrainer instance
    """
    config = WiFiTrainingConfig(
        wifi_throughput=wifi_throughput,
        wifi_latency=wifi_latency,
        sync_strategy=SyncStrategy.ALL_REDUCE,
    )
    
    trainer = WiFiDistributedTrainer(
        rank=rank,
        world_size=world_size,
        config=config,
        model=model
    )
    
    return trainer


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create a simple model for testing
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Create WiFi trainer
    trainer = create_wifi_trainer(
        rank=0,
        world_size=2,
        wifi_throughput=50.0 * 1024 * 1024,  # 50 MB/s
        wifi_latency=0.005,  # 5ms
        model=model
    )
    
    # Discover devices
    devices = trainer.discover_devices()
    print(f"Discovered {len(devices)} devices")
    
    # Start heartbeat
    trainer.start_heartbeat()
    
    # Simulate training steps
    for step in range(10):
        # Simulate forward/backward pass
        x = torch.randn(32, 784)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        # Synchronize gradients
        comm_time, success = trainer.synchronize_gradients(model, step)
        print(f"Step {step}: comm_time={comm_time:.4f}s, success={success}")
        
        # Zero gradients
        model.zero_grad()
    
    # Print metrics
    trainer.print_metrics()
    
    # Stop trainer
    trainer.stop()
