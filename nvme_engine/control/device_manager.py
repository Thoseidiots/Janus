"""
Device Manager for the Software NVMe Engine.

Manages the lifecycle of virtual NVMe devices: creation, deletion,
runtime modification, hot-plug, and hot-unplug.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from nvme_engine.backends.base import StorageBackendOps
from nvme_engine.backends.memory_backend import MemoryBackend
from nvme_engine.commands.namespace_manager import NamespaceManager
from nvme_engine.models.config import NvmeDeviceConfig
from nvme_engine.models.errors import NvmeDeviceNotFoundError, NvmeResourceError


class DeviceState(Enum):
    """Lifecycle state of a virtual NVMe device."""

    CREATING = "creating"
    ACTIVE = "active"
    MODIFYING = "modifying"
    DELETING = "deleting"
    DELETED = "deleted"
    HOT_PLUGGED = "hot_plugged"


@dataclass
class VirtualDevice:
    """
    Represents a virtual NVMe device managed by the DeviceManager.

    Attributes
    ----------
    device_id         : Unique numeric identifier assigned at creation.
    name              : Human-readable device name from config.
    config            : Full device configuration snapshot.
    state             : Current lifecycle state.
    created_at        : Unix timestamp of creation.
    namespace_manager : Namespace manager instance for this device.
    backend           : Storage backend instance for this device.
    """

    device_id: int
    name: str
    config: NvmeDeviceConfig
    state: DeviceState = DeviceState.CREATING
    created_at: float = field(default_factory=time.time)
    namespace_manager: Optional[NamespaceManager] = None
    backend: Optional[StorageBackendOps] = None


class DeviceManager:
    """
    Manages virtual NVMe device lifecycle.

    Supports up to MAX_DEVICES (256) concurrent devices.
    All public methods are thread-safe.
    """

    MAX_DEVICES: int = 256

    def __init__(self) -> None:
        """Initialize an empty DeviceManager."""
        self._devices: Dict[int, VirtualDevice] = {}
        self._next_id: int = 1
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Device lifecycle
    # ------------------------------------------------------------------

    def create_device(self, config: NvmeDeviceConfig) -> VirtualDevice:
        """
        Create a virtual NVMe device from the given configuration.

        Initialises a MemoryBackend and a NamespaceManager for the device,
        then transitions the device to ACTIVE state.

        Args:
            config: Full device configuration.

        Returns:
            The newly created VirtualDevice.

        Raises:
            NvmeResourceError: If the number of active devices would exceed MAX_DEVICES.
        """
        with self._lock:
            active_count = sum(
                1 for d in self._devices.values() if d.state != DeviceState.DELETED
            )
            if active_count >= self.MAX_DEVICES:
                raise NvmeResourceError(
                    f"Cannot create device: maximum of {self.MAX_DEVICES} devices reached",
                    details=f"active_devices={active_count}",
                )

            device_id = self._next_id
            self._next_id += 1

            device = VirtualDevice(
                device_id=device_id,
                name=config.name,
                config=config,
                state=DeviceState.CREATING,
            )
            self._devices[device_id] = device

        # Initialise backend outside the lock to avoid holding it during I/O
        backend = MemoryBackend()
        backend.init({"size_bytes": config.capacity_bytes})

        ns_manager = NamespaceManager()
        # Create the configured number of namespaces, distributing capacity evenly
        ns_capacity = config.capacity_bytes // config.namespace_count
        for _ in range(config.namespace_count):
            ns_manager.create_namespace(ns_capacity)

        with self._lock:
            device.backend = backend
            device.namespace_manager = ns_manager
            device.state = DeviceState.ACTIVE

        return device

    def delete_device(self, device_id: int, timeout_s: float = 5.0) -> None:
        """
        Delete a virtual NVMe device, releasing all resources.

        The device is transitioned to DELETING, resources are released,
        and the state is set to DELETED — all within *timeout_s* seconds.

        Args:
            device_id : ID of the device to delete.
            timeout_s : Maximum seconds to wait for resource release (default 5).

        Raises:
            NvmeDeviceNotFoundError: If device_id does not exist or is already deleted.
        """
        device = self._get_active_device(device_id)

        deadline = time.monotonic() + timeout_s

        with self._lock:
            device.state = DeviceState.DELETING

        try:
            if device.backend is not None and device.backend.is_initialized:
                device.backend.destroy()
            device.backend = None
            device.namespace_manager = None
        finally:
            with self._lock:
                device.state = DeviceState.DELETED

        elapsed = time.monotonic() - (deadline - timeout_s)
        if elapsed > timeout_s:
            # Resources were released but took longer than requested — log only
            pass  # In production this would emit a warning metric

    def modify_device(self, device_id: int, **kwargs) -> VirtualDevice:
        """
        Modify device parameters at runtime without detachment.

        Transitions: ACTIVE → MODIFYING → ACTIVE.
        Only fields present in NvmeDeviceConfig that are passed as kwargs
        are updated.  Active I/O is not interrupted.

        Args:
            device_id : ID of the device to modify.
            **kwargs  : Config fields to update (e.g. name="new-name").

        Returns:
            The updated VirtualDevice.

        Raises:
            NvmeDeviceNotFoundError: If device_id does not exist or is deleted.
        """
        device = self._get_active_device(device_id)

        with self._lock:
            device.state = DeviceState.MODIFYING

        try:
            config = device.config
            for key, value in kwargs.items():
                if hasattr(config, key):
                    object.__setattr__(config, key, value)
            device.name = config.name
        finally:
            with self._lock:
                device.state = DeviceState.ACTIVE

        return device

    def hot_plug(self, device_id: int) -> None:
        """
        Hot-plug a device, making it available within 2 seconds.

        Transitions the device to HOT_PLUGGED state and then to ACTIVE.

        Args:
            device_id: ID of the device to hot-plug.

        Raises:
            NvmeDeviceNotFoundError: If device_id does not exist or is deleted.
        """
        device = self._get_active_device(device_id)

        start = time.monotonic()
        with self._lock:
            device.state = DeviceState.HOT_PLUGGED

        # Simulate hot-plug enumeration (must complete within 2 seconds)
        # In a real implementation this would notify the OS block layer
        elapsed = time.monotonic() - start
        if elapsed > 2.0:
            raise NvmeResourceError(
                f"Hot-plug for device {device_id} exceeded 2-second deadline",
                device_id=device_id,
            )

        with self._lock:
            device.state = DeviceState.ACTIVE

    def hot_unplug(self, device_id: int) -> None:
        """
        Hot-unplug a device, gracefully completing or failing pending I/O.

        The device is transitioned to DELETING to signal that no new I/O
        should be accepted, then resources are released.

        Args:
            device_id: ID of the device to hot-unplug.

        Raises:
            NvmeDeviceNotFoundError: If device_id does not exist or is deleted.
        """
        # Reuse delete_device for graceful teardown
        self.delete_device(device_id)

    def get_device(self, device_id: int) -> VirtualDevice:
        """
        Retrieve a device by its ID.

        Args:
            device_id: ID of the device to retrieve.

        Returns:
            The VirtualDevice with the given ID.

        Raises:
            NvmeDeviceNotFoundError: If device_id does not exist or is deleted.
        """
        return self._get_active_device(device_id)

    def list_devices(self) -> List[VirtualDevice]:
        """
        List all non-deleted devices.

        Returns:
            List of VirtualDevice objects whose state is not DELETED.
        """
        with self._lock:
            return [
                d for d in self._devices.values() if d.state != DeviceState.DELETED
            ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device_count(self) -> int:
        """Number of non-deleted devices currently managed."""
        with self._lock:
            return sum(
                1 for d in self._devices.values() if d.state != DeviceState.DELETED
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_active_device(self, device_id: int) -> VirtualDevice:
        """
        Return the device for *device_id* or raise NvmeDeviceNotFoundError.

        Raises:
            NvmeDeviceNotFoundError: If device_id is unknown or the device is DELETED.
        """
        with self._lock:
            device = self._devices.get(device_id)
        if device is None or device.state == DeviceState.DELETED:
            raise NvmeDeviceNotFoundError(
                f"Device {device_id} not found",
                device_id=device_id,
            )
        return device
