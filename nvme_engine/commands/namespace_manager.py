"""
NVMe Namespace Manager.

Handles namespace lifecycle (create, delete, attach, detach) and
NVMe Reservations for multi-host access coordination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from nvme_engine.models.errors import NvmeConfigError, NvmePermissionError, NvmeDeviceNotFoundError


class ReservationType(Enum):
    WRITE_EXCLUSIVE = "write_exclusive"
    EXCLUSIVE_ACCESS = "exclusive_access"
    WRITE_EXCLUSIVE_REGISTRANTS_ONLY = "write_exclusive_registrants_only"
    EXCLUSIVE_ACCESS_REGISTRANTS_ONLY = "exclusive_access_registrants_only"


@dataclass
class Namespace:
    namespace_id: int
    capacity_bytes: int
    block_size: int = 4096
    read_only: bool = False
    active: bool = True

    @property
    def block_count(self) -> int:
        return self.capacity_bytes // self.block_size


@dataclass
class Reservation:
    holder_host_id: str
    reservation_type: ReservationType
    registrants: List[str] = field(default_factory=list)


class NamespaceManager:
    """Manages NVMe namespaces: create, delete, attach, detach."""

    def __init__(self) -> None:
        self._namespaces: Dict[int, Namespace] = {}
        self._attached: Dict[int, bool] = {}
        self._next_id: int = 1

    def create_namespace(self, capacity_bytes: int, block_size: int = 4096) -> Namespace:
        if capacity_bytes <= 0:
            raise NvmeConfigError(f"capacity_bytes must be positive, got {capacity_bytes}")
        if block_size <= 0:
            raise NvmeConfigError(f"block_size must be positive, got {block_size}")
        ns = Namespace(
            namespace_id=self._next_id,
            capacity_bytes=capacity_bytes,
            block_size=block_size,
        )
        self._namespaces[self._next_id] = ns
        self._attached[self._next_id] = False
        self._next_id += 1
        return ns

    def delete_namespace(self, namespace_id: int) -> None:
        if namespace_id not in self._namespaces:
            raise NvmeDeviceNotFoundError(f"Namespace {namespace_id} not found")
        del self._namespaces[namespace_id]
        self._attached.pop(namespace_id, None)

    def attach_namespace(self, namespace_id: int) -> None:
        if namespace_id not in self._namespaces:
            raise NvmeDeviceNotFoundError(f"Namespace {namespace_id} not found")
        self._attached[namespace_id] = True
        self._namespaces[namespace_id].active = True

    def detach_namespace(self, namespace_id: int) -> None:
        if namespace_id not in self._namespaces:
            raise NvmeDeviceNotFoundError(f"Namespace {namespace_id} not found")
        self._attached[namespace_id] = False
        self._namespaces[namespace_id].active = False

    def get_namespace(self, namespace_id: int) -> Namespace:
        ns = self._namespaces.get(namespace_id)
        if ns is None:
            raise NvmeDeviceNotFoundError(f"Namespace {namespace_id} not found")
        return ns

    def list_namespaces(self) -> List[Namespace]:
        return list(self._namespaces.values())

    def is_attached(self, namespace_id: int) -> bool:
        return self._attached.get(namespace_id, False)


class ReservationManager:
    """NVMe Reservations for multi-host access coordination."""

    def __init__(self) -> None:
        self._registrants: Dict[int, List[str]] = {}   # ns_id -> [host_ids]
        self._reservations: Dict[int, Reservation] = {}  # ns_id -> Reservation

    def register(self, host_id: str, namespace_id: int) -> None:
        if namespace_id not in self._registrants:
            self._registrants[namespace_id] = []
        if host_id not in self._registrants[namespace_id]:
            self._registrants[namespace_id].append(host_id)

    def reserve(self, host_id: str, namespace_id: int, reservation_type: ReservationType) -> None:
        registrants = self._registrants.get(namespace_id, [])
        if host_id not in registrants:
            raise NvmePermissionError(f"Host {host_id} is not registered for namespace {namespace_id}")
        self._reservations[namespace_id] = Reservation(
            holder_host_id=host_id,
            reservation_type=reservation_type,
            registrants=list(registrants),
        )

    def release(self, host_id: str, namespace_id: int) -> None:
        res = self._reservations.get(namespace_id)
        if res is None:
            return
        if res.holder_host_id != host_id:
            raise NvmePermissionError(f"Host {host_id} does not hold the reservation for namespace {namespace_id}")
        del self._reservations[namespace_id]

    def preempt(self, host_id: str, namespace_id: int) -> None:
        registrants = self._registrants.get(namespace_id, [])
        if host_id not in registrants:
            raise NvmePermissionError(f"Host {host_id} is not registered for namespace {namespace_id}")
        # Preempt: take over the reservation regardless of current holder
        self._reservations[namespace_id] = Reservation(
            holder_host_id=host_id,
            reservation_type=ReservationType.EXCLUSIVE_ACCESS,
            registrants=list(registrants),
        )

    def get_reservation(self, namespace_id: int) -> Optional[dict]:
        res = self._reservations.get(namespace_id)
        if res is None:
            return None
        return {
            "holder_host_id": res.holder_host_id,
            "reservation_type": res.reservation_type.value,
            "registrants": res.registrants,
        }

    def unregister(self, host_id: str, namespace_id: int) -> None:
        registrants = self._registrants.get(namespace_id, [])
        if host_id in registrants:
            registrants.remove(host_id)
