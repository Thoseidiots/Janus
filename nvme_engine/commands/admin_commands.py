"""
NVMe Admin Command implementations (NVMe 1.4+).

Covers: Identify, Get/Set Features, Create/Delete I/O Queue,
        Abort, and Async Event Request.
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Admin command type enumeration
# ---------------------------------------------------------------------------


class AdminCommandType(str, Enum):
    """NVMe 1.4+ admin command opcodes (symbolic names)."""

    IDENTIFY = "identify"
    GET_FEATURES = "get_features"
    SET_FEATURES = "set_features"
    CREATE_IO_QUEUE = "create_io_queue"
    DELETE_IO_QUEUE = "delete_io_queue"
    ABORT = "abort"
    ASYNC_EVENT_REQUEST = "async_event_request"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AdminCommandResult:
    """
    Result returned by every admin command handler.

    Attributes
    ----------
    success      : True when the command completed without error.
    status_code  : NVMe status code (0 = success, non-zero = error).
    data         : Optional payload bytes (e.g. Identify data structure).
    error_message: Human-readable error description (empty on success).
    """

    success: bool
    status_code: int  # 0 = success
    data: Optional[bytes] = None
    error_message: str = ""


# ---------------------------------------------------------------------------
# Feature store (simple in-memory key/value)
# ---------------------------------------------------------------------------


class FeatureStore:
    """
    Lightweight in-memory store for NVMe feature values.

    Feature IDs follow the NVMe 1.4 specification numbering.
    Default values are pre-populated for common features.
    """

    # NVMe 1.4 feature IDs (subset)
    ARBITRATION = 0x01
    POWER_MANAGEMENT = 0x02
    LBA_RANGE_TYPE = 0x03
    TEMPERATURE_THRESHOLD = 0x04
    ERROR_RECOVERY = 0x05
    VOLATILE_WRITE_CACHE = 0x06
    NUMBER_OF_QUEUES = 0x07
    INTERRUPT_COALESCING = 0x08
    INTERRUPT_VECTOR_CONFIG = 0x09
    WRITE_ATOMICITY_NORMAL = 0x0A
    ASYNC_EVENT_CONFIG = 0x0B

    _DEFAULTS: Dict[int, int] = {
        ARBITRATION: 0x00000000,
        POWER_MANAGEMENT: 0x00000000,
        TEMPERATURE_THRESHOLD: 0x0096,  # 150 °C
        ERROR_RECOVERY: 0x00000000,
        VOLATILE_WRITE_CACHE: 0x00000001,  # enabled
        NUMBER_OF_QUEUES: 0x003F003F,  # 64 SQ / 64 CQ
        INTERRUPT_COALESCING: 0x00000000,
        WRITE_ATOMICITY_NORMAL: 0x00000000,
        ASYNC_EVENT_CONFIG: 0x00000000,
    }

    def __init__(self) -> None:
        self._store: Dict[int, int] = dict(self._DEFAULTS)

    def get(self, feature_id: int) -> Optional[int]:
        """Return the current value for *feature_id*, or None if unknown."""
        return self._store.get(feature_id)

    def set(self, feature_id: int, value: int) -> None:
        """Persist *value* for *feature_id*."""
        self._store[feature_id] = value

    def all_features(self) -> Dict[int, int]:
        """Return a copy of all stored feature values."""
        return dict(self._store)


# ---------------------------------------------------------------------------
# I/O queue registry
# ---------------------------------------------------------------------------


@dataclass
class IoQueue:
    """Descriptor for a created I/O submission or completion queue."""

    queue_id: int
    queue_type: str  # "submission" | "completion"
    depth: int
    created_at: float = field(default_factory=time.time)


class IoQueueRegistry:
    """
    Tracks created I/O submission and completion queues.

    Queue IDs are 1-based (queue 0 is the admin queue).
    Maximum queue depth is 65 535 per the NVMe specification.
    """

    MAX_QUEUE_DEPTH = 65_535
    MIN_QUEUE_DEPTH = 1

    def __init__(self) -> None:
        self._queues: Dict[int, IoQueue] = {}

    def create(self, queue_id: int, queue_type: str, depth: int) -> IoQueue:
        """
        Register a new I/O queue.

        Args:
            queue_id  : 1-based queue identifier.
            queue_type: "submission" or "completion".
            depth     : Number of entries (1 – 65 535).

        Returns:
            IoQueue: The newly created queue descriptor.

        Raises:
            ValueError: If parameters are out of range or queue already exists.
        """
        if queue_id < 1:
            raise ValueError(f"queue_id must be >= 1, got {queue_id}")
        if queue_type not in ("submission", "completion"):
            raise ValueError(f"queue_type must be 'submission' or 'completion', got {queue_type!r}")
        if not (self.MIN_QUEUE_DEPTH <= depth <= self.MAX_QUEUE_DEPTH):
            raise ValueError(
                f"depth must be {self.MIN_QUEUE_DEPTH}–{self.MAX_QUEUE_DEPTH}, got {depth}"
            )
        if queue_id in self._queues:
            raise ValueError(f"Queue {queue_id} already exists")

        q = IoQueue(queue_id=queue_id, queue_type=queue_type, depth=depth)
        self._queues[queue_id] = q
        return q

    def delete(self, queue_id: int) -> None:
        """
        Remove a queue from the registry.

        Args:
            queue_id: Queue to remove.

        Raises:
            KeyError: If the queue does not exist.
        """
        if queue_id not in self._queues:
            raise KeyError(f"Queue {queue_id} does not exist")
        del self._queues[queue_id]

    def get(self, queue_id: int) -> Optional[IoQueue]:
        """Return the queue descriptor, or None if not found."""
        return self._queues.get(queue_id)

    def list_queues(self) -> List[IoQueue]:
        """Return all registered queues."""
        return list(self._queues.values())


# ---------------------------------------------------------------------------
# Pending async event requests
# ---------------------------------------------------------------------------


class AsyncEventQueue:
    """
    Simple FIFO queue for pending Async Event Request commands.

    In a real NVMe controller the host submits AER commands and the
    controller completes them when an event occurs.  Here we track
    pending requests and deliver synthetic events.
    """

    def __init__(self) -> None:
        self._pending: List[Dict[str, Any]] = []
        self._events: List[Dict[str, Any]] = []

    def submit(self, request_id: int) -> None:
        """Record a pending AER command from the host."""
        self._pending.append({"request_id": request_id, "submitted_at": time.time()})

    def post_event(self, event_type: str, event_info: str) -> Optional[Dict[str, Any]]:
        """
        Post an asynchronous event.

        If a pending AER command is available it is completed immediately;
        otherwise the event is queued for the next AER submission.

        Returns the completed AER descriptor, or None if no pending command.
        """
        event = {"event_type": event_type, "event_info": event_info, "timestamp": time.time()}
        if self._pending:
            pending = self._pending.pop(0)
            return {**pending, **event, "completed": True}
        self._events.append(event)
        return None

    def pending_count(self) -> int:
        """Number of outstanding AER commands."""
        return len(self._pending)

    def queued_event_count(self) -> int:
        """Number of events waiting for an AER command."""
        return len(self._events)


# ---------------------------------------------------------------------------
# Admin command processor
# ---------------------------------------------------------------------------


class AdminCommandProcessor:
    """
    Processes NVMe 1.4+ admin commands.

    Each ``handle_*`` method corresponds to one AdminCommandType and
    returns an AdminCommandResult.
    """

    # Identify CNS (Controller or Namespace Structure) values
    CNS_NAMESPACE = 0x00
    CNS_CONTROLLER = 0x01
    CNS_ACTIVE_NAMESPACE_LIST = 0x02

    def __init__(self) -> None:
        self._features = FeatureStore()
        self._queues = IoQueueRegistry()
        self._aer = AsyncEventQueue()
        # Simulated controller identity fields
        self._controller_id = 1
        self._model_number = "SoftNVMe-v1.4"
        self._serial_number = "SN-0000000001"
        self._firmware_rev = "1.0.0"

    # ------------------------------------------------------------------
    # Public dispatch
    # ------------------------------------------------------------------

    def process(
        self,
        cmd_type: AdminCommandType,
        params: Dict[str, Any],
    ) -> AdminCommandResult:
        """
        Dispatch an admin command to the appropriate handler.

        Args:
            cmd_type: The admin command to execute.
            params  : Command-specific parameters.

        Returns:
            AdminCommandResult with success/failure details.
        """
        handlers = {
            AdminCommandType.IDENTIFY: self._handle_identify,
            AdminCommandType.GET_FEATURES: self._handle_get_features,
            AdminCommandType.SET_FEATURES: self._handle_set_features,
            AdminCommandType.CREATE_IO_QUEUE: self._handle_create_io_queue,
            AdminCommandType.DELETE_IO_QUEUE: self._handle_delete_io_queue,
            AdminCommandType.ABORT: self._handle_abort,
            AdminCommandType.ASYNC_EVENT_REQUEST: self._handle_async_event_request,
        }
        handler = handlers.get(cmd_type)
        if handler is None:
            return AdminCommandResult(
                success=False,
                status_code=1,
                error_message=f"Unknown admin command: {cmd_type}",
            )
        try:
            return handler(params)
        except Exception as exc:  # noqa: BLE001
            return AdminCommandResult(
                success=False,
                status_code=1,
                error_message=str(exc),
            )

    # ------------------------------------------------------------------
    # Individual command handlers
    # ------------------------------------------------------------------

    def _handle_identify(self, params: Dict[str, Any]) -> AdminCommandResult:
        """
        Identify command (CNS 0x00 = namespace, 0x01 = controller).

        Returns a JSON-encoded identify data structure as bytes.
        """
        cns = params.get("cns", self.CNS_CONTROLLER)

        if cns == self.CNS_CONTROLLER:
            identify_data = {
                "controller_id": self._controller_id,
                "model_number": self._model_number,
                "serial_number": self._serial_number,
                "firmware_revision": self._firmware_rev,
                "nvme_version": "1.4",
                "max_queue_entries": IoQueueRegistry.MAX_QUEUE_DEPTH,
                "number_of_namespaces": params.get("namespace_count", 1),
            }
        elif cns == self.CNS_NAMESPACE:
            ns_id = params.get("namespace_id", 1)
            capacity = params.get("capacity_bytes", 0)
            block_size = params.get("block_size", 4096)
            identify_data = {
                "namespace_id": ns_id,
                "namespace_size": capacity // block_size if block_size else 0,
                "namespace_capacity": capacity // block_size if block_size else 0,
                "lba_format": {"block_size": block_size, "metadata_size": 0},
            }
        elif cns == self.CNS_ACTIVE_NAMESPACE_LIST:
            ns_list = params.get("active_namespaces", [])
            identify_data = {"active_namespaces": ns_list}
        else:
            return AdminCommandResult(
                success=False,
                status_code=2,
                error_message=f"Unsupported CNS value: {cns}",
            )

        payload = json.dumps(identify_data).encode()
        return AdminCommandResult(success=True, status_code=0, data=payload)

    def _handle_get_features(self, params: Dict[str, Any]) -> AdminCommandResult:
        """
        Get Features command.

        Params:
            feature_id (int): NVMe feature identifier.

        Returns the feature value encoded as 4-byte little-endian in ``data``.
        """
        feature_id = params.get("feature_id")
        if feature_id is None:
            return AdminCommandResult(
                success=False,
                status_code=2,
                error_message="Missing required parameter: feature_id",
            )

        value = self._features.get(int(feature_id))
        if value is None:
            return AdminCommandResult(
                success=False,
                status_code=2,
                error_message=f"Unknown feature_id: {feature_id:#x}",
            )

        payload = struct.pack("<I", value)
        return AdminCommandResult(success=True, status_code=0, data=payload)

    def _handle_set_features(self, params: Dict[str, Any]) -> AdminCommandResult:
        """
        Set Features command.

        Params:
            feature_id (int): NVMe feature identifier.
            value      (int): New feature value.
        """
        feature_id = params.get("feature_id")
        value = params.get("value")
        if feature_id is None or value is None:
            return AdminCommandResult(
                success=False,
                status_code=2,
                error_message="Missing required parameters: feature_id and value",
            )

        self._features.set(int(feature_id), int(value))
        return AdminCommandResult(success=True, status_code=0)

    def _handle_create_io_queue(self, params: Dict[str, Any]) -> AdminCommandResult:
        """
        Create I/O Submission or Completion Queue.

        Params:
            queue_id   (int): 1-based queue identifier.
            queue_type (str): "submission" or "completion".
            depth      (int): Queue depth (1 – 65 535).
        """
        queue_id = params.get("queue_id")
        queue_type = params.get("queue_type", "submission")
        depth = params.get("depth", 256)

        if queue_id is None:
            return AdminCommandResult(
                success=False,
                status_code=2,
                error_message="Missing required parameter: queue_id",
            )

        queue = self._queues.create(int(queue_id), str(queue_type), int(depth))
        payload = json.dumps(
            {
                "queue_id": queue.queue_id,
                "queue_type": queue.queue_type,
                "depth": queue.depth,
            }
        ).encode()
        return AdminCommandResult(success=True, status_code=0, data=payload)

    def _handle_delete_io_queue(self, params: Dict[str, Any]) -> AdminCommandResult:
        """
        Delete I/O Queue.

        Params:
            queue_id (int): Queue to delete.
        """
        queue_id = params.get("queue_id")
        if queue_id is None:
            return AdminCommandResult(
                success=False,
                status_code=2,
                error_message="Missing required parameter: queue_id",
            )

        self._queues.delete(int(queue_id))
        return AdminCommandResult(success=True, status_code=0)

    def _handle_abort(self, params: Dict[str, Any]) -> AdminCommandResult:
        """
        Abort command.

        Params:
            submission_queue_id (int): SQ containing the command to abort.
            command_id          (int): Command identifier to abort.

        In this simulation the abort always succeeds (no in-flight commands).
        """
        sqid = params.get("submission_queue_id", 0)
        cid = params.get("command_id", 0)
        payload = json.dumps({"aborted_sqid": sqid, "aborted_cid": cid}).encode()
        return AdminCommandResult(success=True, status_code=0, data=payload)

    def _handle_async_event_request(self, params: Dict[str, Any]) -> AdminCommandResult:
        """
        Async Event Request command.

        Params:
            request_id (int): Host-assigned identifier for this AER command.

        The command is queued; it will be completed when an event occurs.
        """
        request_id = params.get("request_id", 0)
        self._aer.submit(int(request_id))
        payload = json.dumps(
            {"pending_aer_count": self._aer.pending_count()}
        ).encode()
        return AdminCommandResult(success=True, status_code=0, data=payload)

    # ------------------------------------------------------------------
    # Accessors for testing / introspection
    # ------------------------------------------------------------------

    @property
    def features(self) -> FeatureStore:
        """Expose the feature store for direct inspection."""
        return self._features

    @property
    def queues(self) -> IoQueueRegistry:
        """Expose the queue registry for direct inspection."""
        return self._queues

    @property
    def aer(self) -> AsyncEventQueue:
        """Expose the AER queue for direct inspection."""
        return self._aer
