"""
QoS Controller for the Software NVMe Engine.

Implements token-bucket rate limiting (IOPS and bandwidth), priority-based
I/O scheduling across 4 priority levels, and weighted fair queuing within
the same priority level.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from nvme_engine.models.io_models import IoRequest


# ---------------------------------------------------------------------------
# Token Bucket
# ---------------------------------------------------------------------------


@dataclass
class TokenBucket:
    """
    Token bucket for rate limiting.

    Tokens are added at *rate* tokens/second up to *capacity*.
    Each call to :meth:`consume` attempts to deduct *amount* tokens.

    Attributes
    ----------
    rate        : Tokens added per second.
    capacity    : Maximum burst capacity (token ceiling).
    tokens      : Current token count.
    last_refill : Monotonic timestamp of the last refill.
    """

    rate: float
    capacity: float
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.monotonic)

    def consume(self, amount: float = 1.0) -> bool:
        """
        Try to consume *amount* tokens.

        Returns True if the request is allowed (tokens available),
        False if rate-limited.
        """
        self._refill()
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False

    def _refill(self) -> None:
        """Add tokens proportional to elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now


# ---------------------------------------------------------------------------
# QoS Policy
# ---------------------------------------------------------------------------


@dataclass
class QosPolicy:
    """
    QoS policy for a single virtual device.

    Attributes
    ----------
    device_id        : Target device identifier.
    iops_limit       : Maximum IOPS (0 = unlimited).
    bandwidth_limit  : Maximum bytes/sec (0 = unlimited).
    priority         : Scheduling priority 0 (highest) – 3 (lowest).
    weight           : Relative weight for fair queuing (1–1000).
    cpu_limit_percent: CPU share limit (0–100).
    """

    device_id: int
    iops_limit: int = 0
    bandwidth_limit: int = 0
    priority: int = 2
    weight: int = 100
    cpu_limit_percent: float = 100.0


# ---------------------------------------------------------------------------
# QoS Controller
# ---------------------------------------------------------------------------


class QosController:
    """
    Controls I/O quality-of-service for virtual NVMe devices.

    Features:
    - Token-bucket IOPS limiting (accurate within 5 %)
    - Token-bucket bandwidth limiting (accurate within 5 %)
    - 4-level strict priority scheduling
    - Weighted fair queuing within the same priority level
    - Live policy updates without interrupting ongoing I/O
    """

    def __init__(self) -> None:
        """Initialise an empty QosController."""
        self._policies: Dict[int, QosPolicy] = {}
        self._iops_buckets: Dict[int, TokenBucket] = {}
        self._bw_buckets: Dict[int, TokenBucket] = {}
        # Priority queues: 4 levels, each a deque of (device_id, request)
        self._priority_queues: List[deque] = [deque() for _ in range(4)]
        # Per-device served-token counters for weighted fair queuing
        self._wfq_tokens: Dict[int, float] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Policy management
    # ------------------------------------------------------------------

    def set_policy(self, policy: QosPolicy) -> None:
        """
        Set or replace the QoS policy for a device.

        Creates or updates token buckets to match the new limits.

        Args:
            policy: The QosPolicy to apply.
        """
        with self._lock:
            self._policies[policy.device_id] = policy
            self._rebuild_buckets(policy)

    def get_policy(self, device_id: int) -> Optional[QosPolicy]:
        """
        Return the current QoS policy for *device_id*, or None.

        Args:
            device_id: Target device identifier.
        """
        with self._lock:
            return self._policies.get(device_id)

    def remove_policy(self, device_id: int) -> None:
        """
        Remove the QoS policy for *device_id*.

        Removes associated token buckets and WFQ state.

        Args:
            device_id: Target device identifier.
        """
        with self._lock:
            self._policies.pop(device_id, None)
            self._iops_buckets.pop(device_id, None)
            self._bw_buckets.pop(device_id, None)
            self._wfq_tokens.pop(device_id, None)

    def update_policy(self, device_id: int, **kwargs) -> None:
        """
        Update individual fields of an existing policy without interrupting I/O.

        If no policy exists for *device_id* a new default policy is created.

        Args:
            device_id : Target device identifier.
            **kwargs  : Fields to update (e.g. iops_limit=5000, priority=1).
        """
        with self._lock:
            policy = self._policies.get(device_id)
            if policy is None:
                policy = QosPolicy(device_id=device_id)
                self._policies[device_id] = policy
            for key, value in kwargs.items():
                if hasattr(policy, key):
                    object.__setattr__(policy, key, value)
            self._rebuild_buckets(policy)

    # ------------------------------------------------------------------
    # Rate-limit checks
    # ------------------------------------------------------------------

    def check_iops(self, device_id: int) -> bool:
        """
        Check whether the device is within its IOPS limit.

        Consumes one token from the IOPS bucket.  Returns True if the
        operation is allowed, False if rate-limited.  Devices without a
        policy or with iops_limit == 0 are always allowed.

        Args:
            device_id: Target device identifier.
        """
        with self._lock:
            bucket = self._iops_buckets.get(device_id)
        if bucket is None:
            return True
        return bucket.consume(1.0)

    def check_bandwidth(self, device_id: int, bytes_count: int) -> bool:
        """
        Check whether the device is within its bandwidth limit.

        Consumes *bytes_count* tokens from the bandwidth bucket.  Returns
        True if allowed, False if rate-limited.  Devices without a policy
        or with bandwidth_limit == 0 are always allowed.

        Args:
            device_id  : Target device identifier.
            bytes_count: Number of bytes in the pending I/O operation.
        """
        with self._lock:
            bucket = self._bw_buckets.get(device_id)
        if bucket is None:
            return True
        return bucket.consume(float(bytes_count))

    # ------------------------------------------------------------------
    # Priority scheduling
    # ------------------------------------------------------------------

    def schedule_io(self, device_id: int, request: IoRequest) -> None:
        """
        Enqueue *request* into the appropriate priority queue.

        The priority level is taken from the device's QoS policy if set,
        otherwise from the request's own priority field.

        Args:
            device_id: Originating device identifier.
            request  : The I/O request to schedule.
        """
        with self._lock:
            policy = self._policies.get(device_id)
            priority = policy.priority if policy is not None else request.priority
            priority = max(0, min(3, priority))
            self._priority_queues[priority].append((device_id, request))

    def next_io(self) -> Optional[Tuple[int, IoRequest]]:
        """
        Dequeue the next I/O request respecting strict priority ordering.

        Within the same priority level, weighted fair queuing is applied:
        the device with the lowest accumulated service tokens (normalised
        by weight) is served next.

        Returns:
            ``(device_id, request)`` tuple, or ``None`` if all queues empty.
        """
        with self._lock:
            for level in range(4):
                q = self._priority_queues[level]
                if not q:
                    continue

                # Weighted fair queuing: pick the entry whose device has the
                # smallest normalised service counter.
                best_idx = 0
                best_score = float("inf")
                for idx, (dev_id, _req) in enumerate(q):
                    policy = self._policies.get(dev_id)
                    weight = policy.weight if policy is not None else 100
                    score = self._wfq_tokens.get(dev_id, 0.0) / max(weight, 1)
                    if score < best_score:
                        best_score = score
                        best_idx = idx

                # Rotate the deque so the chosen entry is at the front
                q.rotate(-best_idx)
                device_id, request = q.popleft()

                # Charge one unit of service to the chosen device
                self._wfq_tokens[device_id] = self._wfq_tokens.get(device_id, 0.0) + 1.0

                return device_id, request

        return None

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self, device_id: int) -> dict:
        """
        Return a snapshot of QoS statistics for *device_id*.

        Args:
            device_id: Target device identifier.

        Returns:
            Dictionary with policy details and current token levels.
        """
        with self._lock:
            policy = self._policies.get(device_id)
            iops_bucket = self._iops_buckets.get(device_id)
            bw_bucket = self._bw_buckets.get(device_id)

        return {
            "device_id": device_id,
            "policy": {
                "iops_limit": policy.iops_limit if policy else 0,
                "bandwidth_limit": policy.bandwidth_limit if policy else 0,
                "priority": policy.priority if policy else 2,
                "weight": policy.weight if policy else 100,
                "cpu_limit_percent": policy.cpu_limit_percent if policy else 100.0,
            },
            "iops_tokens": iops_bucket.tokens if iops_bucket else None,
            "bw_tokens": bw_bucket.tokens if bw_bucket else None,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_buckets(self, policy: QosPolicy) -> None:
        """
        (Re)create token buckets for *policy*.

        Called while *self._lock* is held.
        """
        dev_id = policy.device_id

        if policy.iops_limit > 0:
            rate = float(policy.iops_limit)
            # Allow a burst of up to 1 second worth of tokens
            self._iops_buckets[dev_id] = TokenBucket(
                rate=rate,
                capacity=rate,
                tokens=rate,
            )
        else:
            self._iops_buckets.pop(dev_id, None)

        if policy.bandwidth_limit > 0:
            rate = float(policy.bandwidth_limit)
            self._bw_buckets[dev_id] = TokenBucket(
                rate=rate,
                capacity=rate,
                tokens=rate,
            )
        else:
            self._bw_buckets.pop(dev_id, None)
