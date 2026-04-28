"""
Dynamic Allocator for the Software NVMe Engine.

Monitors CPU, memory, and I/O bandwidth utilisation and triggers
scale-up / scale-down actions based on configurable thresholds.
Supports both threshold-based and predictive scaling policies.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PSUTIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class ScalingPolicy(Enum):
    """Scaling decision strategy."""

    THRESHOLD = "threshold"
    PREDICTIVE = "predictive"


@dataclass
class ResourceSnapshot:
    """
    Point-in-time resource utilisation snapshot.

    Attributes
    ----------
    timestamp            : Monotonic timestamp of the snapshot.
    cpu_percent          : CPU utilisation percentage (0–100).
    memory_percent       : Memory utilisation percentage (0–100).
    io_bandwidth_percent : I/O bandwidth utilisation percentage (0–100).
    """

    timestamp: float
    cpu_percent: float
    memory_percent: float
    io_bandwidth_percent: float


@dataclass
class AllocatorConfig:
    """
    Configuration for the DynamicAllocator.

    Attributes
    ----------
    scale_up_threshold    : Utilisation % that triggers scale-up.
    scale_down_threshold  : Utilisation % that triggers scale-down.
    scale_up_duration_s   : Seconds utilisation must exceed threshold before scaling up.
    scale_down_duration_s : Seconds utilisation must stay below threshold before scaling down.
    rebalance_interval_s  : How often the background thread checks for rebalancing.
    policy                : Scaling decision strategy.
    weights               : Per-device resource allocation weights.
    """

    scale_up_threshold: float = 75.0
    scale_down_threshold: float = 25.0
    scale_up_duration_s: float = 10.0
    scale_down_duration_s: float = 60.0
    rebalance_interval_s: float = 5.0
    policy: ScalingPolicy = ScalingPolicy.THRESHOLD
    weights: Dict[int, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dynamic Allocator
# ---------------------------------------------------------------------------


class DynamicAllocator:
    """
    Monitors resource utilisation and drives scaling decisions.

    A background thread periodically samples CPU, memory, and I/O
    bandwidth via *psutil* and appends snapshots to an internal history
    buffer.  Callers can query :meth:`should_scale_up` /
    :meth:`should_scale_down` to decide whether to allocate or release
    resources.
    """

    # Maximum number of snapshots to retain in history
    _MAX_HISTORY: int = 1000

    def __init__(self, config: Optional[AllocatorConfig] = None) -> None:
        """
        Initialise the allocator.

        Args:
            config: Optional configuration; defaults to AllocatorConfig().
        """
        self._config = config or AllocatorConfig()
        self._history: List[ResourceSnapshot] = []
        self._device_weights: Dict[int, float] = dict(self._config.weights)
        self._queue_assignments: Dict[int, int] = {}  # queue_id -> cpu_core
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
        self._cpu_count: int = max(1, os.cpu_count() or 1)

        # Baseline I/O counters for bandwidth % estimation
        self._last_io_counters: Optional[object] = None
        self._last_io_time: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background resource-monitoring thread."""
        with self._lock:
            if self._running:
                return
            self._running = True

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="nvme-dynamic-allocator",
            daemon=True,
        )
        self._monitor_thread.start()

    def stop(self) -> None:
        """Stop the background monitoring thread and wait for it to exit."""
        with self._lock:
            self._running = False

        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=10.0)
        self._monitor_thread = None

    # ------------------------------------------------------------------
    # Utilisation
    # ------------------------------------------------------------------

    def get_utilization(self) -> ResourceSnapshot:
        """
        Sample current resource utilisation.

        Returns:
            A fresh ResourceSnapshot with CPU, memory, and I/O bandwidth %.
        """
        cpu_pct = self._sample_cpu()
        mem_pct = self._sample_memory()
        io_pct = self._sample_io_bandwidth()

        return ResourceSnapshot(
            timestamp=time.monotonic(),
            cpu_percent=cpu_pct,
            memory_percent=mem_pct,
            io_bandwidth_percent=io_pct,
        )

    # ------------------------------------------------------------------
    # Scaling decisions
    # ------------------------------------------------------------------

    def should_scale_up(self) -> bool:
        """
        Return True if CPU or memory has been above *scale_up_threshold*
        continuously for at least *scale_up_duration_s* seconds.

        Implementation: all snapshots within the last *scale_up_duration_s*
        seconds must exceed the threshold, AND there must be at least one
        snapshot older than *scale_up_duration_s* seconds (proving the
        condition has been sustained for the full window).
        """
        with self._lock:
            history = list(self._history)

        if not history:
            return False

        threshold = self._config.scale_up_threshold
        duration = self._config.scale_up_duration_s
        now = time.monotonic()
        cutoff = now - duration

        # Snapshots within the window
        recent = [s for s in history if s.timestamp >= cutoff]
        if not recent:
            return False

        # There must be history that predates the window (or the window is
        # fully covered by the injected history span)
        oldest = history[0].timestamp
        if (now - oldest) < duration:
            return False

        # All snapshots in the window must exceed the threshold
        return all(
            s.cpu_percent > threshold or s.memory_percent > threshold
            for s in recent
        )

    def should_scale_down(self) -> bool:
        """
        Return True if CPU and memory have both been below
        *scale_down_threshold* continuously for at least
        *scale_down_duration_s* seconds.
        """
        with self._lock:
            history = list(self._history)

        if not history:
            return False

        threshold = self._config.scale_down_threshold
        duration = self._config.scale_down_duration_s
        now = time.monotonic()
        cutoff = now - duration

        recent = [s for s in history if s.timestamp >= cutoff]
        if not recent:
            return False

        oldest = history[0].timestamp
        if (now - oldest) < duration:
            return False

        return all(
            s.cpu_percent < threshold and s.memory_percent < threshold
            for s in recent
        )

    # ------------------------------------------------------------------
    # Scaling actions
    # ------------------------------------------------------------------

    def scale_up(self) -> dict:
        """
        Allocate additional resources.

        Returns a dictionary describing what was allocated.
        """
        snapshot = self.get_utilization()
        allocated = {
            "action": "scale_up",
            "timestamp": time.monotonic(),
            "cpu_percent_at_trigger": snapshot.cpu_percent,
            "memory_percent_at_trigger": snapshot.memory_percent,
            "additional_queues": 1,
        }
        return allocated

    def scale_down(self) -> dict:
        """
        Release unused resources.

        Returns a dictionary describing what was released.
        """
        snapshot = self.get_utilization()
        released = {
            "action": "scale_down",
            "timestamp": time.monotonic(),
            "cpu_percent_at_trigger": snapshot.cpu_percent,
            "memory_percent_at_trigger": snapshot.memory_percent,
            "released_queues": 1,
        }
        return released

    # ------------------------------------------------------------------
    # Device weights
    # ------------------------------------------------------------------

    def set_device_weight(self, device_id: int, weight: float) -> None:
        """
        Set the resource allocation weight for *device_id*.

        Args:
            device_id: Target device identifier.
            weight   : Allocation weight in the range 1–1000.

        Raises:
            ValueError: If weight is outside [1, 1000].
        """
        if not (1 <= weight <= 1000):
            raise ValueError(f"weight must be 1–1000, got {weight}")
        with self._lock:
            self._device_weights[device_id] = float(weight)

    def get_device_allocation(self, device_id: int) -> float:
        """
        Return the proportional resource allocation for *device_id*.

        Computed as ``weight(device) / sum(all weights)``.  Returns 0.0
        if no weights are registered.

        Args:
            device_id: Target device identifier.
        """
        with self._lock:
            weights = dict(self._device_weights)

        total = sum(weights.values())
        if total == 0.0:
            return 0.0
        return weights.get(device_id, 0.0) / total

    # ------------------------------------------------------------------
    # Queue-to-CPU assignment
    # ------------------------------------------------------------------

    def assign_queue_to_cpu(self, queue_id: int, cpu_core: int) -> None:
        """
        Assign *queue_id* to *cpu_core*.

        Args:
            queue_id : Queue identifier.
            cpu_core : CPU core index (0-based).
        """
        with self._lock:
            self._queue_assignments[queue_id] = cpu_core

    def rebalance_queues(self) -> Dict[int, int]:
        """
        Rebalance queue-to-CPU assignments for even distribution.

        Distributes queues across available CPU cores using round-robin.
        Must complete within 5 seconds.

        Returns:
            New ``{queue_id: cpu_core}`` mapping.
        """
        start = time.monotonic()

        with self._lock:
            queue_ids = sorted(self._queue_assignments.keys())
            new_assignments: Dict[int, int] = {}
            for idx, qid in enumerate(queue_ids):
                new_assignments[qid] = idx % self._cpu_count
            self._queue_assignments = new_assignments

        elapsed = time.monotonic() - start
        if elapsed > 5.0:
            raise RuntimeError(
                f"rebalance_queues exceeded 5-second deadline ({elapsed:.2f}s)"
            )

        return dict(new_assignments)

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_history(self, last_n: int = 10) -> List[ResourceSnapshot]:
        """
        Return the last *last_n* resource snapshots.

        Args:
            last_n: Maximum number of snapshots to return.
        """
        with self._lock:
            return list(self._history[-last_n:])

    # ------------------------------------------------------------------
    # Predictive scaling
    # ------------------------------------------------------------------

    def predict_scale_up(self) -> bool:
        """
        Predictive scaling: return True if the utilisation trend suggests
        that CPU or memory will exceed *scale_up_threshold* within the
        next monitoring interval.

        Uses a simple linear regression over the recent history window.
        """
        with self._lock:
            history = list(self._history[-20:])  # last 20 samples

        if len(history) < 3:
            return False

        threshold = self._config.scale_up_threshold

        # Compute linear trend for CPU and memory
        cpu_trend = self._linear_trend([s.cpu_percent for s in history])
        mem_trend = self._linear_trend([s.memory_percent for s in history])

        # Predict next value
        next_cpu = history[-1].cpu_percent + cpu_trend
        next_mem = history[-1].memory_percent + mem_trend

        return next_cpu > threshold or next_mem > threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        """Background thread: periodically sample utilisation."""
        while True:
            with self._lock:
                if not self._running:
                    break

            snapshot = self.get_utilization()

            with self._lock:
                self._history.append(snapshot)
                # Trim history to avoid unbounded growth
                if len(self._history) > self._MAX_HISTORY:
                    self._history = self._history[-self._MAX_HISTORY:]

            time.sleep(self._config.rebalance_interval_s)

    def _sample_cpu(self) -> float:
        """Return current CPU utilisation percentage."""
        if _PSUTIL_AVAILABLE:
            try:
                return psutil.cpu_percent(interval=None)
            except Exception:
                pass
        return 0.0

    def _sample_memory(self) -> float:
        """Return current memory utilisation percentage."""
        if _PSUTIL_AVAILABLE:
            try:
                return psutil.virtual_memory().percent
            except Exception:
                pass
        return 0.0

    def _sample_io_bandwidth(self) -> float:
        """
        Estimate I/O bandwidth utilisation as a percentage.

        Uses the delta of psutil disk I/O counters between calls.
        Returns 0.0 if psutil is unavailable or on the first call.
        """
        if not _PSUTIL_AVAILABLE:
            return 0.0

        try:
            counters = psutil.disk_io_counters()
            if counters is None:
                return 0.0

            now = time.monotonic()
            if self._last_io_counters is None:
                self._last_io_counters = counters
                self._last_io_time = now
                return 0.0

            elapsed = now - self._last_io_time
            if elapsed <= 0:
                return 0.0

            prev = self._last_io_counters
            bytes_delta = (
                (counters.read_bytes - prev.read_bytes)
                + (counters.write_bytes - prev.write_bytes)
            )
            self._last_io_counters = counters
            self._last_io_time = now

            # Normalise against a 1 GB/s reference bandwidth → percentage
            reference_bps = 1024 * 1024 * 1024
            bps = bytes_delta / elapsed
            return min(100.0, (bps / reference_bps) * 100.0)

        except Exception:
            return 0.0

    @staticmethod
    def _linear_trend(values: List[float]) -> float:
        """
        Compute the slope of a simple linear regression over *values*.

        Returns the per-step change (positive = increasing trend).
        """
        n = len(values)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return 0.0
        return numerator / denominator
