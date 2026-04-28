"""
Telemetry data models for the Software NVMe Engine.

Covers: LatencyHistogram, TelemetryMetrics
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Number of microsecond buckets in a latency histogram.
# Buckets 0-98 cover 0-1μs through 98-99μs; bucket 99 is the ">99μs" overflow.
HISTOGRAM_BUCKET_COUNT = 100


@dataclass
class LatencyHistogram:
    """
    Per-operation latency histogram with microsecond precision.

    Buckets
    -------
    buckets[i]  : count of operations with latency in [i, i+1) μs  (i = 0..98)
    buckets[99] : count of operations with latency >= 99 μs (overflow bucket)

    Percentile fields (p50, p95, p99, p999) are in microseconds and are
    computed / updated externally (e.g., by the TelemetryCollector).
    """

    buckets: List[int] = field(
        default_factory=lambda: [0] * HISTOGRAM_BUCKET_COUNT
    )
    p50: int = 0
    p95: int = 0
    p99: int = 0
    p999: int = 0

    def __post_init__(self) -> None:
        if len(self.buckets) != HISTOGRAM_BUCKET_COUNT:
            raise ValueError(
                f"buckets must have exactly {HISTOGRAM_BUCKET_COUNT} entries, "
                f"got {len(self.buckets)}"
            )
        for i, v in enumerate(self.buckets):
            if v < 0:
                raise ValueError(
                    f"bucket[{i}] must be >= 0, got {v}"
                )
        for name, val in (("p50", self.p50), ("p95", self.p95),
                          ("p99", self.p99), ("p999", self.p999)):
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val}")

    def record(self, latency_us: int) -> None:
        """
        Record a single observation.

        Parameters
        ----------
        latency_us : Latency in whole microseconds (>= 0).
        """
        if latency_us < 0:
            raise ValueError(f"latency_us must be >= 0, got {latency_us}")
        idx = min(latency_us, HISTOGRAM_BUCKET_COUNT - 1)
        self.buckets[idx] += 1

    @property
    def total_count(self) -> int:
        """Total number of observations recorded."""
        return sum(self.buckets)

    def compute_percentiles(self) -> None:
        """
        Recompute p50, p95, p99, p999 from the current bucket counts.

        Uses linear interpolation within the overflow bucket for values
        that fall in bucket 99 (>= 99 μs).
        """
        total = self.total_count
        if total == 0:
            self.p50 = self.p95 = self.p99 = self.p999 = 0
            return

        targets = {
            "p50": 0.50,
            "p95": 0.95,
            "p99": 0.99,
            "p999": 0.999,
        }
        results: Dict[str, int] = {}
        cumulative = 0
        for pname, fraction in targets.items():
            threshold = fraction * total
            cumulative = 0
            for i, count in enumerate(self.buckets):
                cumulative += count
                if cumulative >= threshold:
                    results[pname] = i
                    break
            else:
                results[pname] = HISTOGRAM_BUCKET_COUNT - 1

        self.p50 = results["p50"]
        self.p95 = results["p95"]
        self.p99 = results["p99"]
        self.p999 = results["p999"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "buckets": list(self.buckets),
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
            "p999": self.p999,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LatencyHistogram":
        return cls(
            buckets=list(data.get("buckets", [0] * HISTOGRAM_BUCKET_COUNT)),
            p50=data.get("p50", 0),
            p95=data.get("p95", 0),
            p99=data.get("p99", 0),
            p999=data.get("p999", 0),
        )


@dataclass
class TelemetryMetrics:
    """
    Snapshot of telemetry metrics for a single virtual NVMe device.

    All bandwidth values are in bytes/second; all IOPS values are
    operations/second; latency values are in microseconds.
    """

    timestamp_ns: int = field(default_factory=lambda: time.time_ns())

    # I/O throughput
    read_iops: int = 0
    write_iops: int = 0
    read_bandwidth: int = 0    # bytes/sec
    write_bandwidth: int = 0   # bytes/sec
    queue_depth: int = 0

    # Latency histograms
    read_latency: LatencyHistogram = field(default_factory=LatencyHistogram)
    write_latency: LatencyHistogram = field(default_factory=LatencyHistogram)

    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    cache_hit_rate: float = 0.0

    # Resource utilisation
    cpu_utilization: float = 0.0      # 0.0 – 1.0
    memory_used_bytes: int = 0
    io_bandwidth_used: int = 0        # bytes/sec

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError(
                f"timestamp_ns must be >= 0, got {self.timestamp_ns}"
            )
        for name, val in (
            ("read_iops", self.read_iops),
            ("write_iops", self.write_iops),
            ("read_bandwidth", self.read_bandwidth),
            ("write_bandwidth", self.write_bandwidth),
            ("queue_depth", self.queue_depth),
            ("cache_hits", self.cache_hits),
            ("cache_misses", self.cache_misses),
            ("prefetch_hits", self.prefetch_hits),
            ("prefetch_misses", self.prefetch_misses),
            ("memory_used_bytes", self.memory_used_bytes),
            ("io_bandwidth_used", self.io_bandwidth_used),
        ):
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val}")
        if not (0.0 <= self.cache_hit_rate <= 1.0):
            raise ValueError(
                f"cache_hit_rate must be 0.0-1.0, got {self.cache_hit_rate}"
            )
        if not (0.0 <= self.cpu_utilization <= 1.0):
            raise ValueError(
                f"cpu_utilization must be 0.0-1.0, got {self.cpu_utilization}"
            )

    def update_cache_hit_rate(self) -> None:
        """Recompute cache_hit_rate from cache_hits and cache_misses."""
        total = self.cache_hits + self.cache_misses
        self.cache_hit_rate = self.cache_hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "read_iops": self.read_iops,
            "write_iops": self.write_iops,
            "read_bandwidth": self.read_bandwidth,
            "write_bandwidth": self.write_bandwidth,
            "queue_depth": self.queue_depth,
            "read_latency": self.read_latency.to_dict(),
            "write_latency": self.write_latency.to_dict(),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "prefetch_hits": self.prefetch_hits,
            "prefetch_misses": self.prefetch_misses,
            "cache_hit_rate": self.cache_hit_rate,
            "cpu_utilization": self.cpu_utilization,
            "memory_used_bytes": self.memory_used_bytes,
            "io_bandwidth_used": self.io_bandwidth_used,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TelemetryMetrics":
        return cls(
            timestamp_ns=data.get("timestamp_ns", time.time_ns()),
            read_iops=data.get("read_iops", 0),
            write_iops=data.get("write_iops", 0),
            read_bandwidth=data.get("read_bandwidth", 0),
            write_bandwidth=data.get("write_bandwidth", 0),
            queue_depth=data.get("queue_depth", 0),
            read_latency=LatencyHistogram.from_dict(
                data.get("read_latency", {})
            ),
            write_latency=LatencyHistogram.from_dict(
                data.get("write_latency", {})
            ),
            cache_hits=data.get("cache_hits", 0),
            cache_misses=data.get("cache_misses", 0),
            prefetch_hits=data.get("prefetch_hits", 0),
            prefetch_misses=data.get("prefetch_misses", 0),
            cache_hit_rate=data.get("cache_hit_rate", 0.0),
            cpu_utilization=data.get("cpu_utilization", 0.0),
            memory_used_bytes=data.get("memory_used_bytes", 0),
            io_bandwidth_used=data.get("io_bandwidth_used", 0),
        )
