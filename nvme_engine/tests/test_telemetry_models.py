"""
Unit tests for telemetry data models.

Tests: LatencyHistogram, TelemetryMetrics
"""

import time

import pytest

from nvme_engine.models.telemetry import (
    HISTOGRAM_BUCKET_COUNT,
    LatencyHistogram,
    TelemetryMetrics,
)


# ---------------------------------------------------------------------------
# LatencyHistogram Tests
# ---------------------------------------------------------------------------


class TestLatencyHistogram:
    """Tests for LatencyHistogram."""

    def test_default_construction(self):
        """Test creating a default LatencyHistogram."""
        hist = LatencyHistogram()
        assert len(hist.buckets) == HISTOGRAM_BUCKET_COUNT
        assert all(b == 0 for b in hist.buckets)
        assert hist.p50 == 0
        assert hist.p95 == 0
        assert hist.p99 == 0
        assert hist.p999 == 0

    def test_custom_buckets(self):
        """Test creating a histogram with custom bucket values."""
        buckets = [i for i in range(HISTOGRAM_BUCKET_COUNT)]
        hist = LatencyHistogram(buckets=buckets, p50=10, p95=50, p99=90, p999=99)
        assert hist.buckets == buckets
        assert hist.p50 == 10
        assert hist.p95 == 50
        assert hist.p99 == 90
        assert hist.p999 == 99

    def test_invalid_bucket_count(self):
        """Test that wrong number of buckets raises ValueError."""
        with pytest.raises(ValueError, match="buckets must have exactly 100 entries"):
            LatencyHistogram(buckets=[0] * 50)

    def test_invalid_bucket_value(self):
        """Test that negative bucket values raise ValueError."""
        buckets = [0] * HISTOGRAM_BUCKET_COUNT
        buckets[50] = -1
        with pytest.raises(ValueError, match="bucket\\[50\\] must be >= 0"):
            LatencyHistogram(buckets=buckets)

    def test_invalid_percentile(self):
        """Test that negative percentile values raise ValueError."""
        with pytest.raises(ValueError, match="p50 must be >= 0"):
            LatencyHistogram(p50=-1)

    def test_record_single_observation(self):
        """Test recording a single latency observation."""
        hist = LatencyHistogram()
        hist.record(5)
        assert hist.buckets[5] == 1
        assert sum(hist.buckets) == 1

    def test_record_multiple_observations(self):
        """Test recording multiple latency observations."""
        hist = LatencyHistogram()
        for i in range(10):
            hist.record(i)
        
        for i in range(10):
            assert hist.buckets[i] == 1
        assert sum(hist.buckets) == 10

    def test_record_overflow_bucket(self):
        """Test that latencies >= 99μs go into overflow bucket."""
        hist = LatencyHistogram()
        hist.record(99)
        hist.record(100)
        hist.record(1000)
        hist.record(999999)
        
        assert hist.buckets[99] == 4
        assert sum(hist.buckets) == 4

    def test_record_negative_latency(self):
        """Test that negative latency raises ValueError."""
        hist = LatencyHistogram()
        with pytest.raises(ValueError, match="latency_us must be >= 0"):
            hist.record(-1)

    def test_total_count_property(self):
        """Test total_count property."""
        hist = LatencyHistogram()
        assert hist.total_count == 0
        
        hist.record(1)
        hist.record(2)
        hist.record(3)
        assert hist.total_count == 3

    def test_compute_percentiles_empty(self):
        """Test computing percentiles with no data."""
        hist = LatencyHistogram()
        hist.compute_percentiles()
        assert hist.p50 == 0
        assert hist.p95 == 0
        assert hist.p99 == 0
        assert hist.p999 == 0

    def test_compute_percentiles_single_value(self):
        """Test computing percentiles with single value."""
        hist = LatencyHistogram()
        hist.record(10)
        hist.compute_percentiles()
        assert hist.p50 == 10
        assert hist.p95 == 10
        assert hist.p99 == 10
        assert hist.p999 == 10

    def test_compute_percentiles_uniform_distribution(self):
        """Test computing percentiles with uniform distribution."""
        hist = LatencyHistogram()
        # Record 100 observations: 0-99 μs
        for i in range(100):
            hist.record(i)
        
        hist.compute_percentiles()
        assert 45 <= hist.p50 <= 55  # Around 50th percentile
        assert 90 <= hist.p95 <= 99  # Around 95th percentile
        assert 95 <= hist.p99 <= 99  # Around 99th percentile

    def test_compute_percentiles_skewed_distribution(self):
        """Test computing percentiles with skewed distribution."""
        hist = LatencyHistogram()
        # Most observations at low latency
        for _ in range(900):
            hist.record(1)
        for _ in range(100):
            hist.record(50)
        
        hist.compute_percentiles()
        assert hist.p50 == 1  # 50% are at 1μs
        assert hist.p95 >= 1  # 95% includes some at 50μs

    def test_serialization(self):
        """Test to_dict and from_dict round-trip."""
        hist = LatencyHistogram()
        hist.record(5)
        hist.record(10)
        hist.record(15)
        hist.compute_percentiles()
        
        data = hist.to_dict()
        assert "buckets" in data
        assert "p50" in data
        assert "p95" in data
        assert "p99" in data
        assert "p999" in data
        
        restored = LatencyHistogram.from_dict(data)
        assert restored.buckets == hist.buckets
        assert restored.p50 == hist.p50
        assert restored.p95 == hist.p95
        assert restored.p99 == hist.p99
        assert restored.p999 == hist.p999

    def test_realistic_latency_distribution(self):
        """Test with realistic latency distribution."""
        hist = LatencyHistogram()
        
        # Simulate realistic NVMe latencies
        # Most operations: 5-15 μs
        for _ in range(800):
            hist.record(10)
        
        # Some slower: 20-30 μs
        for _ in range(150):
            hist.record(25)
        
        # Few outliers: 50-100 μs
        for _ in range(50):
            hist.record(75)
        
        hist.compute_percentiles()
        
        assert hist.total_count == 1000
        assert hist.p50 == 10  # Median at fast operations
        assert hist.p95 >= 10  # 95th percentile includes slower ops
        assert hist.p99 >= 25  # 99th percentile includes outliers


# ---------------------------------------------------------------------------
# TelemetryMetrics Tests
# ---------------------------------------------------------------------------


class TestTelemetryMetrics:
    """Tests for TelemetryMetrics."""

    def test_default_construction(self):
        """Test creating default TelemetryMetrics."""
        metrics = TelemetryMetrics()
        assert metrics.read_iops == 0
        assert metrics.write_iops == 0
        assert metrics.read_bandwidth == 0
        assert metrics.write_bandwidth == 0
        assert metrics.queue_depth == 0
        assert isinstance(metrics.read_latency, LatencyHistogram)
        assert isinstance(metrics.write_latency, LatencyHistogram)
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0
        assert metrics.prefetch_hits == 0
        assert metrics.prefetch_misses == 0
        assert metrics.cache_hit_rate == 0.0
        assert metrics.cpu_utilization == 0.0
        assert metrics.memory_used_bytes == 0
        assert metrics.io_bandwidth_used == 0

    def test_timestamp_is_recent(self):
        """Test that timestamp is set to current time."""
        before = time.time_ns()
        metrics = TelemetryMetrics()
        after = time.time_ns()
        assert before <= metrics.timestamp_ns <= after

    def test_custom_values(self):
        """Test creating TelemetryMetrics with custom values."""
        read_hist = LatencyHistogram()
        write_hist = LatencyHistogram()
        
        metrics = TelemetryMetrics(
            timestamp_ns=1000000000,
            read_iops=50000,
            write_iops=30000,
            read_bandwidth=500 * 1024 * 1024,
            write_bandwidth=300 * 1024 * 1024,
            queue_depth=128,
            read_latency=read_hist,
            write_latency=write_hist,
            cache_hits=8000,
            cache_misses=2000,
            prefetch_hits=1000,
            prefetch_misses=500,
            cache_hit_rate=0.8,
            cpu_utilization=0.75,
            memory_used_bytes=16 * 1024 * 1024 * 1024,
            io_bandwidth_used=800 * 1024 * 1024
        )
        
        assert metrics.read_iops == 50000
        assert metrics.write_iops == 30000
        assert metrics.cache_hit_rate == 0.8
        assert metrics.cpu_utilization == 0.75

    def test_invalid_negative_values(self):
        """Test that negative values raise ValueError."""
        with pytest.raises(ValueError, match="read_iops must be >= 0"):
            TelemetryMetrics(read_iops=-1)
        
        with pytest.raises(ValueError, match="write_bandwidth must be >= 0"):
            TelemetryMetrics(write_bandwidth=-1)
        
        with pytest.raises(ValueError, match="cache_hits must be >= 0"):
            TelemetryMetrics(cache_hits=-1)

    def test_invalid_cache_hit_rate(self):
        """Test that invalid cache_hit_rate raises ValueError."""
        with pytest.raises(ValueError, match="cache_hit_rate must be 0.0-1.0"):
            TelemetryMetrics(cache_hit_rate=-0.1)
        
        with pytest.raises(ValueError, match="cache_hit_rate must be 0.0-1.0"):
            TelemetryMetrics(cache_hit_rate=1.1)

    def test_invalid_cpu_utilization(self):
        """Test that invalid cpu_utilization raises ValueError."""
        with pytest.raises(ValueError, match="cpu_utilization must be 0.0-1.0"):
            TelemetryMetrics(cpu_utilization=-0.1)
        
        with pytest.raises(ValueError, match="cpu_utilization must be 0.0-1.0"):
            TelemetryMetrics(cpu_utilization=1.5)

    def test_update_cache_hit_rate_no_accesses(self):
        """Test updating cache hit rate with no cache accesses."""
        metrics = TelemetryMetrics()
        metrics.update_cache_hit_rate()
        assert metrics.cache_hit_rate == 0.0

    def test_update_cache_hit_rate_all_hits(self):
        """Test updating cache hit rate with all hits."""
        metrics = TelemetryMetrics(cache_hits=1000, cache_misses=0)
        metrics.update_cache_hit_rate()
        assert metrics.cache_hit_rate == 1.0

    def test_update_cache_hit_rate_all_misses(self):
        """Test updating cache hit rate with all misses."""
        metrics = TelemetryMetrics(cache_hits=0, cache_misses=1000)
        metrics.update_cache_hit_rate()
        assert metrics.cache_hit_rate == 0.0

    def test_update_cache_hit_rate_mixed(self):
        """Test updating cache hit rate with mixed hits/misses."""
        metrics = TelemetryMetrics(cache_hits=800, cache_misses=200)
        metrics.update_cache_hit_rate()
        assert metrics.cache_hit_rate == 0.8

    def test_serialization(self):
        """Test to_dict and from_dict round-trip."""
        metrics = TelemetryMetrics(
            timestamp_ns=2000000000,
            read_iops=10000,
            write_iops=5000,
            read_bandwidth=100 * 1024 * 1024,
            write_bandwidth=50 * 1024 * 1024,
            queue_depth=64,
            cache_hits=500,
            cache_misses=100,
            cache_hit_rate=0.833,
            cpu_utilization=0.5,
            memory_used_bytes=8 * 1024 * 1024 * 1024,
            io_bandwidth_used=150 * 1024 * 1024
        )
        
        data = metrics.to_dict()
        
        assert data["timestamp_ns"] == 2000000000
        assert data["read_iops"] == 10000
        assert data["write_iops"] == 5000
        assert data["cache_hit_rate"] == 0.833
        assert "read_latency" in data
        assert "write_latency" in data
        
        restored = TelemetryMetrics.from_dict(data)
        assert restored.timestamp_ns == metrics.timestamp_ns
        assert restored.read_iops == metrics.read_iops
        assert restored.write_iops == metrics.write_iops
        assert restored.cache_hit_rate == metrics.cache_hit_rate
        assert restored.cpu_utilization == metrics.cpu_utilization

    def test_realistic_metrics_snapshot(self):
        """Test with realistic metrics values."""
        # Simulate a busy NVMe device
        read_hist = LatencyHistogram()
        for _ in range(1000):
            read_hist.record(8)  # 8μs average read latency
        
        write_hist = LatencyHistogram()
        for _ in range(500):
            write_hist.record(12)  # 12μs average write latency
        
        metrics = TelemetryMetrics(
            read_iops=100_000,
            write_iops=50_000,
            read_bandwidth=1 * 1024 * 1024 * 1024,  # 1 GB/s
            write_bandwidth=512 * 1024 * 1024,      # 512 MB/s
            queue_depth=256,
            read_latency=read_hist,
            write_latency=write_hist,
            cache_hits=80_000,
            cache_misses=20_000,
            prefetch_hits=10_000,
            prefetch_misses=5_000,
            cpu_utilization=0.65,
            memory_used_bytes=12 * 1024 * 1024 * 1024,  # 12 GB
            io_bandwidth_used=1536 * 1024 * 1024        # 1.5 GB/s
        )
        
        metrics.update_cache_hit_rate()
        
        assert metrics.read_iops == 100_000
        assert metrics.write_iops == 50_000
        assert metrics.cache_hit_rate == 0.8
        assert metrics.cpu_utilization == 0.65
        assert metrics.read_latency.total_count == 1000
        assert metrics.write_latency.total_count == 500


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestTelemetryIntegration:
    """Integration tests for telemetry models."""

    def test_histogram_in_metrics(self):
        """Test that histograms work correctly within metrics."""
        metrics = TelemetryMetrics()
        
        # Record some latencies
        for i in range(100):
            metrics.read_latency.record(i % 20)
            metrics.write_latency.record(i % 30)
        
        # Compute percentiles
        metrics.read_latency.compute_percentiles()
        metrics.write_latency.compute_percentiles()
        
        assert metrics.read_latency.total_count == 100
        assert metrics.write_latency.total_count == 100
        assert metrics.read_latency.p50 > 0
        assert metrics.write_latency.p50 > 0

    def test_metrics_evolution_over_time(self):
        """Test metrics changing over time."""
        # Initial snapshot
        metrics1 = TelemetryMetrics(
            timestamp_ns=1000000000,
            read_iops=10000,
            cache_hits=800,
            cache_misses=200
        )
        metrics1.update_cache_hit_rate()
        
        # Later snapshot with increased load
        metrics2 = TelemetryMetrics(
            timestamp_ns=2000000000,
            read_iops=50000,
            cache_hits=4000,
            cache_misses=1000
        )
        metrics2.update_cache_hit_rate()
        
        assert metrics2.timestamp_ns > metrics1.timestamp_ns
        assert metrics2.read_iops > metrics1.read_iops
        assert metrics2.cache_hit_rate == metrics1.cache_hit_rate  # Same ratio

    def test_complete_telemetry_workflow(self):
        """Test complete telemetry collection workflow."""
        # Create metrics
        metrics = TelemetryMetrics()
        
        # Simulate I/O operations
        for i in range(1000):
            # Record latencies
            if i % 2 == 0:
                metrics.read_latency.record(10)
                metrics.cache_hits += 1
            else:
                metrics.write_latency.record(15)
                metrics.cache_misses += 1
        
        # Update derived metrics
        metrics.read_iops = 500
        metrics.write_iops = 500
        metrics.queue_depth = 128
        metrics.cpu_utilization = 0.6
        metrics.update_cache_hit_rate()
        
        # Compute percentiles
        metrics.read_latency.compute_percentiles()
        metrics.write_latency.compute_percentiles()
        
        # Verify
        assert metrics.read_latency.total_count == 500
        assert metrics.write_latency.total_count == 500
        assert metrics.cache_hit_rate == 0.5
        assert metrics.read_latency.p50 == 10
        assert metrics.write_latency.p50 == 15
        
        # Serialize
        data = metrics.to_dict()
        restored = TelemetryMetrics.from_dict(data)
        
        assert restored.cache_hit_rate == metrics.cache_hit_rate
        assert restored.read_latency.total_count == metrics.read_latency.total_count
