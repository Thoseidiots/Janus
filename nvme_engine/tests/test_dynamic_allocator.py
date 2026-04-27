"""
Tests for nvme_engine.control.dynamic_allocator.

Covers: start/stop, utilisation snapshots, scale-up/down triggers,
        device weights, queue-to-CPU assignment, rebalancing,
        predictive scaling, and property-based tests.
"""

from __future__ import annotations

import time
from typing import List

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nvme_engine.control.dynamic_allocator import (
    AllocatorConfig,
    DynamicAllocator,
    ResourceSnapshot,
    ScalingPolicy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_allocator(
    scale_up_threshold: float = 75.0,
    scale_down_threshold: float = 25.0,
    scale_up_duration_s: float = 10.0,
    scale_down_duration_s: float = 60.0,
    rebalance_interval_s: float = 5.0,
    policy: ScalingPolicy = ScalingPolicy.THRESHOLD,
) -> DynamicAllocator:
    cfg = AllocatorConfig(
        scale_up_threshold=scale_up_threshold,
        scale_down_threshold=scale_down_threshold,
        scale_up_duration_s=scale_up_duration_s,
        scale_down_duration_s=scale_down_duration_s,
        rebalance_interval_s=rebalance_interval_s,
        policy=policy,
    )
    return DynamicAllocator(cfg)


def _inject_snapshots(
    allocator: DynamicAllocator,
    cpu: float,
    mem: float,
    count: int = 20,
    span_s: float = 15.0,
) -> None:
    """Inject synthetic snapshots into the allocator's history.

    Injects *count* snapshots spread over *span_s* seconds ending at now,
    plus one extra anchor snapshot at ``now - span_s - 1`` so that the
    duration-check in should_scale_up/down can see history older than the
    required window.
    """
    now = time.monotonic()
    start = now - span_s
    step = span_s / max(count - 1, 1)
    with allocator._lock:
        # Anchor snapshot older than the window
        allocator._history.append(
            ResourceSnapshot(
                timestamp=start - 1.0,
                cpu_percent=cpu,
                memory_percent=mem,
                io_bandwidth_percent=0.0,
            )
        )
        for i in range(count):
            allocator._history.append(
                ResourceSnapshot(
                    timestamp=start + i * step,
                    cpu_percent=cpu,
                    memory_percent=mem,
                    io_bandwidth_percent=0.0,
                )
            )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_start_and_stop(self):
        alloc = _make_allocator(rebalance_interval_s=0.1)
        alloc.start()
        assert alloc._running is True
        alloc.stop()
        assert alloc._running is False

    def test_start_idempotent(self):
        alloc = _make_allocator(rebalance_interval_s=0.1)
        alloc.start()
        alloc.start()  # second call should not raise
        alloc.stop()

    def test_stop_without_start(self):
        alloc = _make_allocator()
        alloc.stop()  # should not raise

    def test_background_thread_collects_snapshots(self):
        alloc = _make_allocator(rebalance_interval_s=0.05)
        alloc.start()
        time.sleep(0.3)
        alloc.stop()
        assert len(alloc.get_history(100)) > 0


# ---------------------------------------------------------------------------
# Utilisation snapshot
# ---------------------------------------------------------------------------


class TestGetUtilization:
    def test_returns_resource_snapshot(self):
        alloc = _make_allocator()
        snap = alloc.get_utilization()
        assert isinstance(snap, ResourceSnapshot)

    def test_snapshot_has_timestamp(self):
        alloc = _make_allocator()
        before = time.monotonic()
        snap = alloc.get_utilization()
        after = time.monotonic()
        assert before <= snap.timestamp <= after

    def test_snapshot_cpu_in_range(self):
        alloc = _make_allocator()
        snap = alloc.get_utilization()
        assert 0.0 <= snap.cpu_percent <= 100.0

    def test_snapshot_memory_in_range(self):
        alloc = _make_allocator()
        snap = alloc.get_utilization()
        assert 0.0 <= snap.memory_percent <= 100.0

    def test_snapshot_io_in_range(self):
        alloc = _make_allocator()
        snap = alloc.get_utilization()
        assert 0.0 <= snap.io_bandwidth_percent <= 100.0


# ---------------------------------------------------------------------------
# Scale-up / scale-down triggers
# ---------------------------------------------------------------------------


class TestScaleTriggers:
    def test_should_scale_up_true_when_high_utilization(self):
        alloc = _make_allocator(scale_up_threshold=50.0, scale_up_duration_s=10.0)
        _inject_snapshots(alloc, cpu=90.0, mem=90.0, count=20, span_s=15.0)
        assert alloc.should_scale_up() is True

    def test_should_scale_up_false_when_low_utilization(self):
        alloc = _make_allocator(scale_up_threshold=75.0, scale_up_duration_s=10.0)
        _inject_snapshots(alloc, cpu=10.0, mem=10.0, count=20, span_s=15.0)
        assert alloc.should_scale_up() is False

    def test_should_scale_up_false_when_no_history(self):
        alloc = _make_allocator()
        assert alloc.should_scale_up() is False

    def test_should_scale_up_false_when_duration_not_met(self):
        alloc = _make_allocator(scale_up_threshold=50.0, scale_up_duration_s=30.0)
        # Only 5 seconds of history, need 30
        _inject_snapshots(alloc, cpu=90.0, mem=90.0, count=5, span_s=5.0)
        assert alloc.should_scale_up() is False

    def test_should_scale_down_true_when_low_utilization(self):
        alloc = _make_allocator(scale_down_threshold=50.0, scale_down_duration_s=10.0)
        _inject_snapshots(alloc, cpu=5.0, mem=5.0, count=20, span_s=15.0)
        assert alloc.should_scale_down() is True

    def test_should_scale_down_false_when_high_utilization(self):
        alloc = _make_allocator(scale_down_threshold=25.0, scale_down_duration_s=10.0)
        _inject_snapshots(alloc, cpu=80.0, mem=80.0, count=20, span_s=15.0)
        assert alloc.should_scale_down() is False

    def test_should_scale_down_false_when_no_history(self):
        alloc = _make_allocator()
        assert alloc.should_scale_down() is False

    def test_should_scale_down_false_when_duration_not_met(self):
        alloc = _make_allocator(scale_down_threshold=50.0, scale_down_duration_s=120.0)
        _inject_snapshots(alloc, cpu=5.0, mem=5.0, count=5, span_s=5.0)
        assert alloc.should_scale_down() is False


# ---------------------------------------------------------------------------
# Scale actions
# ---------------------------------------------------------------------------


class TestScaleActions:
    def test_scale_up_returns_dict(self):
        alloc = _make_allocator()
        result = alloc.scale_up()
        assert isinstance(result, dict)
        assert result["action"] == "scale_up"

    def test_scale_up_has_timestamp(self):
        alloc = _make_allocator()
        before = time.monotonic()
        result = alloc.scale_up()
        after = time.monotonic()
        assert before <= result["timestamp"] <= after

    def test_scale_down_returns_dict(self):
        alloc = _make_allocator()
        result = alloc.scale_down()
        assert isinstance(result, dict)
        assert result["action"] == "scale_down"

    def test_scale_down_has_timestamp(self):
        alloc = _make_allocator()
        before = time.monotonic()
        result = alloc.scale_down()
        after = time.monotonic()
        assert before <= result["timestamp"] <= after


# ---------------------------------------------------------------------------
# Device weights
# ---------------------------------------------------------------------------


class TestDeviceWeights:
    def test_set_and_get_weight(self):
        alloc = _make_allocator()
        alloc.set_device_weight(1, 500.0)
        assert alloc._device_weights[1] == 500.0

    def test_weight_out_of_range_raises(self):
        alloc = _make_allocator()
        with pytest.raises(ValueError):
            alloc.set_device_weight(1, 0.0)
        with pytest.raises(ValueError):
            alloc.set_device_weight(1, 1001.0)

    def test_get_allocation_proportional(self):
        alloc = _make_allocator()
        alloc.set_device_weight(1, 300.0)
        alloc.set_device_weight(2, 700.0)
        alloc1 = alloc.get_device_allocation(1)
        alloc2 = alloc.get_device_allocation(2)
        assert abs(alloc1 - 0.3) < 0.01
        assert abs(alloc2 - 0.7) < 0.01

    def test_get_allocation_sums_to_one(self):
        alloc = _make_allocator()
        for dev_id in range(1, 6):
            alloc.set_device_weight(dev_id, float(dev_id * 100))
        total = sum(alloc.get_device_allocation(dev_id) for dev_id in range(1, 6))
        assert abs(total - 1.0) < 0.001

    def test_get_allocation_no_weights_returns_zero(self):
        alloc = _make_allocator()
        assert alloc.get_device_allocation(1) == 0.0

    def test_get_allocation_unknown_device_returns_zero(self):
        alloc = _make_allocator()
        alloc.set_device_weight(1, 100.0)
        assert alloc.get_device_allocation(99) == 0.0


# ---------------------------------------------------------------------------
# Queue-to-CPU assignment
# ---------------------------------------------------------------------------


class TestQueueAssignment:
    def test_assign_queue_to_cpu(self):
        alloc = _make_allocator()
        alloc.assign_queue_to_cpu(0, 2)
        assert alloc._queue_assignments[0] == 2

    def test_rebalance_distributes_evenly(self):
        alloc = _make_allocator()
        cpu_count = alloc._cpu_count
        for qid in range(cpu_count * 2):
            alloc.assign_queue_to_cpu(qid, 0)
        mapping = alloc.rebalance_queues()
        # Each CPU core should appear at least once
        cores_used = set(mapping.values())
        assert len(cores_used) == min(cpu_count, len(mapping))

    def test_rebalance_returns_mapping(self):
        alloc = _make_allocator()
        alloc.assign_queue_to_cpu(0, 0)
        alloc.assign_queue_to_cpu(1, 0)
        result = alloc.rebalance_queues()
        assert isinstance(result, dict)
        assert 0 in result
        assert 1 in result

    def test_rebalance_completes_within_5s(self):
        alloc = _make_allocator()
        for qid in range(100):
            alloc.assign_queue_to_cpu(qid, 0)
        start = time.monotonic()
        alloc.rebalance_queues()
        assert time.monotonic() - start < 5.0

    def test_rebalance_empty_queues(self):
        alloc = _make_allocator()
        result = alloc.rebalance_queues()
        assert result == {}


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


class TestHistory:
    def test_get_history_empty_initially(self):
        alloc = _make_allocator()
        assert alloc.get_history() == []

    def test_get_history_returns_last_n(self):
        alloc = _make_allocator()
        _inject_snapshots(alloc, cpu=50.0, mem=50.0, count=20, span_s=20.0)
        history = alloc.get_history(last_n=5)
        assert len(history) == 5

    def test_get_history_default_10(self):
        alloc = _make_allocator()
        _inject_snapshots(alloc, cpu=50.0, mem=50.0, count=20, span_s=20.0)
        history = alloc.get_history()
        assert len(history) == 10


# ---------------------------------------------------------------------------
# Predictive scaling
# ---------------------------------------------------------------------------


class TestPredictiveScaling:
    def test_predict_scale_up_true_on_rising_trend(self):
        alloc = _make_allocator(scale_up_threshold=75.0)
        now = time.monotonic()
        with alloc._lock:
            for i in range(10):
                alloc._history.append(
                    ResourceSnapshot(
                        timestamp=now - (10 - i),
                        cpu_percent=60.0 + i * 2.0,  # rising: 60→78
                        memory_percent=50.0,
                        io_bandwidth_percent=0.0,
                    )
                )
        assert alloc.predict_scale_up() is True

    def test_predict_scale_up_false_on_flat_low(self):
        alloc = _make_allocator(scale_up_threshold=75.0)
        _inject_snapshots(alloc, cpu=20.0, mem=20.0, count=10, span_s=10.0)
        assert alloc.predict_scale_up() is False

    def test_predict_scale_up_false_with_insufficient_history(self):
        alloc = _make_allocator()
        # Inject only 1 snapshot (below the 3-sample minimum)
        now = time.monotonic()
        with alloc._lock:
            alloc._history.append(
                ResourceSnapshot(
                    timestamp=now - 1.0,
                    cpu_percent=90.0,
                    memory_percent=90.0,
                    io_bandwidth_percent=0.0,
                )
            )
        # Only 1 sample — need at least 3
        assert alloc.predict_scale_up() is False


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


@given(
    cpu=st.floats(min_value=0.0, max_value=100.0),
    mem=st.floats(min_value=0.0, max_value=100.0),
)
@settings(max_examples=30, deadline=5000)
def test_property7_utilization_snapshot_in_range(cpu: float, mem: float):
    """Property 7: Utilisation snapshot values are always in [0, 100]."""
    alloc = _make_allocator()
    snap = alloc.get_utilization()
    assert 0.0 <= snap.cpu_percent <= 100.0
    assert 0.0 <= snap.memory_percent <= 100.0
    assert 0.0 <= snap.io_bandwidth_percent <= 100.0


@given(
    threshold=st.floats(min_value=50.0, max_value=90.0),
    duration=st.floats(min_value=5.0, max_value=15.0),
)
@settings(max_examples=20, deadline=5000)
def test_property34_scale_up_trigger(threshold: float, duration: float):
    """Property 34: should_scale_up returns True when threshold exceeded for duration."""
    alloc = _make_allocator(
        scale_up_threshold=threshold,
        scale_up_duration_s=duration,
    )
    _inject_snapshots(
        alloc,
        cpu=threshold + 10.0,
        mem=threshold + 10.0,
        count=20,
        span_s=duration + 5.0,
    )
    assert alloc.should_scale_up() is True


@given(
    threshold=st.floats(min_value=10.0, max_value=40.0),
    duration=st.floats(min_value=5.0, max_value=15.0),
)
@settings(max_examples=20, deadline=5000)
def test_property35_scale_down_trigger(threshold: float, duration: float):
    """Property 35: should_scale_down returns True when below threshold for duration."""
    alloc = _make_allocator(
        scale_down_threshold=threshold,
        scale_down_duration_s=duration,
    )
    _inject_snapshots(
        alloc,
        cpu=threshold - 5.0,
        mem=threshold - 5.0,
        count=20,
        span_s=duration + 5.0,
    )
    assert alloc.should_scale_down() is True


@given(queue_count=st.integers(min_value=1, max_value=50))
@settings(max_examples=20, deadline=5000)
def test_property37_rebalance_within_5s(queue_count: int):
    """Property 37: rebalance_queues completes within 5 seconds."""
    alloc = _make_allocator()
    for qid in range(queue_count):
        alloc.assign_queue_to_cpu(qid, 0)
    start = time.monotonic()
    alloc.rebalance_queues()
    assert time.monotonic() - start < 5.0


@given(
    weights=st.lists(
        st.floats(min_value=1.0, max_value=1000.0), min_size=2, max_size=10
    )
)
@settings(max_examples=30, deadline=5000)
def test_property38_weighted_distribution(weights: List[float]):
    """Property 38: Proportional allocations sum to 1.0."""
    alloc = _make_allocator()
    for dev_id, w in enumerate(weights, start=1):
        alloc.set_device_weight(dev_id, w)
    total = sum(alloc.get_device_allocation(dev_id) for dev_id in range(1, len(weights) + 1))
    assert abs(total - 1.0) < 0.001


@given(
    cpu=st.floats(min_value=0.0, max_value=100.0),
    mem=st.floats(min_value=0.0, max_value=100.0),
)
@settings(max_examples=20, deadline=5000)
def test_property33_monitoring_records_snapshots(cpu: float, mem: float):
    """Property 33: Injected snapshots are retrievable from history."""
    alloc = _make_allocator()
    _inject_snapshots(alloc, cpu=cpu, mem=mem, count=5, span_s=5.0)
    history = alloc.get_history(last_n=5)
    assert len(history) == 5
    for snap in history:
        assert abs(snap.cpu_percent - cpu) < 0.001
        assert abs(snap.memory_percent - mem) < 0.001
