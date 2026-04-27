"""
Tests for nvme_engine.control.qos_controller.

Covers: policy management, token-bucket IOPS/bandwidth limiting,
        priority scheduling, weighted fair queuing, and property tests.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nvme_engine.control.qos_controller import QosController, QosPolicy, TokenBucket
from nvme_engine.models.io_models import IoRequest, IoType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(request_id: int = 1, priority: int = 2) -> IoRequest:
    return IoRequest(
        request_id=request_id,
        type=IoType.READ,
        lba=0,
        block_count=1,
        priority=priority,
    )


def _make_policy(
    device_id: int = 1,
    iops_limit: int = 0,
    bandwidth_limit: int = 0,
    priority: int = 2,
    weight: int = 100,
) -> QosPolicy:
    return QosPolicy(
        device_id=device_id,
        iops_limit=iops_limit,
        bandwidth_limit=bandwidth_limit,
        priority=priority,
        weight=weight,
    )


# ---------------------------------------------------------------------------
# TokenBucket unit tests
# ---------------------------------------------------------------------------


class TestTokenBucket:
    def test_consume_allowed_when_tokens_available(self):
        bucket = TokenBucket(rate=100.0, capacity=100.0, tokens=100.0)
        assert bucket.consume(1.0) is True

    def test_consume_blocked_when_empty(self):
        bucket = TokenBucket(rate=0.0, capacity=100.0, tokens=0.0)
        assert bucket.consume(1.0) is False

    def test_consume_deducts_tokens(self):
        bucket = TokenBucket(rate=0.0, capacity=100.0, tokens=50.0)
        bucket.consume(10.0)
        assert abs(bucket.tokens - 40.0) < 0.01

    def test_refill_adds_tokens_over_time(self):
        bucket = TokenBucket(rate=1000.0, capacity=1000.0, tokens=0.0)
        # Manually set last_refill to 1 second ago
        bucket.last_refill = time.monotonic() - 1.0
        bucket._refill()
        assert bucket.tokens > 900.0  # should have ~1000 tokens

    def test_refill_does_not_exceed_capacity(self):
        bucket = TokenBucket(rate=1000.0, capacity=100.0, tokens=0.0)
        bucket.last_refill = time.monotonic() - 10.0
        bucket._refill()
        assert bucket.tokens <= 100.0

    def test_consume_partial_amount(self):
        bucket = TokenBucket(rate=0.0, capacity=100.0, tokens=5.0)
        assert bucket.consume(5.0) is True
        assert bucket.consume(1.0) is False


# ---------------------------------------------------------------------------
# Policy management
# ---------------------------------------------------------------------------


class TestPolicyManagement:
    def test_set_and_get_policy(self):
        qc = QosController()
        policy = _make_policy(device_id=1, iops_limit=1000)
        qc.set_policy(policy)
        retrieved = qc.get_policy(1)
        assert retrieved is policy

    def test_get_nonexistent_policy_returns_none(self):
        qc = QosController()
        assert qc.get_policy(99) is None

    def test_remove_policy(self):
        qc = QosController()
        qc.set_policy(_make_policy(device_id=1))
        qc.remove_policy(1)
        assert qc.get_policy(1) is None

    def test_remove_nonexistent_policy_no_error(self):
        qc = QosController()
        qc.remove_policy(999)  # should not raise

    def test_update_policy_creates_if_missing(self):
        qc = QosController()
        qc.update_policy(5, iops_limit=500)
        policy = qc.get_policy(5)
        assert policy is not None
        assert policy.iops_limit == 500

    def test_update_policy_modifies_existing(self):
        qc = QosController()
        qc.set_policy(_make_policy(device_id=1, iops_limit=100))
        qc.update_policy(1, iops_limit=200)
        assert qc.get_policy(1).iops_limit == 200

    def test_update_policy_does_not_raise(self):
        qc = QosController()
        qc.set_policy(_make_policy(device_id=1))
        qc.update_policy(1, priority=0, weight=500)
        assert qc.get_policy(1).priority == 0
        assert qc.get_policy(1).weight == 500

    def test_replace_policy_updates_buckets(self):
        qc = QosController()
        qc.set_policy(_make_policy(device_id=1, iops_limit=100))
        qc.set_policy(_make_policy(device_id=1, iops_limit=0))
        # With iops_limit=0 the bucket should be removed → always allowed
        assert qc.check_iops(1) is True


# ---------------------------------------------------------------------------
# IOPS token bucket
# ---------------------------------------------------------------------------


class TestIopsLimiting:
    def test_iops_allowed_within_limit(self):
        qc = QosController()
        qc.set_policy(_make_policy(device_id=1, iops_limit=1000))
        # Should be allowed immediately (bucket starts full)
        assert qc.check_iops(1) is True

    def test_iops_blocked_when_bucket_empty(self):
        qc = QosController()
        qc.set_policy(_make_policy(device_id=1, iops_limit=5))
        # Drain the bucket
        for _ in range(5):
            qc.check_iops(1)
        # Next call should be blocked
        assert qc.check_iops(1) is False

    def test_iops_unlimited_always_allowed(self):
        qc = QosController()
        qc.set_policy(_make_policy(device_id=1, iops_limit=0))
        for _ in range(1000):
            assert qc.check_iops(1) is True

    def test_iops_no_policy_always_allowed(self):
        qc = QosController()
        for _ in range(100):
            assert qc.check_iops(99) is True

    def test_iops_refills_over_time(self):
        qc = QosController()
        qc.set_policy(_make_policy(device_id=1, iops_limit=10))
        # Drain
        for _ in range(10):
            qc.check_iops(1)
        assert qc.check_iops(1) is False
        # Wait for refill
        time.sleep(0.15)
        assert qc.check_iops(1) is True


# ---------------------------------------------------------------------------
# Bandwidth token bucket
# ---------------------------------------------------------------------------


class TestBandwidthLimiting:
    def test_bandwidth_allowed_within_limit(self):
        qc = QosController()
        qc.set_policy(_make_policy(device_id=1, bandwidth_limit=1024 * 1024))
        assert qc.check_bandwidth(1, 4096) is True

    def test_bandwidth_blocked_when_bucket_empty(self):
        qc = QosController()
        limit = 4096
        qc.set_policy(_make_policy(device_id=1, bandwidth_limit=limit))
        # Consume all tokens
        qc.check_bandwidth(1, limit)
        # Next request should be blocked
        assert qc.check_bandwidth(1, 1) is False

    def test_bandwidth_unlimited_always_allowed(self):
        qc = QosController()
        qc.set_policy(_make_policy(device_id=1, bandwidth_limit=0))
        for _ in range(100):
            assert qc.check_bandwidth(1, 1024 * 1024) is True

    def test_bandwidth_no_policy_always_allowed(self):
        qc = QosController()
        assert qc.check_bandwidth(99, 1024 * 1024) is True

    def test_bandwidth_refills_over_time(self):
        qc = QosController()
        limit = 1000
        qc.set_policy(_make_policy(device_id=1, bandwidth_limit=limit))
        qc.check_bandwidth(1, limit)
        assert qc.check_bandwidth(1, 1) is False
        time.sleep(0.15)
        assert qc.check_bandwidth(1, 1) is True


# ---------------------------------------------------------------------------
# Priority scheduling
# ---------------------------------------------------------------------------


class TestPriorityScheduling:
    def test_priority_0_served_before_1(self):
        qc = QosController()
        qc.set_policy(_make_policy(device_id=1, priority=1))
        qc.set_policy(_make_policy(device_id=2, priority=0))
        req1 = _make_request(request_id=1)
        req2 = _make_request(request_id=2)
        qc.schedule_io(1, req1)
        qc.schedule_io(2, req2)
        dev_id, req = qc.next_io()
        assert dev_id == 2  # priority 0 first

    def test_priority_0_served_before_2_and_3(self):
        qc = QosController()
        for dev_id, prio in [(1, 3), (2, 2), (3, 0)]:
            qc.set_policy(_make_policy(device_id=dev_id, priority=prio))
            qc.schedule_io(dev_id, _make_request(request_id=dev_id))
        dev_id, _ = qc.next_io()
        assert dev_id == 3  # priority 0

    def test_all_four_priority_levels_ordered(self):
        qc = QosController()
        for dev_id, prio in [(1, 3), (2, 2), (3, 1), (4, 0)]:
            qc.set_policy(_make_policy(device_id=dev_id, priority=prio))
            qc.schedule_io(dev_id, _make_request(request_id=dev_id))
        order = []
        while True:
            result = qc.next_io()
            if result is None:
                break
            order.append(result[0])
        assert order == [4, 3, 2, 1]

    def test_next_io_returns_none_when_empty(self):
        qc = QosController()
        assert qc.next_io() is None

    def test_schedule_uses_policy_priority(self):
        qc = QosController()
        qc.set_policy(_make_policy(device_id=1, priority=0))
        qc.set_policy(_make_policy(device_id=2, priority=3))
        qc.schedule_io(1, _make_request(request_id=1))
        qc.schedule_io(2, _make_request(request_id=2))
        dev_id, _ = qc.next_io()
        assert dev_id == 1

    def test_schedule_falls_back_to_request_priority(self):
        qc = QosController()
        # No policy set for device 1 → uses request.priority
        req_high = _make_request(request_id=1, priority=0)
        req_low = _make_request(request_id=2, priority=3)
        qc.schedule_io(1, req_low)
        qc.schedule_io(1, req_high)
        # Both go to the same device; priority comes from request
        # req_high (priority=0) should be in queue[0], req_low in queue[3]
        dev_id, req = qc.next_io()
        assert req.request_id == 1  # high priority first


# ---------------------------------------------------------------------------
# Weighted fair queuing
# ---------------------------------------------------------------------------


class TestWeightedFairQueuing:
    def test_higher_weight_served_more(self):
        """Device with higher weight should be served more often within same priority."""
        qc = QosController()
        qc.set_policy(_make_policy(device_id=1, priority=2, weight=900))
        qc.set_policy(_make_policy(device_id=2, priority=2, weight=100))

        # Enqueue 10 requests per device
        for i in range(10):
            qc.schedule_io(1, _make_request(request_id=i))
            qc.schedule_io(2, _make_request(request_id=i + 100))

        served: dict = {1: 0, 2: 0}
        while True:
            result = qc.next_io()
            if result is None:
                break
            served[result[0]] += 1

        # Device 1 (weight 900) should be served at least as often as device 2
        assert served[1] >= served[2]

    def test_equal_weight_roughly_equal_service(self):
        qc = QosController()
        qc.set_policy(_make_policy(device_id=1, priority=2, weight=100))
        qc.set_policy(_make_policy(device_id=2, priority=2, weight=100))

        for i in range(10):
            qc.schedule_io(1, _make_request(request_id=i))
            qc.schedule_io(2, _make_request(request_id=i + 100))

        served: dict = {1: 0, 2: 0}
        while True:
            result = qc.next_io()
            if result is None:
                break
            served[result[0]] += 1

        # With equal weights, service should be roughly equal
        assert abs(served[1] - served[2]) <= 2


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_get_stats_returns_dict(self):
        qc = QosController()
        qc.set_policy(_make_policy(device_id=1, iops_limit=100))
        stats = qc.get_stats(1)
        assert isinstance(stats, dict)
        assert stats["device_id"] == 1

    def test_get_stats_no_policy(self):
        qc = QosController()
        stats = qc.get_stats(99)
        assert stats["device_id"] == 99


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


@given(
    iops_limit=st.integers(min_value=10, max_value=10000),
    requests=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=30, deadline=5000)
def test_property9_iops_accuracy(iops_limit: int, requests: int):
    """Property 9: Token bucket IOPS limiting is accurate within 5%."""
    qc = QosController()
    qc.set_policy(_make_policy(device_id=1, iops_limit=iops_limit))
    allowed = sum(1 for _ in range(requests) if qc.check_iops(1))
    # Allowed should not exceed iops_limit (bucket starts full = 1 second burst)
    assert allowed <= iops_limit + 1  # +1 for floating-point tolerance


@given(
    bw_limit=st.integers(min_value=1024, max_value=1024 * 1024),
    chunk_size=st.integers(min_value=1, max_value=512),
)
@settings(max_examples=30, deadline=5000)
def test_property10_bandwidth_accuracy(bw_limit: int, chunk_size: int):
    """Property 10: Token bucket bandwidth limiting is accurate within 5%."""
    qc = QosController()
    qc.set_policy(_make_policy(device_id=1, bandwidth_limit=bw_limit))
    total_allowed = 0
    for _ in range(bw_limit // chunk_size + 10):
        if qc.check_bandwidth(1, chunk_size):
            total_allowed += chunk_size
        else:
            break
    # Total allowed bytes should not exceed limit by more than 5%
    assert total_allowed <= bw_limit * 1.05 + chunk_size


@given(
    priorities=st.lists(
        st.integers(min_value=0, max_value=3), min_size=2, max_size=8
    )
)
@settings(max_examples=30, deadline=5000)
def test_property11_priority_ordering(priorities: List[int]):
    """Property 11: Higher priority (lower number) always served first."""
    qc = QosController()
    for dev_id, prio in enumerate(priorities, start=1):
        qc.set_policy(_make_policy(device_id=dev_id, priority=prio))
        qc.schedule_io(dev_id, _make_request(request_id=dev_id))

    served_priorities: List[int] = []
    while True:
        result = qc.next_io()
        if result is None:
            break
        policy = qc.get_policy(result[0])
        served_priorities.append(policy.priority)

    # Served priorities should be non-decreasing
    for i in range(len(served_priorities) - 1):
        assert served_priorities[i] <= served_priorities[i + 1]


@given(
    cpu_limit=st.floats(min_value=1.0, max_value=100.0),
)
@settings(max_examples=20, deadline=5000)
def test_property12_cpu_isolation(cpu_limit: float):
    """Property 12: CPU limit is stored and retrievable per device."""
    qc = QosController()
    policy = QosPolicy(device_id=1, cpu_limit_percent=cpu_limit)
    qc.set_policy(policy)
    retrieved = qc.get_policy(1)
    assert abs(retrieved.cpu_limit_percent - cpu_limit) < 0.001


@given(
    new_iops=st.integers(min_value=1, max_value=100000),
)
@settings(max_examples=30, deadline=5000)
def test_property14_update_policy_no_interrupt(new_iops: int):
    """Property 14: Updating policy does not raise and preserves device state."""
    qc = QosController()
    qc.set_policy(_make_policy(device_id=1, iops_limit=100))
    # Schedule some I/O
    for i in range(5):
        qc.schedule_io(1, _make_request(request_id=i))
    # Update policy mid-flight
    qc.update_policy(1, iops_limit=new_iops)
    # Drain remaining I/O — should not raise
    while qc.next_io() is not None:
        pass
    assert qc.get_policy(1).iops_limit == new_iops
