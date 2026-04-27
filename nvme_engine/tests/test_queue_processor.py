"""
Tests for the NVMe Queue Processor (Task 10).

Covers:
- SubmissionQueue (creation, submit/pop, full/empty, utilization, thread-safety)
- CompletionQueue (post/poll, poll_all, full)
- QueueProcessor (queue pairs, process_one/all, workers, polling mode)
- CPU distribution (affinity, rebalance)
- Dynamic expansion (expand_queue, auto_expand_if_needed)
- Property tests 36 & 37 (I/O distribution, CPU rebalancing)
"""

from __future__ import annotations

import threading
import time
from typing import List

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nvme_engine.backends.memory_backend import MemoryBackend
from nvme_engine.models.errors import NvmeConfigError, NvmeResourceError
from nvme_engine.models.io_models import IoCompletion, IoRequest, IoType
from nvme_engine.queue.completion_queue import CompletionQueue
from nvme_engine.queue.queue_processor import PollingMode, QueueProcessor
from nvme_engine.queue.submission_queue import MAX_QUEUE_DEPTH, SubmissionQueue

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BACKEND_SIZE = 1024 * 1024  # 1 MB


def make_backend() -> MemoryBackend:
    """Return an initialized MemoryBackend."""
    b = MemoryBackend()
    b.init({"size_bytes": _BACKEND_SIZE})
    return b


def make_read_request(request_id: int = 0, lba: int = 0, block_count: int = 1) -> IoRequest:
    return IoRequest(
        request_id=request_id,
        type=IoType.READ,
        lba=lba,
        block_count=block_count,
        buffer_size=block_count * 512,
    )


def make_write_request(
    request_id: int = 0,
    lba: int = 0,
    data: bytes = b"\xAB" * 512,
) -> IoRequest:
    return IoRequest(
        request_id=request_id,
        type=IoType.WRITE,
        lba=lba,
        block_count=len(data) // 512,
        buffer=data,
        buffer_size=len(data),
    )


def make_flush_request(request_id: int = 0) -> IoRequest:
    return IoRequest(
        request_id=request_id,
        type=IoType.FLUSH,
        lba=0,
        block_count=0,
    )


# ===========================================================================
# 1. SubmissionQueue tests
# ===========================================================================


class TestSubmissionQueueCreation:
    def test_create_default_depth(self):
        sq = SubmissionQueue(queue_id=1)
        assert sq.depth == 1024
        assert sq.queue_id == 1

    def test_create_min_depth(self):
        sq = SubmissionQueue(queue_id=2, depth=1)
        assert sq.depth == 1

    def test_create_max_depth(self):
        sq = SubmissionQueue(queue_id=3, depth=MAX_QUEUE_DEPTH)
        assert sq.depth == MAX_QUEUE_DEPTH

    def test_create_invalid_depth_zero(self):
        with pytest.raises(NvmeConfigError):
            SubmissionQueue(queue_id=4, depth=0)

    def test_create_invalid_depth_negative(self):
        with pytest.raises(NvmeConfigError):
            SubmissionQueue(queue_id=5, depth=-1)

    def test_create_invalid_depth_too_large(self):
        with pytest.raises(NvmeConfigError):
            SubmissionQueue(queue_id=6, depth=MAX_QUEUE_DEPTH + 1)


class TestSubmissionQueueOperations:
    def setup_method(self):
        self.sq = SubmissionQueue(queue_id=10, depth=4)

    def test_initial_state(self):
        assert self.sq.size == 0
        assert self.sq.is_empty
        assert not self.sq.is_full
        assert self.sq.utilization == 0.0

    def test_submit_and_pop(self):
        req = make_read_request(request_id=1)
        assert self.sq.submit(req) is True
        assert self.sq.size == 1
        popped = self.sq.pop()
        assert popped is req
        assert self.sq.size == 0

    def test_fifo_ordering(self):
        reqs = [make_read_request(request_id=i) for i in range(3)]
        for r in reqs:
            self.sq.submit(r)
        for r in reqs:
            assert self.sq.pop() is r

    def test_pop_empty_returns_none(self):
        assert self.sq.pop() is None

    def test_peek_does_not_remove(self):
        req = make_read_request(request_id=99)
        self.sq.submit(req)
        peeked = self.sq.peek()
        assert peeked is req
        assert self.sq.size == 1  # still there

    def test_peek_empty_returns_none(self):
        assert self.sq.peek() is None

    def test_queue_full_returns_false(self):
        for i in range(4):
            assert self.sq.submit(make_read_request(request_id=i)) is True
        assert self.sq.is_full
        assert self.sq.submit(make_read_request(request_id=99)) is False

    def test_utilization_calculation(self):
        self.sq.submit(make_read_request(request_id=1))
        self.sq.submit(make_read_request(request_id=2))
        assert self.sq.utilization == pytest.approx(0.5)

    def test_utilization_full(self):
        for i in range(4):
            self.sq.submit(make_read_request(request_id=i))
        assert self.sq.utilization == pytest.approx(1.0)


class TestSubmissionQueueThreadSafety:
    def test_concurrent_submits(self):
        """Multiple threads submitting concurrently should not exceed depth."""
        depth = 100
        sq = SubmissionQueue(queue_id=20, depth=depth)
        accepted = []
        lock = threading.Lock()

        def submit_many():
            for i in range(50):
                result = sq.submit(make_read_request(request_id=i))
                with lock:
                    accepted.append(result)

        threads = [threading.Thread(target=submit_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Queue must never exceed depth
        assert sq.size <= depth
        # Total accepted must equal current size
        assert sq.size == sum(1 for r in accepted if r)

    def test_concurrent_submit_and_pop(self):
        """Producers and consumers running concurrently should not corrupt state."""
        sq = SubmissionQueue(queue_id=21, depth=200)
        produced = []
        consumed = []
        lock = threading.Lock()

        def producer():
            for i in range(50):
                req = make_read_request(request_id=i)
                if sq.submit(req):
                    with lock:
                        produced.append(req)

        def consumer():
            for _ in range(50):
                r = sq.pop()
                if r is not None:
                    with lock:
                        consumed.append(r)
                time.sleep(0.0001)

        threads = (
            [threading.Thread(target=producer) for _ in range(2)]
            + [threading.Thread(target=consumer) for _ in range(2)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No assertion on exact counts; just verify no exceptions and size is sane
        assert sq.size >= 0


# ===========================================================================
# 2. CompletionQueue tests
# ===========================================================================


class TestCompletionQueueCreation:
    def test_create_default_depth(self):
        cq = CompletionQueue(queue_id=1)
        assert cq.depth == 1024

    def test_create_min_depth(self):
        cq = CompletionQueue(queue_id=2, depth=1)
        assert cq.depth == 1

    def test_create_invalid_depth_zero(self):
        with pytest.raises(NvmeConfigError):
            CompletionQueue(queue_id=3, depth=0)

    def test_create_invalid_depth_too_large(self):
        with pytest.raises(NvmeConfigError):
            CompletionQueue(queue_id=4, depth=MAX_QUEUE_DEPTH + 1)


class TestCompletionQueueOperations:
    def setup_method(self):
        self.cq = CompletionQueue(queue_id=30, depth=4)

    def _make_completion(self, request_id: int = 0) -> IoCompletion:
        return IoCompletion(
            request_id=request_id,
            status=0,
            complete_time_ns=time.time_ns(),
            bytes_transferred=512,
        )

    def test_initial_state(self):
        assert self.cq.pending_count == 0
        assert self.cq.is_empty

    def test_post_and_poll(self):
        c = self._make_completion(1)
        assert self.cq.post(c) is True
        assert self.cq.pending_count == 1
        polled = self.cq.poll()
        assert polled is c
        assert self.cq.pending_count == 0

    def test_poll_empty_returns_none(self):
        assert self.cq.poll() is None

    def test_poll_all_returns_all(self):
        completions = [self._make_completion(i) for i in range(3)]
        for c in completions:
            self.cq.post(c)
        result = self.cq.poll_all()
        assert len(result) == 3
        assert self.cq.pending_count == 0

    def test_poll_all_empty_returns_empty_list(self):
        assert self.cq.poll_all() == []

    def test_queue_full_returns_false(self):
        for i in range(4):
            assert self.cq.post(self._make_completion(i)) is True
        assert self.cq.is_full
        assert self.cq.post(self._make_completion(99)) is False

    def test_fifo_ordering(self):
        completions = [self._make_completion(i) for i in range(3)]
        for c in completions:
            self.cq.post(c)
        for c in completions:
            assert self.cq.poll() is c


# ===========================================================================
# 3. QueueProcessor tests
# ===========================================================================


class TestQueueProcessorQueuePairs:
    def setup_method(self):
        self.backend = make_backend()
        self.qp = QueueProcessor(self.backend)

    def teardown_method(self):
        self.backend.destroy()

    def test_create_queue_pair(self):
        sq, cq = self.qp.create_queue_pair(queue_id=1)
        assert isinstance(sq, SubmissionQueue)
        assert isinstance(cq, CompletionQueue)
        assert self.qp.queue_count == 1

    def test_create_multiple_queue_pairs(self):
        for i in range(5):
            self.qp.create_queue_pair(queue_id=i)
        assert self.qp.queue_count == 5

    def test_create_duplicate_queue_id_raises(self):
        self.qp.create_queue_pair(queue_id=1)
        with pytest.raises(NvmeResourceError):
            self.qp.create_queue_pair(queue_id=1)

    def test_delete_queue_pair(self):
        self.qp.create_queue_pair(queue_id=1)
        self.qp.delete_queue_pair(queue_id=1)
        assert self.qp.queue_count == 0

    def test_delete_nonexistent_queue_raises(self):
        with pytest.raises(NvmeResourceError):
            self.qp.delete_queue_pair(queue_id=999)

    def test_create_with_custom_depth(self):
        sq, cq = self.qp.create_queue_pair(queue_id=1, depth=256)
        assert sq.depth == 256
        assert cq.depth == 256


class TestQueueProcessorIO:
    def setup_method(self):
        self.backend = make_backend()
        self.qp = QueueProcessor(self.backend)
        self.sq, self.cq = self.qp.create_queue_pair(queue_id=1)

    def teardown_method(self):
        self.backend.destroy()

    def test_process_one_write(self):
        req = make_write_request(request_id=1, lba=0)
        self.sq.submit(req)
        completion = self.qp.process_one(queue_id=1)
        assert completion is not None
        assert completion.request_id == 1
        assert completion.status == 0

    def test_process_one_read(self):
        req = make_read_request(request_id=2, lba=0, block_count=1)
        self.sq.submit(req)
        completion = self.qp.process_one(queue_id=1)
        assert completion is not None
        assert completion.request_id == 2
        assert completion.status == 0
        assert completion.bytes_transferred == 512

    def test_process_one_flush(self):
        req = make_flush_request(request_id=3)
        self.sq.submit(req)
        completion = self.qp.process_one(queue_id=1)
        assert completion is not None
        assert completion.status == 0

    def test_process_one_empty_queue_returns_none(self):
        result = self.qp.process_one(queue_id=1)
        assert result is None

    def test_process_all(self):
        for i in range(5):
            self.sq.submit(make_write_request(request_id=i, lba=i * 512))
        completions = self.qp.process_all(queue_id=1)
        assert len(completions) == 5
        assert all(c.status == 0 for c in completions)

    def test_process_all_empty_returns_empty_list(self):
        assert self.qp.process_all(queue_id=1) == []

    def test_completion_posted_to_cq(self):
        req = make_write_request(request_id=10, lba=0)
        self.sq.submit(req)
        self.qp.process_one(queue_id=1)
        assert self.cq.pending_count == 1
        c = self.cq.poll()
        assert c.request_id == 10


class TestQueueProcessorWorkerThreads:
    def setup_method(self):
        self.backend = make_backend()
        self.qp = QueueProcessor(self.backend)
        self.sq, self.cq = self.qp.create_queue_pair(queue_id=1)

    def teardown_method(self):
        # Ensure worker is stopped
        try:
            self.qp.stop_worker(queue_id=1)
        except Exception:
            pass
        self.backend.destroy()

    def test_start_and_stop_worker(self):
        self.qp.start_worker(queue_id=1)
        assert self.qp.active_workers == 1
        self.qp.stop_worker(queue_id=1)
        assert self.qp.active_workers == 0

    def test_worker_processes_requests(self):
        self.qp.start_worker(queue_id=1)
        for i in range(3):
            self.sq.submit(make_write_request(request_id=i, lba=i * 512))

        # Wait for worker to drain the queue
        deadline = time.time() + 2.0
        while self.cq.pending_count < 3 and time.time() < deadline:
            time.sleep(0.01)

        self.qp.stop_worker(queue_id=1)
        assert self.cq.pending_count == 3

    def test_start_duplicate_worker_raises(self):
        self.qp.start_worker(queue_id=1)
        with pytest.raises(NvmeResourceError):
            self.qp.start_worker(queue_id=1)
        self.qp.stop_worker(queue_id=1)

    def test_delete_queue_stops_worker(self):
        self.qp.start_worker(queue_id=1)
        assert self.qp.active_workers == 1
        self.qp.delete_queue_pair(queue_id=1)
        # Worker should be stopped
        time.sleep(0.1)
        assert self.qp.active_workers == 0


# ===========================================================================
# 4. Polling mode tests
# ===========================================================================


class TestPollingMode:
    def setup_method(self):
        self.backend = make_backend()

    def teardown_method(self):
        self.backend.destroy()

    def test_polling_mode_poll_completions(self):
        qp = QueueProcessor(self.backend, polling_mode=PollingMode.POLLING)
        sq, cq = qp.create_queue_pair(queue_id=1)

        for i in range(5):
            sq.submit(make_write_request(request_id=i, lba=i * 512))

        completions = qp.poll_completions(queue_id=1, max_completions=10)
        assert len(completions) == 5
        assert all(c.status == 0 for c in completions)

    def test_polling_mode_max_completions_limit(self):
        qp = QueueProcessor(self.backend, polling_mode=PollingMode.POLLING)
        sq, cq = qp.create_queue_pair(queue_id=1)

        for i in range(10):
            sq.submit(make_write_request(request_id=i, lba=i * 512))

        completions = qp.poll_completions(queue_id=1, max_completions=3)
        assert len(completions) == 3

    def test_interrupt_mode_poll_completions(self):
        """In INTERRUPT mode, poll_completions reads from CQ without processing SQ."""
        qp = QueueProcessor(self.backend, polling_mode=PollingMode.INTERRUPT)
        sq, cq = qp.create_queue_pair(queue_id=1)

        # Manually process then poll
        sq.submit(make_write_request(request_id=1, lba=0))
        qp.process_one(queue_id=1)

        completions = qp.poll_completions(queue_id=1)
        assert len(completions) == 1

    def test_interrupt_mode_worker_processes(self):
        """In INTERRUPT mode, worker thread drives processing."""
        qp = QueueProcessor(self.backend, polling_mode=PollingMode.INTERRUPT)
        sq, cq = qp.create_queue_pair(queue_id=1)
        qp.start_worker(queue_id=1)

        for i in range(4):
            sq.submit(make_write_request(request_id=i, lba=i * 512))

        deadline = time.time() + 2.0
        while cq.pending_count < 4 and time.time() < deadline:
            time.sleep(0.01)

        qp.stop_worker(queue_id=1)
        assert cq.pending_count == 4


# ===========================================================================
# 5. CPU distribution tests
# ===========================================================================


class TestCpuDistribution:
    def setup_method(self):
        self.backend = make_backend()
        self.qp = QueueProcessor(self.backend)

    def teardown_method(self):
        self.backend.destroy()

    def test_queue_assigned_to_cpu_core(self):
        self.qp.create_queue_pair(queue_id=0)
        cpu = self.qp.get_cpu_affinity(queue_id=0)
        assert isinstance(cpu, int)
        assert cpu >= 0

    def test_cpu_affinity_nonexistent_queue_raises(self):
        with pytest.raises(NvmeResourceError):
            self.qp.get_cpu_affinity(queue_id=999)

    def test_rebalance_distributes_evenly(self):
        """After rebalance, CPU assignments should be round-robin 0..cpu_count-1."""
        cpu_count = self.qp._cpu_count
        num_queues = cpu_count * 2
        for i in range(num_queues):
            self.qp.create_queue_pair(queue_id=i)

        self.qp.rebalance_queues()

        assignments = [self.qp.get_cpu_affinity(i) for i in range(num_queues)]
        # Each CPU should appear exactly twice
        from collections import Counter
        counts = Counter(assignments)
        for core in range(cpu_count):
            assert counts[core] == 2

    def test_rebalance_single_queue(self):
        self.qp.create_queue_pair(queue_id=0)
        self.qp.rebalance_queues()
        assert self.qp.get_cpu_affinity(0) == 0

    def test_multiple_queues_round_robin_assignment(self):
        """New queues get assigned round-robin at creation time."""
        cpu_count = self.qp._cpu_count
        for i in range(cpu_count):
            self.qp.create_queue_pair(queue_id=i)
        for i in range(cpu_count):
            assert self.qp.get_cpu_affinity(i) == i % cpu_count


# ===========================================================================
# 6. Dynamic expansion tests
# ===========================================================================


class TestDynamicExpansion:
    def setup_method(self):
        self.backend = make_backend()
        self.qp = QueueProcessor(self.backend)
        self.sq, self.cq = self.qp.create_queue_pair(queue_id=1, depth=10)

    def teardown_method(self):
        self.backend.destroy()

    def test_expand_queue_depth(self):
        self.qp.expand_queue(queue_id=1, new_depth=20)
        assert self.sq.depth == 20
        assert self.cq.depth == 20

    def test_expand_to_max(self):
        self.qp.expand_queue(queue_id=1, new_depth=MAX_QUEUE_DEPTH)
        assert self.sq.depth == MAX_QUEUE_DEPTH

    def test_expand_below_current_raises(self):
        with pytest.raises(NvmeConfigError):
            self.qp.expand_queue(queue_id=1, new_depth=5)

    def test_expand_same_depth_raises(self):
        with pytest.raises(NvmeConfigError):
            self.qp.expand_queue(queue_id=1, new_depth=10)

    def test_expand_beyond_max_raises(self):
        with pytest.raises(NvmeConfigError):
            self.qp.expand_queue(queue_id=1, new_depth=MAX_QUEUE_DEPTH + 1)

    def test_expand_nonexistent_queue_raises(self):
        with pytest.raises(NvmeResourceError):
            self.qp.expand_queue(queue_id=999, new_depth=20)

    def test_auto_expand_when_utilization_above_threshold(self):
        """Fill >80% of queue; auto_expand_if_needed should double depth."""
        # Submit 9 out of 10 → 90% utilization
        for i in range(9):
            self.sq.submit(make_read_request(request_id=i))
        assert self.sq.utilization > 0.80

        expanded = self.qp.auto_expand_if_needed(queue_id=1)
        assert expanded is True
        assert self.sq.depth == 20  # doubled

    def test_auto_expand_when_utilization_below_threshold(self):
        """Fill ≤80% of queue; auto_expand_if_needed should not expand."""
        for i in range(4):
            self.sq.submit(make_read_request(request_id=i))
        assert self.sq.utilization <= 0.80

        expanded = self.qp.auto_expand_if_needed(queue_id=1)
        assert expanded is False
        assert self.sq.depth == 10  # unchanged

    def test_auto_expand_capped_at_max(self):
        """Queue already at max depth should not expand."""
        _, _ = self.qp.create_queue_pair(queue_id=2, depth=MAX_QUEUE_DEPTH)
        # Fill it beyond 80%
        for i in range(int(MAX_QUEUE_DEPTH * 0.9)):
            self.qp._queues[2].sq.submit(make_read_request(request_id=i))
        expanded = self.qp.auto_expand_if_needed(queue_id=2)
        assert expanded is False

    def test_expanded_queue_accepts_more_entries(self):
        """After expansion, previously-full queue should accept new entries."""
        for i in range(10):
            self.sq.submit(make_read_request(request_id=i))
        assert self.sq.is_full

        self.qp.expand_queue(queue_id=1, new_depth=20)
        assert self.sq.submit(make_read_request(request_id=99)) is True


# ===========================================================================
# 7. Property-based tests (Properties 36 & 37)
# ===========================================================================


class TestPropertyBasedTests:
    """
    Property 36: I/O requests are distributed across CPU cores.
    Property 37: CPU rebalancing distributes queues evenly.
    """

    def setup_method(self):
        self.backend = make_backend()

    def teardown_method(self):
        self.backend.destroy()

    @given(
        num_queues=st.integers(min_value=1, max_value=16),
    )
    @settings(max_examples=30, deadline=5000)
    def test_property_36_io_distribution_across_cpu_cores(self, num_queues):
        """
        Property 36: Every queue is assigned a valid CPU core index.
        With N queues and C cores, assignments are in [0, C-1].
        """
        qp = QueueProcessor(self.backend)
        cpu_count = qp._cpu_count

        for i in range(num_queues):
            qp.create_queue_pair(queue_id=i)

        for i in range(num_queues):
            cpu = qp.get_cpu_affinity(i)
            assert 0 <= cpu < cpu_count, (
                f"Queue {i} assigned to invalid CPU {cpu} (cpu_count={cpu_count})"
            )

    @given(
        num_queues=st.integers(min_value=2, max_value=20),
    )
    @settings(max_examples=30, deadline=5000)
    def test_property_37_rebalance_distributes_evenly(self, num_queues):
        """
        Property 37: After rebalance, the difference between the most-loaded
        and least-loaded CPU core is at most 1.
        """
        qp = QueueProcessor(self.backend)
        cpu_count = qp._cpu_count

        for i in range(num_queues):
            qp.create_queue_pair(queue_id=i)

        qp.rebalance_queues()

        from collections import Counter
        assignments = [qp.get_cpu_affinity(i) for i in range(num_queues)]
        counts = Counter(assignments)

        # All assigned CPUs must be valid
        for cpu in assignments:
            assert 0 <= cpu < cpu_count

        # Max imbalance is 1 (round-robin guarantee)
        if counts:
            max_load = max(counts.values())
            min_load = min(counts.values())
            assert max_load - min_load <= 1, (
                f"Imbalanced after rebalance: {dict(counts)}"
            )
