"""
NVMe Queue Processor.

Drives I/O requests from submission queues through the storage backend
and posts completions to completion queues.

Features:
- Polling and interrupt modes
- Background worker threads per queue
- CPU affinity tracking (round-robin assignment)
- Dynamic queue depth expansion
"""

from __future__ import annotations

import os
import threading
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple

from nvme_engine.backends.base import StorageBackendOps
from nvme_engine.models.errors import NvmeConfigError, NvmeIoError, NvmeResourceError
from nvme_engine.models.io_models import IoCompletion, IoRequest, IoType
from nvme_engine.queue.completion_queue import CompletionQueue
from nvme_engine.queue.submission_queue import MAX_QUEUE_DEPTH, SubmissionQueue

# Utilization threshold that triggers auto-expansion
_EXPANSION_THRESHOLD = 0.80

# How much to grow the queue on expansion (double, capped at max)
_EXPANSION_FACTOR = 2

# Worker thread sleep interval when idle (seconds)
_WORKER_IDLE_SLEEP = 0.0001  # 100 µs


class PollingMode(Enum):
    """I/O completion detection strategy."""

    INTERRUPT = "interrupt"  # Worker thread signals completion
    POLLING = "polling"      # Caller busy-polls for completions


class _QueuePair:
    """Internal container for a matched submission + completion queue."""

    __slots__ = ("sq", "cq", "cpu_core")

    def __init__(
        self,
        sq: SubmissionQueue,
        cq: CompletionQueue,
        cpu_core: int,
    ) -> None:
        self.sq = sq
        self.cq = cq
        self.cpu_core = cpu_core


class QueueProcessor:
    """
    NVMe Queue Processor.

    Manages queue pairs and drives I/O through the storage backend.
    """

    def __init__(
        self,
        backend: StorageBackendOps,
        polling_mode: PollingMode = PollingMode.INTERRUPT,
    ) -> None:
        """
        Create a QueueProcessor.

        Args:
            backend: Initialized storage backend to execute I/O against.
            polling_mode: Completion detection strategy.
        """
        self._backend = backend
        self._polling_mode = polling_mode

        # queue_id → _QueuePair
        self._queues: Dict[int, _QueuePair] = {}
        self._queues_lock = threading.Lock()

        # queue_id → worker Thread
        self._workers: Dict[int, threading.Thread] = {}
        # queue_id → stop event
        self._stop_events: Dict[int, threading.Event] = {}

        # Number of logical CPU cores available
        self._cpu_count: int = max(1, os.cpu_count() or 1)

    # ------------------------------------------------------------------
    # Queue pair management
    # ------------------------------------------------------------------

    def create_queue_pair(
        self, queue_id: int, depth: int = 1024
    ) -> Tuple[SubmissionQueue, CompletionQueue]:
        """
        Create a matched submission + completion queue pair.

        Args:
            queue_id: Unique identifier for the pair.
            depth: Initial queue depth (1 – 65535).

        Returns:
            Tuple of (SubmissionQueue, CompletionQueue).

        Raises:
            NvmeResourceError: If queue_id already exists.
            NvmeConfigError: If depth is invalid.
        """
        with self._queues_lock:
            if queue_id in self._queues:
                raise NvmeResourceError(
                    f"Queue pair {queue_id} already exists",
                    details=f"queue_id={queue_id}",
                )

            cpu_core = self._assign_cpu(queue_id)
            sq = SubmissionQueue(queue_id, depth)
            cq = CompletionQueue(queue_id, depth)
            self._queues[queue_id] = _QueuePair(sq, cq, cpu_core)

        return sq, cq

    def delete_queue_pair(self, queue_id: int) -> None:
        """
        Delete a queue pair, stopping any associated worker thread.

        Args:
            queue_id: Queue pair to delete.

        Raises:
            NvmeResourceError: If queue_id does not exist.
        """
        # Stop worker first (outside queues_lock to avoid deadlock)
        if queue_id in self._workers:
            self.stop_worker(queue_id)

        with self._queues_lock:
            if queue_id not in self._queues:
                raise NvmeResourceError(
                    f"Queue pair {queue_id} does not exist",
                    details=f"queue_id={queue_id}",
                )
            del self._queues[queue_id]

    # ------------------------------------------------------------------
    # I/O processing
    # ------------------------------------------------------------------

    def process_one(self, queue_id: int) -> Optional[IoCompletion]:
        """
        Dequeue and process one request from the submission queue.

        Args:
            queue_id: Target queue pair.

        Returns:
            IoCompletion if a request was processed; None if queue empty.

        Raises:
            NvmeResourceError: If queue_id does not exist.
        """
        pair = self._get_pair(queue_id)
        request = pair.sq.pop()
        if request is None:
            return None

        completion = self._execute_request(request)
        pair.cq.post(completion)
        return completion

    def process_all(self, queue_id: int) -> List[IoCompletion]:
        """
        Process all pending requests in the submission queue.

        Args:
            queue_id: Target queue pair.

        Returns:
            List of IoCompletion objects for every processed request.

        Raises:
            NvmeResourceError: If queue_id does not exist.
        """
        completions: List[IoCompletion] = []
        while True:
            completion = self.process_one(queue_id)
            if completion is None:
                break
            completions.append(completion)
        return completions

    def poll_completions(
        self, queue_id: int, max_completions: int = 32
    ) -> List[IoCompletion]:
        """
        Poll for completions (polling mode).

        In POLLING mode this also drains the submission queue first so
        that completions are immediately available.

        Args:
            queue_id: Target queue pair.
            max_completions: Maximum number of completions to return.

        Returns:
            List of IoCompletion objects (up to max_completions).

        Raises:
            NvmeResourceError: If queue_id does not exist.
        """
        pair = self._get_pair(queue_id)

        if self._polling_mode == PollingMode.POLLING:
            # Process pending submissions inline
            self.process_all(queue_id)

        results: List[IoCompletion] = []
        for _ in range(max_completions):
            c = pair.cq.poll()
            if c is None:
                break
            results.append(c)
        return results

    # ------------------------------------------------------------------
    # Worker threads
    # ------------------------------------------------------------------

    def start_worker(self, queue_id: int) -> None:
        """
        Start a background worker thread that continuously drains the
        submission queue for the given queue pair.

        Args:
            queue_id: Target queue pair.

        Raises:
            NvmeResourceError: If queue_id does not exist or worker already running.
        """
        self._get_pair(queue_id)  # validate existence

        if queue_id in self._workers and self._workers[queue_id].is_alive():
            raise NvmeResourceError(
                f"Worker for queue {queue_id} is already running",
                details=f"queue_id={queue_id}",
            )

        stop_event = threading.Event()
        self._stop_events[queue_id] = stop_event

        thread = threading.Thread(
            target=self._worker_loop,
            args=(queue_id, stop_event),
            name=f"nvme-worker-{queue_id}",
            daemon=True,
        )
        self._workers[queue_id] = thread
        thread.start()

    def stop_worker(self, queue_id: int) -> None:
        """
        Stop the background worker thread for a queue pair.

        Blocks until the thread has exited.

        Args:
            queue_id: Target queue pair.
        """
        stop_event = self._stop_events.get(queue_id)
        if stop_event is not None:
            stop_event.set()

        thread = self._workers.get(queue_id)
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)

        self._workers.pop(queue_id, None)
        self._stop_events.pop(queue_id, None)

    # ------------------------------------------------------------------
    # CPU distribution
    # ------------------------------------------------------------------

    def get_cpu_affinity(self, queue_id: int) -> int:
        """
        Return the CPU core assigned to this queue (round-robin).

        Args:
            queue_id: Target queue pair.

        Returns:
            CPU core index (0-based).

        Raises:
            NvmeResourceError: If queue_id does not exist.
        """
        pair = self._get_pair(queue_id)
        return pair.cpu_core

    def rebalance_queues(self) -> None:
        """
        Rebalance queue-to-CPU assignments using round-robin across all
        currently registered queues.
        """
        with self._queues_lock:
            for idx, pair in enumerate(self._queues.values()):
                pair.cpu_core = idx % self._cpu_count

    # ------------------------------------------------------------------
    # Dynamic queue depth expansion
    # ------------------------------------------------------------------

    def expand_queue(self, queue_id: int, new_depth: int) -> None:
        """
        Expand the depth of a queue pair.

        Args:
            queue_id: Target queue pair.
            new_depth: New depth; must be > current depth and <= 65535.

        Raises:
            NvmeResourceError: If queue_id does not exist.
            NvmeConfigError: If new_depth is invalid.
        """
        pair = self._get_pair(queue_id)
        current_depth = pair.sq.depth

        if new_depth <= current_depth:
            raise NvmeConfigError(
                f"new_depth {new_depth} must be greater than current depth {current_depth}",
                details=f"queue_id={queue_id}",
            )
        if new_depth > MAX_QUEUE_DEPTH:
            raise NvmeConfigError(
                f"new_depth {new_depth} exceeds maximum {MAX_QUEUE_DEPTH}",
                details=f"queue_id={queue_id}",
            )

        pair.sq.depth = new_depth
        pair.cq.depth = new_depth

    def auto_expand_if_needed(self, queue_id: int) -> bool:
        """
        Check if the submission queue utilization exceeds 80% and expand
        it if so (doubles depth, capped at 65535).

        Args:
            queue_id: Target queue pair.

        Returns:
            True if expansion occurred; False otherwise.

        Raises:
            NvmeResourceError: If queue_id does not exist.
        """
        pair = self._get_pair(queue_id)

        if pair.sq.utilization <= _EXPANSION_THRESHOLD:
            return False

        current_depth = pair.sq.depth
        if current_depth >= MAX_QUEUE_DEPTH:
            return False  # Already at maximum

        new_depth = min(current_depth * _EXPANSION_FACTOR, MAX_QUEUE_DEPTH)
        self.expand_queue(queue_id, new_depth)
        return True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def queue_count(self) -> int:
        """Number of registered queue pairs."""
        with self._queues_lock:
            return len(self._queues)

    @property
    def active_workers(self) -> int:
        """Number of currently running worker threads."""
        return sum(
            1 for t in self._workers.values() if t.is_alive()
        )

    @property
    def polling_mode(self) -> PollingMode:
        """The polling mode this processor was created with."""
        return self._polling_mode

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_pair(self, queue_id: int) -> _QueuePair:
        """Return the _QueuePair for queue_id or raise NvmeResourceError."""
        with self._queues_lock:
            pair = self._queues.get(queue_id)
        if pair is None:
            raise NvmeResourceError(
                f"Queue pair {queue_id} does not exist",
                details=f"queue_id={queue_id}",
            )
        return pair

    def _assign_cpu(self, queue_id: int) -> int:
        """Assign a CPU core to a new queue using round-robin."""
        # Called while _queues_lock is held
        return len(self._queues) % self._cpu_count

    def _execute_request(self, request: IoRequest) -> IoCompletion:
        """
        Execute a single IoRequest against the backend.

        Returns an IoCompletion with status 0 on success or non-zero on error.
        """
        complete_time_ns = time.time_ns()
        bytes_transferred = 0
        status = 0

        try:
            if request.type == IoType.READ:
                length = request.block_count * 512 if request.block_count else request.buffer_size
                if length == 0:
                    length = 512  # default single sector
                data = self._backend.read(request.lba, length)
                bytes_transferred = len(data)

            elif request.type == IoType.WRITE:
                data = request.buffer
                if not data and request.buffer_size:
                    data = b"\x00" * request.buffer_size
                self._backend.write(request.lba, data)
                bytes_transferred = len(data)

            elif request.type == IoType.FLUSH:
                self._backend.flush()

            elif request.type == IoType.TRIM:
                length = request.block_count * 512 if request.block_count else request.buffer_size
                if length == 0:
                    length = 512
                self._backend.trim(request.lba, length)
                bytes_transferred = length

            complete_time_ns = time.time_ns()

        except Exception:
            status = 1  # Generic I/O error status

        return IoCompletion(
            request_id=request.request_id,
            status=status,
            complete_time_ns=complete_time_ns,
            bytes_transferred=bytes_transferred,
        )

    def _worker_loop(self, queue_id: int, stop_event: threading.Event) -> None:
        """
        Background worker: continuously drain the submission queue until
        the stop event is set.
        """
        while not stop_event.is_set():
            try:
                pair = self._get_pair(queue_id)
            except NvmeResourceError:
                # Queue was deleted; exit gracefully
                break

            request = pair.sq.pop()
            if request is None:
                # Nothing to do; yield CPU briefly
                stop_event.wait(timeout=_WORKER_IDLE_SLEEP)
                continue

            completion = self._execute_request(request)
            pair.cq.post(completion)
