"""
NVMe Submission Queue.

Holds IoRequests waiting to be processed by the queue processor.
Thread-safe implementation using collections.deque + threading.Lock.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Optional

from nvme_engine.models.errors import NvmeConfigError
from nvme_engine.models.io_models import IoRequest

# NVMe spec: max queue depth is 65,535 entries
MAX_QUEUE_DEPTH = 65_535
MIN_QUEUE_DEPTH = 1


class SubmissionQueue:
    """
    NVMe Submission Queue.

    Holds IoRequests waiting to be processed.
    Max depth: 65,535 entries (NVMe spec limit).
    Thread-safe using a deque + lock.
    """

    def __init__(self, queue_id: int, depth: int = 1024) -> None:
        """
        Create a submission queue.

        Args:
            queue_id: Unique identifier for this queue.
            depth: Maximum number of entries (1 <= depth <= 65535).

        Raises:
            NvmeConfigError: If depth is out of range.
        """
        if not (MIN_QUEUE_DEPTH <= depth <= MAX_QUEUE_DEPTH):
            raise NvmeConfigError(
                f"Queue depth {depth} out of range [{MIN_QUEUE_DEPTH}, {MAX_QUEUE_DEPTH}]",
                details=f"queue_id={queue_id}",
            )

        self._queue_id = queue_id
        self._depth = depth
        self._entries: deque[IoRequest] = deque()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, request: IoRequest) -> bool:
        """
        Submit a request to the queue.

        Args:
            request: The IoRequest to enqueue.

        Returns:
            True if the request was accepted; False if the queue is full.
        """
        with self._lock:
            if len(self._entries) >= self._depth:
                return False
            self._entries.append(request)
            return True

    def pop(self) -> Optional[IoRequest]:
        """
        Remove and return the next request (FIFO).

        Returns:
            The next IoRequest, or None if the queue is empty.
        """
        with self._lock:
            if not self._entries:
                return None
            return self._entries.popleft()

    def peek(self) -> Optional[IoRequest]:
        """
        Return the next request without removing it.

        Returns:
            The next IoRequest, or None if the queue is empty.
        """
        with self._lock:
            if not self._entries:
                return None
            return self._entries[0]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def queue_id(self) -> int:
        """Unique identifier for this queue."""
        return self._queue_id

    @property
    def depth(self) -> int:
        """Maximum number of entries this queue can hold."""
        return self._depth

    @depth.setter
    def depth(self, new_depth: int) -> None:
        """
        Update the queue depth (used by expand_queue).

        Args:
            new_depth: New maximum depth.

        Raises:
            NvmeConfigError: If new_depth is out of range.
        """
        if not (MIN_QUEUE_DEPTH <= new_depth <= MAX_QUEUE_DEPTH):
            raise NvmeConfigError(
                f"Queue depth {new_depth} out of range [{MIN_QUEUE_DEPTH}, {MAX_QUEUE_DEPTH}]",
                details=f"queue_id={self._queue_id}",
            )
        with self._lock:
            self._depth = new_depth

    @property
    def size(self) -> int:
        """Current number of entries in the queue."""
        with self._lock:
            return len(self._entries)

    @property
    def utilization(self) -> float:
        """Current utilization as a fraction (0.0 – 1.0)."""
        with self._lock:
            return len(self._entries) / self._depth

    @property
    def is_full(self) -> bool:
        """True if the queue has reached its depth limit."""
        with self._lock:
            return len(self._entries) >= self._depth

    @property
    def is_empty(self) -> bool:
        """True if the queue contains no entries."""
        with self._lock:
            return len(self._entries) == 0

    def __repr__(self) -> str:
        return (
            f"SubmissionQueue(id={self._queue_id}, "
            f"size={self.size}/{self._depth})"
        )
