"""
NVMe Completion Queue.

Holds IoCompletions ready to be consumed by the host.
Thread-safe implementation using collections.deque + threading.Lock.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import List, Optional

from nvme_engine.models.errors import NvmeConfigError
from nvme_engine.models.io_models import IoCompletion

MAX_QUEUE_DEPTH = 65_535
MIN_QUEUE_DEPTH = 1


class CompletionQueue:
    """
    NVMe Completion Queue.

    Holds IoCompletions ready to be consumed by the host.
    Thread-safe using a deque + lock.
    """

    def __init__(self, queue_id: int, depth: int = 1024) -> None:
        """
        Create a completion queue.

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
        self._entries: deque[IoCompletion] = deque()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def post(self, completion: IoCompletion) -> bool:
        """
        Post a completion entry to the queue.

        Args:
            completion: The IoCompletion to enqueue.

        Returns:
            True if posted successfully; False if the queue is full.
        """
        with self._lock:
            if len(self._entries) >= self._depth:
                return False
            self._entries.append(completion)
            return True

    def poll(self) -> Optional[IoCompletion]:
        """
        Remove and return the next completion (FIFO).

        Returns:
            The next IoCompletion, or None if the queue is empty.
        """
        with self._lock:
            if not self._entries:
                return None
            return self._entries.popleft()

    def poll_all(self) -> List[IoCompletion]:
        """
        Remove and return all available completions.

        Returns:
            List of IoCompletion objects (may be empty).
        """
        with self._lock:
            results = list(self._entries)
            self._entries.clear()
            return results

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
        """Update the queue depth."""
        if not (MIN_QUEUE_DEPTH <= new_depth <= MAX_QUEUE_DEPTH):
            raise NvmeConfigError(
                f"Queue depth {new_depth} out of range [{MIN_QUEUE_DEPTH}, {MAX_QUEUE_DEPTH}]",
                details=f"queue_id={self._queue_id}",
            )
        with self._lock:
            self._depth = new_depth

    @property
    def pending_count(self) -> int:
        """Number of completions waiting to be polled."""
        with self._lock:
            return len(self._entries)

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
            f"CompletionQueue(id={self._queue_id}, "
            f"pending={self.pending_count}/{self._depth})"
        )
