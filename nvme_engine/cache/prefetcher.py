"""
Sequential Prefetcher for the Software NVMe Cache Manager.

Detects sequential access patterns and returns a list of LBAs to prefetch
ahead of the current access stream.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Deque, List


class Prefetcher:
    """
    Detects sequential access patterns and prefetches ahead.

    Tracks the last ``window`` LBAs accessed. If they form a sequential
    pattern (each LBA = previous + block_size), the next ``prefetch_count``
    blocks are returned as prefetch candidates.
    """

    def __init__(
        self,
        block_size: int = 4096,
        window: int = 4,
        prefetch_count: int = 4,
    ) -> None:
        """
        Initialize the prefetcher.

        Args:
            block_size:     Size of each block in bytes (stride for sequential detection).
            window:         Number of recent accesses to examine for pattern detection.
            prefetch_count: Number of blocks to prefetch when a pattern is detected.
        """
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if prefetch_count <= 0:
            raise ValueError(f"prefetch_count must be positive, got {prefetch_count}")

        self._block_size = block_size
        self._window = window
        self._prefetch_count = prefetch_count

        # Circular buffer of recent LBA accesses
        self._history: Deque[int] = deque(maxlen=window)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_access(self, lba: int) -> List[int]:
        """
        Record an LBA access.

        Returns a list of LBAs to prefetch if a sequential pattern is
        detected, otherwise returns an empty list.

        Args:
            lba: The LBA that was just accessed.

        Returns:
            List of LBAs to prefetch (may be empty).
        """
        with self._lock:
            self._history.append(lba)

            if not self._is_sequential_locked():
                return []

            # Prefetch the next prefetch_count blocks after the last access
            last_lba = self._history[-1]
            return [
                last_lba + self._block_size * (i + 1)
                for i in range(self._prefetch_count)
            ]

    def is_sequential(self) -> bool:
        """Check if recent accesses form a sequential pattern."""
        with self._lock:
            return self._is_sequential_locked()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_sequential_locked(self) -> bool:
        """
        Check sequential pattern without acquiring the lock (caller holds it).

        Returns True only when the history buffer is full and every
        consecutive pair differs by exactly block_size.
        """
        if len(self._history) < self._window:
            return False

        history_list = list(self._history)
        for i in range(1, len(history_list)):
            if history_list[i] - history_list[i - 1] != self._block_size:
                return False
        return True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def block_size(self) -> int:
        """Block size used for sequential stride detection."""
        return self._block_size

    @property
    def window(self) -> int:
        """Number of recent accesses examined for pattern detection."""
        return self._window

    @property
    def prefetch_count(self) -> int:
        """Number of blocks prefetched when a pattern is detected."""
        return self._prefetch_count

    @property
    def history(self) -> List[int]:
        """Copy of the current access history."""
        with self._lock:
            return list(self._history)
