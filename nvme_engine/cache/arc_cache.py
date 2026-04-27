"""
Adaptive Replacement Cache (ARC) algorithm implementation.

ARC dynamically balances between recency (LRU) and frequency (LFU) by
maintaining four lists:
  T1 - recently accessed once (recency)
  T2 - accessed more than once (frequency)
  B1 - ghost entries evicted from T1 (history, no data)
  B2 - ghost entries evicted from T2 (history, no data)

The target size p for T1 adapts based on ghost-list hits:
  - Hit in B1 → increase p (favor recency)
  - Hit in B2 → decrease p (favor frequency)
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Optional


class ArcCache:
    """
    Adaptive Replacement Cache (ARC) algorithm.

    Thread-safe implementation using a single lock.
    """

    def __init__(self, max_size: int) -> None:
        """
        Initialize ARC cache.

        Args:
            max_size: Maximum number of cache entries (T1 + T2 combined).
        """
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")

        self.max_size = max_size
        self.p: int = 0  # target size for T1

        # Cache lists (lba -> data)
        self.t1: OrderedDict[int, bytes] = OrderedDict()  # recent, once
        self.t2: OrderedDict[int, bytes] = OrderedDict()  # frequent

        # Ghost lists (lba -> None, no data stored)
        self.b1: OrderedDict[int, None] = OrderedDict()  # ghost T1
        self.b2: OrderedDict[int, None] = OrderedDict()  # ghost T2

        # Statistics
        self._hits: int = 0
        self._misses: int = 0

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, lba: int) -> Optional[bytes]:
        """
        Get data for LBA. Returns None on miss. Updates ARC state.

        On a hit in T1, the entry is promoted to T2 (accessed more than once).
        On a hit in T2, the entry is moved to the MRU end of T2.
        """
        with self._lock:
            # Hit in T1 → promote to T2
            if lba in self.t1:
                data = self.t1.pop(lba)
                self.t2[lba] = data
                self.t2.move_to_end(lba)
                self._hits += 1
                return data

            # Hit in T2 → move to MRU end
            if lba in self.t2:
                self.t2.move_to_end(lba)
                self._hits += 1
                return self.t2[lba]

            # Miss
            self._misses += 1
            return None

    def put(self, lba: int, data: bytes) -> None:
        """
        Insert or update LBA in cache. Evicts as needed per ARC policy.

        Handles four cases:
          1. LBA in T1 or T2 → update data in place
          2. LBA in B1 (ghost) → adapt p upward, replace, move to T2
          3. LBA in B2 (ghost) → adapt p downward, replace, move to T2
          4. New LBA → insert into T1
        """
        with self._lock:
            # Case: already in T1 → update data, keep in T1
            if lba in self.t1:
                self.t1[lba] = data
                self.t1.move_to_end(lba)
                return

            # Case: already in T2 → update data, move to MRU end
            if lba in self.t2:
                self.t2[lba] = data
                self.t2.move_to_end(lba)
                return

            # Case: ghost hit in B1 → adapt p upward
            if lba in self.b1:
                # Increase p: favor recency
                delta = max(1, len(self.b2) // max(len(self.b1), 1))
                self.p = min(self.p + delta, self.max_size)
                self.b1.pop(lba)
                self._replace(in_b2=False)
                self.t2[lba] = data
                self.t2.move_to_end(lba)
                return

            # Case: ghost hit in B2 → adapt p downward
            if lba in self.b2:
                # Decrease p: favor frequency
                delta = max(1, len(self.b1) // max(len(self.b2), 1))
                self.p = max(self.p - delta, 0)
                self.b2.pop(lba)
                self._replace(in_b2=True)
                self.t2[lba] = data
                self.t2.move_to_end(lba)
                return

            # New entry: insert into T1
            # If cache is full, evict
            total = len(self.t1) + len(self.t2)
            if total >= self.max_size:
                self._replace(in_b2=False)
                # Also trim ghost lists if total directory is too large
                total_dir = len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2)
                if total_dir >= 2 * self.max_size:
                    if self.b1:
                        self.b1.popitem(last=False)
                    elif self.b2:
                        self.b2.popitem(last=False)

            self.t1[lba] = data
            self.t1.move_to_end(lba)

    def invalidate(self, lba: int) -> None:
        """Remove LBA from all lists (T1, T2, B1, B2)."""
        with self._lock:
            self.t1.pop(lba, None)
            self.t2.pop(lba, None)
            self.b1.pop(lba, None)
            self.b2.pop(lba, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _replace(self, in_b2: bool) -> None:
        """
        ARC replacement policy.

        Evicts the LRU entry from T1 or T2 based on the target p:
          - If |T1| > p (or T2 is empty and T1 is non-empty), evict from T1 → B1
          - Otherwise evict from T2 → B2

        Args:
            in_b2: True if the triggering ghost hit was in B2 (used to break ties).
        """
        t1_len = len(self.t1)
        t2_len = len(self.t2)

        if t1_len == 0 and t2_len == 0:
            return

        # Decide which list to evict from
        evict_from_t1 = (
            t1_len > 0
            and (
                t1_len > self.p
                or (in_b2 and t1_len == self.p)
                or t2_len == 0
            )
        )

        if evict_from_t1:
            # Evict LRU from T1 → add to B1 ghost
            lba, _ = self.t1.popitem(last=False)
            self.b1[lba] = None
            self.b1.move_to_end(lba)
        else:
            # Evict LRU from T2 → add to B2 ghost
            lba, _ = self.t2.popitem(last=False)
            self.b2[lba] = None
            self.b2.move_to_end(lba)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def t1_size(self) -> int:
        """Number of entries in T1."""
        return len(self.t1)

    @property
    def t2_size(self) -> int:
        """Number of entries in T2."""
        return len(self.t2)

    @property
    def cache_size(self) -> int:
        """Total number of cached entries (T1 + T2)."""
        return len(self.t1) + len(self.t2)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction [0.0, 1.0]."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def hits(self) -> int:
        """Total cache hits."""
        return self._hits

    @property
    def misses(self) -> int:
        """Total cache misses."""
        return self._misses
