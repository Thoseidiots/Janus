"""
Cache Manager for the Software NVMe Engine.

Provides a three-tier write-back cache with:
  - ARC (Adaptive Replacement Cache) algorithm
  - Sequential prefetching
  - Write coalescing
  - Non-blocking tier migration
  - Cache coherency guarantees
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

from nvme_engine.backends.base import StorageBackendOps
from nvme_engine.cache.arc_cache import ArcCache
from nvme_engine.cache.prefetcher import Prefetcher
from nvme_engine.models.config import CacheConfig


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class CacheTier(Enum):
    """Cache tier levels, ordered from fastest to slowest."""

    MEMORY = 0       # Fastest – in-memory cache
    FAST_STORAGE = 1  # Medium – fast SSD simulation
    BACKING = 2      # Slowest – backing storage (the actual backend)


class CacheEntryState(Enum):
    """State of a cache entry."""

    CLEAN = "CLEAN"          # Data matches backing store
    DIRTY = "DIRTY"          # Data has been written but not flushed
    PREFETCHED = "PREFETCHED"  # Data was prefetched speculatively


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    """A single entry in the cache."""

    lba: int
    data: bytes
    tier: CacheTier
    state: CacheEntryState
    access_count: int = 0
    last_access_time: float = field(default_factory=time.time)
    creation_time: float = field(default_factory=time.time)


@dataclass
class CacheStats:
    """Aggregate statistics for the cache manager."""

    hits: int = 0
    misses: int = 0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    evictions: int = 0
    promotions: int = 0
    demotions: int = 0
    dirty_entries: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction [0.0, 1.0]."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Write Coalescer
# ---------------------------------------------------------------------------


class WriteCoalescer:
    """
    Coalesces multiple writes to the same LBA range.

    Buffers writes and merges overlapping/adjacent writes before flushing
    to the backing store.
    """

    def __init__(self, flush_interval_s: float = 30.0) -> None:
        """
        Initialize the write coalescer.

        Args:
            flush_interval_s: Maximum time (seconds) before a forced flush.
                              Must be <= 30 seconds per spec.
        """
        if flush_interval_s <= 0:
            raise ValueError(
                f"flush_interval_s must be positive, got {flush_interval_s}"
            )
        if flush_interval_s > 30.0:
            raise ValueError(
                f"flush_interval_s must be <= 30.0 seconds, got {flush_interval_s}"
            )

        self._flush_interval_s = flush_interval_s
        self._pending: Dict[int, bytes] = {}  # lba -> data
        self._last_flush: float = time.time()
        self._lock = threading.Lock()

    def coalesce(self, lba: int, data: bytes) -> None:
        """
        Buffer a write for coalescing.

        If the same LBA is written multiple times, only the latest data
        is retained (last-write-wins).

        Args:
            lba:  Logical Block Address.
            data: Data to write.
        """
        with self._lock:
            self._pending[lba] = data

    def flush(self, backend: StorageBackendOps) -> int:
        """
        Flush all pending writes to the backend.

        Args:
            backend: The storage backend to flush to.

        Returns:
            Number of writes flushed.
        """
        with self._lock:
            if not self._pending:
                self._last_flush = time.time()
                return 0

            # Snapshot and clear pending under the lock
            pending_snapshot = dict(self._pending)
            self._pending.clear()
            self._last_flush = time.time()

        # Write outside the lock to avoid holding it during I/O
        count = 0
        for lba, data in pending_snapshot.items():
            backend.write(lba, data)
            count += 1

        return count

    def should_flush(self) -> bool:
        """Return True if the flush interval has elapsed."""
        with self._lock:
            return (time.time() - self._last_flush) >= self._flush_interval_s

    @property
    def pending_count(self) -> int:
        """Number of pending (unflushed) writes."""
        with self._lock:
            return len(self._pending)

    @property
    def flush_interval_s(self) -> float:
        """Configured flush interval in seconds."""
        return self._flush_interval_s


# ---------------------------------------------------------------------------
# Cache Manager
# ---------------------------------------------------------------------------


class CacheManager:
    """
    Three-tier write-back cache manager.

    Integrates ARC caching, sequential prefetching, and write coalescing.
    Supports non-blocking tier migration via a background thread.

    Tier hierarchy (fastest → slowest):
      MEMORY → FAST_STORAGE → BACKING
    """

    def __init__(
        self,
        backend: StorageBackendOps,
        config: CacheConfig,
    ) -> None:
        """
        Initialize the cache manager.

        Args:
            backend: The backing storage backend.
            config:  Cache configuration.
        """
        block_size = 4096
        max_entries = max(1, config.size_bytes // block_size)

        self._arc = ArcCache(max_size=max_entries)
        self._prefetcher = Prefetcher(
            block_size=block_size,
            window=max(2, config.prefetch_threshold),
            prefetch_count=config.prefetch_threshold,
        )
        self._coalescer = WriteCoalescer(
            flush_interval_s=min(config.flush_interval_ms / 1000.0, 30.0)
        )
        self._backend = backend
        self._config = config
        self._stats = CacheStats()
        self._block_size = block_size

        # Metadata for tier tracking and dirty state
        # lba -> CacheEntry (lightweight metadata only; data lives in ARC)
        self._metadata: Dict[int, CacheEntry] = {}
        self._metadata_lock = threading.Lock()

        # Set of LBAs that are prefetched (for prefetch-hit tracking)
        self._prefetched_lbas: Set[int] = set()

        # Background migration thread
        self._migration_queue: List[tuple] = []  # (lba, target_tier)
        self._migration_lock = threading.Lock()
        self._migration_event = threading.Event()
        self._shutdown = threading.Event()
        self._migration_thread = threading.Thread(
            target=self._migration_worker,
            name="cache-migration",
            daemon=True,
        )
        self._migration_thread.start()

        # Background flush thread (write-back)
        self._flush_thread = threading.Thread(
            target=self._flush_worker,
            name="cache-flush",
            daemon=True,
        )
        self._flush_thread.start()

    # ------------------------------------------------------------------
    # Public I/O API
    # ------------------------------------------------------------------

    def read(self, lba: int, length: int) -> bytes:
        """
        Read data with cache.

        Checks ARC first; on a miss, fetches from the backend and caches
        the result. Also triggers prefetching for sequential patterns.

        Args:
            lba:    Starting Logical Block Address (byte offset).
            length: Number of bytes to read.

        Returns:
            Data bytes.
        """
        # Align to block boundaries for cache lookup
        block_lba = self._align_lba(lba)

        # Check ARC cache
        cached = self._arc.get(block_lba)
        if cached is not None:
            # Track whether this was a prefetch hit
            with self._metadata_lock:
                if block_lba in self._prefetched_lbas:
                    self._stats.prefetch_hits += 1
                    self._prefetched_lbas.discard(block_lba)
                    # Update state from PREFETCHED to CLEAN
                    if block_lba in self._metadata:
                        self._metadata[block_lba].state = CacheEntryState.CLEAN
                else:
                    self._stats.hits += 1

                if block_lba in self._metadata:
                    entry = self._metadata[block_lba]
                    entry.access_count += 1
                    entry.last_access_time = time.time()

            # Trigger prefetch detection
            self._trigger_prefetch(block_lba)
            return cached[:length]

        # Cache miss – fetch from backend
        self._stats.misses += 1
        data = self._backend.read(lba, length)

        # Pad to block size for caching
        block_data = self._backend.read(block_lba, self._block_size)
        self._arc.put(block_lba, block_data)

        with self._metadata_lock:
            self._metadata[block_lba] = CacheEntry(
                lba=block_lba,
                data=block_data,
                tier=CacheTier.MEMORY,
                state=CacheEntryState.CLEAN,
                access_count=1,
            )

        # Trigger prefetch detection
        self._trigger_prefetch(block_lba)
        return data

    def write(self, lba: int, data: bytes) -> None:
        """
        Write data with write-back caching.

        Writes to ARC and the write coalescer. The coalescer will flush
        to the backend within flush_interval_ms.

        Args:
            lba:  Starting Logical Block Address (byte offset).
            data: Data to write.
        """
        block_lba = self._align_lba(lba)

        # Invalidate any stale prefetched data for this block
        with self._metadata_lock:
            self._prefetched_lbas.discard(block_lba)

        # Write to ARC cache
        # For partial writes, merge with existing cached data
        existing = self._arc.get(block_lba)
        if existing is not None:
            # Merge: overlay the new data at the correct offset
            offset = lba - block_lba
            block_data = bytearray(existing)
            block_data[offset: offset + len(data)] = data
            block_data = bytes(block_data)
        else:
            # No existing cache entry; build a block-aligned buffer
            # We need to read the existing block from backend to merge
            try:
                existing_backend = self._backend.read(block_lba, self._block_size)
                block_data = bytearray(existing_backend)
            except Exception:
                block_data = bytearray(self._block_size)
            offset = lba - block_lba
            block_data[offset: offset + len(data)] = data
            block_data = bytes(block_data)

        self._arc.put(block_lba, block_data)

        with self._metadata_lock:
            if block_lba in self._metadata:
                entry = self._metadata[block_lba]
                entry.data = block_data
                entry.state = CacheEntryState.DIRTY
                entry.access_count += 1
                entry.last_access_time = time.time()
            else:
                self._metadata[block_lba] = CacheEntry(
                    lba=block_lba,
                    data=block_data,
                    tier=CacheTier.MEMORY,
                    state=CacheEntryState.DIRTY,
                    access_count=1,
                )
            self._stats.dirty_entries = self._count_dirty_locked()

        # Buffer in coalescer for write-back
        self._coalescer.coalesce(lba, data)

    def flush(self) -> None:
        """
        Flush all dirty data to the backend immediately.

        Writes all dirty cache entries and clears the coalescer.
        """
        # Flush dirty entries from metadata
        with self._metadata_lock:
            dirty_lbas = [
                lba
                for lba, entry in self._metadata.items()
                if entry.state == CacheEntryState.DIRTY
            ]

        for lba in dirty_lbas:
            cached = self._arc.get(lba)
            if cached is not None:
                self._backend.write(lba, cached)
            with self._metadata_lock:
                if lba in self._metadata:
                    self._metadata[lba].state = CacheEntryState.CLEAN

        # Flush coalescer
        self._coalescer.flush(self._backend)

        with self._metadata_lock:
            self._stats.dirty_entries = self._count_dirty_locked()

    def invalidate(self, lba: int, length: int) -> None:
        """
        Invalidate cache entries covering the given byte range.

        Args:
            lba:    Starting Logical Block Address.
            length: Number of bytes to invalidate.
        """
        start_block = self._align_lba(lba)
        end_byte = lba + length
        current = start_block

        while current < end_byte:
            self._arc.invalidate(current)
            with self._metadata_lock:
                self._metadata.pop(current, None)
                self._prefetched_lbas.discard(current)
            current += self._block_size

        with self._metadata_lock:
            self._stats.dirty_entries = self._count_dirty_locked()

    def promote(self, lba: int) -> bool:
        """
        Manually promote an entry to a faster tier.

        Schedules a non-blocking migration to MEMORY tier.

        Args:
            lba: Logical Block Address to promote.

        Returns:
            True if the entry exists and promotion was scheduled, False otherwise.
        """
        block_lba = self._align_lba(lba)

        with self._metadata_lock:
            if block_lba not in self._metadata:
                return False
            entry = self._metadata[block_lba]
            if entry.tier == CacheTier.MEMORY:
                # Already at the fastest tier
                return False
            target_tier = CacheTier(entry.tier.value - 1)

        # Schedule non-blocking migration
        with self._migration_lock:
            self._migration_queue.append((block_lba, target_tier, "promote"))
        self._migration_event.set()
        return True

    def demote(self, lba: int) -> bool:
        """
        Manually demote an entry to a slower tier.

        Schedules a non-blocking migration to FAST_STORAGE or BACKING tier.

        Args:
            lba: Logical Block Address to demote.

        Returns:
            True if the entry exists and demotion was scheduled, False otherwise.
        """
        block_lba = self._align_lba(lba)

        with self._metadata_lock:
            if block_lba not in self._metadata:
                return False
            entry = self._metadata[block_lba]
            if entry.tier == CacheTier.BACKING:
                # Already at the slowest tier
                return False
            target_tier = CacheTier(entry.tier.value + 1)

        # Schedule non-blocking migration
        with self._migration_lock:
            self._migration_queue.append((block_lba, target_tier, "demote"))
        self._migration_event.set()
        return True

    def shutdown(self) -> None:
        """
        Shut down background threads gracefully.

        Flushes all dirty data before stopping.
        """
        self.flush()
        self._shutdown.set()
        self._migration_event.set()
        self._migration_thread.join(timeout=5.0)
        self._flush_thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> CacheStats:
        """Current cache statistics."""
        with self._metadata_lock:
            self._stats.dirty_entries = self._count_dirty_locked()
        return self._stats

    @property
    def arc(self) -> ArcCache:
        """The underlying ARC cache instance."""
        return self._arc

    @property
    def prefetcher(self) -> Prefetcher:
        """The sequential prefetcher instance."""
        return self._prefetcher

    @property
    def coalescer(self) -> WriteCoalescer:
        """The write coalescer instance."""
        return self._coalescer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _align_lba(self, lba: int) -> int:
        """Align an LBA down to the nearest block boundary."""
        return (lba // self._block_size) * self._block_size

    def _count_dirty_locked(self) -> int:
        """Count dirty entries (caller must hold _metadata_lock)."""
        return sum(
            1
            for entry in self._metadata.values()
            if entry.state == CacheEntryState.DIRTY
        )

    def _trigger_prefetch(self, block_lba: int) -> None:
        """
        Record the access in the prefetcher and issue prefetch reads
        for any detected sequential pattern.
        """
        prefetch_lbas = self._prefetcher.record_access(block_lba)
        for plba in prefetch_lbas:
            # Only prefetch if not already cached
            if self._arc.get(plba) is None:
                try:
                    pdata = self._backend.read(plba, self._block_size)
                    self._arc.put(plba, pdata)
                    with self._metadata_lock:
                        self._metadata[plba] = CacheEntry(
                            lba=plba,
                            data=pdata,
                            tier=CacheTier.MEMORY,
                            state=CacheEntryState.PREFETCHED,
                            access_count=0,
                        )
                        self._prefetched_lbas.add(plba)
                        self._stats.prefetch_misses += 1
                except Exception:
                    pass  # Prefetch failures are non-fatal

    def _migration_worker(self) -> None:
        """Background thread that processes tier migration requests."""
        while not self._shutdown.is_set():
            self._migration_event.wait(timeout=1.0)
            self._migration_event.clear()

            if self._shutdown.is_set():
                break

            # Drain the migration queue
            while True:
                with self._migration_lock:
                    if not self._migration_queue:
                        break
                    task = self._migration_queue.pop(0)

                lba, target_tier, direction = task
                self._execute_migration(lba, target_tier, direction)

    def _execute_migration(
        self, lba: int, target_tier: CacheTier, direction: str
    ) -> None:
        """
        Execute a single tier migration.

        Uses a minimal critical section: only the metadata update is locked.
        The actual data movement happens outside the lock.
        """
        # Read current data (non-blocking, outside lock)
        data = self._arc.get(lba)
        if data is None:
            return  # Entry was evicted before migration ran

        # Update metadata tier (minimal critical section)
        with self._metadata_lock:
            if lba not in self._metadata:
                return
            entry = self._metadata[lba]
            old_tier = entry.tier
            entry.tier = target_tier

            if direction == "promote":
                self._stats.promotions += 1
            else:
                self._stats.demotions += 1

        # If demoting to BACKING, flush to backend
        if target_tier == CacheTier.BACKING and direction == "demote":
            try:
                self._backend.write(lba, data)
                with self._metadata_lock:
                    if lba in self._metadata:
                        self._metadata[lba].state = CacheEntryState.CLEAN
            except Exception:
                # Restore tier on failure
                with self._metadata_lock:
                    if lba in self._metadata:
                        self._metadata[lba].tier = old_tier

    def _flush_worker(self) -> None:
        """Background thread that periodically flushes dirty data."""
        while not self._shutdown.is_set():
            # Sleep in small increments to allow clean shutdown
            time.sleep(min(self._coalescer.flush_interval_s / 10.0, 1.0))

            if self._shutdown.is_set():
                break

            if self._coalescer.should_flush():
                try:
                    self._coalescer.flush(self._backend)
                except Exception:
                    pass  # Flush errors are logged elsewhere
