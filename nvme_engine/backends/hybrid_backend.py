"""
Hybrid Backend for the Software NVMe Engine.

Provides multi-tier storage with automatic hot/cold data tiering and
migration between backends based on access frequency.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nvme_engine.backends.base import BackendStats, StorageBackendOps
from nvme_engine.models.errors import NvmeBackendError, NvmeIoError


@dataclass
class TieringPolicy:
    """
    Determines which tier data belongs to based on access frequency.

    Attributes:
        hot_threshold: Minimum access count to be considered hot data.
        cold_threshold: Maximum access count to be considered cold data.
        migration_interval_s: How often (in seconds) to run migration.
    """

    hot_threshold: int = 5
    cold_threshold: int = 0
    migration_interval_s: float = 30.0


class HybridBackend(StorageBackendOps):
    """
    Multi-backend storage with automatic tiering.

    Tier 0 = fastest (e.g., memory)
    Tier 1 = medium (e.g., file)
    Tier N = slowest (e.g., network)

    Hot data (frequently accessed) lives in tier 0.
    Cold data (infrequently accessed) lives in the highest tier.

    Data migration runs periodically in a background thread:
    - Hot LBAs are promoted from slow tiers to tier 0.
    - Cold LBAs are demoted from tier 0 to the slowest tier.
    """

    def __init__(self) -> None:
        """Initialize the hybrid backend."""
        super().__init__()
        self._backends: List[StorageBackendOps] = []
        self._policy: TieringPolicy = TieringPolicy()
        self._access_counts: Dict[int, int] = {}
        self._lba_tier: Dict[int, int] = {}
        self._block_size: int = 4096
        self._size_bytes: int = 0
        self._migration_thread: Optional[threading.Thread] = None
        self._stop_migration = threading.Event()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self, config: Dict[str, Any]) -> None:
        """
        Initialize the hybrid backend with configuration.

        Args:
            config: Configuration dictionary with keys:
                - backends: List of already-initialized StorageBackendOps instances.
                - size_bytes: Total addressable size in bytes (must match backends).
                - block_size: Block size in bytes (default: 4096).
                - hot_threshold: Access count to be considered hot (default: 5).
                - cold_threshold: Access count to be considered cold (default: 0).
                - migration_interval_s: Migration interval in seconds (default: 30.0).
                - auto_migrate: Whether to start background migration (default: True).

        Raises:
            NvmeBackendError: If initialization fails.
        """
        if self._initialized:
            raise NvmeBackendError("Hybrid backend already initialized")

        try:
            backends = config.get("backends", [])
            if not backends or len(backends) < 1:
                raise ValueError("At least one backend must be provided")

            for b in backends:
                if not isinstance(b, StorageBackendOps):
                    raise ValueError("All backends must be StorageBackendOps instances")
                if not b.is_initialized:
                    raise ValueError("All backends must be initialized before use")

            self._backends = list(backends)

            self._size_bytes = config.get("size_bytes", 0)
            if self._size_bytes <= 0:
                raise ValueError("size_bytes must be positive")

            self._block_size = config.get("block_size", 4096)
            if self._block_size <= 0:
                raise ValueError("block_size must be positive")

            self._policy = TieringPolicy(
                hot_threshold=config.get("hot_threshold", 5),
                cold_threshold=config.get("cold_threshold", 0),
                migration_interval_s=config.get("migration_interval_s", 30.0),
            )

            auto_migrate = config.get("auto_migrate", True)
            if auto_migrate and len(self._backends) > 1:
                self._start_migration_thread()

            self._initialized = True

        except Exception as e:
            raise NvmeBackendError(
                f"Failed to initialize hybrid backend: {e}",
                details=str(e),
            )

    def destroy(self) -> None:
        """
        Destroy the hybrid backend and stop background migration.

        Note: Individual backends are NOT destroyed here; the caller is
        responsible for destroying them.

        Raises:
            NvmeBackendError: If cleanup fails.
        """
        if not self._initialized:
            return

        try:
            self._stop_migration_thread()
            self._access_counts.clear()
            self._lba_tier.clear()
            self._backends = []
            self._initialized = False

        except Exception as e:
            raise NvmeBackendError(
                f"Failed to destroy hybrid backend: {e}",
                details=str(e),
            )

    # ------------------------------------------------------------------
    # Background migration thread
    # ------------------------------------------------------------------

    def _start_migration_thread(self) -> None:
        """Start the background data migration thread."""
        self._stop_migration.clear()
        self._migration_thread = threading.Thread(
            target=self._migration_loop,
            daemon=True,
            name="hybrid-migration",
        )
        self._migration_thread.start()

    def _stop_migration_thread(self) -> None:
        """Stop the background data migration thread."""
        self._stop_migration.set()
        if self._migration_thread is not None and self._migration_thread.is_alive():
            self._migration_thread.join(timeout=5.0)
        self._migration_thread = None

    def _migration_loop(self) -> None:
        """Background loop that periodically migrates data between tiers."""
        while not self._stop_migration.wait(timeout=self._policy.migration_interval_s):
            try:
                self._migrate_cold_data()
                self._promote_hot_data()
            except Exception:
                # Migration errors are non-fatal; log and continue
                pass

    # ------------------------------------------------------------------
    # Tier helpers
    # ------------------------------------------------------------------

    def _tier_for_lba(self, lba: int) -> int:
        """
        Return the current tier index for an LBA.

        Defaults to the slowest tier (last backend) if unknown.
        """
        return self._lba_tier.get(lba, len(self._backends) - 1)

    def _backend_for_lba(self, lba: int) -> StorageBackendOps:
        """Return the backend that currently holds data for an LBA."""
        tier = self._tier_for_lba(lba)
        return self._backends[tier]

    def _increment_access(self, lba: int) -> int:
        """Increment and return the access count for an LBA."""
        count = self._access_counts.get(lba, 0) + 1
        self._access_counts[lba] = count
        return count

    # ------------------------------------------------------------------
    # I/O operations
    # ------------------------------------------------------------------

    def read(self, lba: int, length: int) -> bytes:
        """
        Read data from the appropriate tier.

        Increments the access count for the LBA and may trigger promotion
        if the hot threshold is reached.

        Args:
            lba: Logical Block Address (byte offset).
            length: Number of bytes to read.

        Returns:
            bytes: Data read from storage.

        Raises:
            NvmeIoError: If read operation fails.
        """
        if not self._initialized:
            raise NvmeIoError("Hybrid backend not initialized")

        start_time = time.perf_counter()

        try:
            if lba < 0 or lba >= self._size_bytes:
                raise ValueError(f"LBA {lba} out of bounds (size: {self._size_bytes})")
            if length < 0:
                raise ValueError(f"Invalid length: {length}")

            with self._lock:
                count = self._increment_access(lba)
                backend = self._backend_for_lba(lba)

            data = backend.read(lba, length)

            # Promote if hot threshold reached and not already in tier 0
            with self._lock:
                if (
                    count >= self._policy.hot_threshold
                    and self._tier_for_lba(lba) > 0
                    and len(self._backends) > 1
                ):
                    self._promote_lba(lba, data)

            # Update stats
            self._stats.total_reads += 1
            self._stats.bytes_read += len(data)
            end_time = time.perf_counter()
            latency_us = (end_time - start_time) * 1_000_000
            total_ops = self._stats.total_reads
            self._stats.avg_read_latency_us = (
                self._stats.avg_read_latency_us * (total_ops - 1) + latency_us
            ) / total_ops

            return data

        except Exception as e:
            self._stats.read_errors += 1
            raise NvmeIoError(
                f"Hybrid read failed at LBA {lba}: {e}",
                details=str(e),
                lba=lba,
            )

    def write(self, lba: int, data: bytes) -> None:
        """
        Write data to tier 0 (fastest backend).

        New writes always go to the fastest tier. The LBA tier mapping is
        updated accordingly.

        Args:
            lba: Logical Block Address (byte offset).
            data: Data to write.

        Raises:
            NvmeIoError: If write operation fails.
        """
        if not self._initialized:
            raise NvmeIoError("Hybrid backend not initialized")

        start_time = time.perf_counter()

        try:
            if lba < 0 or lba >= self._size_bytes:
                raise ValueError(f"LBA {lba} out of bounds (size: {self._size_bytes})")
            if len(data) == 0:
                return
            if lba + len(data) > self._size_bytes:
                raise ValueError(
                    f"Write would exceed storage size: {lba + len(data)} > {self._size_bytes}"
                )

            self._wal_append(lba, data)

            with self._lock:
                # Write to tier 0 (fastest)
                self._backends[0].write(lba, data)
                old_tier = self._lba_tier.get(lba)
                self._lba_tier[lba] = 0
                self._increment_access(lba)

                # If data was previously in a slower tier, invalidate it there
                # (best-effort; we don't propagate errors from cleanup)
                if old_tier is not None and old_tier > 0:
                    try:
                        self._backends[old_tier].trim(lba, len(data))
                    except Exception:
                        pass

            self._stats.total_writes += 1
            self._stats.bytes_written += len(data)
            end_time = time.perf_counter()
            latency_us = (end_time - start_time) * 1_000_000
            total_ops = self._stats.total_writes
            self._stats.avg_write_latency_us = (
                self._stats.avg_write_latency_us * (total_ops - 1) + latency_us
            ) / total_ops

        except Exception as e:
            self._stats.write_errors += 1
            raise NvmeIoError(
                f"Hybrid write failed at LBA {lba}: {e}",
                details=str(e),
                lba=lba,
            )

    def flush(self) -> None:
        """
        Flush all backends.

        Raises:
            NvmeIoError: If any flush fails.
        """
        if not self._initialized:
            raise NvmeIoError("Hybrid backend not initialized")

        errors = []
        for backend in self._backends:
            try:
                backend.flush()
            except Exception as e:
                errors.append(str(e))

        self._wal_clear()
        self._stats.total_flushes += 1

        if errors:
            raise NvmeIoError(
                f"Hybrid flush had errors: {'; '.join(errors)}",
                details="; ".join(errors),
            )

    def trim(self, lba: int, length: int) -> None:
        """
        Trim a range across all backends.

        Args:
            lba: Starting Logical Block Address.
            length: Number of bytes to trim.

        Raises:
            NvmeIoError: If trim operation fails.
        """
        if not self._initialized:
            raise NvmeIoError("Hybrid backend not initialized")

        try:
            if lba < 0 or lba >= self._size_bytes:
                raise ValueError(f"LBA {lba} out of bounds (size: {self._size_bytes})")
            if length < 0:
                raise ValueError(f"Invalid length: {length}")

            with self._lock:
                for backend in self._backends:
                    try:
                        backend.trim(lba, length)
                    except Exception:
                        pass

                # Remove tier and access tracking for trimmed LBAs
                self._lba_tier.pop(lba, None)
                self._access_counts.pop(lba, None)

            self._stats.total_trims += 1

        except Exception as e:
            raise NvmeIoError(
                f"Hybrid trim failed at LBA {lba}: {e}",
                details=str(e),
                lba=lba,
            )

    # ------------------------------------------------------------------
    # Snapshot operations
    # ------------------------------------------------------------------

    def snapshot_create(self, name: str) -> None:
        """
        Create a snapshot on all backends.

        Args:
            name: Unique name for the snapshot.

        Raises:
            NvmeBackendError: If snapshot creation fails on any backend.
        """
        if not self._initialized:
            raise NvmeBackendError("Hybrid backend not initialized")

        created: List[int] = []
        try:
            for i, backend in enumerate(self._backends):
                backend.snapshot_create(name)
                created.append(i)
        except Exception as e:
            # Roll back already-created snapshots
            for i in created:
                try:
                    self._backends[i].snapshot_delete(name)
                except Exception:
                    pass
            raise NvmeBackendError(
                f"Failed to create snapshot '{name}': {e}",
                details=str(e),
            )

    def snapshot_delete(self, name: str) -> None:
        """
        Delete a snapshot from all backends.

        Args:
            name: Name of the snapshot to delete.

        Raises:
            NvmeBackendError: If snapshot deletion fails.
        """
        if not self._initialized:
            raise NvmeBackendError("Hybrid backend not initialized")

        errors = []
        for backend in self._backends:
            try:
                backend.snapshot_delete(name)
            except Exception as e:
                errors.append(str(e))

        if errors:
            raise NvmeBackendError(
                f"Failed to delete snapshot '{name}': {'; '.join(errors)}",
                details="; ".join(errors),
            )

    def snapshot_restore(self, name: str) -> None:
        """
        Restore all backends to a snapshot state.

        Args:
            name: Name of the snapshot to restore.

        Raises:
            NvmeBackendError: If snapshot restoration fails.
        """
        if not self._initialized:
            raise NvmeBackendError("Hybrid backend not initialized")

        errors = []
        for backend in self._backends:
            try:
                backend.snapshot_restore(name)
            except Exception as e:
                errors.append(str(e))

        if errors:
            raise NvmeBackendError(
                f"Failed to restore snapshot '{name}': {'; '.join(errors)}",
                details="; ".join(errors),
            )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> BackendStats:
        """
        Return aggregated statistics from all backends plus hybrid-level stats.

        Returns:
            BackendStats: Aggregated statistics.
        """
        aggregated = BackendStats()
        for backend in self._backends:
            s = backend.get_stats()
            aggregated.total_reads += s.total_reads
            aggregated.total_writes += s.total_writes
            aggregated.total_flushes += s.total_flushes
            aggregated.total_trims += s.total_trims
            aggregated.bytes_read += s.bytes_read
            aggregated.bytes_written += s.bytes_written
            aggregated.read_errors += s.read_errors
            aggregated.write_errors += s.write_errors
            aggregated.checksum_errors += s.checksum_errors

        # Add hybrid-level counters (reads/writes tracked at this layer)
        aggregated.total_reads += self._stats.total_reads
        aggregated.total_writes += self._stats.total_writes
        aggregated.total_flushes += self._stats.total_flushes
        aggregated.total_trims += self._stats.total_trims
        aggregated.bytes_read += self._stats.bytes_read
        aggregated.bytes_written += self._stats.bytes_written
        aggregated.read_errors += self._stats.read_errors
        aggregated.write_errors += self._stats.write_errors
        aggregated.avg_read_latency_us = self._stats.avg_read_latency_us
        aggregated.avg_write_latency_us = self._stats.avg_write_latency_us

        return aggregated

    # ------------------------------------------------------------------
    # Data migration helpers
    # ------------------------------------------------------------------

    def _promote_lba(self, lba: int, data: bytes) -> None:
        """
        Promote an LBA from its current tier to tier 0.

        Must be called with self._lock held.

        Args:
            lba: LBA to promote.
            data: Current data for the LBA (avoids an extra read).
        """
        current_tier = self._tier_for_lba(lba)
        if current_tier == 0:
            return

        try:
            self._backends[0].write(lba, data)
            # Best-effort cleanup of old tier
            try:
                self._backends[current_tier].trim(lba, len(data))
            except Exception:
                pass
            self._lba_tier[lba] = 0
        except Exception:
            # Promotion failure is non-fatal; data remains in current tier
            pass

    def _promote_hot_data(self) -> None:
        """
        Scan all tracked LBAs and promote hot ones to tier 0.

        Hot = access count >= hot_threshold and currently not in tier 0.
        """
        if len(self._backends) < 2:
            return

        with self._lock:
            hot_lbas = [
                lba
                for lba, count in self._access_counts.items()
                if count >= self._policy.hot_threshold and self._tier_for_lba(lba) > 0
            ]

        for lba in hot_lbas:
            with self._lock:
                current_tier = self._tier_for_lba(lba)
                if current_tier == 0:
                    continue
                try:
                    data = self._backends[current_tier].read(lba, self._block_size)
                    self._promote_lba(lba, data)
                except Exception:
                    pass

    def _migrate_cold_data(self) -> None:
        """
        Scan all tracked LBAs and demote cold ones from tier 0 to the slowest tier.

        Cold = access count <= cold_threshold and currently in tier 0.
        """
        if len(self._backends) < 2:
            return

        slowest_tier = len(self._backends) - 1

        with self._lock:
            cold_lbas = [
                lba
                for lba, count in self._access_counts.items()
                if count <= self._policy.cold_threshold and self._tier_for_lba(lba) == 0
            ]

        for lba in cold_lbas:
            with self._lock:
                if self._tier_for_lba(lba) != 0:
                    continue
                try:
                    data = self._backends[0].read(lba, self._block_size)
                    self._backends[slowest_tier].write(lba, data)
                    try:
                        self._backends[0].trim(lba, self._block_size)
                    except Exception:
                        pass
                    self._lba_tier[lba] = slowest_tier
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def backends(self) -> List[StorageBackendOps]:
        """Return the list of tier backends."""
        return list(self._backends)

    @property
    def tier_count(self) -> int:
        """Return the number of tiers."""
        return len(self._backends)

    @property
    def size_bytes(self) -> int:
        """Return the total addressable size in bytes."""
        return self._size_bytes

    @property
    def policy(self) -> TieringPolicy:
        """Return the current tiering policy."""
        return self._policy
