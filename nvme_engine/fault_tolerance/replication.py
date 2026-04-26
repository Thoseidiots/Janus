"""
Replication-based fault tolerance for the Software NVMe Engine.

Provides N-way replication with automatic failover, checksum-based
corruption detection, and point-in-time snapshot consistency.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from nvme_engine.backends.base import BackendStats, StorageBackendOps
from nvme_engine.models.errors import (
    NvmeBackendError,
    NvmeDataCorruptionError,
    NvmeIoError,
)


@dataclass
class ReplicationConfig:
    """
    Configuration for the replicated backend.

    Attributes:
        copy_count: Number of replicas (must match the number of backends provided).
        failover_timeout_ms: Maximum time allowed for failover in milliseconds.
        verify_on_read: Whether to verify checksum on every read.
    """

    copy_count: int = 2
    failover_timeout_ms: int = 100
    verify_on_read: bool = True


class ReplicatedBackend(StorageBackendOps):
    """
    Fault-tolerant backend with N-way replication.

    Writes go to ALL replica backends simultaneously.
    Reads come from the primary (first healthy replica).
    On failure or corruption, automatically fails over to the next replica.

    Failover completes within ``failover_timeout_ms`` milliseconds.
    """

    def __init__(self) -> None:
        """Initialize the replicated backend."""
        super().__init__()
        self._backends: List[StorageBackendOps] = []
        self._primary_index: int = 0
        self._config: ReplicationConfig = ReplicationConfig()
        self._size_bytes: int = 0
        # Per-(lba, length) checksums stored by this layer for corruption detection
        # Value: (lba, length, checksum_bytes)
        self._checksums: Dict[tuple, tuple] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self, config: Dict[str, Any]) -> None:
        """
        Initialize the replicated backend.

        Args:
            config: Configuration dictionary with keys:
                - backends: List of already-initialized StorageBackendOps instances.
                - size_bytes: Total addressable size in bytes.
                - copy_count: Expected number of replicas (default: len(backends)).
                - failover_timeout_ms: Failover timeout in ms (default: 100).
                - verify_on_read: Verify checksum on read (default: True).

        Raises:
            NvmeBackendError: If initialization fails.
        """
        if self._initialized:
            raise NvmeBackendError("Replicated backend already initialized")

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
            self._primary_index = 0

            self._size_bytes = config.get("size_bytes", 0)
            if self._size_bytes <= 0:
                raise ValueError("size_bytes must be positive")

            self._config = ReplicationConfig(
                copy_count=config.get("copy_count", len(backends)),
                failover_timeout_ms=config.get("failover_timeout_ms", 100),
                verify_on_read=config.get("verify_on_read", True),
            )

            self._initialized = True

        except Exception as e:
            raise NvmeBackendError(
                f"Failed to initialize replicated backend: {e}",
                details=str(e),
            )

    def destroy(self) -> None:
        """
        Destroy the replicated backend.

        Note: Individual replica backends are NOT destroyed here; the caller
        is responsible for destroying them.

        Raises:
            NvmeBackendError: If cleanup fails.
        """
        if not self._initialized:
            return

        try:
            self._backends = []
            self._checksums.clear()
            self._primary_index = 0
            self._initialized = False

        except Exception as e:
            raise NvmeBackendError(
                f"Failed to destroy replicated backend: {e}",
                details=str(e),
            )

    # ------------------------------------------------------------------
    # Health and failover
    # ------------------------------------------------------------------

    def _detect_failure(self, backend: StorageBackendOps) -> bool:
        """
        Check whether a backend is healthy by attempting a zero-length read.

        Args:
            backend: Backend to check.

        Returns:
            bool: True if the backend appears healthy, False otherwise.
        """
        try:
            # A zero-length read at LBA 0 is a lightweight health probe.
            backend.read(0, 0)
            return True
        except Exception:
            return False

    def _failover(self) -> bool:
        """
        Switch the primary to the next healthy backend.

        Iterates through all backends starting after the current primary.
        Failover must complete within ``failover_timeout_ms`` milliseconds.

        Returns:
            bool: True if a healthy replica was found, False otherwise.

        Raises:
            NvmeBackendError: If no healthy replica is available.
        """
        deadline = time.perf_counter() + self._config.failover_timeout_ms / 1000.0
        n = len(self._backends)

        for offset in range(1, n):
            if time.perf_counter() >= deadline:
                break
            candidate = (self._primary_index + offset) % n
            if self._detect_failure(self._backends[candidate]):
                self._primary_index = candidate
                return True

        raise NvmeBackendError(
            "Failover failed: no healthy replica available within timeout",
            details=f"Checked {n - 1} replicas",
        )

    @property
    def primary_index(self) -> int:
        """Return the index of the current primary backend."""
        return self._primary_index

    @property
    def replica_count(self) -> int:
        """Return the total number of replica backends."""
        return len(self._backends)

    # ------------------------------------------------------------------
    # Checksum helpers
    # ------------------------------------------------------------------

    def _compute_checksum(self, data: bytes) -> bytes:
        """Compute a SHA-256 checksum for the given data."""
        return hashlib.sha256(data).digest()

    def _store_checksum(self, lba: int, data: bytes) -> None:
        """
        Store the checksum for an (lba, length) pair.

        Also invalidates any previously stored checksums whose range
        overlaps with this write, since their data may have changed.
        """
        length = len(data)
        write_end = lba + length

        # Invalidate overlapping checksums
        stale = [
            key
            for key, (stored_lba, stored_len, _) in self._checksums.items()
            if not (stored_lba + stored_len <= lba or stored_lba >= write_end)
        ]
        for key in stale:
            del self._checksums[key]

        key = (lba, length)
        self._checksums[key] = (lba, length, self._compute_checksum(data))

    def _verify_data(self, lba: int, data: bytes) -> bool:
        """
        Verify data against the stored checksum for an (lba, length) pair.

        Returns True if no checksum is stored (first read) or if it matches.
        Returns False if the checksum does not match.
        """
        key = (lba, len(data))
        entry = self._checksums.get(key)
        if entry is None:
            return True
        _, _, expected = entry
        return self._compute_checksum(data) == expected

    # ------------------------------------------------------------------
    # I/O operations
    # ------------------------------------------------------------------

    def read(self, lba: int, length: int) -> bytes:
        """
        Read data from the primary backend with optional checksum verification.

        On corruption or failure, automatically fails over to the next replica.

        Args:
            lba: Logical Block Address (byte offset).
            length: Number of bytes to read.

        Returns:
            bytes: Data read from storage.

        Raises:
            NvmeIoError: If all replicas fail.
            NvmeDataCorruptionError: If corruption is detected and no healthy
                replica can provide clean data.
        """
        if not self._initialized:
            raise NvmeIoError("Replicated backend not initialized")

        start_time = time.perf_counter()

        if lba < 0 or lba >= self._size_bytes:
            raise NvmeIoError(
                f"LBA {lba} out of bounds (size: {self._size_bytes})",
                lba=lba,
            )
        if length < 0:
            raise NvmeIoError(f"Invalid length: {length}", lba=lba)

        # Try each replica in order starting from the current primary.
        # We iterate through ALL backends exactly once, tracking which ones
        # we have already tried so we never revisit a failed/corrupt replica.
        n = len(self._backends)
        last_error: Optional[Exception] = None
        start_idx = self._primary_index

        for attempt in range(n):
            idx = (start_idx + attempt) % n
            backend = self._backends[idx]

            try:
                data = backend.read(lba, length)

                if self._config.verify_on_read and not self._verify_data(lba, data):
                    # Corruption detected on this replica — record and try next
                    self._stats.checksum_errors += 1
                    last_error = NvmeDataCorruptionError(
                        f"Checksum mismatch on replica {idx} at LBA {lba}",
                        lba=lba,
                    )
                    continue

                # Success — update primary to this replica if we failed over
                if attempt > 0:
                    self._primary_index = idx

                self._stats.total_reads += 1
                self._stats.bytes_read += len(data)
                end_time = time.perf_counter()
                latency_us = (end_time - start_time) * 1_000_000
                total_ops = self._stats.total_reads
                self._stats.avg_read_latency_us = (
                    self._stats.avg_read_latency_us * (total_ops - 1) + latency_us
                ) / total_ops

                return data

            except (NvmeDataCorruptionError, NvmeIoError, Exception) as e:
                last_error = e
                # Continue to next replica

        # All replicas exhausted
        self._stats.read_errors += 1
        if isinstance(last_error, NvmeDataCorruptionError):
            raise last_error
        raise NvmeIoError(
            f"All replicas failed for read at LBA {lba}: {last_error}",
            details=str(last_error),
            lba=lba,
        )

    def write(self, lba: int, data: bytes) -> None:
        """
        Write data to ALL replica backends.

        The write succeeds only if a majority of replicas accept it.
        If fewer than a majority succeed, an error is raised.

        Args:
            lba: Logical Block Address (byte offset).
            data: Data to write.

        Raises:
            NvmeIoError: If a majority of replicas fail.
        """
        if not self._initialized:
            raise NvmeIoError("Replicated backend not initialized")

        start_time = time.perf_counter()

        if lba < 0 or lba >= self._size_bytes:
            raise NvmeIoError(
                f"LBA {lba} out of bounds (size: {self._size_bytes})",
                lba=lba,
            )
        length = len(data)
        if length == 0:
            return
        if lba + length > self._size_bytes:
            raise NvmeIoError(
                f"Write would exceed storage size: {lba + length} > {self._size_bytes}",
                lba=lba,
            )

        self._wal_append(lba, data)

        successes = 0
        errors: List[str] = []

        for backend in self._backends:
            try:
                backend.write(lba, data)
                successes += 1
            except Exception as e:
                errors.append(str(e))

        majority = (len(self._backends) // 2) + 1

        if successes < majority:
            self._stats.write_errors += 1
            raise NvmeIoError(
                f"Write failed: only {successes}/{len(self._backends)} replicas succeeded "
                f"(need {majority}): {'; '.join(errors)}",
                details="; ".join(errors),
                lba=lba,
            )

        # Store checksum for future read verification
        self._store_checksum(lba, data)

        self._stats.total_writes += 1
        self._stats.bytes_written += length
        end_time = time.perf_counter()
        latency_us = (end_time - start_time) * 1_000_000
        total_ops = self._stats.total_writes
        self._stats.avg_write_latency_us = (
            self._stats.avg_write_latency_us * (total_ops - 1) + latency_us
        ) / total_ops

    def flush(self) -> None:
        """
        Flush all replica backends.

        Raises:
            NvmeIoError: If any flush fails.
        """
        if not self._initialized:
            raise NvmeIoError("Replicated backend not initialized")

        errors: List[str] = []
        for backend in self._backends:
            try:
                backend.flush()
            except Exception as e:
                errors.append(str(e))

        self._wal_clear()
        self._stats.total_flushes += 1

        if errors:
            raise NvmeIoError(
                f"Flush errors on some replicas: {'; '.join(errors)}",
                details="; ".join(errors),
            )

    def trim(self, lba: int, length: int) -> None:
        """
        Trim a range across all replica backends.

        Args:
            lba: Starting Logical Block Address.
            length: Number of bytes to trim.

        Raises:
            NvmeIoError: If trim fails on all replicas.
        """
        if not self._initialized:
            raise NvmeIoError("Replicated backend not initialized")

        if lba < 0 or lba >= self._size_bytes:
            raise NvmeIoError(
                f"LBA {lba} out of bounds (size: {self._size_bytes})",
                lba=lba,
            )
        if length < 0:
            raise NvmeIoError(f"Invalid length: {length}", lba=lba)

        errors: List[str] = []
        for backend in self._backends:
            try:
                backend.trim(lba, length)
            except Exception as e:
                errors.append(str(e))

        # Remove stored checksums for trimmed range
        stale = [
            key
            for key, (stored_lba, stored_len, _) in self._checksums.items()
            if not (stored_lba + stored_len <= lba or stored_lba >= lba + length)
        ]
        for key in stale:
            del self._checksums[key]
        self._stats.total_trims += 1

        if len(errors) == len(self._backends):
            raise NvmeIoError(
                f"Trim failed on all replicas at LBA {lba}: {'; '.join(errors)}",
                details="; ".join(errors),
                lba=lba,
            )

    # ------------------------------------------------------------------
    # Snapshot operations
    # ------------------------------------------------------------------

    def snapshot_create(self, name: str) -> None:
        """
        Create a point-in-time snapshot on all replica backends.

        Snapshots are created atomically across replicas. If any backend
        fails, already-created snapshots are rolled back.

        Args:
            name: Unique name for the snapshot.

        Raises:
            NvmeBackendError: If snapshot creation fails.
        """
        if not self._initialized:
            raise NvmeBackendError("Replicated backend not initialized")

        created: List[int] = []
        try:
            for i, backend in enumerate(self._backends):
                backend.snapshot_create(name)
                created.append(i)
        except Exception as e:
            # Roll back
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
        Delete a snapshot from all replica backends.

        Args:
            name: Name of the snapshot to delete.

        Raises:
            NvmeBackendError: If snapshot deletion fails on all replicas.
        """
        if not self._initialized:
            raise NvmeBackendError("Replicated backend not initialized")

        errors: List[str] = []
        for backend in self._backends:
            try:
                backend.snapshot_delete(name)
            except Exception as e:
                errors.append(str(e))

        if len(errors) == len(self._backends):
            raise NvmeBackendError(
                f"Failed to delete snapshot '{name}' on all replicas: {'; '.join(errors)}",
                details="; ".join(errors),
            )

    def snapshot_restore(self, name: str) -> None:
        """
        Restore all replica backends to a snapshot state.

        Args:
            name: Name of the snapshot to restore.

        Raises:
            NvmeBackendError: If snapshot restoration fails.
        """
        if not self._initialized:
            raise NvmeBackendError("Replicated backend not initialized")

        errors: List[str] = []
        for backend in self._backends:
            try:
                backend.snapshot_restore(name)
            except Exception as e:
                errors.append(str(e))

        if errors:
            raise NvmeBackendError(
                f"Failed to restore snapshot '{name}' on some replicas: {'; '.join(errors)}",
                details="; ".join(errors),
            )

        # Invalidate stored checksums since data has been restored
        self._checksums.clear()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> BackendStats:
        """
        Return aggregated statistics from all replica backends.

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

        # Add replication-layer counters
        aggregated.total_reads += self._stats.total_reads
        aggregated.total_writes += self._stats.total_writes
        aggregated.total_flushes += self._stats.total_flushes
        aggregated.total_trims += self._stats.total_trims
        aggregated.bytes_read += self._stats.bytes_read
        aggregated.bytes_written += self._stats.bytes_written
        aggregated.read_errors += self._stats.read_errors
        aggregated.write_errors += self._stats.write_errors
        aggregated.checksum_errors += self._stats.checksum_errors
        aggregated.avg_read_latency_us = self._stats.avg_read_latency_us
        aggregated.avg_write_latency_us = self._stats.avg_write_latency_us

        return aggregated
