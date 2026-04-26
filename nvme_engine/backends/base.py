"""
Storage Backend Abstraction Layer for the Software NVMe Engine.

This module defines the abstract base class for all storage backends,
providing a unified interface for memory, file, network, and hybrid backends.
"""

from __future__ import annotations

import hashlib
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BackendStats:
    """Statistics for a storage backend."""

    total_reads: int = 0
    total_writes: int = 0
    total_flushes: int = 0
    total_trims: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    read_errors: int = 0
    write_errors: int = 0
    checksum_errors: int = 0
    avg_read_latency_us: float = 0.0
    avg_write_latency_us: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_reads": self.total_reads,
            "total_writes": self.total_writes,
            "total_flushes": self.total_flushes,
            "total_trims": self.total_trims,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "read_errors": self.read_errors,
            "write_errors": self.write_errors,
            "checksum_errors": self.checksum_errors,
            "avg_read_latency_us": self.avg_read_latency_us,
            "avg_write_latency_us": self.avg_write_latency_us,
        }


@dataclass
class WalEntry:
    """Write-Ahead Log entry for write ordering guarantees."""

    sequence_number: int
    lba: int
    length: int
    checksum: bytes
    data: bytes

    def to_bytes(self) -> bytes:
        """Serialize WAL entry to bytes."""
        header = struct.pack(
            "<QQI32s",
            self.sequence_number,
            self.lba,
            self.length,
            self.checksum,
        )
        return header + self.data

    @classmethod
    def from_bytes(cls, data: bytes) -> "WalEntry":
        """Deserialize WAL entry from bytes."""
        header_size = struct.calcsize("<QQI32s")
        if len(data) < header_size:
            raise ValueError(f"Invalid WAL entry: too short ({len(data)} bytes)")

        seq_num, lba, length, checksum = struct.unpack("<QQI32s", data[:header_size])
        payload = data[header_size : header_size + length]

        if len(payload) != length:
            raise ValueError(
                f"Invalid WAL entry: expected {length} bytes, got {len(payload)}"
            )

        return cls(
            sequence_number=seq_num,
            lba=lba,
            length=length,
            checksum=checksum,
            data=payload,
        )


class StorageBackendOps(ABC):
    """
    Abstract base class for all storage backend implementations.

    All storage backends must implement this interface to provide:
    - Initialization and cleanup
    - Read/write operations with checksums
    - Flush and trim operations
    - Snapshot management
    - Statistics collection
    - Write ordering guarantees via WAL
    """

    def __init__(self) -> None:
        """Initialize the storage backend."""
        self._stats = BackendStats()
        self._wal: List[WalEntry] = []
        self._wal_sequence = 0
        self._initialized = False

    @abstractmethod
    def init(self, config: Dict[str, Any]) -> None:
        """
        Initialize the storage backend with the given configuration.

        Args:
            config: Backend-specific configuration dictionary

        Raises:
            NvmeBackendError: If initialization fails
        """
        pass

    @abstractmethod
    def destroy(self) -> None:
        """
        Destroy the storage backend and release all resources.

        This method must ensure all resources (memory, file handles,
        network connections) are properly released.

        Raises:
            NvmeBackendError: If cleanup fails
        """
        pass

    @abstractmethod
    def read(self, lba: int, length: int) -> bytes:
        """
        Read data from the storage backend.

        Args:
            lba: Logical Block Address to read from
            length: Number of bytes to read

        Returns:
            bytes: Data read from storage

        Raises:
            NvmeIoError: If read operation fails
            NvmeDataCorruptionError: If checksum verification fails
        """
        pass

    @abstractmethod
    def write(self, lba: int, data: bytes) -> None:
        """
        Write data to the storage backend.

        This method must:
        1. Calculate checksum for the data
        2. Add entry to WAL for write ordering
        3. Perform the actual write
        4. Update statistics

        Args:
            lba: Logical Block Address to write to
            data: Data to write

        Raises:
            NvmeIoError: If write operation fails
        """
        pass

    @abstractmethod
    def flush(self) -> None:
        """
        Flush all pending writes to persistent storage.

        This method must ensure all data in write-back caches
        is persisted to the backing storage.

        Raises:
            NvmeIoError: If flush operation fails
        """
        pass

    @abstractmethod
    def trim(self, lba: int, length: int) -> None:
        """
        Trim (deallocate) a range of blocks.

        Args:
            lba: Starting Logical Block Address
            length: Number of bytes to trim

        Raises:
            NvmeIoError: If trim operation fails
        """
        pass

    @abstractmethod
    def snapshot_create(self, name: str) -> None:
        """
        Create a point-in-time snapshot of the storage.

        Args:
            name: Unique name for the snapshot

        Raises:
            NvmeBackendError: If snapshot creation fails
        """
        pass

    @abstractmethod
    def snapshot_delete(self, name: str) -> None:
        """
        Delete a previously created snapshot.

        Args:
            name: Name of the snapshot to delete

        Raises:
            NvmeBackendError: If snapshot deletion fails
        """
        pass

    @abstractmethod
    def snapshot_restore(self, name: str) -> None:
        """
        Restore storage to a previous snapshot state.

        Args:
            name: Name of the snapshot to restore

        Raises:
            NvmeBackendError: If snapshot restoration fails
        """
        pass

    def get_stats(self) -> BackendStats:
        """
        Get current statistics for the storage backend.

        Returns:
            BackendStats: Current backend statistics
        """
        return self._stats

    # Helper methods for checksum and WAL management

    def _calculate_checksum(self, data: bytes) -> bytes:
        """
        Calculate SHA-256 checksum for data.

        Args:
            data: Data to checksum

        Returns:
            bytes: 32-byte SHA-256 checksum
        """
        return hashlib.sha256(data).digest()

    def _verify_checksum(self, data: bytes, expected_checksum: bytes) -> bool:
        """
        Verify data checksum.

        Args:
            data: Data to verify
            expected_checksum: Expected checksum value

        Returns:
            bool: True if checksum matches, False otherwise
        """
        actual_checksum = self._calculate_checksum(data)
        return actual_checksum == expected_checksum

    def _wal_append(self, lba: int, data: bytes) -> None:
        """
        Append a write operation to the Write-Ahead Log.

        Args:
            lba: Logical Block Address being written
            data: Data being written
        """
        checksum = self._calculate_checksum(data)
        entry = WalEntry(
            sequence_number=self._wal_sequence,
            lba=lba,
            length=len(data),
            checksum=checksum,
            data=data,
        )
        self._wal.append(entry)
        self._wal_sequence += 1

    def _wal_clear(self) -> None:
        """Clear the Write-Ahead Log after successful flush."""
        self._wal.clear()

    def _wal_replay(self) -> None:
        """
        Replay Write-Ahead Log entries in order.

        This method is called during recovery to ensure write ordering
        guarantees are maintained after a crash.
        """
        # Sort by sequence number to ensure ordering
        sorted_entries = sorted(self._wal, key=lambda e: e.sequence_number)

        for entry in sorted_entries:
            # Verify checksum before replaying
            if not self._verify_checksum(entry.data, entry.checksum):
                self._stats.checksum_errors += 1
                continue

            # Replay the write
            try:
                self.write(entry.lba, entry.data)
            except Exception:
                # Log error but continue with other entries
                self._stats.write_errors += 1

    @property
    def is_initialized(self) -> bool:
        """Check if the backend is initialized."""
        return self._initialized

    def __enter__(self) -> "StorageBackendOps":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensure cleanup."""
        if self._initialized:
            self.destroy()
