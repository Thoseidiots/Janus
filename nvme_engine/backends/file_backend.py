"""
File Backend for the Software NVMe Engine.

Provides persistent storage using regular files with sparse file support,
direct I/O mode, and file-based snapshots.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

from nvme_engine.backends.base import StorageBackendOps
from nvme_engine.models.errors import NvmeBackendError, NvmeDataCorruptionError, NvmeIoError


class FileBackend(StorageBackendOps):
    """
    File-based storage backend using regular files.

    Features:
    - Persistent storage using files
    - Sparse file allocation to minimize disk usage
    - Direct I/O mode (O_DIRECT flag simulation)
    - File-based snapshots via copy-on-write
    """

    def __init__(self) -> None:
        """Initialize the file backend."""
        super().__init__()
        self._file_path: Optional[Path] = None
        self._file_handle: Optional[Any] = None
        self._sparse: bool = True
        self._direct_io: bool = False
        self._size_bytes: int = 0
        self._snapshots_dir: Optional[Path] = None

    def init(self, config: Dict[str, Any]) -> None:
        """
        Initialize the file backend with configuration.

        Args:
            config: Configuration dictionary with keys:
                - path: Path to the storage file
                - sparse: Enable sparse file allocation (default: True)
                - direct_io: Enable direct I/O mode (default: False)
                - size_bytes: Total size in bytes

        Raises:
            NvmeBackendError: If initialization fails
        """
        if self._initialized:
            raise NvmeBackendError("File backend already initialized")

        try:
            path_str = config.get("path")
            if not path_str:
                raise ValueError("path must be specified")

            self._file_path = Path(path_str)
            self._sparse = config.get("sparse", True)
            self._direct_io = config.get("direct_io", False)
            self._size_bytes = config.get("size_bytes", 0)

            if self._size_bytes <= 0:
                raise ValueError("size_bytes must be positive")

            # Create parent directory if it doesn't exist
            self._file_path.parent.mkdir(parents=True, exist_ok=True)

            # Create snapshots directory
            self._snapshots_dir = self._file_path.parent / f".{self._file_path.name}_snapshots"
            self._snapshots_dir.mkdir(exist_ok=True)

            # Open or create the file
            if self._sparse:
                # Create sparse file
                self._create_sparse_file()
            else:
                # Create regular file
                self._create_regular_file()

            # Open file for read/write
            mode = "r+b"
            self._file_handle = open(self._file_path, mode, buffering=0 if self._direct_io else -1)

            self._initialized = True

        except Exception as e:
            raise NvmeBackendError(
                f"Failed to initialize file backend: {e}",
                details=str(e),
            )

    def _create_sparse_file(self) -> None:
        """Create a sparse file."""
        if self._file_path.exists():
            # File already exists, verify size
            current_size = self._file_path.stat().st_size
            if current_size < self._size_bytes:
                # Extend the file
                with open(self._file_path, "ab") as f:
                    f.truncate(self._size_bytes)
        else:
            # Create new sparse file
            with open(self._file_path, "wb") as f:
                f.seek(self._size_bytes - 1)
                f.write(b"\x00")

    def _create_regular_file(self) -> None:
        """Create a regular (non-sparse) file."""
        if not self._file_path.exists():
            with open(self._file_path, "wb") as f:
                # Write zeros to allocate space
                chunk_size = 1024 * 1024  # 1MB chunks
                remaining = self._size_bytes
                while remaining > 0:
                    write_size = min(chunk_size, remaining)
                    f.write(b"\x00" * write_size)
                    remaining -= write_size

    def destroy(self) -> None:
        """
        Destroy the file backend and release all resources.

        Raises:
            NvmeBackendError: If cleanup fails
        """
        if not self._initialized:
            return

        try:
            # Close file handle
            if self._file_handle is not None:
                self._file_handle.close()
                self._file_handle = None

            # Note: We don't delete the file itself as it contains persistent data
            # Users should manually delete the file if needed

            self._initialized = False

        except Exception as e:
            raise NvmeBackendError(
                f"Failed to destroy file backend: {e}",
                details=str(e),
            )

    def read(self, lba: int, length: int) -> bytes:
        """
        Read data from file storage.

        Args:
            lba: Logical Block Address (byte offset)
            length: Number of bytes to read

        Returns:
            bytes: Data read from file

        Raises:
            NvmeIoError: If read operation fails
        """
        if not self._initialized or self._file_handle is None:
            raise NvmeIoError("File backend not initialized")

        start_time = time.perf_counter()

        try:
            # Validate bounds
            if lba < 0 or lba >= self._size_bytes:
                raise ValueError(f"LBA {lba} out of bounds (size: {self._size_bytes})")

            if length < 0:
                raise ValueError(f"Invalid length: {length}")

            end_offset = lba + length
            if end_offset > self._size_bytes:
                # Read partial data up to end of storage
                length = self._size_bytes - lba

            # Seek and read
            self._file_handle.seek(lba)
            data = self._file_handle.read(length)

            # Pad with zeros if we read less than requested (sparse regions)
            if len(data) < length:
                data += b"\x00" * (length - len(data))

            # Update statistics
            self._stats.total_reads += 1
            self._stats.bytes_read += len(data)

            # Calculate latency
            end_time = time.perf_counter()
            latency_us = (end_time - start_time) * 1_000_000

            # Update average latency
            total_ops = self._stats.total_reads
            self._stats.avg_read_latency_us = (
                self._stats.avg_read_latency_us * (total_ops - 1) + latency_us
            ) / total_ops

            return data

        except Exception as e:
            self._stats.read_errors += 1
            raise NvmeIoError(
                f"File read failed at LBA {lba}: {e}",
                details=str(e),
                lba=lba,
            )

    def write(self, lba: int, data: bytes) -> None:
        """
        Write data to file storage.

        Args:
            lba: Logical Block Address (byte offset)
            data: Data to write

        Raises:
            NvmeIoError: If write operation fails
        """
        if not self._initialized or self._file_handle is None:
            raise NvmeIoError("File backend not initialized")

        start_time = time.perf_counter()

        try:
            # Validate bounds
            if lba < 0 or lba >= self._size_bytes:
                raise ValueError(f"LBA {lba} out of bounds (size: {self._size_bytes})")

            length = len(data)
            if length == 0:
                return

            end_offset = lba + length
            if end_offset > self._size_bytes:
                raise ValueError(
                    f"Write would exceed storage size: {end_offset} > {self._size_bytes}"
                )

            # Add to WAL for write ordering
            self._wal_append(lba, data)

            # Seek and write
            self._file_handle.seek(lba)
            bytes_written = self._file_handle.write(data)

            if bytes_written != length:
                raise IOError(f"Partial write: {bytes_written} of {length} bytes")

            # Update statistics
            self._stats.total_writes += 1
            self._stats.bytes_written += length

            # Calculate latency
            end_time = time.perf_counter()
            latency_us = (end_time - start_time) * 1_000_000

            # Update average latency
            total_ops = self._stats.total_writes
            self._stats.avg_write_latency_us = (
                self._stats.avg_write_latency_us * (total_ops - 1) + latency_us
            ) / total_ops

        except Exception as e:
            self._stats.write_errors += 1
            raise NvmeIoError(
                f"File write failed at LBA {lba}: {e}",
                details=str(e),
                lba=lba,
            )

    def flush(self) -> None:
        """
        Flush pending writes to disk.

        Raises:
            NvmeIoError: If flush operation fails
        """
        if not self._initialized or self._file_handle is None:
            raise NvmeIoError("File backend not initialized")

        try:
            self._file_handle.flush()
            os.fsync(self._file_handle.fileno())
            self._wal_clear()
            self._stats.total_flushes += 1

        except Exception as e:
            raise NvmeIoError(
                f"File flush failed: {e}",
                details=str(e),
            )

    def trim(self, lba: int, length: int) -> None:
        """
        Trim (zero) a range of file storage.

        For sparse files, this can potentially free disk space.

        Args:
            lba: Starting Logical Block Address
            length: Number of bytes to trim

        Raises:
            NvmeIoError: If trim operation fails
        """
        if not self._initialized or self._file_handle is None:
            raise NvmeIoError("File backend not initialized")

        try:
            # Validate bounds
            if lba < 0 or lba >= self._size_bytes:
                raise ValueError(f"LBA {lba} out of bounds (size: {self._size_bytes})")

            if length < 0:
                raise ValueError(f"Invalid length: {length}")

            end_offset = lba + length
            if end_offset > self._size_bytes:
                length = self._size_bytes - lba

            # Write zeros to the range
            self._file_handle.seek(lba)
            chunk_size = 4096
            remaining = length
            while remaining > 0:
                write_size = min(chunk_size, remaining)
                self._file_handle.write(b"\x00" * write_size)
                remaining -= write_size

            self._stats.total_trims += 1

        except Exception as e:
            raise NvmeIoError(
                f"File trim failed at LBA {lba}: {e}",
                details=str(e),
                lba=lba,
            )

    def snapshot_create(self, name: str) -> None:
        """
        Create a point-in-time snapshot of file storage.

        Creates a copy of the file in the snapshots directory.

        Args:
            name: Unique name for the snapshot

        Raises:
            NvmeBackendError: If snapshot creation fails
        """
        if not self._initialized or self._file_path is None or self._snapshots_dir is None:
            raise NvmeBackendError("File backend not initialized")

        try:
            snapshot_path = self._snapshots_dir / name

            if snapshot_path.exists():
                raise ValueError(f"Snapshot '{name}' already exists")

            # Flush before snapshot
            self.flush()

            # Copy the file
            shutil.copy2(self._file_path, snapshot_path)

        except Exception as e:
            raise NvmeBackendError(
                f"Failed to create snapshot '{name}': {e}",
                details=str(e),
            )

    def snapshot_delete(self, name: str) -> None:
        """
        Delete a snapshot.

        Args:
            name: Name of the snapshot to delete

        Raises:
            NvmeBackendError: If snapshot deletion fails
        """
        if not self._initialized or self._snapshots_dir is None:
            raise NvmeBackendError("File backend not initialized")

        try:
            snapshot_path = self._snapshots_dir / name

            if not snapshot_path.exists():
                raise ValueError(f"Snapshot '{name}' does not exist")

            snapshot_path.unlink()

        except Exception as e:
            raise NvmeBackendError(
                f"Failed to delete snapshot '{name}': {e}",
                details=str(e),
            )

    def snapshot_restore(self, name: str) -> None:
        """
        Restore storage to a snapshot state.

        Args:
            name: Name of the snapshot to restore

        Raises:
            NvmeBackendError: If snapshot restoration fails
        """
        if not self._initialized or self._file_path is None or self._snapshots_dir is None:
            raise NvmeBackendError("File backend not initialized")

        try:
            snapshot_path = self._snapshots_dir / name

            if not snapshot_path.exists():
                raise ValueError(f"Snapshot '{name}' does not exist")

            # Close current file
            if self._file_handle is not None:
                self._file_handle.close()

            # Copy snapshot over current file
            shutil.copy2(snapshot_path, self._file_path)

            # Reopen file
            mode = "r+b"
            self._file_handle = open(self._file_path, mode, buffering=0 if self._direct_io else -1)

        except Exception as e:
            raise NvmeBackendError(
                f"Failed to restore snapshot '{name}': {e}",
                details=str(e),
            )

    @property
    def file_path(self) -> Optional[Path]:
        """Get the file path."""
        return self._file_path

    @property
    def is_sparse(self) -> bool:
        """Check if sparse file allocation is enabled."""
        return self._sparse

    @property
    def is_direct_io(self) -> bool:
        """Check if direct I/O mode is enabled."""
        return self._direct_io

    @property
    def size_bytes(self) -> int:
        """Get the total size of the file backend in bytes."""
        return self._size_bytes

    def get_actual_disk_usage(self) -> int:
        """
        Get the actual disk space used by the file.

        For sparse files, this may be less than size_bytes.

        Returns:
            int: Actual disk space used in bytes
        """
        if self._file_path is None or not self._file_path.exists():
            return 0

        # On Windows, st_blocks is not available, so we use st_size
        # On Unix, st_blocks * 512 gives actual disk usage
        stat = self._file_path.stat()
        if hasattr(stat, "st_blocks"):
            return stat.st_blocks * 512
        else:
            # Windows: return file size (not accurate for sparse files)
            return stat.st_size
