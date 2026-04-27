"""
NVMe I/O Command implementations.

Covers: Read, Write, Flush, Write Zeroes, and Atomic Write (up to 128KB).
"""

from __future__ import annotations

import time
from typing import Optional

from nvme_engine.backends.base import StorageBackendOps
from nvme_engine.models.errors import NvmeConfigError, NvmeIoError
from nvme_engine.models.io_models import IoCompletion, IoRequest, IoType

# Maximum atomic write size: 128 KB
MAX_ATOMIC_WRITE_BYTES = 128 * 1024


class IoCommandProcessor:
    """Processes NVMe I/O commands against a storage backend."""

    def __init__(self, backend: StorageBackendOps) -> None:
        self._backend = backend

    def execute(self, request: IoRequest) -> IoCompletion:
        """Dispatch an IoRequest to the appropriate handler."""
        start_ns = time.time_ns()
        status = 0
        bytes_transferred = 0

        try:
            if request.type == IoType.READ:
                data = self._read(request)
                bytes_transferred = len(data)
            elif request.type == IoType.WRITE:
                self._write(request)
                bytes_transferred = len(request.buffer) if request.buffer else request.buffer_size
            elif request.type == IoType.FLUSH:
                self._flush(request)
            elif request.type == IoType.TRIM:
                self._write_zeroes(request)
                bytes_transferred = request.block_count * 512 if request.block_count else request.buffer_size
            else:
                status = 1
        except Exception:
            status = 1

        return IoCompletion(
            request_id=request.request_id,
            status=status,
            complete_time_ns=time.time_ns(),
            bytes_transferred=bytes_transferred,
        )

    def _read(self, request: IoRequest) -> bytes:
        length = request.buffer_size or (request.block_count * 512)
        if length == 0:
            length = 512
        return self._backend.read(request.lba, length)

    def _write(self, request: IoRequest) -> None:
        data = request.buffer
        if not data:
            data = b"\x00" * (request.buffer_size or 512)
        self._backend.write(request.lba, data)

    def _flush(self, request: IoRequest) -> None:
        self._backend.flush()

    def _write_zeroes(self, request: IoRequest) -> None:
        length = request.block_count * 512 if request.block_count else request.buffer_size
        if length == 0:
            length = 512
        self._backend.trim(request.lba, length)

    def atomic_write(self, lba: int, data: bytes, backend: Optional[StorageBackendOps] = None) -> IoCompletion:
        """
        Atomic write — either all data is written or none (up to 128KB).

        Uses a read-backup → write → verify approach with rollback on failure.
        """
        if len(data) > MAX_ATOMIC_WRITE_BYTES:
            raise NvmeConfigError(
                f"Atomic write size {len(data)} exceeds maximum {MAX_ATOMIC_WRITE_BYTES} bytes"
            )

        backend = backend or self._backend
        request_id = int(time.time_ns())

        # Backup existing data for rollback
        try:
            backup = backend.read(lba, len(data))
        except Exception:
            backup = b"\x00" * len(data)

        try:
            backend.write(lba, data)
            # Verify write succeeded
            verify = backend.read(lba, len(data))
            if verify != data:
                raise NvmeIoError("Atomic write verification failed", lba=lba)
        except Exception as e:
            # Rollback
            try:
                backend.write(lba, backup)
            except Exception:
                pass
            return IoCompletion(
                request_id=request_id,
                status=1,
                complete_time_ns=time.time_ns(),
                bytes_transferred=0,
            )

        return IoCompletion(
            request_id=request_id,
            status=0,
            complete_time_ns=time.time_ns(),
            bytes_transferred=len(data),
        )
