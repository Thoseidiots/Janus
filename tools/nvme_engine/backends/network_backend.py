"""
Network Backend for the Software NVMe Engine.

Provides remote storage access via TCP with connection pooling and retry logic.
"""

from __future__ import annotations

import socket
import struct
import time
from typing import Any, Dict, List, Optional

from nvme_engine.backends.base import StorageBackendOps
from nvme_engine.models.errors import NvmeBackendError, NvmeIoError


# Protocol constants
PROTOCOL_VERSION = 1
CMD_READ = 1
CMD_WRITE = 2
CMD_FLUSH = 3
CMD_TRIM = 4
CMD_SNAPSHOT_CREATE = 5
CMD_SNAPSHOT_DELETE = 6
CMD_SNAPSHOT_RESTORE = 7

RESPONSE_OK = 0
RESPONSE_ERROR = 1


class NetworkBackend(StorageBackendOps):
    """
    Network-based storage backend using TCP.

    Features:
    - TCP-based remote storage protocol
    - Connection pooling for performance
    - Automatic retry logic for transient failures
    - Graceful network failure handling
    """

    def __init__(self) -> None:
        """Initialize the network backend."""
        super().__init__()
        self._host: str = ""
        self._port: int = 0
        self._connection_pool: List[socket.socket] = []
        self._pool_size: int = 4
        self._retry_count: int = 3
        self._retry_delay_ms: int = 100
        self._timeout_seconds: float = 5.0
        self._size_bytes: int = 0

    def init(self, config: Dict[str, Any]) -> None:
        """
        Initialize the network backend with configuration.

        Args:
            config: Configuration dictionary with keys:
                - host: Remote server hostname or IP
                - port: Remote server port
                - pool_size: Connection pool size (default: 4)
                - retry_count: Number of retries for failed operations (default: 3)
                - retry_delay_ms: Delay between retries in milliseconds (default: 100)
                - timeout_seconds: Socket timeout in seconds (default: 5.0)
                - size_bytes: Total size in bytes

        Raises:
            NvmeBackendError: If initialization fails
        """
        if self._initialized:
            raise NvmeBackendError("Network backend already initialized")

        try:
            self._host = config.get("host", "")
            if not self._host:
                raise ValueError("host must be specified")

            self._port = config.get("port", 0)
            if not (1 <= self._port <= 65535):
                raise ValueError(f"port must be 1-65535, got {self._port}")

            self._pool_size = config.get("pool_size", 4)
            self._retry_count = config.get("retry_count", 3)
            self._retry_delay_ms = config.get("retry_delay_ms", 100)
            self._timeout_seconds = config.get("timeout_seconds", 5.0)
            self._size_bytes = config.get("size_bytes", 0)

            if self._size_bytes <= 0:
                raise ValueError("size_bytes must be positive")

            # Initialize connection pool
            self._init_connection_pool()

            self._initialized = True

        except Exception as e:
            raise NvmeBackendError(
                f"Failed to initialize network backend: {e}",
                details=str(e),
            )

    def _init_connection_pool(self) -> None:
        """Initialize the connection pool."""
        for _ in range(self._pool_size):
            try:
                conn = self._create_connection()
                self._connection_pool.append(conn)
            except Exception:
                # If we can't create initial connections, that's okay
                # We'll create them on-demand
                pass

    def _create_connection(self) -> socket.socket:
        """
        Create a new connection to the remote server.

        Returns:
            socket.socket: Connected socket

        Raises:
            NvmeBackendError: If connection fails
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self._timeout_seconds)
            sock.connect((self._host, self._port))
            return sock
        except Exception as e:
            raise NvmeBackendError(
                f"Failed to connect to {self._host}:{self._port}: {e}",
                details=str(e),
            )

    def _get_connection(self) -> socket.socket:
        """
        Get a connection from the pool or create a new one.

        Returns:
            socket.socket: Connected socket
        """
        if self._connection_pool:
            return self._connection_pool.pop()
        else:
            return self._create_connection()

    def _return_connection(self, conn: socket.socket) -> None:
        """
        Return a connection to the pool.

        Args:
            conn: Socket to return to pool
        """
        if len(self._connection_pool) < self._pool_size:
            self._connection_pool.append(conn)
        else:
            # Pool is full, close the connection
            try:
                conn.close()
            except Exception:
                pass

    def _close_connection(self, conn: socket.socket) -> None:
        """
        Close a connection (used when connection is bad).

        Args:
            conn: Socket to close
        """
        try:
            conn.close()
        except Exception:
            pass

    def destroy(self) -> None:
        """
        Destroy the network backend and release all resources.

        Raises:
            NvmeBackendError: If cleanup fails
        """
        if not self._initialized:
            return

        try:
            # Close all connections in pool
            for conn in self._connection_pool:
                try:
                    conn.close()
                except Exception:
                    pass

            self._connection_pool.clear()
            self._initialized = False

        except Exception as e:
            raise NvmeBackendError(
                f"Failed to destroy network backend: {e}",
                details=str(e),
            )

    def _send_request(self, cmd: int, lba: int = 0, data: bytes = b"", name: str = "") -> bytes:
        """
        Send a request to the remote server with retry logic.

        Args:
            cmd: Command code
            lba: Logical Block Address
            data: Data payload
            name: Name parameter (for snapshots)

        Returns:
            bytes: Response data

        Raises:
            NvmeIoError: If request fails after retries
        """
        last_error = None

        for attempt in range(self._retry_count):
            conn = None
            try:
                conn = self._get_connection()

                # Build request packet
                # Format: version(1) cmd(1) lba(8) data_len(4) name_len(1) name(var) data(var)
                name_bytes = name.encode("utf-8")
                header = struct.pack(
                    "<BBQIB",
                    PROTOCOL_VERSION,
                    cmd,
                    lba,
                    len(data),
                    len(name_bytes),
                )
                request = header + name_bytes + data

                # Send request
                conn.sendall(request)

                # Receive response
                # Format: status(1) data_len(4) data(var)
                response_header = self._recv_exact(conn, 5)
                status, response_len = struct.unpack("<BI", response_header)

                response_data = b""
                if response_len > 0:
                    response_data = self._recv_exact(conn, response_len)

                # Return connection to pool
                self._return_connection(conn)

                if status == RESPONSE_OK:
                    return response_data
                else:
                    # Server returned error
                    error_msg = response_data.decode("utf-8", errors="replace")
                    raise NvmeIoError(f"Server error: {error_msg}")

            except Exception as e:
                last_error = e
                if conn is not None:
                    self._close_connection(conn)

                # Wait before retry
                if attempt < self._retry_count - 1:
                    time.sleep(self._retry_delay_ms / 1000.0)

        # All retries failed
        raise NvmeIoError(
            f"Network request failed after {self._retry_count} retries: {last_error}",
            details=str(last_error),
        )

    def _recv_exact(self, conn: socket.socket, length: int) -> bytes:
        """
        Receive exactly the specified number of bytes.

        Args:
            conn: Socket to receive from
            length: Number of bytes to receive

        Returns:
            bytes: Received data

        Raises:
            IOError: If connection closed or timeout
        """
        data = b""
        while len(data) < length:
            chunk = conn.recv(length - len(data))
            if not chunk:
                raise IOError("Connection closed")
            data += chunk
        return data

    def read(self, lba: int, length: int) -> bytes:
        """
        Read data from remote storage.

        Args:
            lba: Logical Block Address (byte offset)
            length: Number of bytes to read

        Returns:
            bytes: Data read from remote storage

        Raises:
            NvmeIoError: If read operation fails
        """
        if not self._initialized:
            raise NvmeIoError("Network backend not initialized")

        start_time = time.perf_counter()

        try:
            # Validate bounds
            if lba < 0 or lba >= self._size_bytes:
                raise ValueError(f"LBA {lba} out of bounds (size: {self._size_bytes})")

            if length < 0:
                raise ValueError(f"Invalid length: {length}")

            # Send read request (length encoded in data field as 4 bytes)
            length_bytes = struct.pack("<I", length)
            data = self._send_request(CMD_READ, lba, length_bytes)

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
                f"Network read failed at LBA {lba}: {e}",
                details=str(e),
                lba=lba,
            )

    def write(self, lba: int, data: bytes) -> None:
        """
        Write data to remote storage.

        Args:
            lba: Logical Block Address (byte offset)
            data: Data to write

        Raises:
            NvmeIoError: If write operation fails
        """
        if not self._initialized:
            raise NvmeIoError("Network backend not initialized")

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

            # Send write request
            self._send_request(CMD_WRITE, lba, data)

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
                f"Network write failed at LBA {lba}: {e}",
                details=str(e),
                lba=lba,
            )

    def flush(self) -> None:
        """
        Flush pending writes to remote storage.

        Raises:
            NvmeIoError: If flush operation fails
        """
        if not self._initialized:
            raise NvmeIoError("Network backend not initialized")

        try:
            self._send_request(CMD_FLUSH)
            self._wal_clear()
            self._stats.total_flushes += 1

        except Exception as e:
            raise NvmeIoError(
                f"Network flush failed: {e}",
                details=str(e),
            )

    def trim(self, lba: int, length: int) -> None:
        """
        Trim a range of remote storage.

        Args:
            lba: Starting Logical Block Address
            length: Number of bytes to trim

        Raises:
            NvmeIoError: If trim operation fails
        """
        if not self._initialized:
            raise NvmeIoError("Network backend not initialized")

        try:
            # Validate bounds
            if lba < 0 or lba >= self._size_bytes:
                raise ValueError(f"LBA {lba} out of bounds (size: {self._size_bytes})")

            if length < 0:
                raise ValueError(f"Invalid length: {length}")

            # Send trim request (length encoded in data field)
            length_bytes = struct.pack("<I", length)
            self._send_request(CMD_TRIM, lba, length_bytes)

            self._stats.total_trims += 1

        except Exception as e:
            raise NvmeIoError(
                f"Network trim failed at LBA {lba}: {e}",
                details=str(e),
                lba=lba,
            )

    def snapshot_create(self, name: str) -> None:
        """
        Create a snapshot on remote storage.

        Args:
            name: Unique name for the snapshot

        Raises:
            NvmeBackendError: If snapshot creation fails
        """
        if not self._initialized:
            raise NvmeBackendError("Network backend not initialized")

        try:
            self._send_request(CMD_SNAPSHOT_CREATE, name=name)

        except Exception as e:
            raise NvmeBackendError(
                f"Failed to create snapshot '{name}': {e}",
                details=str(e),
            )

    def snapshot_delete(self, name: str) -> None:
        """
        Delete a snapshot on remote storage.

        Args:
            name: Name of the snapshot to delete

        Raises:
            NvmeBackendError: If snapshot deletion fails
        """
        if not self._initialized:
            raise NvmeBackendError("Network backend not initialized")

        try:
            self._send_request(CMD_SNAPSHOT_DELETE, name=name)

        except Exception as e:
            raise NvmeBackendError(
                f"Failed to delete snapshot '{name}': {e}",
                details=str(e),
            )

    def snapshot_restore(self, name: str) -> None:
        """
        Restore storage to a snapshot state on remote storage.

        Args:
            name: Name of the snapshot to restore

        Raises:
            NvmeBackendError: If snapshot restoration fails
        """
        if not self._initialized:
            raise NvmeBackendError("Network backend not initialized")

        try:
            self._send_request(CMD_SNAPSHOT_RESTORE, name=name)

        except Exception as e:
            raise NvmeBackendError(
                f"Failed to restore snapshot '{name}': {e}",
                details=str(e),
            )

    @property
    def host(self) -> str:
        """Get the remote host."""
        return self._host

    @property
    def port(self) -> int:
        """Get the remote port."""
        return self._port

    @property
    def pool_size(self) -> int:
        """Get the connection pool size."""
        return self._pool_size

    @property
    def size_bytes(self) -> int:
        """Get the total size of the network backend in bytes."""
        return self._size_bytes
