"""
Unit tests for the Network Backend.

Note: These tests use mocking since we don't have a real network server.
"""

import socket
import struct
from unittest.mock import MagicMock, patch

import pytest

from nvme_engine.backends.network_backend import (
    CMD_READ,
    CMD_WRITE,
    CMD_FLUSH,
    CMD_TRIM,
    CMD_SNAPSHOT_CREATE,
    RESPONSE_OK,
    NetworkBackend,
)
from nvme_engine.models.errors import NvmeBackendError, NvmeIoError


class MockSocket:
    """Mock socket for testing."""

    def __init__(self):
        self.sent_data = b""
        self.response_data = b""
        self.closed = False

    def settimeout(self, timeout):
        pass

    def connect(self, address):
        pass

    def sendall(self, data):
        self.sent_data += data

    def recv(self, size):
        if not self.response_data:
            return b""
        chunk = self.response_data[:size]
        self.response_data = self.response_data[size:]
        return chunk

    def close(self):
        self.closed = True


class TestNetworkBackendBasic:
    """Basic unit tests for Network Backend."""

    def test_initialization(self):
        """Test network backend initialization."""
        backend = NetworkBackend()
        assert not backend.is_initialized

        with patch("socket.socket") as mock_socket_class:
            mock_socket = MockSocket()
            mock_socket_class.return_value = mock_socket

            config = {
                "host": "localhost",
                "port": 9000,
                "size_bytes": 1024 * 1024,
                "pool_size": 2,
            }
            backend.init(config)

            assert backend.is_initialized
            assert backend.host == "localhost"
            assert backend.port == 9000
            assert backend.size_bytes == 1024 * 1024
            assert backend.pool_size == 2

            backend.destroy()

    def test_initialization_invalid_host(self):
        """Test initialization with empty host fails."""
        backend = NetworkBackend()

        with pytest.raises(NvmeBackendError, match="host must be specified"):
            backend.init({"port": 9000, "size_bytes": 1024})

    def test_initialization_invalid_port(self):
        """Test initialization with invalid port."""
        backend = NetworkBackend()

        with pytest.raises(NvmeBackendError, match="port must be 1-65535"):
            backend.init({"host": "localhost", "port": 0, "size_bytes": 1024})

        with pytest.raises(NvmeBackendError, match="port must be 1-65535"):
            backend.init({"host": "localhost", "port": 70000, "size_bytes": 1024})

    def test_initialization_invalid_size(self):
        """Test initialization with invalid size."""
        backend = NetworkBackend()

        with pytest.raises(NvmeBackendError, match="size_bytes must be positive"):
            backend.init({"host": "localhost", "port": 9000, "size_bytes": 0})

    def test_double_initialization(self):
        """Test that double initialization fails."""
        backend = NetworkBackend()

        with patch("socket.socket") as mock_socket_class:
            mock_socket = MockSocket()
            mock_socket_class.return_value = mock_socket

            config = {"host": "localhost", "port": 9000, "size_bytes": 1024}
            backend.init(config)

            with pytest.raises(NvmeBackendError, match="already initialized"):
                backend.init(config)

            backend.destroy()

    def test_destroy(self):
        """Test backend destruction."""
        backend = NetworkBackend()

        with patch("socket.socket") as mock_socket_class:
            mock_socket = MockSocket()
            mock_socket_class.return_value = mock_socket

            backend.init({"host": "localhost", "port": 9000, "size_bytes": 1024})
            assert backend.is_initialized

            backend.destroy()
            assert not backend.is_initialized
            assert mock_socket.closed

    def test_read_basic(self):
        """Test basic read operation."""
        backend = NetworkBackend()

        with patch("socket.socket") as mock_socket_class:
            mock_socket = MockSocket()
            mock_socket_class.return_value = mock_socket

            backend.init({"host": "localhost", "port": 9000, "size_bytes": 4096})

            # Prepare mock response
            response_data = b"test data"
            response_header = struct.pack("<BI", RESPONSE_OK, len(response_data))
            mock_socket.response_data = response_header + response_data

            # Perform read
            data = backend.read(0, 9)

            assert data == response_data
            assert backend.get_stats().total_reads == 1
            assert backend.get_stats().bytes_read == 9

            backend.destroy()

    def test_write_basic(self):
        """Test basic write operation."""
        backend = NetworkBackend()

        with patch("socket.socket") as mock_socket_class:
            mock_socket = MockSocket()
            mock_socket_class.return_value = mock_socket

            backend.init({"host": "localhost", "port": 9000, "size_bytes": 4096})

            # Prepare mock response (empty success)
            response_header = struct.pack("<BI", RESPONSE_OK, 0)
            mock_socket.response_data = response_header

            # Perform write
            data = b"test data"
            backend.write(0, data)

            assert backend.get_stats().total_writes == 1
            assert backend.get_stats().bytes_written == 9

            backend.destroy()

    def test_write_out_of_bounds(self):
        """Test write beyond storage size fails."""
        backend = NetworkBackend()

        with patch("socket.socket") as mock_socket_class:
            mock_socket = MockSocket()
            mock_socket_class.return_value = mock_socket

            backend.init({"host": "localhost", "port": 9000, "size_bytes": 1024})

            with pytest.raises(NvmeIoError, match="exceed storage size"):
                backend.write(1000, b"X" * 100)

            backend.destroy()

    def test_flush(self):
        """Test flush operation."""
        backend = NetworkBackend()

        with patch("socket.socket") as mock_socket_class:
            mock_socket = MockSocket()
            mock_socket_class.return_value = mock_socket

            backend.init({"host": "localhost", "port": 9000, "size_bytes": 4096})

            # Prepare mock response
            response_header = struct.pack("<BI", RESPONSE_OK, 0)
            mock_socket.response_data = response_header

            backend.flush()

            assert backend.get_stats().total_flushes == 1

            backend.destroy()

    def test_trim(self):
        """Test trim operation."""
        backend = NetworkBackend()

        with patch("socket.socket") as mock_socket_class:
            mock_socket = MockSocket()
            mock_socket_class.return_value = mock_socket

            backend.init({"host": "localhost", "port": 9000, "size_bytes": 4096})

            # Prepare mock response
            response_header = struct.pack("<BI", RESPONSE_OK, 0)
            mock_socket.response_data = response_header

            backend.trim(0, 100)

            assert backend.get_stats().total_trims == 1

            backend.destroy()

    def test_snapshot_create(self):
        """Test snapshot creation."""
        backend = NetworkBackend()

        with patch("socket.socket") as mock_socket_class:
            mock_socket = MockSocket()
            mock_socket_class.return_value = mock_socket

            backend.init({"host": "localhost", "port": 9000, "size_bytes": 4096})

            # Prepare mock response
            response_header = struct.pack("<BI", RESPONSE_OK, 0)
            mock_socket.response_data = response_header

            backend.snapshot_create("snap1")

            backend.destroy()

    def test_snapshot_delete(self):
        """Test snapshot deletion."""
        backend = NetworkBackend()

        with patch("socket.socket") as mock_socket_class:
            mock_socket = MockSocket()
            mock_socket_class.return_value = mock_socket

            backend.init({"host": "localhost", "port": 9000, "size_bytes": 4096})

            # Prepare mock response
            response_header = struct.pack("<BI", RESPONSE_OK, 0)
            mock_socket.response_data = response_header

            backend.snapshot_delete("snap1")

            backend.destroy()

    def test_snapshot_restore(self):
        """Test snapshot restoration."""
        backend = NetworkBackend()

        with patch("socket.socket") as mock_socket_class:
            mock_socket = MockSocket()
            mock_socket_class.return_value = mock_socket

            backend.init({"host": "localhost", "port": 9000, "size_bytes": 4096})

            # Prepare mock response
            response_header = struct.pack("<BI", RESPONSE_OK, 0)
            mock_socket.response_data = response_header

            backend.snapshot_restore("snap1")

            backend.destroy()

    def test_connection_pooling(self):
        """Test that connection pooling works."""
        backend = NetworkBackend()

        with patch("socket.socket") as mock_socket_class:
            # Track created sockets
            created_sockets = []

            def create_socket(*args, **kwargs):
                sock = MockSocket()
                created_sockets.append(sock)
                return sock

            mock_socket_class.side_effect = create_socket

            backend.init({
                "host": "localhost",
                "port": 9000,
                "size_bytes": 4096,
                "pool_size": 2,
            })

            # Should have created 2 connections for the pool
            assert len(created_sockets) == 2

            backend.destroy()

            # All connections should be closed
            for sock in created_sockets:
                assert sock.closed

    def test_retry_logic(self):
        """Test that retry logic works on failures."""
        backend = NetworkBackend()

        with patch("socket.socket") as mock_socket_class:
            call_count = [0]

            def create_socket(*args, **kwargs):
                call_count[0] += 1
                sock = MockSocket()

                # First 2 attempts fail, third succeeds
                if call_count[0] < 3:
                    # Simulate connection failure
                    def failing_sendall(data):
                        raise IOError("Connection failed")

                    sock.sendall = failing_sendall
                else:
                    # Success on third attempt
                    response_header = struct.pack("<BI", RESPONSE_OK, 0)
                    sock.response_data = response_header

                return sock

            mock_socket_class.side_effect = create_socket

            backend.init({
                "host": "localhost",
                "port": 9000,
                "size_bytes": 4096,
                "pool_size": 0,  # No initial pool
                "retry_count": 3,
                "retry_delay_ms": 10,
            })

            # This should succeed after retries
            backend.flush()

            # Should have tried 3 times (2 failures + 1 success)
            assert call_count[0] == 3

            backend.destroy()

    def test_operations_before_init_fail(self):
        """Test that operations before init fail."""
        backend = NetworkBackend()

        with pytest.raises(NvmeIoError, match="not initialized"):
            backend.read(0, 100)

        with pytest.raises(NvmeIoError, match="not initialized"):
            backend.write(0, b"data")

        with pytest.raises(NvmeIoError, match="not initialized"):
            backend.flush()
