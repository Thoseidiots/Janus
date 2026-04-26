"""
Unit tests for I/O request and completion data models.

Tests: IoRequest, IoCompletion, IoType
"""

import pytest

from nvme_engine.models.io_models import IoCompletion, IoRequest, IoType


# ---------------------------------------------------------------------------
# IoType Tests
# ---------------------------------------------------------------------------


class TestIoType:
    """Tests for IoType enum."""

    def test_all_types_defined(self):
        """Test that all expected I/O types are defined."""
        assert IoType.READ == "READ"
        assert IoType.WRITE == "WRITE"
        assert IoType.FLUSH == "FLUSH"
        assert IoType.TRIM == "TRIM"

    def test_string_conversion(self):
        """Test converting strings to IoType."""
        assert IoType("READ") == IoType.READ
        assert IoType("WRITE") == IoType.WRITE
        assert IoType("FLUSH") == IoType.FLUSH
        assert IoType("TRIM") == IoType.TRIM


# ---------------------------------------------------------------------------
# IoRequest Tests
# ---------------------------------------------------------------------------


class TestIoRequest:
    """Tests for IoRequest."""

    def test_minimal_construction(self):
        """Test creating a minimal IoRequest."""
        req = IoRequest(
            request_id=1,
            type=IoType.READ,
            lba=0,
            block_count=8
        )
        assert req.request_id == 1
        assert req.type == IoType.READ
        assert req.lba == 0
        assert req.block_count == 8
        assert req.buffer == b""
        assert req.buffer_size == 0
        assert req.priority == 2  # default
        assert req.submit_time_ns == 0
        assert req.complete_time_ns == 0

    def test_full_construction(self):
        """Test creating a full IoRequest with all fields."""
        data = b"test data"
        req = IoRequest(
            request_id=42,
            type=IoType.WRITE,
            lba=0x1000,
            block_count=16,
            buffer=data,
            buffer_size=len(data),
            priority=0,
            submit_time_ns=1000000,
            complete_time_ns=1500000
        )
        assert req.request_id == 42
        assert req.type == IoType.WRITE
        assert req.lba == 0x1000
        assert req.block_count == 16
        assert req.buffer == data
        assert req.buffer_size == len(data)
        assert req.priority == 0
        assert req.submit_time_ns == 1000000
        assert req.complete_time_ns == 1500000

    def test_invalid_request_id(self):
        """Test that negative request_id raises ValueError."""
        with pytest.raises(ValueError, match="request_id must be >= 0"):
            IoRequest(
                request_id=-1,
                type=IoType.READ,
                lba=0,
                block_count=1
            )

    def test_invalid_lba(self):
        """Test that negative lba raises ValueError."""
        with pytest.raises(ValueError, match="lba must be >= 0"):
            IoRequest(
                request_id=1,
                type=IoType.READ,
                lba=-1,
                block_count=1
            )

    def test_invalid_block_count(self):
        """Test that negative block_count raises ValueError."""
        with pytest.raises(ValueError, match="block_count must be >= 0"):
            IoRequest(
                request_id=1,
                type=IoType.READ,
                lba=0,
                block_count=-1
            )

    def test_invalid_priority(self):
        """Test that invalid priority raises ValueError."""
        with pytest.raises(ValueError, match="priority must be 0-3"):
            IoRequest(
                request_id=1,
                type=IoType.READ,
                lba=0,
                block_count=1,
                priority=-1
            )
        with pytest.raises(ValueError, match="priority must be 0-3"):
            IoRequest(
                request_id=1,
                type=IoType.READ,
                lba=0,
                block_count=1,
                priority=4
            )

    def test_invalid_submit_time(self):
        """Test that negative submit_time_ns raises ValueError."""
        with pytest.raises(ValueError, match="submit_time_ns must be >= 0"):
            IoRequest(
                request_id=1,
                type=IoType.READ,
                lba=0,
                block_count=1,
                submit_time_ns=-1
            )

    def test_invalid_complete_time(self):
        """Test that negative complete_time_ns raises ValueError."""
        with pytest.raises(ValueError, match="complete_time_ns must be >= 0"):
            IoRequest(
                request_id=1,
                type=IoType.READ,
                lba=0,
                block_count=1,
                complete_time_ns=-1
            )

    def test_type_string_conversion(self):
        """Test that type string is converted to IoType enum."""
        req = IoRequest(
            request_id=1,
            type="WRITE",
            lba=0,
            block_count=1
        )
        assert req.type == IoType.WRITE

    def test_latency_property_not_completed(self):
        """Test latency property when request not completed."""
        req = IoRequest(
            request_id=1,
            type=IoType.READ,
            lba=0,
            block_count=1,
            submit_time_ns=1000000,
            complete_time_ns=0
        )
        assert req.latency_ns is None

    def test_latency_property_completed(self):
        """Test latency property when request completed."""
        req = IoRequest(
            request_id=1,
            type=IoType.READ,
            lba=0,
            block_count=1,
            submit_time_ns=1000000,
            complete_time_ns=1500000
        )
        assert req.latency_ns == 500000

    def test_serialization_without_buffer(self):
        """Test to_dict and from_dict round-trip without buffer."""
        req = IoRequest(
            request_id=10,
            type=IoType.FLUSH,
            lba=0,
            block_count=0,
            priority=1,
            submit_time_ns=2000000
        )
        data = req.to_dict()
        
        assert data["request_id"] == 10
        assert data["type"] == "FLUSH"
        assert data["lba"] == 0
        assert data["block_count"] == 0
        assert data["priority"] == 1
        assert data["submit_time_ns"] == 2000000
        
        restored = IoRequest.from_dict(data)
        assert restored.request_id == req.request_id
        assert restored.type == req.type
        assert restored.lba == req.lba
        assert restored.block_count == req.block_count
        assert restored.priority == req.priority
        assert restored.submit_time_ns == req.submit_time_ns

    def test_serialization_with_buffer(self):
        """Test to_dict and from_dict round-trip with buffer."""
        buffer_data = b"Hello, NVMe!"
        req = IoRequest(
            request_id=20,
            type=IoType.WRITE,
            lba=0x2000,
            block_count=4,
            buffer=buffer_data,
            buffer_size=len(buffer_data),
            priority=0
        )
        data = req.to_dict()
        
        assert data["buffer"] == buffer_data.hex()
        assert data["buffer_size"] == len(buffer_data)
        
        restored = IoRequest.from_dict(data)
        assert restored.buffer == buffer_data
        assert restored.buffer_size == len(buffer_data)

    def test_read_request(self):
        """Test creating a READ request."""
        req = IoRequest(
            request_id=1,
            type=IoType.READ,
            lba=0x100,
            block_count=8
        )
        assert req.type == IoType.READ
        assert req.buffer == b""  # READ requests don't have data

    def test_write_request(self):
        """Test creating a WRITE request."""
        data = b"x" * 4096
        req = IoRequest(
            request_id=2,
            type=IoType.WRITE,
            lba=0x200,
            block_count=8,
            buffer=data,
            buffer_size=len(data)
        )
        assert req.type == IoType.WRITE
        assert len(req.buffer) == 4096

    def test_flush_request(self):
        """Test creating a FLUSH request."""
        req = IoRequest(
            request_id=3,
            type=IoType.FLUSH,
            lba=0,
            block_count=0
        )
        assert req.type == IoType.FLUSH
        assert req.lba == 0
        assert req.block_count == 0

    def test_trim_request(self):
        """Test creating a TRIM request."""
        req = IoRequest(
            request_id=4,
            type=IoType.TRIM,
            lba=0x1000,
            block_count=128
        )
        assert req.type == IoType.TRIM
        assert req.lba == 0x1000
        assert req.block_count == 128


# ---------------------------------------------------------------------------
# IoCompletion Tests
# ---------------------------------------------------------------------------


class TestIoCompletion:
    """Tests for IoCompletion."""

    def test_construction(self):
        """Test creating an IoCompletion."""
        completion = IoCompletion(
            request_id=1,
            status=0,
            complete_time_ns=1500000,
            bytes_transferred=4096
        )
        assert completion.request_id == 1
        assert completion.status == 0
        assert completion.complete_time_ns == 1500000
        assert completion.bytes_transferred == 4096

    def test_invalid_request_id(self):
        """Test that negative request_id raises ValueError."""
        with pytest.raises(ValueError, match="request_id must be >= 0"):
            IoCompletion(
                request_id=-1,
                status=0,
                complete_time_ns=1000000,
                bytes_transferred=0
            )

    def test_invalid_status(self):
        """Test that negative status raises ValueError."""
        with pytest.raises(ValueError, match="status must be >= 0"):
            IoCompletion(
                request_id=1,
                status=-1,
                complete_time_ns=1000000,
                bytes_transferred=0
            )

    def test_invalid_complete_time(self):
        """Test that negative complete_time_ns raises ValueError."""
        with pytest.raises(ValueError, match="complete_time_ns must be >= 0"):
            IoCompletion(
                request_id=1,
                status=0,
                complete_time_ns=-1,
                bytes_transferred=0
            )

    def test_invalid_bytes_transferred(self):
        """Test that negative bytes_transferred raises ValueError."""
        with pytest.raises(ValueError, match="bytes_transferred must be >= 0"):
            IoCompletion(
                request_id=1,
                status=0,
                complete_time_ns=1000000,
                bytes_transferred=-1
            )

    def test_is_success_property_true(self):
        """Test is_success property when status is 0."""
        completion = IoCompletion(
            request_id=1,
            status=0,
            complete_time_ns=1000000,
            bytes_transferred=4096
        )
        assert completion.is_success is True

    def test_is_success_property_false(self):
        """Test is_success property when status is non-zero."""
        completion = IoCompletion(
            request_id=1,
            status=1,
            complete_time_ns=1000000,
            bytes_transferred=0
        )
        assert completion.is_success is False

    def test_serialization(self):
        """Test to_dict and from_dict round-trip."""
        completion = IoCompletion(
            request_id=42,
            status=0,
            complete_time_ns=2500000,
            bytes_transferred=8192
        )
        data = completion.to_dict()
        
        assert data["request_id"] == 42
        assert data["status"] == 0
        assert data["complete_time_ns"] == 2500000
        assert data["bytes_transferred"] == 8192
        
        restored = IoCompletion.from_dict(data)
        assert restored.request_id == completion.request_id
        assert restored.status == completion.status
        assert restored.complete_time_ns == completion.complete_time_ns
        assert restored.bytes_transferred == completion.bytes_transferred

    def test_successful_completion(self):
        """Test a successful I/O completion."""
        completion = IoCompletion(
            request_id=10,
            status=0,
            complete_time_ns=3000000,
            bytes_transferred=16384
        )
        assert completion.is_success
        assert completion.bytes_transferred == 16384

    def test_failed_completion(self):
        """Test a failed I/O completion."""
        completion = IoCompletion(
            request_id=11,
            status=5,  # some error code
            complete_time_ns=3000000,
            bytes_transferred=0
        )
        assert not completion.is_success
        assert completion.status == 5


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestIoIntegration:
    """Integration tests for I/O request and completion."""

    def test_request_completion_pairing(self):
        """Test that request and completion can be paired by request_id."""
        req = IoRequest(
            request_id=100,
            type=IoType.READ,
            lba=0x1000,
            block_count=8,
            submit_time_ns=1000000
        )
        
        completion = IoCompletion(
            request_id=100,
            status=0,
            complete_time_ns=1500000,
            bytes_transferred=4096
        )
        
        assert req.request_id == completion.request_id
        
        # Update request with completion time
        req.complete_time_ns = completion.complete_time_ns
        assert req.latency_ns == 500000

    def test_multiple_requests_different_priorities(self):
        """Test creating multiple requests with different priorities."""
        requests = [
            IoRequest(request_id=i, type=IoType.READ, lba=i*8, block_count=8, priority=i % 4)
            for i in range(10)
        ]
        
        priorities = [req.priority for req in requests]
        assert min(priorities) == 0
        assert max(priorities) == 3

    def test_large_buffer_write(self):
        """Test WRITE request with large buffer."""
        large_buffer = b"x" * (1024 * 1024)  # 1 MB
        req = IoRequest(
            request_id=200,
            type=IoType.WRITE,
            lba=0,
            block_count=2048,
            buffer=large_buffer,
            buffer_size=len(large_buffer)
        )
        assert len(req.buffer) == 1024 * 1024
        assert req.buffer_size == 1024 * 1024

    def test_zero_block_operations(self):
        """Test operations with zero blocks (FLUSH)."""
        req = IoRequest(
            request_id=300,
            type=IoType.FLUSH,
            lba=0,
            block_count=0
        )
        assert req.block_count == 0
        
        completion = IoCompletion(
            request_id=300,
            status=0,
            complete_time_ns=2000000,
            bytes_transferred=0
        )
        assert completion.bytes_transferred == 0
        assert completion.is_success

    def test_request_lifecycle(self):
        """Test complete request lifecycle from submission to completion."""
        # Create request
        req = IoRequest(
            request_id=400,
            type=IoType.WRITE,
            lba=0x5000,
            block_count=16,
            buffer=b"data" * 1024,
            buffer_size=4096,
            priority=1,
            submit_time_ns=5000000
        )
        
        # Initially not completed
        assert req.complete_time_ns == 0
        assert req.latency_ns is None
        
        # Create completion
        completion = IoCompletion(
            request_id=400,
            status=0,
            complete_time_ns=5010000,
            bytes_transferred=4096
        )
        
        # Update request with completion
        req.complete_time_ns = completion.complete_time_ns
        
        # Now completed
        assert req.latency_ns == 10000  # 10 microseconds
        assert completion.is_success
        assert completion.bytes_transferred == req.buffer_size
