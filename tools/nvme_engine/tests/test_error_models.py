"""
Unit tests for error data models.

Tests: NvmeError and all concrete error subclasses
"""

import time

import pytest

from nvme_engine.models.errors import (
    ERROR_CODE_TO_CLASS,
    NvmeBackendError,
    NvmeConfigError,
    NvmeDataCorruptionError,
    NvmeDeviceNotFoundError,
    NvmeError,
    NvmeErrorCode,
    NvmeIoError,
    NvmePermissionError,
    NvmeResourceError,
    NvmeTimeoutError,
)


# ---------------------------------------------------------------------------
# NvmeError Base Class Tests
# ---------------------------------------------------------------------------


class TestNvmeError:
    """Tests for the base NvmeError class."""

    def test_basic_construction(self):
        """Test creating a basic NvmeError."""
        error = NvmeError("Test error message")
        assert error.message == "Test error message"
        assert error.code == NvmeErrorCode.NVME_ERR_IO_FAILURE  # default
        assert error.details == ""
        assert error.device_id is None
        assert error.lba is None
        assert isinstance(error.timestamp, float)

    def test_full_construction(self):
        """Test creating an NvmeError with all fields."""
        error = NvmeError(
            message="Full error",
            code=NvmeErrorCode.NVME_ERR_DATA_CORRUPTION,
            details="Checksum mismatch",
            device_id=42,
            lba=0x1000
        )
        assert error.message == "Full error"
        assert error.code == NvmeErrorCode.NVME_ERR_DATA_CORRUPTION
        assert error.details == "Checksum mismatch"
        assert error.device_id == 42
        assert error.lba == 0x1000

    def test_timestamp_is_recent(self):
        """Test that timestamp is set to current time."""
        before = time.time()
        error = NvmeError("Test")
        after = time.time()
        assert before <= error.timestamp <= after

    def test_str_representation(self):
        """Test string representation of error."""
        error = NvmeError(
            message="Test error",
            code=NvmeErrorCode.NVME_ERR_IO_FAILURE,
            details="Additional info",
            device_id=1,
            lba=0x100
        )
        error_str = str(error)
        assert "NVME_ERR_IO_FAILURE" in error_str
        assert "Test error" in error_str
        assert "Additional info" in error_str
        assert "Device: 1" in error_str
        assert "LBA: 0x100" in error_str

    def test_repr_representation(self):
        """Test repr representation of error."""
        error = NvmeError("Test", device_id=5, lba=0x200)
        repr_str = repr(error)
        assert "NvmeError" in repr_str
        assert "code=NVME_ERR_IO_FAILURE" in repr_str
        assert "device_id=5" in repr_str
        assert "lba=512" in repr_str

    def test_to_dict(self):
        """Test serialization to dictionary."""
        error = NvmeError(
            message="Serialization test",
            code=NvmeErrorCode.NVME_ERR_TIMEOUT,
            details="Operation timed out",
            device_id=10,
            lba=0x500
        )
        data = error.to_dict()
        
        assert data["code"] == NvmeErrorCode.NVME_ERR_TIMEOUT.value
        assert data["code_name"] == "NVME_ERR_TIMEOUT"
        assert data["message"] == "Serialization test"
        assert data["details"] == "Operation timed out"
        assert data["device_id"] == 10
        assert data["lba"] == 0x500
        assert isinstance(data["timestamp"], float)

    def test_exception_behavior(self):
        """Test that NvmeError can be raised and caught as an exception."""
        with pytest.raises(NvmeError) as exc_info:
            raise NvmeError("Test exception")
        
        assert exc_info.value.message == "Test exception"
        assert str(exc_info.value.args[0]) == "Test exception"


# ---------------------------------------------------------------------------
# Concrete Error Subclass Tests
# ---------------------------------------------------------------------------


class TestNvmeConfigError:
    """Tests for NvmeConfigError."""

    def test_construction(self):
        """Test creating an NvmeConfigError."""
        error = NvmeConfigError("Invalid configuration")
        assert error.code == NvmeErrorCode.NVME_ERR_INVALID_CONFIG
        assert error.message == "Invalid configuration"

    def test_with_details(self):
        """Test NvmeConfigError with details."""
        error = NvmeConfigError(
            message="Bad config",
            details="Queue depth must be positive",
            device_id=1
        )
        assert error.code == NvmeErrorCode.NVME_ERR_INVALID_CONFIG
        assert error.details == "Queue depth must be positive"
        assert error.device_id == 1

    def test_exception_hierarchy(self):
        """Test that NvmeConfigError is an NvmeError."""
        error = NvmeConfigError("Test")
        assert isinstance(error, NvmeError)
        assert isinstance(error, Exception)


class TestNvmeResourceError:
    """Tests for NvmeResourceError."""

    def test_construction(self):
        """Test creating an NvmeResourceError."""
        error = NvmeResourceError("Out of memory")
        assert error.code == NvmeErrorCode.NVME_ERR_RESOURCE_EXHAUSTED
        assert error.message == "Out of memory"

    def test_exception_hierarchy(self):
        """Test that NvmeResourceError is an NvmeError."""
        error = NvmeResourceError("Test")
        assert isinstance(error, NvmeError)


class TestNvmeIoError:
    """Tests for NvmeIoError."""

    def test_construction(self):
        """Test creating an NvmeIoError."""
        error = NvmeIoError("I/O operation failed", lba=0x1000)
        assert error.code == NvmeErrorCode.NVME_ERR_IO_FAILURE
        assert error.message == "I/O operation failed"
        assert error.lba == 0x1000

    def test_exception_hierarchy(self):
        """Test that NvmeIoError is an NvmeError."""
        error = NvmeIoError("Test")
        assert isinstance(error, NvmeError)


class TestNvmeDataCorruptionError:
    """Tests for NvmeDataCorruptionError."""

    def test_construction(self):
        """Test creating an NvmeDataCorruptionError."""
        error = NvmeDataCorruptionError(
            message="Checksum mismatch",
            details="Expected: 0xABCD, Got: 0x1234",
            device_id=5,
            lba=0x2000
        )
        assert error.code == NvmeErrorCode.NVME_ERR_DATA_CORRUPTION
        assert error.message == "Checksum mismatch"
        assert error.details == "Expected: 0xABCD, Got: 0x1234"
        assert error.device_id == 5
        assert error.lba == 0x2000

    def test_exception_hierarchy(self):
        """Test that NvmeDataCorruptionError is an NvmeError."""
        error = NvmeDataCorruptionError("Test")
        assert isinstance(error, NvmeError)


class TestNvmeTimeoutError:
    """Tests for NvmeTimeoutError."""

    def test_construction(self):
        """Test creating an NvmeTimeoutError."""
        error = NvmeTimeoutError("Operation timed out after 30s")
        assert error.code == NvmeErrorCode.NVME_ERR_TIMEOUT
        assert error.message == "Operation timed out after 30s"

    def test_exception_hierarchy(self):
        """Test that NvmeTimeoutError is an NvmeError."""
        error = NvmeTimeoutError("Test")
        assert isinstance(error, NvmeError)


class TestNvmePermissionError:
    """Tests for NvmePermissionError."""

    def test_construction(self):
        """Test creating an NvmePermissionError."""
        error = NvmePermissionError(
            message="Access denied",
            details="User 'guest' does not have write permission",
            device_id=3
        )
        assert error.code == NvmeErrorCode.NVME_ERR_PERMISSION_DENIED
        assert error.message == "Access denied"
        assert error.details == "User 'guest' does not have write permission"
        assert error.device_id == 3

    def test_exception_hierarchy(self):
        """Test that NvmePermissionError is an NvmeError."""
        error = NvmePermissionError("Test")
        assert isinstance(error, NvmeError)


class TestNvmeDeviceNotFoundError:
    """Tests for NvmeDeviceNotFoundError."""

    def test_construction(self):
        """Test creating an NvmeDeviceNotFoundError."""
        error = NvmeDeviceNotFoundError(
            message="Device not found",
            details="Device ID 99 does not exist",
            device_id=99
        )
        assert error.code == NvmeErrorCode.NVME_ERR_DEVICE_NOT_FOUND
        assert error.message == "Device not found"
        assert error.details == "Device ID 99 does not exist"
        assert error.device_id == 99

    def test_exception_hierarchy(self):
        """Test that NvmeDeviceNotFoundError is an NvmeError."""
        error = NvmeDeviceNotFoundError("Test")
        assert isinstance(error, NvmeError)


class TestNvmeBackendError:
    """Tests for NvmeBackendError."""

    def test_construction(self):
        """Test creating an NvmeBackendError."""
        error = NvmeBackendError(
            message="Backend failure",
            details="Network connection lost",
            device_id=7
        )
        assert error.code == NvmeErrorCode.NVME_ERR_BACKEND_FAILURE
        assert error.message == "Backend failure"
        assert error.details == "Network connection lost"
        assert error.device_id == 7

    def test_exception_hierarchy(self):
        """Test that NvmeBackendError is an NvmeError."""
        error = NvmeBackendError("Test")
        assert isinstance(error, NvmeError)


# ---------------------------------------------------------------------------
# Error Code Mapping Tests
# ---------------------------------------------------------------------------


class TestErrorCodeMapping:
    """Tests for ERROR_CODE_TO_CLASS mapping."""

    def test_all_error_codes_mapped(self):
        """Test that all error codes (except SUCCESS) are mapped to classes."""
        expected_codes = [
            NvmeErrorCode.NVME_ERR_INVALID_CONFIG,
            NvmeErrorCode.NVME_ERR_RESOURCE_EXHAUSTED,
            NvmeErrorCode.NVME_ERR_IO_FAILURE,
            NvmeErrorCode.NVME_ERR_DATA_CORRUPTION,
            NvmeErrorCode.NVME_ERR_TIMEOUT,
            NvmeErrorCode.NVME_ERR_PERMISSION_DENIED,
            NvmeErrorCode.NVME_ERR_DEVICE_NOT_FOUND,
            NvmeErrorCode.NVME_ERR_BACKEND_FAILURE,
        ]
        
        for code in expected_codes:
            assert code in ERROR_CODE_TO_CLASS, f"Error code {code} not in mapping"

    def test_mapping_correctness(self):
        """Test that error codes map to correct exception classes."""
        assert ERROR_CODE_TO_CLASS[NvmeErrorCode.NVME_ERR_INVALID_CONFIG] == NvmeConfigError
        assert ERROR_CODE_TO_CLASS[NvmeErrorCode.NVME_ERR_RESOURCE_EXHAUSTED] == NvmeResourceError
        assert ERROR_CODE_TO_CLASS[NvmeErrorCode.NVME_ERR_IO_FAILURE] == NvmeIoError
        assert ERROR_CODE_TO_CLASS[NvmeErrorCode.NVME_ERR_DATA_CORRUPTION] == NvmeDataCorruptionError
        assert ERROR_CODE_TO_CLASS[NvmeErrorCode.NVME_ERR_TIMEOUT] == NvmeTimeoutError
        assert ERROR_CODE_TO_CLASS[NvmeErrorCode.NVME_ERR_PERMISSION_DENIED] == NvmePermissionError
        assert ERROR_CODE_TO_CLASS[NvmeErrorCode.NVME_ERR_DEVICE_NOT_FOUND] == NvmeDeviceNotFoundError
        assert ERROR_CODE_TO_CLASS[NvmeErrorCode.NVME_ERR_BACKEND_FAILURE] == NvmeBackendError

    def test_success_code_not_mapped(self):
        """Test that SUCCESS code is not in the error mapping."""
        assert NvmeErrorCode.SUCCESS not in ERROR_CODE_TO_CLASS


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestErrorIntegration:
    """Integration tests for error handling."""

    def test_catch_specific_error(self):
        """Test catching a specific error type."""
        with pytest.raises(NvmeConfigError) as exc_info:
            raise NvmeConfigError("Bad config")
        
        assert exc_info.value.code == NvmeErrorCode.NVME_ERR_INVALID_CONFIG

    def test_catch_base_error(self):
        """Test catching any NvmeError."""
        with pytest.raises(NvmeError):
            raise NvmeTimeoutError("Timeout")

    def test_multiple_error_types(self):
        """Test that different error types have different codes."""
        errors = [
            NvmeConfigError("Config"),
            NvmeResourceError("Resource"),
            NvmeIoError("IO"),
            NvmeDataCorruptionError("Corruption"),
            NvmeTimeoutError("Timeout"),
            NvmePermissionError("Permission"),
            NvmeDeviceNotFoundError("NotFound"),
            NvmeBackendError("Backend"),
        ]
        
        codes = [e.code for e in errors]
        assert len(set(codes)) == len(codes), "All error codes should be unique"

    def test_error_with_context(self):
        """Test error with full context information."""
        try:
            raise NvmeIoError(
                message="Read failed",
                details="Disk sector unreadable",
                device_id=1,
                lba=0x10000
            )
        except NvmeError as e:
            assert e.message == "Read failed"
            assert e.details == "Disk sector unreadable"
            assert e.device_id == 1
            assert e.lba == 0x10000
            assert e.code == NvmeErrorCode.NVME_ERR_IO_FAILURE
