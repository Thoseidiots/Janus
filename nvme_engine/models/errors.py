"""
Error class hierarchy for the Software NVMe Engine.

All engine-specific exceptions derive from NvmeError, which carries a
structured error code, human-readable message, optional detail string,
and optional context (device_id, lba).
"""

from __future__ import annotations

import time
from enum import IntEnum
from typing import Optional


# ---------------------------------------------------------------------------
# Error codes
# ---------------------------------------------------------------------------


class NvmeErrorCode(IntEnum):
    """Enumeration of all NVMe engine error codes."""

    SUCCESS = 0
    NVME_ERR_INVALID_CONFIG = 1
    NVME_ERR_RESOURCE_EXHAUSTED = 2
    NVME_ERR_IO_FAILURE = 3
    NVME_ERR_DATA_CORRUPTION = 4
    NVME_ERR_TIMEOUT = 5
    NVME_ERR_PERMISSION_DENIED = 6
    NVME_ERR_DEVICE_NOT_FOUND = 7
    NVME_ERR_BACKEND_FAILURE = 8


# ---------------------------------------------------------------------------
# Base exception
# ---------------------------------------------------------------------------


class NvmeError(Exception):
    """
    Base exception for all Software NVMe Engine errors.

    Attributes
    ----------
    code       : Structured error code from NvmeErrorCode.
    message    : Short human-readable description.
    details    : Optional extended diagnostic information.
    timestamp  : Unix timestamp (float) when the error was created.
    device_id  : Optional device identifier associated with the error.
    lba        : Optional Logical Block Address for I/O errors.
    """

    def __init__(
        self,
        message: str,
        code: NvmeErrorCode = NvmeErrorCode.NVME_ERR_IO_FAILURE,
        details: str = "",
        device_id: Optional[int] = None,
        lba: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details
        self.timestamp: float = time.time()
        self.device_id = device_id
        self.lba = lba

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"code={self.code.name}, "
            f"message={self.message!r}, "
            f"device_id={self.device_id}, "
            f"lba={self.lba})"
        )

    def __str__(self) -> str:
        parts = [f"[{self.code.name}] {self.message}"]
        if self.details:
            parts.append(f"Details: {self.details}")
        if self.device_id is not None:
            parts.append(f"Device: {self.device_id}")
        if self.lba is not None:
            parts.append(f"LBA: {self.lba:#x}")
        return " | ".join(parts)

    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "code_name": self.code.name,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "device_id": self.device_id,
            "lba": self.lba,
        }


# ---------------------------------------------------------------------------
# Concrete error subclasses
# ---------------------------------------------------------------------------


class NvmeConfigError(NvmeError):
    """
    Raised when a device configuration is invalid or cannot be applied.

    Error code: NVME_ERR_INVALID_CONFIG
    """

    def __init__(
        self,
        message: str,
        details: str = "",
        device_id: Optional[int] = None,
        lba: Optional[int] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=NvmeErrorCode.NVME_ERR_INVALID_CONFIG,
            details=details,
            device_id=device_id,
            lba=lba,
        )


class NvmeResourceError(NvmeError):
    """
    Raised when a required resource (memory, queue slots, etc.) is exhausted.

    Error code: NVME_ERR_RESOURCE_EXHAUSTED
    """

    def __init__(
        self,
        message: str,
        details: str = "",
        device_id: Optional[int] = None,
        lba: Optional[int] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=NvmeErrorCode.NVME_ERR_RESOURCE_EXHAUSTED,
            details=details,
            device_id=device_id,
            lba=lba,
        )


class NvmeIoError(NvmeError):
    """
    Raised when an I/O operation fails at the backend or command-handler level.

    Error code: NVME_ERR_IO_FAILURE
    """

    def __init__(
        self,
        message: str,
        details: str = "",
        device_id: Optional[int] = None,
        lba: Optional[int] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=NvmeErrorCode.NVME_ERR_IO_FAILURE,
            details=details,
            device_id=device_id,
            lba=lba,
        )


class NvmeDataCorruptionError(NvmeError):
    """
    Raised when end-to-end checksum verification detects data corruption.

    Error code: NVME_ERR_DATA_CORRUPTION
    """

    def __init__(
        self,
        message: str,
        details: str = "",
        device_id: Optional[int] = None,
        lba: Optional[int] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=NvmeErrorCode.NVME_ERR_DATA_CORRUPTION,
            details=details,
            device_id=device_id,
            lba=lba,
        )


class NvmeTimeoutError(NvmeError):
    """
    Raised when an operation exceeds its configured timeout.

    Error code: NVME_ERR_TIMEOUT
    """

    def __init__(
        self,
        message: str,
        details: str = "",
        device_id: Optional[int] = None,
        lba: Optional[int] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=NvmeErrorCode.NVME_ERR_TIMEOUT,
            details=details,
            device_id=device_id,
            lba=lba,
        )


class NvmePermissionError(NvmeError):
    """
    Raised when an operation is denied due to access control policy.

    Error code: NVME_ERR_PERMISSION_DENIED
    """

    def __init__(
        self,
        message: str,
        details: str = "",
        device_id: Optional[int] = None,
        lba: Optional[int] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=NvmeErrorCode.NVME_ERR_PERMISSION_DENIED,
            details=details,
            device_id=device_id,
            lba=lba,
        )


class NvmeDeviceNotFoundError(NvmeError):
    """
    Raised when a referenced virtual NVMe device does not exist.

    Error code: NVME_ERR_DEVICE_NOT_FOUND
    """

    def __init__(
        self,
        message: str,
        details: str = "",
        device_id: Optional[int] = None,
        lba: Optional[int] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=NvmeErrorCode.NVME_ERR_DEVICE_NOT_FOUND,
            details=details,
            device_id=device_id,
            lba=lba,
        )


class NvmeBackendError(NvmeError):
    """
    Raised when a storage backend operation fails (e.g., disk I/O error,
    network disconnection, backend process crash).

    Error code: NVME_ERR_BACKEND_FAILURE
    """

    def __init__(
        self,
        message: str,
        details: str = "",
        device_id: Optional[int] = None,
        lba: Optional[int] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=NvmeErrorCode.NVME_ERR_BACKEND_FAILURE,
            details=details,
            device_id=device_id,
            lba=lba,
        )


# ---------------------------------------------------------------------------
# Convenience mapping: error code → exception class
# ---------------------------------------------------------------------------

ERROR_CODE_TO_CLASS: dict = {
    NvmeErrorCode.NVME_ERR_INVALID_CONFIG: NvmeConfigError,
    NvmeErrorCode.NVME_ERR_RESOURCE_EXHAUSTED: NvmeResourceError,
    NvmeErrorCode.NVME_ERR_IO_FAILURE: NvmeIoError,
    NvmeErrorCode.NVME_ERR_DATA_CORRUPTION: NvmeDataCorruptionError,
    NvmeErrorCode.NVME_ERR_TIMEOUT: NvmeTimeoutError,
    NvmeErrorCode.NVME_ERR_PERMISSION_DENIED: NvmePermissionError,
    NvmeErrorCode.NVME_ERR_DEVICE_NOT_FOUND: NvmeDeviceNotFoundError,
    NvmeErrorCode.NVME_ERR_BACKEND_FAILURE: NvmeBackendError,
}
