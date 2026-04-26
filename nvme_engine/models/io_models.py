"""
I/O request and completion data models for the Software NVMe Engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class IoType(str, Enum):
    """Type of an I/O operation."""

    READ = "READ"
    WRITE = "WRITE"
    FLUSH = "FLUSH"
    TRIM = "TRIM"


@dataclass
class IoRequest:
    """
    Represents a single I/O request submitted to a virtual NVMe device.

    Fields
    ------
    request_id      : Unique identifier for this request.
    type            : The I/O operation type (READ, WRITE, FLUSH, TRIM).
    lba             : Logical Block Address (starting block).
    block_count     : Number of blocks to transfer.
    buffer          : Raw data bytes (used for WRITE; empty for READ/FLUSH/TRIM).
    buffer_size     : Size of the buffer in bytes.
    priority        : I/O priority (0 = highest, 3 = lowest).
    submit_time_ns  : Submission timestamp in nanoseconds.
    complete_time_ns: Completion timestamp in nanoseconds (0 until completed).
    """

    request_id: int
    type: IoType
    lba: int
    block_count: int
    buffer: bytes = field(default=b"")
    buffer_size: int = 0
    priority: int = 2
    submit_time_ns: int = 0
    complete_time_ns: int = 0

    def __post_init__(self) -> None:
        if self.request_id < 0:
            raise ValueError(f"request_id must be >= 0, got {self.request_id}")
        if not isinstance(self.type, IoType):
            self.type = IoType(self.type)
        if self.lba < 0:
            raise ValueError(f"lba must be >= 0, got {self.lba}")
        if self.block_count < 0:
            raise ValueError(f"block_count must be >= 0, got {self.block_count}")
        if self.buffer_size < 0:
            raise ValueError(f"buffer_size must be >= 0, got {self.buffer_size}")
        if not (0 <= self.priority <= 3):
            raise ValueError(f"priority must be 0-3, got {self.priority}")
        if self.submit_time_ns < 0:
            raise ValueError(
                f"submit_time_ns must be >= 0, got {self.submit_time_ns}"
            )
        if self.complete_time_ns < 0:
            raise ValueError(
                f"complete_time_ns must be >= 0, got {self.complete_time_ns}"
            )

    @property
    def latency_ns(self) -> Optional[int]:
        """Return the I/O latency in nanoseconds, or None if not yet completed."""
        if self.complete_time_ns == 0:
            return None
        return self.complete_time_ns - self.submit_time_ns

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "type": self.type.value,
            "lba": self.lba,
            "block_count": self.block_count,
            "buffer": self.buffer.hex(),
            "buffer_size": self.buffer_size,
            "priority": self.priority,
            "submit_time_ns": self.submit_time_ns,
            "complete_time_ns": self.complete_time_ns,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IoRequest":
        raw_buffer = data.get("buffer", "")
        buffer = bytes.fromhex(raw_buffer) if isinstance(raw_buffer, str) else raw_buffer
        return cls(
            request_id=data["request_id"],
            type=IoType(data["type"]),
            lba=data["lba"],
            block_count=data["block_count"],
            buffer=buffer,
            buffer_size=data.get("buffer_size", 0),
            priority=data.get("priority", 2),
            submit_time_ns=data.get("submit_time_ns", 0),
            complete_time_ns=data.get("complete_time_ns", 0),
        )


@dataclass
class IoCompletion:
    """
    Represents the completion record for an I/O request.

    Fields
    ------
    request_id       : Matches the originating IoRequest.request_id.
    status           : NVMe status code (0 = success).
    complete_time_ns : Completion timestamp in nanoseconds.
    bytes_transferred: Number of bytes actually transferred.
    """

    request_id: int
    status: int
    complete_time_ns: int
    bytes_transferred: int

    def __post_init__(self) -> None:
        if self.request_id < 0:
            raise ValueError(f"request_id must be >= 0, got {self.request_id}")
        if self.status < 0:
            raise ValueError(f"status must be >= 0, got {self.status}")
        if self.complete_time_ns < 0:
            raise ValueError(
                f"complete_time_ns must be >= 0, got {self.complete_time_ns}"
            )
        if self.bytes_transferred < 0:
            raise ValueError(
                f"bytes_transferred must be >= 0, got {self.bytes_transferred}"
            )

    @property
    def is_success(self) -> bool:
        """Return True if the completion indicates success (status == 0)."""
        return self.status == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status,
            "complete_time_ns": self.complete_time_ns,
            "bytes_transferred": self.bytes_transferred,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IoCompletion":
        return cls(
            request_id=data["request_id"],
            status=data["status"],
            complete_time_ns=data["complete_time_ns"],
            bytes_transferred=data["bytes_transferred"],
        )
