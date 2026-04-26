"""
Configuration data models for the Software NVMe Engine.

Covers: NvmeDeviceConfig, BackendConfig, QosConfig, CacheConfig,
        SecurityConfig, FeatureFlags, PerformanceConfig, and related types.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class BackendType(str, Enum):
    """Storage backend type."""

    MEMORY = "MEMORY"
    FILE = "FILE"
    NETWORK = "NETWORK"
    HYBRID = "HYBRID"


class TransportType(str, Enum):
    """Network transport type for the network backend."""

    RDMA = "RDMA"
    TCP = "TCP"


class CachePolicy(str, Enum):
    """Cache eviction / replacement policy."""

    ARC = "ARC"
    LRU = "LRU"
    LFU = "LFU"
    TWO_Q = "TWO_Q"


class EncryptionAlgorithm(str, Enum):
    """Encryption algorithm for data-at-rest."""

    AES256 = "AES256"
    AES128 = "AES128"


# ---------------------------------------------------------------------------
# Backend sub-configurations
# ---------------------------------------------------------------------------


@dataclass
class MemoryBackendConfig:
    """Configuration for a memory-based storage backend."""

    size_bytes: int
    numa_node: int = 0

    def __post_init__(self) -> None:
        if self.size_bytes <= 0:
            raise ValueError(f"size_bytes must be positive, got {self.size_bytes}")
        if self.numa_node < 0:
            raise ValueError(f"numa_node must be >= 0, got {self.numa_node}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "size_bytes": self.size_bytes,
            "numa_node": self.numa_node,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryBackendConfig":
        return cls(
            size_bytes=data["size_bytes"],
            numa_node=data.get("numa_node", 0),
        )


@dataclass
class FileBackendConfig:
    """Configuration for a file-based storage backend."""

    path: str
    sparse: bool = True
    direct_io: bool = False

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("path must not be empty")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "sparse": self.sparse,
            "direct_io": self.direct_io,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileBackendConfig":
        return cls(
            path=data["path"],
            sparse=data.get("sparse", True),
            direct_io=data.get("direct_io", False),
        )


@dataclass
class NetworkBackendConfig:
    """Configuration for a network-based storage backend."""

    host: str
    port: int
    transport: TransportType = TransportType.TCP

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("host must not be empty")
        if not (1 <= self.port <= 65535):
            raise ValueError(f"port must be 1-65535, got {self.port}")
        if not isinstance(self.transport, TransportType):
            self.transport = TransportType(self.transport)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "transport": self.transport.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NetworkBackendConfig":
        return cls(
            host=data["host"],
            port=data["port"],
            transport=TransportType(data.get("transport", "TCP")),
        )


@dataclass
class HybridBackendConfig:
    """Configuration for a hybrid (multi-backend) storage backend."""

    backends: List[BackendConfig] = field(default_factory=list)
    tiering_policy: str = "hot_cold"

    def __post_init__(self) -> None:
        if len(self.backends) < 2:
            raise ValueError(
                f"hybrid backend requires at least 2 backends, got {len(self.backends)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backends": [b.to_dict() for b in self.backends],
            "tiering_policy": self.tiering_policy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridBackendConfig":
        backends = [BackendConfig.from_dict(b) for b in data.get("backends", [])]
        return cls(
            backends=backends,
            tiering_policy=data.get("tiering_policy", "hot_cold"),
        )


@dataclass
class BackendConfig:
    """
    Top-level backend configuration.

    Exactly one of memory/file/network/hybrid must be set, matching the type.
    """

    type: BackendType
    memory: Optional[MemoryBackendConfig] = None
    file: Optional[FileBackendConfig] = None
    network: Optional[NetworkBackendConfig] = None
    hybrid: Optional[HybridBackendConfig] = None

    def __post_init__(self) -> None:
        if not isinstance(self.type, BackendType):
            self.type = BackendType(self.type)
        self._validate_backend_match()

    def _validate_backend_match(self) -> None:
        """Ensure the sub-config matches the declared type."""
        type_to_field = {
            BackendType.MEMORY: "memory",
            BackendType.FILE: "file",
            BackendType.NETWORK: "network",
            BackendType.HYBRID: "hybrid",
        }
        required_field = type_to_field[self.type]
        if getattr(self, required_field) is None:
            raise ValueError(
                f"BackendConfig type is {self.type.value} but '{required_field}' config is None"
            )

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"type": self.type.value}
        if self.memory is not None:
            d["memory"] = self.memory.to_dict()
        if self.file is not None:
            d["file"] = self.file.to_dict()
        if self.network is not None:
            d["network"] = self.network.to_dict()
        if self.hybrid is not None:
            d["hybrid"] = self.hybrid.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackendConfig":
        backend_type = BackendType(data["type"])
        memory = (
            MemoryBackendConfig.from_dict(data["memory"])
            if "memory" in data
            else None
        )
        file_ = (
            FileBackendConfig.from_dict(data["file"]) if "file" in data else None
        )
        network = (
            NetworkBackendConfig.from_dict(data["network"])
            if "network" in data
            else None
        )
        hybrid = (
            HybridBackendConfig.from_dict(data["hybrid"])
            if "hybrid" in data
            else None
        )
        return cls(
            type=backend_type,
            memory=memory,
            file=file_,
            network=network,
            hybrid=hybrid,
        )


# ---------------------------------------------------------------------------
# Performance configuration
# ---------------------------------------------------------------------------


@dataclass
class PerformanceConfig:
    """Performance tuning parameters for a virtual NVMe device."""

    max_iops: int = 1_000_000
    max_bandwidth_bytes_per_sec: int = 10 * 1024 * 1024 * 1024  # 10 GB/s
    target_latency_us: int = 10
    polling_mode: bool = False
    zero_copy: bool = False

    def __post_init__(self) -> None:
        if self.max_iops <= 0:
            raise ValueError(f"max_iops must be positive, got {self.max_iops}")
        if self.max_bandwidth_bytes_per_sec <= 0:
            raise ValueError(
                f"max_bandwidth_bytes_per_sec must be positive, got {self.max_bandwidth_bytes_per_sec}"
            )
        if self.target_latency_us <= 0:
            raise ValueError(
                f"target_latency_us must be positive, got {self.target_latency_us}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_iops": self.max_iops,
            "max_bandwidth_bytes_per_sec": self.max_bandwidth_bytes_per_sec,
            "target_latency_us": self.target_latency_us,
            "polling_mode": self.polling_mode,
            "zero_copy": self.zero_copy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceConfig":
        return cls(
            max_iops=data.get("max_iops", 1_000_000),
            max_bandwidth_bytes_per_sec=data.get(
                "max_bandwidth_bytes_per_sec", 10 * 1024 * 1024 * 1024
            ),
            target_latency_us=data.get("target_latency_us", 10),
            polling_mode=data.get("polling_mode", False),
            zero_copy=data.get("zero_copy", False),
        )


# ---------------------------------------------------------------------------
# QoS configuration
# ---------------------------------------------------------------------------


@dataclass
class QosConfig:
    """Quality-of-Service configuration for a virtual NVMe device."""

    priority: int = 2          # 0 (highest) – 3 (lowest)
    weight: int = 100          # 1 – 1000
    iops_limit: int = 0        # 0 = unlimited
    bandwidth_limit: int = 0   # 0 = unlimited (bytes/sec)
    isolation_enabled: bool = False

    def __post_init__(self) -> None:
        if not (0 <= self.priority <= 3):
            raise ValueError(f"priority must be 0-3, got {self.priority}")
        if not (1 <= self.weight <= 1000):
            raise ValueError(f"weight must be 1-1000, got {self.weight}")
        if self.iops_limit < 0:
            raise ValueError(f"iops_limit must be >= 0, got {self.iops_limit}")
        if self.bandwidth_limit < 0:
            raise ValueError(
                f"bandwidth_limit must be >= 0, got {self.bandwidth_limit}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "priority": self.priority,
            "weight": self.weight,
            "iops_limit": self.iops_limit,
            "bandwidth_limit": self.bandwidth_limit,
            "isolation_enabled": self.isolation_enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QosConfig":
        return cls(
            priority=data.get("priority", 2),
            weight=data.get("weight", 100),
            iops_limit=data.get("iops_limit", 0),
            bandwidth_limit=data.get("bandwidth_limit", 0),
            isolation_enabled=data.get("isolation_enabled", False),
        )


# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------


@dataclass
class CacheConfig:
    """Cache configuration for a virtual NVMe device."""

    enabled: bool = True
    size_bytes: int = 16 * 1024 * 1024 * 1024  # 16 GB
    policy: CachePolicy = CachePolicy.ARC
    prefetch_threshold: int = 4   # consecutive sequential blocks before prefetch
    write_back: bool = True
    flush_interval_ms: int = 5_000  # 5 seconds

    def __post_init__(self) -> None:
        if self.size_bytes <= 0:
            raise ValueError(f"size_bytes must be positive, got {self.size_bytes}")
        if not isinstance(self.policy, CachePolicy):
            self.policy = CachePolicy(self.policy)
        if self.prefetch_threshold < 0:
            raise ValueError(
                f"prefetch_threshold must be >= 0, got {self.prefetch_threshold}"
            )
        if self.flush_interval_ms <= 0:
            raise ValueError(
                f"flush_interval_ms must be positive, got {self.flush_interval_ms}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "size_bytes": self.size_bytes,
            "policy": self.policy.value,
            "prefetch_threshold": self.prefetch_threshold,
            "write_back": self.write_back,
            "flush_interval_ms": self.flush_interval_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheConfig":
        return cls(
            enabled=data.get("enabled", True),
            size_bytes=data.get("size_bytes", 16 * 1024 * 1024 * 1024),
            policy=CachePolicy(data.get("policy", "ARC")),
            prefetch_threshold=data.get("prefetch_threshold", 4),
            write_back=data.get("write_back", True),
            flush_interval_ms=data.get("flush_interval_ms", 5_000),
        )


# ---------------------------------------------------------------------------
# Security configuration
# ---------------------------------------------------------------------------


@dataclass
class AccessRule:
    """A single access control rule."""

    subject: str          # user or process identifier
    namespace_id: int     # namespace this rule applies to (0 = all)
    read_allowed: bool = True
    write_allowed: bool = True

    def __post_init__(self) -> None:
        if not self.subject:
            raise ValueError("subject must not be empty")
        if self.namespace_id < 0:
            raise ValueError(f"namespace_id must be >= 0, got {self.namespace_id}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "namespace_id": self.namespace_id,
            "read_allowed": self.read_allowed,
            "write_allowed": self.write_allowed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AccessRule":
        return cls(
            subject=data["subject"],
            namespace_id=data.get("namespace_id", 0),
            read_allowed=data.get("read_allowed", True),
            write_allowed=data.get("write_allowed", True),
        )


@dataclass
class SecurityConfig:
    """Security configuration for a virtual NVMe device."""

    encryption_enabled: bool = False
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES256
    key_id: str = ""
    access_control_enabled: bool = False
    rules: List[AccessRule] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.algorithm, EncryptionAlgorithm):
            self.algorithm = EncryptionAlgorithm(self.algorithm)
        if self.encryption_enabled and not self.key_id:
            raise ValueError("key_id must be set when encryption is enabled")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "encryption_enabled": self.encryption_enabled,
            "algorithm": self.algorithm.value,
            "key_id": self.key_id,
            "access_control_enabled": self.access_control_enabled,
            "rules": [r.to_dict() for r in self.rules],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecurityConfig":
        rules = [AccessRule.from_dict(r) for r in data.get("rules", [])]
        return cls(
            encryption_enabled=data.get("encryption_enabled", False),
            algorithm=EncryptionAlgorithm(data.get("algorithm", "AES256")),
            key_id=data.get("key_id", ""),
            access_control_enabled=data.get("access_control_enabled", False),
            rules=rules,
        )


# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------


@dataclass
class FeatureFlags:
    """Optional feature flags for a virtual NVMe device."""

    atomic_writes: bool = False
    reservations: bool = False
    nvme_of: bool = False
    smart: bool = True
    hot_plug: bool = False
    live_migration: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "atomic_writes": self.atomic_writes,
            "reservations": self.reservations,
            "nvme_of": self.nvme_of,
            "smart": self.smart,
            "hot_plug": self.hot_plug,
            "live_migration": self.live_migration,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureFlags":
        return cls(
            atomic_writes=data.get("atomic_writes", False),
            reservations=data.get("reservations", False),
            nvme_of=data.get("nvme_of", False),
            smart=data.get("smart", True),
            hot_plug=data.get("hot_plug", False),
            live_migration=data.get("live_migration", False),
        )


# ---------------------------------------------------------------------------
# Top-level device configuration
# ---------------------------------------------------------------------------


@dataclass
class NvmeDeviceConfig:
    """
    Complete configuration for a virtual NVMe device.

    This is the root configuration object that aggregates all sub-configurations.
    """

    name: str
    capacity_bytes: int
    namespace_count: int
    max_queue_pairs: int
    queue_depth: int
    backend: BackendConfig
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    qos: QosConfig = field(default_factory=QosConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if len(self.name) > 255:
            raise ValueError(f"name must be <= 255 characters, got {len(self.name)}")
        if self.capacity_bytes <= 0:
            raise ValueError(
                f"capacity_bytes must be positive, got {self.capacity_bytes}"
            )
        if self.namespace_count <= 0:
            raise ValueError(
                f"namespace_count must be positive, got {self.namespace_count}"
            )
        if not (1 <= self.max_queue_pairs <= 65535):
            raise ValueError(
                f"max_queue_pairs must be 1-65535, got {self.max_queue_pairs}"
            )
        if not (1 <= self.queue_depth <= 65535):
            raise ValueError(
                f"queue_depth must be 1-65535, got {self.queue_depth}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "capacity_bytes": self.capacity_bytes,
            "namespace_count": self.namespace_count,
            "max_queue_pairs": self.max_queue_pairs,
            "queue_depth": self.queue_depth,
            "backend": self.backend.to_dict(),
            "performance": self.performance.to_dict(),
            "qos": self.qos.to_dict(),
            "cache": self.cache.to_dict(),
            "security": self.security.to_dict(),
            "features": self.features.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NvmeDeviceConfig":
        return cls(
            name=data["name"],
            capacity_bytes=data["capacity_bytes"],
            namespace_count=data["namespace_count"],
            max_queue_pairs=data["max_queue_pairs"],
            queue_depth=data["queue_depth"],
            backend=BackendConfig.from_dict(data["backend"]),
            performance=PerformanceConfig.from_dict(data.get("performance", {})),
            qos=QosConfig.from_dict(data.get("qos", {})),
            cache=CacheConfig.from_dict(data.get("cache", {})),
            security=SecurityConfig.from_dict(data.get("security", {})),
            features=FeatureFlags.from_dict(data.get("features", {})),
        )
