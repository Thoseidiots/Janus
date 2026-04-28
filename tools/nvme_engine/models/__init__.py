"""
Data models for the Software NVMe Engine.
"""

from nvme_engine.models.config import (
    BackendType,
    TransportType,
    CachePolicy,
    EncryptionAlgorithm,
    MemoryBackendConfig,
    FileBackendConfig,
    NetworkBackendConfig,
    HybridBackendConfig,
    BackendConfig,
    PerformanceConfig,
    QosConfig,
    CacheConfig,
    AccessRule,
    SecurityConfig,
    FeatureFlags,
    NvmeDeviceConfig,
)
from nvme_engine.models.io_models import (
    IoType,
    IoRequest,
    IoCompletion,
)
from nvme_engine.models.errors import (
    NvmeErrorCode,
    NvmeError,
    NvmeConfigError,
    NvmeResourceError,
    NvmeIoError,
    NvmeDataCorruptionError,
    NvmeTimeoutError,
    NvmePermissionError,
    NvmeDeviceNotFoundError,
    NvmeBackendError,
)
from nvme_engine.models.telemetry import (
    LatencyHistogram,
    TelemetryMetrics,
)

__all__ = [
    # Config enums
    "BackendType",
    "TransportType",
    "CachePolicy",
    "EncryptionAlgorithm",
    # Config models
    "MemoryBackendConfig",
    "FileBackendConfig",
    "NetworkBackendConfig",
    "HybridBackendConfig",
    "BackendConfig",
    "PerformanceConfig",
    "QosConfig",
    "CacheConfig",
    "AccessRule",
    "SecurityConfig",
    "FeatureFlags",
    "NvmeDeviceConfig",
    # I/O models
    "IoType",
    "IoRequest",
    "IoCompletion",
    # Error hierarchy
    "NvmeErrorCode",
    "NvmeError",
    "NvmeConfigError",
    "NvmeResourceError",
    "NvmeIoError",
    "NvmeDataCorruptionError",
    "NvmeTimeoutError",
    "NvmePermissionError",
    "NvmeDeviceNotFoundError",
    "NvmeBackendError",
    # Telemetry
    "LatencyHistogram",
    "TelemetryMetrics",
]
