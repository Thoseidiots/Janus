"""
Unit tests for configuration data models.

Tests: NvmeDeviceConfig, BackendConfig, QosConfig, CacheConfig,
       SecurityConfig, FeatureFlags, PerformanceConfig
"""

import pytest

from nvme_engine.models.config import (
    AccessRule,
    BackendConfig,
    BackendType,
    CacheConfig,
    CachePolicy,
    EncryptionAlgorithm,
    FeatureFlags,
    FileBackendConfig,
    HybridBackendConfig,
    MemoryBackendConfig,
    NetworkBackendConfig,
    NvmeDeviceConfig,
    PerformanceConfig,
    QosConfig,
    SecurityConfig,
    TransportType,
)


# ---------------------------------------------------------------------------
# MemoryBackendConfig Tests
# ---------------------------------------------------------------------------


class TestMemoryBackendConfig:
    """Tests for MemoryBackendConfig."""

    def test_valid_construction(self):
        """Test creating a valid memory backend config."""
        config = MemoryBackendConfig(size_bytes=1024 * 1024 * 1024, numa_node=0)
        assert config.size_bytes == 1024 * 1024 * 1024
        assert config.numa_node == 0

    def test_default_numa_node(self):
        """Test default NUMA node is 0."""
        config = MemoryBackendConfig(size_bytes=1024)
        assert config.numa_node == 0

    def test_invalid_size_bytes(self):
        """Test that negative or zero size_bytes raises ValueError."""
        with pytest.raises(ValueError, match="size_bytes must be positive"):
            MemoryBackendConfig(size_bytes=0)
        with pytest.raises(ValueError, match="size_bytes must be positive"):
            MemoryBackendConfig(size_bytes=-1)

    def test_invalid_numa_node(self):
        """Test that negative numa_node raises ValueError."""
        with pytest.raises(ValueError, match="numa_node must be >= 0"):
            MemoryBackendConfig(size_bytes=1024, numa_node=-1)

    def test_serialization(self):
        """Test to_dict and from_dict round-trip."""
        config = MemoryBackendConfig(size_bytes=2048, numa_node=1)
        data = config.to_dict()
        assert data == {"size_bytes": 2048, "numa_node": 1}
        
        restored = MemoryBackendConfig.from_dict(data)
        assert restored.size_bytes == config.size_bytes
        assert restored.numa_node == config.numa_node


# ---------------------------------------------------------------------------
# FileBackendConfig Tests
# ---------------------------------------------------------------------------


class TestFileBackendConfig:
    """Tests for FileBackendConfig."""

    def test_valid_construction(self):
        """Test creating a valid file backend config."""
        config = FileBackendConfig(path="/tmp/nvme.img", sparse=True, direct_io=False)
        assert config.path == "/tmp/nvme.img"
        assert config.sparse is True
        assert config.direct_io is False

    def test_defaults(self):
        """Test default values for sparse and direct_io."""
        config = FileBackendConfig(path="/tmp/test.img")
        assert config.sparse is True
        assert config.direct_io is False

    def test_empty_path(self):
        """Test that empty path raises ValueError."""
        with pytest.raises(ValueError, match="path must not be empty"):
            FileBackendConfig(path="")

    def test_serialization(self):
        """Test to_dict and from_dict round-trip."""
        config = FileBackendConfig(path="/data/nvme.img", sparse=False, direct_io=True)
        data = config.to_dict()
        assert data == {"path": "/data/nvme.img", "sparse": False, "direct_io": True}
        
        restored = FileBackendConfig.from_dict(data)
        assert restored.path == config.path
        assert restored.sparse == config.sparse
        assert restored.direct_io == config.direct_io


# ---------------------------------------------------------------------------
# NetworkBackendConfig Tests
# ---------------------------------------------------------------------------


class TestNetworkBackendConfig:
    """Tests for NetworkBackendConfig."""

    def test_valid_construction(self):
        """Test creating a valid network backend config."""
        config = NetworkBackendConfig(host="192.168.1.100", port=4420, transport=TransportType.TCP)
        assert config.host == "192.168.1.100"
        assert config.port == 4420
        assert config.transport == TransportType.TCP

    def test_default_transport(self):
        """Test default transport is TCP."""
        config = NetworkBackendConfig(host="localhost", port=8080)
        assert config.transport == TransportType.TCP

    def test_empty_host(self):
        """Test that empty host raises ValueError."""
        with pytest.raises(ValueError, match="host must not be empty"):
            NetworkBackendConfig(host="", port=8080)

    def test_invalid_port(self):
        """Test that invalid port raises ValueError."""
        with pytest.raises(ValueError, match="port must be 1-65535"):
            NetworkBackendConfig(host="localhost", port=0)
        with pytest.raises(ValueError, match="port must be 1-65535"):
            NetworkBackendConfig(host="localhost", port=65536)

    def test_transport_string_conversion(self):
        """Test that transport string is converted to enum."""
        config = NetworkBackendConfig(host="localhost", port=8080, transport="RDMA")
        assert config.transport == TransportType.RDMA

    def test_serialization(self):
        """Test to_dict and from_dict round-trip."""
        config = NetworkBackendConfig(host="10.0.0.1", port=9000, transport=TransportType.RDMA)
        data = config.to_dict()
        assert data == {"host": "10.0.0.1", "port": 9000, "transport": "RDMA"}
        
        restored = NetworkBackendConfig.from_dict(data)
        assert restored.host == config.host
        assert restored.port == config.port
        assert restored.transport == config.transport


# ---------------------------------------------------------------------------
# BackendConfig Tests
# ---------------------------------------------------------------------------


class TestBackendConfig:
    """Tests for BackendConfig."""

    def test_memory_backend(self):
        """Test creating a memory backend config."""
        mem_config = MemoryBackendConfig(size_bytes=1024)
        config = BackendConfig(type=BackendType.MEMORY, memory=mem_config)
        assert config.type == BackendType.MEMORY
        assert config.memory == mem_config

    def test_file_backend(self):
        """Test creating a file backend config."""
        file_config = FileBackendConfig(path="/tmp/test.img")
        config = BackendConfig(type=BackendType.FILE, file=file_config)
        assert config.type == BackendType.FILE
        assert config.file == file_config

    def test_type_mismatch_raises_error(self):
        """Test that type mismatch raises ValueError."""
        mem_config = MemoryBackendConfig(size_bytes=1024)
        with pytest.raises(ValueError, match="type is MEMORY but 'memory' config is None"):
            BackendConfig(type=BackendType.MEMORY, memory=None)

    def test_serialization_memory(self):
        """Test serialization of memory backend."""
        mem_config = MemoryBackendConfig(size_bytes=2048, numa_node=1)
        config = BackendConfig(type=BackendType.MEMORY, memory=mem_config)
        data = config.to_dict()
        
        restored = BackendConfig.from_dict(data)
        assert restored.type == BackendType.MEMORY
        assert restored.memory.size_bytes == 2048
        assert restored.memory.numa_node == 1


# ---------------------------------------------------------------------------
# PerformanceConfig Tests
# ---------------------------------------------------------------------------


class TestPerformanceConfig:
    """Tests for PerformanceConfig."""

    def test_defaults(self):
        """Test default performance config values."""
        config = PerformanceConfig()
        assert config.max_iops == 1_000_000
        assert config.max_bandwidth_bytes_per_sec == 10 * 1024 * 1024 * 1024
        assert config.target_latency_us == 10
        assert config.polling_mode is False
        assert config.zero_copy is False

    def test_custom_values(self):
        """Test custom performance config values."""
        config = PerformanceConfig(
            max_iops=500_000,
            max_bandwidth_bytes_per_sec=5 * 1024 * 1024 * 1024,
            target_latency_us=5,
            polling_mode=True,
            zero_copy=True
        )
        assert config.max_iops == 500_000
        assert config.max_bandwidth_bytes_per_sec == 5 * 1024 * 1024 * 1024
        assert config.target_latency_us == 5
        assert config.polling_mode is True
        assert config.zero_copy is True

    def test_invalid_max_iops(self):
        """Test that invalid max_iops raises ValueError."""
        with pytest.raises(ValueError, match="max_iops must be positive"):
            PerformanceConfig(max_iops=0)

    def test_serialization(self):
        """Test to_dict and from_dict round-trip."""
        config = PerformanceConfig(max_iops=100_000, polling_mode=True)
        data = config.to_dict()
        restored = PerformanceConfig.from_dict(data)
        assert restored.max_iops == config.max_iops
        assert restored.polling_mode == config.polling_mode


# ---------------------------------------------------------------------------
# QosConfig Tests
# ---------------------------------------------------------------------------


class TestQosConfig:
    """Tests for QosConfig."""

    def test_defaults(self):
        """Test default QoS config values."""
        config = QosConfig()
        assert config.priority == 2
        assert config.weight == 100
        assert config.iops_limit == 0
        assert config.bandwidth_limit == 0
        assert config.isolation_enabled is False

    def test_valid_priority_range(self):
        """Test valid priority values (0-3)."""
        for priority in range(4):
            config = QosConfig(priority=priority)
            assert config.priority == priority

    def test_invalid_priority(self):
        """Test that invalid priority raises ValueError."""
        with pytest.raises(ValueError, match="priority must be 0-3"):
            QosConfig(priority=-1)
        with pytest.raises(ValueError, match="priority must be 0-3"):
            QosConfig(priority=4)

    def test_valid_weight_range(self):
        """Test valid weight values (1-1000)."""
        config = QosConfig(weight=1)
        assert config.weight == 1
        config = QosConfig(weight=1000)
        assert config.weight == 1000

    def test_invalid_weight(self):
        """Test that invalid weight raises ValueError."""
        with pytest.raises(ValueError, match="weight must be 1-1000"):
            QosConfig(weight=0)
        with pytest.raises(ValueError, match="weight must be 1-1000"):
            QosConfig(weight=1001)

    def test_serialization(self):
        """Test to_dict and from_dict round-trip."""
        config = QosConfig(priority=1, weight=500, iops_limit=10000, isolation_enabled=True)
        data = config.to_dict()
        restored = QosConfig.from_dict(data)
        assert restored.priority == config.priority
        assert restored.weight == config.weight
        assert restored.iops_limit == config.iops_limit
        assert restored.isolation_enabled == config.isolation_enabled


# ---------------------------------------------------------------------------
# CacheConfig Tests
# ---------------------------------------------------------------------------


class TestCacheConfig:
    """Tests for CacheConfig."""

    def test_defaults(self):
        """Test default cache config values."""
        config = CacheConfig()
        assert config.enabled is True
        assert config.size_bytes == 16 * 1024 * 1024 * 1024
        assert config.policy == CachePolicy.ARC
        assert config.prefetch_threshold == 4
        assert config.write_back is True
        assert config.flush_interval_ms == 5_000

    def test_custom_values(self):
        """Test custom cache config values."""
        config = CacheConfig(
            enabled=False,
            size_bytes=8 * 1024 * 1024 * 1024,
            policy=CachePolicy.LRU,
            prefetch_threshold=8,
            write_back=False,
            flush_interval_ms=10_000
        )
        assert config.enabled is False
        assert config.size_bytes == 8 * 1024 * 1024 * 1024
        assert config.policy == CachePolicy.LRU
        assert config.prefetch_threshold == 8
        assert config.write_back is False
        assert config.flush_interval_ms == 10_000

    def test_invalid_size_bytes(self):
        """Test that invalid size_bytes raises ValueError."""
        with pytest.raises(ValueError, match="size_bytes must be positive"):
            CacheConfig(size_bytes=0)

    def test_policy_string_conversion(self):
        """Test that policy string is converted to enum."""
        config = CacheConfig(policy="LFU")
        assert config.policy == CachePolicy.LFU

    def test_serialization(self):
        """Test to_dict and from_dict round-trip."""
        config = CacheConfig(policy=CachePolicy.TWO_Q, prefetch_threshold=10)
        data = config.to_dict()
        restored = CacheConfig.from_dict(data)
        assert restored.policy == config.policy
        assert restored.prefetch_threshold == config.prefetch_threshold


# ---------------------------------------------------------------------------
# SecurityConfig Tests
# ---------------------------------------------------------------------------


class TestSecurityConfig:
    """Tests for SecurityConfig."""

    def test_defaults(self):
        """Test default security config values."""
        config = SecurityConfig()
        assert config.encryption_enabled is False
        assert config.algorithm == EncryptionAlgorithm.AES256
        assert config.key_id == ""
        assert config.access_control_enabled is False
        assert config.rules == []

    def test_encryption_requires_key_id(self):
        """Test that encryption enabled requires key_id."""
        with pytest.raises(ValueError, match="key_id must be set when encryption is enabled"):
            SecurityConfig(encryption_enabled=True, key_id="")

    def test_valid_encryption_config(self):
        """Test valid encryption configuration."""
        config = SecurityConfig(
            encryption_enabled=True,
            algorithm=EncryptionAlgorithm.AES256,
            key_id="key-12345"
        )
        assert config.encryption_enabled is True
        assert config.key_id == "key-12345"

    def test_access_rules(self):
        """Test access control rules."""
        rule1 = AccessRule(subject="user1", namespace_id=1, read_allowed=True, write_allowed=False)
        rule2 = AccessRule(subject="user2", namespace_id=2, read_allowed=True, write_allowed=True)
        config = SecurityConfig(access_control_enabled=True, rules=[rule1, rule2])
        assert len(config.rules) == 2
        assert config.rules[0].subject == "user1"
        assert config.rules[1].subject == "user2"

    def test_serialization(self):
        """Test to_dict and from_dict round-trip."""
        rule = AccessRule(subject="admin", namespace_id=0, read_allowed=True, write_allowed=True)
        config = SecurityConfig(
            encryption_enabled=True,
            algorithm=EncryptionAlgorithm.AES128,
            key_id="test-key",
            access_control_enabled=True,
            rules=[rule]
        )
        data = config.to_dict()
        restored = SecurityConfig.from_dict(data)
        assert restored.encryption_enabled == config.encryption_enabled
        assert restored.algorithm == config.algorithm
        assert restored.key_id == config.key_id
        assert len(restored.rules) == 1
        assert restored.rules[0].subject == "admin"


# ---------------------------------------------------------------------------
# AccessRule Tests
# ---------------------------------------------------------------------------


class TestAccessRule:
    """Tests for AccessRule."""

    def test_valid_construction(self):
        """Test creating a valid access rule."""
        rule = AccessRule(subject="user1", namespace_id=1, read_allowed=True, write_allowed=False)
        assert rule.subject == "user1"
        assert rule.namespace_id == 1
        assert rule.read_allowed is True
        assert rule.write_allowed is False

    def test_defaults(self):
        """Test default values for read_allowed and write_allowed."""
        rule = AccessRule(subject="user1", namespace_id=0)
        assert rule.read_allowed is True
        assert rule.write_allowed is True

    def test_empty_subject(self):
        """Test that empty subject raises ValueError."""
        with pytest.raises(ValueError, match="subject must not be empty"):
            AccessRule(subject="", namespace_id=0)

    def test_invalid_namespace_id(self):
        """Test that negative namespace_id raises ValueError."""
        with pytest.raises(ValueError, match="namespace_id must be >= 0"):
            AccessRule(subject="user1", namespace_id=-1)

    def test_serialization(self):
        """Test to_dict and from_dict round-trip."""
        rule = AccessRule(subject="test", namespace_id=5, read_allowed=False, write_allowed=True)
        data = rule.to_dict()
        restored = AccessRule.from_dict(data)
        assert restored.subject == rule.subject
        assert restored.namespace_id == rule.namespace_id
        assert restored.read_allowed == rule.read_allowed
        assert restored.write_allowed == rule.write_allowed


# ---------------------------------------------------------------------------
# FeatureFlags Tests
# ---------------------------------------------------------------------------


class TestFeatureFlags:
    """Tests for FeatureFlags."""

    def test_defaults(self):
        """Test default feature flag values."""
        flags = FeatureFlags()
        assert flags.atomic_writes is False
        assert flags.reservations is False
        assert flags.nvme_of is False
        assert flags.smart is True
        assert flags.hot_plug is False
        assert flags.live_migration is False

    def test_custom_values(self):
        """Test custom feature flag values."""
        flags = FeatureFlags(
            atomic_writes=True,
            reservations=True,
            nvme_of=True,
            smart=False,
            hot_plug=True,
            live_migration=True
        )
        assert flags.atomic_writes is True
        assert flags.reservations is True
        assert flags.nvme_of is True
        assert flags.smart is False
        assert flags.hot_plug is True
        assert flags.live_migration is True

    def test_serialization(self):
        """Test to_dict and from_dict round-trip."""
        flags = FeatureFlags(atomic_writes=True, hot_plug=True)
        data = flags.to_dict()
        restored = FeatureFlags.from_dict(data)
        assert restored.atomic_writes == flags.atomic_writes
        assert restored.hot_plug == flags.hot_plug


# ---------------------------------------------------------------------------
# NvmeDeviceConfig Tests
# ---------------------------------------------------------------------------


class TestNvmeDeviceConfig:
    """Tests for NvmeDeviceConfig."""

    def test_minimal_valid_config(self):
        """Test creating a minimal valid device config."""
        backend = BackendConfig(
            type=BackendType.MEMORY,
            memory=MemoryBackendConfig(size_bytes=1024 * 1024 * 1024)
        )
        config = NvmeDeviceConfig(
            name="test-device",
            capacity_bytes=10 * 1024 * 1024 * 1024,
            namespace_count=1,
            max_queue_pairs=16,
            queue_depth=256,
            backend=backend
        )
        assert config.name == "test-device"
        assert config.capacity_bytes == 10 * 1024 * 1024 * 1024
        assert config.namespace_count == 1
        assert config.max_queue_pairs == 16
        assert config.queue_depth == 256

    def test_empty_name(self):
        """Test that empty name raises ValueError."""
        backend = BackendConfig(
            type=BackendType.MEMORY,
            memory=MemoryBackendConfig(size_bytes=1024)
        )
        with pytest.raises(ValueError, match="name must not be empty"):
            NvmeDeviceConfig(
                name="",
                capacity_bytes=1024,
                namespace_count=1,
                max_queue_pairs=1,
                queue_depth=1,
                backend=backend
            )

    def test_name_too_long(self):
        """Test that name > 255 characters raises ValueError."""
        backend = BackendConfig(
            type=BackendType.MEMORY,
            memory=MemoryBackendConfig(size_bytes=1024)
        )
        with pytest.raises(ValueError, match="name must be <= 255 characters"):
            NvmeDeviceConfig(
                name="a" * 256,
                capacity_bytes=1024,
                namespace_count=1,
                max_queue_pairs=1,
                queue_depth=1,
                backend=backend
            )

    def test_invalid_capacity(self):
        """Test that invalid capacity raises ValueError."""
        backend = BackendConfig(
            type=BackendType.MEMORY,
            memory=MemoryBackendConfig(size_bytes=1024)
        )
        with pytest.raises(ValueError, match="capacity_bytes must be positive"):
            NvmeDeviceConfig(
                name="test",
                capacity_bytes=0,
                namespace_count=1,
                max_queue_pairs=1,
                queue_depth=1,
                backend=backend
            )

    def test_invalid_queue_pairs(self):
        """Test that invalid max_queue_pairs raises ValueError."""
        backend = BackendConfig(
            type=BackendType.MEMORY,
            memory=MemoryBackendConfig(size_bytes=1024)
        )
        with pytest.raises(ValueError, match="max_queue_pairs must be 1-65535"):
            NvmeDeviceConfig(
                name="test",
                capacity_bytes=1024,
                namespace_count=1,
                max_queue_pairs=0,
                queue_depth=1,
                backend=backend
            )
        with pytest.raises(ValueError, match="max_queue_pairs must be 1-65535"):
            NvmeDeviceConfig(
                name="test",
                capacity_bytes=1024,
                namespace_count=1,
                max_queue_pairs=65536,
                queue_depth=1,
                backend=backend
            )

    def test_full_config_serialization(self):
        """Test complete device config serialization."""
        backend = BackendConfig(
            type=BackendType.FILE,
            file=FileBackendConfig(path="/tmp/test.img", sparse=True)
        )
        config = NvmeDeviceConfig(
            name="full-test",
            capacity_bytes=100 * 1024 * 1024 * 1024,
            namespace_count=4,
            max_queue_pairs=32,
            queue_depth=512,
            backend=backend,
            performance=PerformanceConfig(max_iops=500_000, polling_mode=True),
            qos=QosConfig(priority=1, weight=200),
            cache=CacheConfig(policy=CachePolicy.LRU),
            security=SecurityConfig(),
            features=FeatureFlags(atomic_writes=True, smart=True)
        )
        
        data = config.to_dict()
        restored = NvmeDeviceConfig.from_dict(data)
        
        assert restored.name == config.name
        assert restored.capacity_bytes == config.capacity_bytes
        assert restored.namespace_count == config.namespace_count
        assert restored.max_queue_pairs == config.max_queue_pairs
        assert restored.queue_depth == config.queue_depth
        assert restored.backend.type == BackendType.FILE
        assert restored.performance.max_iops == 500_000
        assert restored.qos.priority == 1
        assert restored.cache.policy == CachePolicy.LRU
        assert restored.features.atomic_writes is True
