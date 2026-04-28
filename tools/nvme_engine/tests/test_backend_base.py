"""
Unit tests for the Storage Backend Abstraction Layer.
"""

import pytest

from nvme_engine.backends.base import (
    BackendStats,
    StorageBackendOps,
    WalEntry,
)


class TestBackendStats:
    """Tests for BackendStats data model."""

    def test_default_construction(self):
        """Test BackendStats with default values."""
        stats = BackendStats()
        assert stats.total_reads == 0
        assert stats.total_writes == 0
        assert stats.bytes_read == 0
        assert stats.bytes_written == 0
        assert stats.read_errors == 0
        assert stats.write_errors == 0
        assert stats.checksum_errors == 0

    def test_custom_values(self):
        """Test BackendStats with custom values."""
        stats = BackendStats(
            total_reads=100,
            total_writes=50,
            bytes_read=4096 * 100,
            bytes_written=4096 * 50,
            avg_read_latency_us=5.5,
            avg_write_latency_us=8.2,
        )
        assert stats.total_reads == 100
        assert stats.total_writes == 50
        assert stats.bytes_read == 409600
        assert stats.bytes_written == 204800
        assert stats.avg_read_latency_us == 5.5
        assert stats.avg_write_latency_us == 8.2

    def test_serialization(self):
        """Test BackendStats serialization to dict."""
        stats = BackendStats(
            total_reads=10,
            total_writes=5,
            bytes_read=1024,
            bytes_written=512,
        )
        data = stats.to_dict()
        assert data["total_reads"] == 10
        assert data["total_writes"] == 5
        assert data["bytes_read"] == 1024
        assert data["bytes_written"] == 512


class TestWalEntry:
    """Tests for Write-Ahead Log entry."""

    def test_construction(self):
        """Test WalEntry construction."""
        data = b"test data"
        checksum = b"0" * 32
        entry = WalEntry(
            sequence_number=1,
            lba=0,
            length=len(data),
            checksum=checksum,
            data=data,
        )
        assert entry.sequence_number == 1
        assert entry.lba == 0
        assert entry.length == 9
        assert entry.checksum == checksum
        assert entry.data == data

    def test_serialization_roundtrip(self):
        """Test WalEntry serialization and deserialization."""
        data = b"test data for WAL entry"
        checksum = b"x" * 32
        entry = WalEntry(
            sequence_number=42,
            lba=1000,
            length=len(data),
            checksum=checksum,
            data=data,
        )

        # Serialize
        serialized = entry.to_bytes()
        assert isinstance(serialized, bytes)
        assert len(serialized) > len(data)

        # Deserialize
        restored = WalEntry.from_bytes(serialized)
        assert restored.sequence_number == 42
        assert restored.lba == 1000
        assert restored.length == len(data)
        assert restored.checksum == checksum
        assert restored.data == data

    def test_deserialization_invalid_short_data(self):
        """Test WalEntry deserialization with too-short data."""
        with pytest.raises(ValueError, match="too short"):
            WalEntry.from_bytes(b"short")

    def test_deserialization_invalid_length_mismatch(self):
        """Test WalEntry deserialization with length mismatch."""
        # Create a valid entry but truncate the data
        data = b"test data"
        checksum = b"0" * 32
        entry = WalEntry(
            sequence_number=1,
            lba=0,
            length=len(data),
            checksum=checksum,
            data=data,
        )
        serialized = entry.to_bytes()

        # Truncate the data portion
        truncated = serialized[:-5]

        with pytest.raises(ValueError, match="expected .* bytes"):
            WalEntry.from_bytes(truncated)


class MockBackend(StorageBackendOps):
    """Mock backend implementation for testing."""

    def __init__(self):
        super().__init__()
        self.storage: Dict[int, bytes] = {}
        self.snapshots: Dict[str, Dict[int, bytes]] = {}
        self.init_called = False
        self.destroy_called = False

    def init(self, config):
        self.init_called = True
        self._initialized = True

    def destroy(self):
        self.destroy_called = True
        self.storage.clear()
        self.snapshots.clear()
        self._initialized = False

    def read(self, lba, length):
        if lba not in self.storage:
            return b"\x00" * length
        data = self.storage[lba]
        self._stats.total_reads += 1
        self._stats.bytes_read += len(data)
        return data[:length]

    def write(self, lba, data):
        self._wal_append(lba, data)
        self.storage[lba] = data
        self._stats.total_writes += 1
        self._stats.bytes_written += len(data)

    def flush(self):
        self._wal_clear()
        self._stats.total_flushes += 1

    def trim(self, lba, length):
        if lba in self.storage:
            del self.storage[lba]
        self._stats.total_trims += 1

    def snapshot_create(self, name):
        self.snapshots[name] = self.storage.copy()

    def snapshot_delete(self, name):
        if name in self.snapshots:
            del self.snapshots[name]

    def snapshot_restore(self, name):
        if name in self.snapshots:
            self.storage = self.snapshots[name].copy()


class TestStorageBackendOps:
    """Tests for StorageBackendOps abstract base class."""

    def test_initialization(self):
        """Test backend initialization."""
        backend = MockBackend()
        assert not backend.is_initialized
        assert backend._stats.total_reads == 0

        backend.init({})
        assert backend.is_initialized
        assert backend.init_called

    def test_destroy(self):
        """Test backend destruction."""
        backend = MockBackend()
        backend.init({})
        assert backend.is_initialized

        backend.destroy()
        assert backend.destroy_called
        assert not backend.is_initialized

    def test_read_write(self):
        """Test basic read/write operations."""
        backend = MockBackend()
        backend.init({})

        # Write data
        data = b"test data"
        backend.write(0, data)
        assert backend._stats.total_writes == 1
        assert backend._stats.bytes_written == len(data)

        # Read data back
        read_data = backend.read(0, len(data))
        assert read_data == data
        assert backend._stats.total_reads == 1
        assert backend._stats.bytes_read == len(data)

    def test_flush(self):
        """Test flush operation."""
        backend = MockBackend()
        backend.init({})

        backend.write(0, b"data")
        assert len(backend._wal) == 1

        backend.flush()
        assert backend._stats.total_flushes == 1
        assert len(backend._wal) == 0

    def test_trim(self):
        """Test trim operation."""
        backend = MockBackend()
        backend.init({})

        backend.write(0, b"data")
        assert 0 in backend.storage

        backend.trim(0, 4)
        assert 0 not in backend.storage
        assert backend._stats.total_trims == 1

    def test_snapshot_create(self):
        """Test snapshot creation."""
        backend = MockBackend()
        backend.init({})

        backend.write(0, b"data1")
        backend.write(1, b"data2")

        backend.snapshot_create("snap1")
        assert "snap1" in backend.snapshots
        assert len(backend.snapshots["snap1"]) == 2

    def test_snapshot_delete(self):
        """Test snapshot deletion."""
        backend = MockBackend()
        backend.init({})

        backend.write(0, b"data")
        backend.snapshot_create("snap1")
        assert "snap1" in backend.snapshots

        backend.snapshot_delete("snap1")
        assert "snap1" not in backend.snapshots

    def test_snapshot_restore(self):
        """Test snapshot restoration."""
        backend = MockBackend()
        backend.init({})

        # Write initial data and create snapshot
        backend.write(0, b"original")
        backend.snapshot_create("snap1")

        # Modify data
        backend.write(0, b"modified")
        assert backend.read(0, 8) == b"modified"

        # Restore snapshot
        backend.snapshot_restore("snap1")
        assert backend.read(0, 8) == b"original"

    def test_checksum_calculation(self):
        """Test checksum calculation."""
        backend = MockBackend()
        data = b"test data"

        checksum1 = backend._calculate_checksum(data)
        assert isinstance(checksum1, bytes)
        assert len(checksum1) == 32  # SHA-256 is 32 bytes

        # Same data should produce same checksum
        checksum2 = backend._calculate_checksum(data)
        assert checksum1 == checksum2

        # Different data should produce different checksum
        checksum3 = backend._calculate_checksum(b"different data")
        assert checksum1 != checksum3

    def test_checksum_verification(self):
        """Test checksum verification."""
        backend = MockBackend()
        data = b"test data"

        checksum = backend._calculate_checksum(data)
        assert backend._verify_checksum(data, checksum)

        # Wrong checksum should fail
        wrong_checksum = b"0" * 32
        assert not backend._verify_checksum(data, wrong_checksum)

    def test_wal_append(self):
        """Test WAL append operation."""
        backend = MockBackend()
        backend.init({})

        data = b"test data"
        backend._wal_append(0, data)

        assert len(backend._wal) == 1
        assert backend._wal[0].lba == 0
        assert backend._wal[0].data == data
        assert backend._wal[0].sequence_number == 0

        # Second append should increment sequence
        backend._wal_append(1, b"more data")
        assert len(backend._wal) == 2
        assert backend._wal[1].sequence_number == 1

    def test_wal_clear(self):
        """Test WAL clear operation."""
        backend = MockBackend()
        backend.init({})

        backend._wal_append(0, b"data1")
        backend._wal_append(1, b"data2")
        assert len(backend._wal) == 2

        backend._wal_clear()
        assert len(backend._wal) == 0

    def test_wal_replay_ordering(self):
        """Test WAL replay maintains write ordering."""
        backend = MockBackend()
        backend.init({})

        # Add entries out of order
        backend._wal.append(
            WalEntry(
                sequence_number=2,
                lba=2,
                length=5,
                checksum=backend._calculate_checksum(b"data3"),
                data=b"data3",
            )
        )
        backend._wal.append(
            WalEntry(
                sequence_number=0,
                lba=0,
                length=5,
                checksum=backend._calculate_checksum(b"data1"),
                data=b"data1",
            )
        )
        backend._wal.append(
            WalEntry(
                sequence_number=1,
                lba=1,
                length=5,
                checksum=backend._calculate_checksum(b"data2"),
                data=b"data2",
            )
        )

        # Clear storage and replay
        backend.storage.clear()
        backend._stats.total_writes = 0

        backend._wal_replay()

        # Should have written all entries
        assert backend._stats.total_writes == 3
        assert backend.storage[0] == b"data1"
        assert backend.storage[1] == b"data2"
        assert backend.storage[2] == b"data3"

    def test_context_manager(self):
        """Test backend as context manager."""
        backend = MockBackend()
        backend.init({})

        with backend as b:
            assert b.is_initialized
            b.write(0, b"data")

        # Should be destroyed after context exit
        assert backend.destroy_called

    def test_get_stats(self):
        """Test get_stats method."""
        backend = MockBackend()
        backend.init({})

        backend.write(0, b"data")
        backend.read(0, 4)
        backend.flush()

        stats = backend.get_stats()
        assert stats.total_writes == 1
        assert stats.total_reads == 1
        assert stats.total_flushes == 1
        assert stats.bytes_written == 4
        assert stats.bytes_read == 4
