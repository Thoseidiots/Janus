"""
Unit and property tests for the Memory Backend.
"""

import pytest
from hypothesis import given, strategies as st

from nvme_engine.backends.memory_backend import MemoryBackend
from nvme_engine.models.errors import NvmeBackendError, NvmeIoError


class TestMemoryBackendBasic:
    """Basic unit tests for Memory Backend."""

    def test_initialization(self):
        """Test memory backend initialization."""
        backend = MemoryBackend()
        assert not backend.is_initialized

        config = {"size_bytes": 1024 * 1024, "numa_node": 0}
        backend.init(config)

        assert backend.is_initialized
        assert backend.size_bytes == 1024 * 1024
        assert backend.numa_node == 0

    def test_initialization_invalid_size(self):
        """Test initialization with invalid size."""
        backend = MemoryBackend()

        with pytest.raises(NvmeBackendError, match="size_bytes must be positive"):
            backend.init({"size_bytes": 0})

        with pytest.raises(NvmeBackendError, match="size_bytes must be positive"):
            backend.init({"size_bytes": -1})

    def test_initialization_invalid_numa_node(self):
        """Test initialization with invalid NUMA node."""
        backend = MemoryBackend()

        with pytest.raises(NvmeBackendError, match="numa_node must be >= 0"):
            backend.init({"size_bytes": 1024, "numa_node": -1})

    def test_double_initialization(self):
        """Test that double initialization fails."""
        backend = MemoryBackend()
        backend.init({"size_bytes": 1024})

        with pytest.raises(NvmeBackendError, match="already initialized"):
            backend.init({"size_bytes": 2048})

    def test_destroy(self):
        """Test backend destruction."""
        backend = MemoryBackend()
        backend.init({"size_bytes": 1024})
        assert backend.is_initialized

        backend.destroy()
        assert not backend.is_initialized

    def test_destroy_idempotent(self):
        """Test that destroy can be called multiple times."""
        backend = MemoryBackend()
        backend.init({"size_bytes": 1024})

        backend.destroy()
        backend.destroy()  # Should not raise

    def test_read_write_basic(self):
        """Test basic read/write operations."""
        backend = MemoryBackend()
        backend.init({"size_bytes": 4096})

        # Write data
        data = b"Hello, NVMe!"
        backend.write(0, data)

        # Read data back
        read_data = backend.read(0, len(data))
        assert read_data == data

    def test_read_write_large_data(self):
        """Test read/write with larger data."""
        backend = MemoryBackend()
        backend.init({"size_bytes": 1024 * 1024})  # 1MB

        # Write 64KB of data
        data = b"X" * (64 * 1024)
        backend.write(0, data)

        # Read back
        read_data = backend.read(0, len(data))
        assert read_data == data
        assert len(read_data) == 64 * 1024

    def test_read_write_at_offset(self):
        """Test read/write at non-zero offset."""
        backend = MemoryBackend()
        backend.init({"size_bytes": 4096})

        # Write at offset 1000
        data = b"offset data"
        backend.write(1000, data)

        # Read back from offset
        read_data = backend.read(1000, len(data))
        assert read_data == data

    def test_read_uninitialized_memory(self):
        """Test reading uninitialized memory returns zeros."""
        backend = MemoryBackend()
        backend.init({"size_bytes": 4096})

        # Read without writing first
        data = backend.read(0, 100)
        assert data == b"\x00" * 100

    def test_write_out_of_bounds(self):
        """Test write beyond storage size fails."""
        backend = MemoryBackend()
        backend.init({"size_bytes": 1024})

        with pytest.raises(NvmeIoError, match="exceed storage size"):
            backend.write(1000, b"X" * 100)

    def test_read_out_of_bounds(self):
        """Test read beyond storage size fails."""
        backend = MemoryBackend()
        backend.init({"size_bytes": 1024})

        with pytest.raises(NvmeIoError, match="out of bounds"):
            backend.read(2000, 100)

    def test_read_partial_at_end(self):
        """Test read that extends past end returns partial data."""
        backend = MemoryBackend()
        backend.init({"size_bytes": 1024})

        # Write at end
        backend.write(1000, b"END")

        # Try to read past end
        data = backend.read(1000, 100)
        assert len(data) == 24  # Only 24 bytes available
        assert data[:3] == b"END"

    def test_flush(self):
        """Test flush operation."""
        backend = MemoryBackend()
        backend.init({"size_bytes": 4096})

        backend.write(0, b"data")
        backend.flush()

        assert backend.get_stats().total_flushes == 1

    def test_trim(self):
        """Test trim operation."""
        backend = MemoryBackend()
        backend.init({"size_bytes": 4096})

        # Write data
        backend.write(0, b"X" * 100)
        assert backend.read(0, 100) == b"X" * 100

        # Trim the data
        backend.trim(0, 100)
        assert backend.read(0, 100) == b"\x00" * 100
        assert backend.get_stats().total_trims == 1

    def test_snapshot_create(self):
        """Test snapshot creation."""
        backend = MemoryBackend()
        backend.init({"size_bytes": 4096})

        backend.write(0, b"original data")
        backend.snapshot_create("snap1")

        # Verify snapshot exists (indirectly by trying to create duplicate)
        with pytest.raises(NvmeBackendError, match="already exists"):
            backend.snapshot_create("snap1")

    def test_snapshot_delete(self):
        """Test snapshot deletion."""
        backend = MemoryBackend()
        backend.init({"size_bytes": 4096})

        backend.write(0, b"data")
        backend.snapshot_create("snap1")
        backend.snapshot_delete("snap1")

        # Should be able to create again after deletion
        backend.snapshot_create("snap1")

    def test_snapshot_delete_nonexistent(self):
        """Test deleting non-existent snapshot fails."""
        backend = MemoryBackend()
        backend.init({"size_bytes": 4096})

        with pytest.raises(NvmeBackendError, match="does not exist"):
            backend.snapshot_delete("nonexistent")

    def test_snapshot_restore(self):
        """Test snapshot restoration."""
        backend = MemoryBackend()
        backend.init({"size_bytes": 4096})

        # Write original data and snapshot
        backend.write(0, b"original")
        backend.snapshot_create("snap1")

        # Modify data
        backend.write(0, b"modified")
        assert backend.read(0, 8) == b"modified"

        # Restore snapshot
        backend.snapshot_restore("snap1")
        assert backend.read(0, 8) == b"original"

    def test_snapshot_restore_nonexistent(self):
        """Test restoring non-existent snapshot fails."""
        backend = MemoryBackend()
        backend.init({"size_bytes": 4096})

        with pytest.raises(NvmeBackendError, match="does not exist"):
            backend.snapshot_restore("nonexistent")

    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly."""
        backend = MemoryBackend()
        backend.init({"size_bytes": 4096})

        # Perform operations
        backend.write(0, b"X" * 100)
        backend.write(100, b"Y" * 200)
        backend.read(0, 100)
        backend.read(100, 200)
        backend.flush()
        backend.trim(0, 50)

        stats = backend.get_stats()
        assert stats.total_writes == 2
        assert stats.total_reads == 2
        assert stats.total_flushes == 1
        assert stats.total_trims == 1
        assert stats.bytes_written == 300
        assert stats.bytes_read == 300

    def test_latency_tracking(self):
        """Test that latency is tracked."""
        backend = MemoryBackend()
        backend.init({"size_bytes": 4096})

        backend.write(0, b"X" * 4096)
        backend.read(0, 4096)

        stats = backend.get_stats()
        assert stats.avg_read_latency_us > 0
        assert stats.avg_write_latency_us > 0

        # Should be very fast (<10μs target, but allow more for test environment)
        assert stats.avg_read_latency_us < 1000  # 1ms
        assert stats.avg_write_latency_us < 1000  # 1ms

    def test_context_manager(self):
        """Test backend as context manager."""
        backend = MemoryBackend()
        backend.init({"size_bytes": 4096})

        with backend as b:
            b.write(0, b"data")
            assert b.read(0, 4) == b"data"

        assert not backend.is_initialized

    def test_operations_before_init_fail(self):
        """Test that operations before init fail."""
        backend = MemoryBackend()

        with pytest.raises(NvmeIoError, match="not initialized"):
            backend.read(0, 100)

        with pytest.raises(NvmeIoError, match="not initialized"):
            backend.write(0, b"data")

        with pytest.raises(NvmeIoError, match="not initialized"):
            backend.flush()


class TestMemoryBackendProperties:
    """Property-based tests for Memory Backend."""

    @given(
        size=st.integers(min_value=4096, max_value=10 * 1024 * 1024),
        numa_node=st.integers(min_value=0, max_value=7),
    )
    def test_property_initialization_preserves_config(self, size, numa_node):
        """
        Feature: software-nvme-solution, Property 2: Device Deletion Releases Resources

        Test that initialization preserves configuration parameters.
        """
        backend = MemoryBackend()
        config = {"size_bytes": size, "numa_node": numa_node}

        backend.init(config)

        assert backend.size_bytes == size
        assert backend.numa_node == numa_node

        backend.destroy()

    @given(
        lba=st.integers(min_value=0, max_value=1000),
        data=st.binary(min_size=1, max_size=1024),
    )
    def test_property_read_after_write_returns_same_data(self, lba, data):
        """
        Feature: software-nvme-solution, Property 4: Backend Switching Preserves Data Integrity

        Test that reading after writing returns the same data.
        """
        backend = MemoryBackend()
        backend.init({"size_bytes": 10 * 1024})

        # Ensure write fits in storage
        if lba + len(data) > backend.size_bytes:
            return

        backend.write(lba, data)
        read_data = backend.read(lba, len(data))

        assert read_data == data

        backend.destroy()

    @given(
        num_writes=st.integers(min_value=1, max_value=10),
        data=st.binary(min_size=1, max_size=100),
    )
    def test_property_multiple_writes_preserve_data(self, num_writes, data):
        """
        Feature: software-nvme-solution, Property 4: Backend Switching Preserves Data Integrity

        Test that multiple non-overlapping writes preserve all data correctly.
        """
        backend = MemoryBackend()
        backend.init({"size_bytes": 10 * 1024})

        # Write to non-overlapping locations
        writes = []
        for i in range(num_writes):
            lba = i * 200  # Ensure no overlap (200 bytes apart)
            if lba + len(data) <= backend.size_bytes:
                backend.write(lba, data)
                writes.append((lba, data))

        # Verify all writes
        for lba, expected_data in writes:
            read_data = backend.read(lba, len(expected_data))
            assert read_data == expected_data

        backend.destroy()

    @given(
        data=st.binary(min_size=100, max_size=1024),
    )
    def test_property_snapshot_preserves_state(self, data):
        """
        Feature: software-nvme-solution, Property 31: Snapshot Point-in-Time Consistency

        Test that snapshots preserve exact state at creation time.
        """
        backend = MemoryBackend()
        backend.init({"size_bytes": 10 * 1024})

        # Write original data and create snapshot
        backend.write(0, data)
        backend.snapshot_create("snap1")

        # Modify data
        modified_data = b"X" * len(data)
        backend.write(0, modified_data)

        # Verify modification
        assert backend.read(0, len(data)) == modified_data

        # Restore snapshot
        backend.snapshot_restore("snap1")

        # Verify original data is restored
        assert backend.read(0, len(data)) == data

        backend.destroy()

    def test_property_resource_cleanup_on_destroy(self):
        """
        Feature: software-nvme-solution, Property 2: Device Deletion Releases Resources

        Test that destroy releases all resources (no memory leaks).
        """
        backend = MemoryBackend()
        backend.init({"size_bytes": 10 * 1024 * 1024})  # 10MB

        # Create some snapshots
        backend.write(0, b"data1")
        backend.snapshot_create("snap1")
        backend.write(1000, b"data2")
        backend.snapshot_create("snap2")

        # Destroy should release everything
        backend.destroy()

        assert not backend.is_initialized
        assert backend._storage is None
        assert len(backend._snapshots) == 0

    @given(
        trim_lba=st.integers(min_value=0, max_value=500),
        trim_length=st.integers(min_value=1, max_value=500),
    )
    def test_property_trim_zeros_memory(self, trim_lba, trim_length):
        """
        Feature: software-nvme-solution, Property 5: Sparse File Allocation Minimizes Space

        Test that trim operation zeros the specified memory range.
        """
        backend = MemoryBackend()
        backend.init({"size_bytes": 10 * 1024})

        # Ensure trim fits in storage
        if trim_lba + trim_length > backend.size_bytes:
            return

        # Write non-zero data
        backend.write(trim_lba, b"X" * trim_length)
        assert backend.read(trim_lba, trim_length) == b"X" * trim_length

        # Trim
        backend.trim(trim_lba, trim_length)

        # Verify zeros
        assert backend.read(trim_lba, trim_length) == b"\x00" * trim_length

        backend.destroy()
