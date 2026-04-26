"""
Unit and property tests for the File Backend.
"""

import os
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from nvme_engine.backends.file_backend import FileBackend
from nvme_engine.models.errors import NvmeBackendError, NvmeIoError


class TestFileBackendBasic:
    """Basic unit tests for File Backend."""

    def test_initialization_sparse(self):
        """Test file backend initialization with sparse file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileBackend()
            config = {
                "path": os.path.join(tmpdir, "test.nvme"),
                "size_bytes": 1024 * 1024,
                "sparse": True,
            }
            backend.init(config)

            assert backend.is_initialized
            assert backend.size_bytes == 1024 * 1024
            assert backend.is_sparse
            assert backend.file_path.exists()

            backend.destroy()

    def test_initialization_regular(self):
        """Test file backend initialization with regular file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileBackend()
            config = {
                "path": os.path.join(tmpdir, "test.nvme"),
                "size_bytes": 4096,  # Small size for regular file
                "sparse": False,
            }
            backend.init(config)

            assert backend.is_initialized
            assert not backend.is_sparse

            backend.destroy()

    def test_initialization_creates_parent_directory(self):
        """Test that initialization creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileBackend()
            nested_path = os.path.join(tmpdir, "subdir1", "subdir2", "test.nvme")
            config = {
                "path": nested_path,
                "size_bytes": 4096,
            }
            backend.init(config)

            assert backend.is_initialized
            assert Path(nested_path).exists()

            backend.destroy()

    def test_initialization_invalid_path(self):
        """Test initialization with empty path fails."""
        backend = FileBackend()

        with pytest.raises(NvmeBackendError, match="path must be specified"):
            backend.init({"size_bytes": 1024})

    def test_initialization_invalid_size(self):
        """Test initialization with invalid size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileBackend()

            with pytest.raises(NvmeBackendError, match="size_bytes must be positive"):
                backend.init({
                    "path": os.path.join(tmpdir, "test.nvme"),
                    "size_bytes": 0,
                })

    def test_double_initialization(self):
        """Test that double initialization fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileBackend()
            config = {
                "path": os.path.join(tmpdir, "test.nvme"),
                "size_bytes": 4096,
            }
            backend.init(config)

            with pytest.raises(NvmeBackendError, match="already initialized"):
                backend.init(config)

            backend.destroy()

    def test_read_write_basic(self):
        """Test basic read/write operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileBackend()
            backend.init({
                "path": os.path.join(tmpdir, "test.nvme"),
                "size_bytes": 4096,
            })

            # Write data
            data = b"Hello, File Backend!"
            backend.write(0, data)

            # Read data back
            read_data = backend.read(0, len(data))
            assert read_data == data

            backend.destroy()

    def test_read_write_persistence(self):
        """Test that data persists across backend instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.nvme")
            data = b"Persistent data"

            # Write data
            backend1 = FileBackend()
            backend1.init({"path": file_path, "size_bytes": 4096})
            backend1.write(0, data)
            backend1.flush()
            backend1.destroy()

            # Read data with new backend instance
            backend2 = FileBackend()
            backend2.init({"path": file_path, "size_bytes": 4096})
            read_data = backend2.read(0, len(data))
            assert read_data == data

            backend2.destroy()

    def test_read_unwritten_sparse_region(self):
        """Test reading unwritten sparse region returns zeros."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileBackend()
            backend.init({
                "path": os.path.join(tmpdir, "test.nvme"),
                "size_bytes": 4096,
                "sparse": True,
            })

            # Read without writing first
            data = backend.read(1000, 100)
            assert data == b"\x00" * 100

            backend.destroy()

    def test_write_out_of_bounds(self):
        """Test write beyond storage size fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileBackend()
            backend.init({
                "path": os.path.join(tmpdir, "test.nvme"),
                "size_bytes": 1024,
            })

            with pytest.raises(NvmeIoError, match="exceed storage size"):
                backend.write(1000, b"X" * 100)

            backend.destroy()

    def test_flush(self):
        """Test flush operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileBackend()
            backend.init({
                "path": os.path.join(tmpdir, "test.nvme"),
                "size_bytes": 4096,
            })

            backend.write(0, b"data")
            backend.flush()

            assert backend.get_stats().total_flushes == 1

            backend.destroy()

    def test_trim(self):
        """Test trim operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileBackend()
            backend.init({
                "path": os.path.join(tmpdir, "test.nvme"),
                "size_bytes": 4096,
            })

            # Write data
            backend.write(0, b"X" * 100)
            assert backend.read(0, 100) == b"X" * 100

            # Trim the data
            backend.trim(0, 100)
            assert backend.read(0, 100) == b"\x00" * 100
            assert backend.get_stats().total_trims == 1

            backend.destroy()

    def test_snapshot_create(self):
        """Test snapshot creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileBackend()
            backend.init({
                "path": os.path.join(tmpdir, "test.nvme"),
                "size_bytes": 4096,
            })

            backend.write(0, b"original data")
            backend.snapshot_create("snap1")

            # Verify snapshot exists (indirectly by trying to create duplicate)
            with pytest.raises(NvmeBackendError, match="already exists"):
                backend.snapshot_create("snap1")

            backend.destroy()

    def test_snapshot_delete(self):
        """Test snapshot deletion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileBackend()
            backend.init({
                "path": os.path.join(tmpdir, "test.nvme"),
                "size_bytes": 4096,
            })

            backend.write(0, b"data")
            backend.snapshot_create("snap1")
            backend.snapshot_delete("snap1")

            # Should be able to create again after deletion
            backend.snapshot_create("snap1")

            backend.destroy()

    def test_snapshot_restore(self):
        """Test snapshot restoration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileBackend()
            backend.init({
                "path": os.path.join(tmpdir, "test.nvme"),
                "size_bytes": 4096,
            })

            # Write original data and snapshot
            backend.write(0, b"original")
            backend.snapshot_create("snap1")

            # Modify data
            backend.write(0, b"modified")
            assert backend.read(0, 8) == b"modified"

            # Restore snapshot
            backend.snapshot_restore("snap1")
            assert backend.read(0, 8) == b"original"

            backend.destroy()

    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileBackend()
            backend.init({
                "path": os.path.join(tmpdir, "test.nvme"),
                "size_bytes": 4096,
            })

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

            backend.destroy()

    def test_context_manager(self):
        """Test backend as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileBackend()
            backend.init({
                "path": os.path.join(tmpdir, "test.nvme"),
                "size_bytes": 4096,
            })

            with backend as b:
                b.write(0, b"data")
                assert b.read(0, 4) == b"data"

            assert not backend.is_initialized


class TestFileBackendProperties:
    """Property-based tests for File Backend."""

    @given(
        lba=st.integers(min_value=0, max_value=1000),
        data=st.binary(min_size=1, max_size=1024),
    )
    def test_property_read_after_write_returns_same_data(self, lba, data):
        """
        Feature: software-nvme-solution, Property 4: Backend Switching Preserves Data Integrity

        Test that reading after writing returns the same data.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileBackend()
            backend.init({
                "path": os.path.join(tmpdir, "test.nvme"),
                "size_bytes": 10 * 1024,
            })

            # Ensure write fits in storage
            if lba + len(data) > backend.size_bytes:
                backend.destroy()
                return

            backend.write(lba, data)
            read_data = backend.read(lba, len(data))

            assert read_data == data

            backend.destroy()

    @given(
        data=st.binary(min_size=100, max_size=1024),
    )
    def test_property_snapshot_preserves_state(self, data):
        """
        Feature: software-nvme-solution, Property 31: Snapshot Point-in-Time Consistency

        Test that snapshots preserve exact state at creation time.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileBackend()
            backend.init({
                "path": os.path.join(tmpdir, "test.nvme"),
                "size_bytes": 10 * 1024,
            })

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

    def test_property_sparse_file_minimizes_disk_usage(self):
        """
        Feature: software-nvme-solution, Property 5: Sparse File Allocation Minimizes Space

        Test that sparse files use less disk space than their logical size.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileBackend()
            logical_size = 10 * 1024 * 1024  # 10MB logical size

            backend.init({
                "path": os.path.join(tmpdir, "test.nvme"),
                "size_bytes": logical_size,
                "sparse": True,
            })

            # Write only 4KB of data
            backend.write(0, b"X" * 4096)
            backend.flush()

            # Get actual disk usage
            actual_usage = backend.get_actual_disk_usage()

            # On systems that support sparse files, actual usage should be much less
            # than logical size. We allow some overhead for metadata.
            # Note: On Windows, this test may not work as expected
            if hasattr(os.stat(backend.file_path), "st_blocks"):
                # Unix-like system with sparse file support
                assert actual_usage < logical_size / 2, \
                    f"Sparse file should use less space: {actual_usage} < {logical_size / 2}"

            backend.destroy()

    @given(
        num_writes=st.integers(min_value=1, max_value=5),
        data=st.binary(min_size=1, max_size=100),
    )
    def test_property_data_persists_across_instances(self, num_writes, data):
        """
        Feature: software-nvme-solution, Property 4: Backend Switching Preserves Data Integrity

        Test that data persists across backend instances (file persistence).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.nvme")

            # Write data with first instance
            backend1 = FileBackend()
            backend1.init({"path": file_path, "size_bytes": 10 * 1024})

            writes = []
            for i in range(num_writes):
                lba = i * 200  # Non-overlapping
                if lba + len(data) <= backend1.size_bytes:
                    backend1.write(lba, data)
                    writes.append((lba, data))

            backend1.flush()
            backend1.destroy()

            # Read data with second instance
            backend2 = FileBackend()
            backend2.init({"path": file_path, "size_bytes": 10 * 1024})

            for lba, expected_data in writes:
                read_data = backend2.read(lba, len(expected_data))
                assert read_data == expected_data

            backend2.destroy()
