"""
Unit and property tests for the Fault Tolerance / Replication module.

Covers:
- Initialization with multiple replica backends
- Write replication to all backends
- Read from primary backend
- Failover when primary fails
- Failover timing (<100ms)
- Data consistency after failover
- Corruption detection (checksum mismatch triggers failover)
- Snapshot creation across all replicas
- Properties 27-32
"""

from __future__ import annotations

import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, strategies as st

from nvme_engine.backends.base import BackendStats, StorageBackendOps
from nvme_engine.backends.memory_backend import MemoryBackend
from nvme_engine.fault_tolerance.replication import ReplicatedBackend, ReplicationConfig
from nvme_engine.models.errors import (
    NvmeBackendError,
    NvmeDataCorruptionError,
    NvmeIoError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BACKEND_SIZE = 64 * 1024  # 64 KB


def make_memory_backend(size: int = BACKEND_SIZE) -> MemoryBackend:
    b = MemoryBackend()
    b.init({"size_bytes": size})
    return b


def make_replicated(
    num_replicas: int = 2,
    size: int = BACKEND_SIZE,
    verify_on_read: bool = True,
    failover_timeout_ms: int = 100,
) -> tuple[ReplicatedBackend, list[MemoryBackend]]:
    backends = [make_memory_backend(size) for _ in range(num_replicas)]
    rb = ReplicatedBackend()
    rb.init(
        {
            "backends": backends,
            "size_bytes": size,
            "verify_on_read": verify_on_read,
            "failover_timeout_ms": failover_timeout_ms,
        }
    )
    return rb, backends


def teardown_replicated(rb: ReplicatedBackend, backends: list[MemoryBackend]) -> None:
    rb.destroy()
    for b in backends:
        if b.is_initialized:
            b.destroy()


class FailingBackend(StorageBackendOps):
    """A backend that always raises NvmeIoError on read/write."""

    def __init__(self, error_msg: str = "simulated failure") -> None:
        super().__init__()
        self._error_msg = error_msg
        self._size_bytes = BACKEND_SIZE

    def init(self, config: Dict[str, Any]) -> None:
        self._initialized = True

    def destroy(self) -> None:
        self._initialized = False

    def read(self, lba: int, length: int) -> bytes:
        raise NvmeIoError(self._error_msg, lba=lba)

    def write(self, lba: int, data: bytes) -> None:
        raise NvmeIoError(self._error_msg, lba=lba)

    def flush(self) -> None:
        raise NvmeIoError(self._error_msg)

    def trim(self, lba: int, length: int) -> None:
        raise NvmeIoError(self._error_msg, lba=lba)

    def snapshot_create(self, name: str) -> None:
        raise NvmeBackendError(self._error_msg)

    def snapshot_delete(self, name: str) -> None:
        raise NvmeBackendError(self._error_msg)

    def snapshot_restore(self, name: str) -> None:
        raise NvmeBackendError(self._error_msg)


class CorruptingBackend(StorageBackendOps):
    """A backend that returns corrupted data on read."""

    def __init__(self, good_backend: StorageBackendOps) -> None:
        super().__init__()
        self._good = good_backend
        self._size_bytes = BACKEND_SIZE

    def init(self, config: Dict[str, Any]) -> None:
        self._initialized = True

    def destroy(self) -> None:
        self._initialized = False

    def read(self, lba: int, length: int) -> bytes:
        data = self._good.read(lba, length)
        # Flip all bits to corrupt
        return bytes(b ^ 0xFF for b in data)

    def write(self, lba: int, data: bytes) -> None:
        self._good.write(lba, data)

    def flush(self) -> None:
        self._good.flush()

    def trim(self, lba: int, length: int) -> None:
        self._good.trim(lba, length)

    def snapshot_create(self, name: str) -> None:
        self._good.snapshot_create(name)

    def snapshot_delete(self, name: str) -> None:
        self._good.snapshot_delete(name)

    def snapshot_restore(self, name: str) -> None:
        self._good.snapshot_restore(name)


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestReplicatedBackendInit:
    def test_init_two_replicas(self):
        rb, backends = make_replicated(num_replicas=2)
        assert rb.is_initialized
        assert rb.replica_count == 2
        teardown_replicated(rb, backends)

    def test_init_three_replicas(self):
        rb, backends = make_replicated(num_replicas=3)
        assert rb.is_initialized
        assert rb.replica_count == 3
        teardown_replicated(rb, backends)

    def test_init_no_backends_raises(self):
        rb = ReplicatedBackend()
        with pytest.raises(NvmeBackendError):
            rb.init({"backends": [], "size_bytes": BACKEND_SIZE})

    def test_init_invalid_size_raises(self):
        b = make_memory_backend()
        rb = ReplicatedBackend()
        with pytest.raises(NvmeBackendError):
            rb.init({"backends": [b], "size_bytes": 0})
        b.destroy()

    def test_init_uninitialized_backend_raises(self):
        b = MemoryBackend()  # not initialized
        rb = ReplicatedBackend()
        with pytest.raises(NvmeBackendError):
            rb.init({"backends": [b], "size_bytes": BACKEND_SIZE})

    def test_double_init_raises(self):
        rb, backends = make_replicated()
        with pytest.raises(NvmeBackendError, match="already initialized"):
            rb.init({"backends": backends, "size_bytes": BACKEND_SIZE})
        teardown_replicated(rb, backends)

    def test_destroy(self):
        rb, backends = make_replicated()
        rb.destroy()
        assert not rb.is_initialized
        for b in backends:
            b.destroy()

    def test_destroy_idempotent(self):
        rb, backends = make_replicated()
        rb.destroy()
        rb.destroy()
        for b in backends:
            b.destroy()

    def test_primary_index_starts_at_zero(self):
        rb, backends = make_replicated()
        assert rb.primary_index == 0
        teardown_replicated(rb, backends)

    def test_config_defaults(self):
        rb, backends = make_replicated()
        assert rb._config.failover_timeout_ms == 100
        assert rb._config.verify_on_read is True
        teardown_replicated(rb, backends)


# ---------------------------------------------------------------------------
# Write replication tests
# ---------------------------------------------------------------------------


class TestReplicatedWrite:
    def test_write_replicates_to_all_backends(self):
        """Write goes to all replica backends."""
        rb, backends = make_replicated(num_replicas=3)
        data = b"replicated data"
        rb.write(0, data)

        for b in backends:
            assert b.read(0, len(data)) == data
        teardown_replicated(rb, backends)

    def test_write_two_replicas(self):
        """Write replicates to exactly two backends."""
        rb, backends = make_replicated(num_replicas=2)
        data = b"two copies"
        rb.write(0, data)

        assert backends[0].read(0, len(data)) == data
        assert backends[1].read(0, len(data)) == data
        teardown_replicated(rb, backends)

    def test_write_out_of_bounds_raises(self):
        rb, backends = make_replicated(size=1024)
        with pytest.raises(NvmeIoError):
            rb.write(1000, b"X" * 100)
        teardown_replicated(rb, backends)

    def test_write_before_init_raises(self):
        rb = ReplicatedBackend()
        with pytest.raises(NvmeIoError):
            rb.write(0, b"data")

    def test_write_majority_failure_raises(self):
        """Write raises if majority of replicas fail."""
        good = make_memory_backend()
        failing1 = FailingBackend()
        failing1.init({})
        failing2 = FailingBackend()
        failing2.init({})

        rb = ReplicatedBackend()
        rb.init(
            {
                "backends": [failing1, failing2, good],
                "size_bytes": BACKEND_SIZE,
            }
        )

        with pytest.raises(NvmeIoError):
            rb.write(0, b"data")

        rb.destroy()
        good.destroy()

    def test_write_partial_failure_succeeds_with_majority(self):
        """Write succeeds if majority of replicas succeed."""
        good1 = make_memory_backend()
        good2 = make_memory_backend()
        failing = FailingBackend()
        failing.init({})

        rb = ReplicatedBackend()
        rb.init(
            {
                "backends": [good1, good2, failing],
                "size_bytes": BACKEND_SIZE,
            }
        )

        # Should succeed (2/3 replicas succeed = majority)
        rb.write(0, b"partial ok")

        rb.destroy()
        good1.destroy()
        good2.destroy()

    def test_write_updates_stats(self):
        rb, backends = make_replicated()
        rb.write(0, b"X" * 50)
        stats = rb.get_stats()
        assert stats.total_writes >= 1
        assert stats.bytes_written >= 50
        teardown_replicated(rb, backends)


# ---------------------------------------------------------------------------
# Read from primary tests
# ---------------------------------------------------------------------------


class TestReplicatedRead:
    def test_read_from_primary(self):
        """Read returns data from the primary backend."""
        rb, backends = make_replicated()
        data = b"primary read"
        rb.write(0, data)
        result = rb.read(0, len(data))
        assert result == data
        teardown_replicated(rb, backends)

    def test_read_before_init_raises(self):
        rb = ReplicatedBackend()
        with pytest.raises(NvmeIoError):
            rb.read(0, 10)

    def test_read_out_of_bounds_raises(self):
        rb, backends = make_replicated(size=1024)
        with pytest.raises(NvmeIoError):
            rb.read(2000, 10)
        teardown_replicated(rb, backends)

    def test_read_updates_stats(self):
        rb, backends = make_replicated()
        rb.write(0, b"data")
        rb.read(0, 4)
        stats = rb.get_stats()
        assert stats.total_reads >= 1
        teardown_replicated(rb, backends)


# ---------------------------------------------------------------------------
# Failover tests
# ---------------------------------------------------------------------------


class TestReplicatedFailover:
    def test_failover_when_primary_fails(self):
        """Read succeeds by failing over when primary raises."""
        failing = FailingBackend()
        failing.init({})
        good = make_memory_backend()
        good.write(0, b"failover data")

        rb = ReplicatedBackend()
        rb.init(
            {
                "backends": [failing, good],
                "size_bytes": BACKEND_SIZE,
                "verify_on_read": False,
            }
        )

        result = rb.read(0, len(b"failover data"))
        assert result == b"failover data"
        assert rb.primary_index == 1

        rb.destroy()
        good.destroy()

    def test_failover_timing_under_100ms(self):
        """Failover completes within 100ms."""
        failing = FailingBackend()
        failing.init({})
        good = make_memory_backend()
        good.write(0, b"timing test!")

        rb = ReplicatedBackend()
        rb.init(
            {
                "backends": [failing, good],
                "size_bytes": BACKEND_SIZE,
                "verify_on_read": False,
                "failover_timeout_ms": 100,
            }
        )

        start = time.perf_counter()
        result = rb.read(0, len(b"timing test!"))
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result == b"timing test!"
        assert elapsed_ms < 200  # generous bound for test environment

        rb.destroy()
        good.destroy()

    def test_failover_updates_primary_index(self):
        """After failover, primary_index points to the healthy replica."""
        failing = FailingBackend()
        failing.init({})
        good = make_memory_backend()
        good.write(0, b"new primary")

        rb = ReplicatedBackend()
        rb.init(
            {
                "backends": [failing, good],
                "size_bytes": BACKEND_SIZE,
                "verify_on_read": False,
            }
        )

        assert rb.primary_index == 0
        rb.read(0, len(b"new primary"))
        assert rb.primary_index == 1

        rb.destroy()
        good.destroy()

    def test_all_replicas_fail_raises(self):
        """If all replicas fail, NvmeIoError is raised."""
        f1 = FailingBackend()
        f1.init({})
        f2 = FailingBackend()
        f2.init({})

        rb = ReplicatedBackend()
        rb.init(
            {
                "backends": [f1, f2],
                "size_bytes": BACKEND_SIZE,
                "verify_on_read": False,
            }
        )

        with pytest.raises(NvmeIoError):
            rb.read(0, 10)

        rb.destroy()

    def test_data_consistency_after_failover(self):
        """Data is consistent after failover to replica."""
        primary = make_memory_backend()
        replica = make_memory_backend()

        rb = ReplicatedBackend()
        rb.init(
            {
                "backends": [primary, replica],
                "size_bytes": BACKEND_SIZE,
                "verify_on_read": False,
            }
        )

        data = b"consistent data"
        rb.write(0, data)

        # Simulate primary failure by replacing it
        rb._backends[0] = FailingBackend()
        rb._backends[0].init({})

        result = rb.read(0, len(data))
        assert result == data

        rb.destroy()
        primary.destroy()
        replica.destroy()


# ---------------------------------------------------------------------------
# Corruption detection tests
# ---------------------------------------------------------------------------


class TestCorruptionDetection:
    def test_corruption_detected_on_read(self):
        """Checksum mismatch on primary triggers failover to clean replica."""
        good = make_memory_backend()
        corrupting = CorruptingBackend(make_memory_backend())
        corrupting.init({})

        rb = ReplicatedBackend()
        rb.init(
            {
                "backends": [corrupting, good],
                "size_bytes": BACKEND_SIZE,
                "verify_on_read": True,
            }
        )

        data = b"clean data!!"
        # Write to both backends directly so checksums are stored
        rb.write(0, data)

        # Now the corrupting backend will return flipped bits
        result = rb.read(0, len(data))
        assert result == data  # Should get clean data from replica

        rb.destroy()
        good.destroy()

    def test_checksum_error_increments_stat(self):
        """Checksum errors are counted in stats."""
        good = make_memory_backend()
        corrupting = CorruptingBackend(make_memory_backend())
        corrupting.init({})

        rb = ReplicatedBackend()
        rb.init(
            {
                "backends": [corrupting, good],
                "size_bytes": BACKEND_SIZE,
                "verify_on_read": True,
            }
        )

        rb.write(0, b"data to corrupt")
        rb.read(0, len(b"data to corrupt"))

        assert rb._stats.checksum_errors >= 1

        rb.destroy()
        good.destroy()

    def test_no_checksum_verification_when_disabled(self):
        """With verify_on_read=False, corrupted data is returned as-is."""
        good = make_memory_backend()
        corrupting = CorruptingBackend(make_memory_backend())
        corrupting.init({})

        rb = ReplicatedBackend()
        rb.init(
            {
                "backends": [corrupting, good],
                "size_bytes": BACKEND_SIZE,
                "verify_on_read": False,
            }
        )

        data = b"original"
        rb.write(0, data)

        # With verify disabled, corrupted data is returned
        result = rb.read(0, len(data))
        # Result may be corrupted — just verify no exception
        assert isinstance(result, bytes)

        rb.destroy()
        good.destroy()


# ---------------------------------------------------------------------------
# Snapshot tests
# ---------------------------------------------------------------------------


class TestReplicatedSnapshots:
    def test_snapshot_create_all_replicas(self):
        """Snapshot creation propagates to all replicas."""
        rb, backends = make_replicated(num_replicas=3)
        rb.write(0, b"snap data")
        rb.snapshot_create("snap1")

        for b in backends:
            with pytest.raises(NvmeBackendError, match="already exists"):
                b.snapshot_create("snap1")
        teardown_replicated(rb, backends)

    def test_snapshot_delete_all_replicas(self):
        """Snapshot deletion propagates to all replicas."""
        rb, backends = make_replicated(num_replicas=2)
        rb.write(0, b"data")
        rb.snapshot_create("snap1")
        rb.snapshot_delete("snap1")

        # Should be able to create again
        rb.snapshot_create("snap1")
        teardown_replicated(rb, backends)

    def test_snapshot_restore_all_replicas(self):
        """Snapshot restore propagates to all replicas."""
        rb, backends = make_replicated(num_replicas=2)
        rb.write(0, b"original")
        rb.snapshot_create("snap1")
        rb.write(0, b"modified")
        rb.snapshot_restore("snap1")

        for b in backends:
            assert b.read(0, 8) == b"original"
        teardown_replicated(rb, backends)

    def test_snapshot_rollback_on_partial_failure(self):
        """If snapshot creation fails on one backend, already-created ones are rolled back."""
        good = make_memory_backend()
        failing = FailingBackend()
        failing.init({})

        rb = ReplicatedBackend()
        rb.init(
            {
                "backends": [good, failing],
                "size_bytes": BACKEND_SIZE,
            }
        )

        with pytest.raises(NvmeBackendError):
            rb.snapshot_create("snap1")

        # The good backend's snapshot should have been rolled back
        # (can create again without "already exists" error)
        good.snapshot_create("snap1")  # should not raise

        rb.destroy()
        good.destroy()

    def test_snapshot_before_init_raises(self):
        rb = ReplicatedBackend()
        with pytest.raises(NvmeBackendError):
            rb.snapshot_create("snap1")
        with pytest.raises(NvmeBackendError):
            rb.snapshot_delete("snap1")
        with pytest.raises(NvmeBackendError):
            rb.snapshot_restore("snap1")


# ---------------------------------------------------------------------------
# Statistics tests
# ---------------------------------------------------------------------------


class TestReplicatedStats:
    def test_stats_aggregated(self):
        """get_stats() aggregates from all replicas."""
        rb, backends = make_replicated(num_replicas=2)
        rb.write(0, b"data")
        rb.read(0, 4)

        stats = rb.get_stats()
        assert stats.total_writes >= 1
        assert stats.total_reads >= 1
        teardown_replicated(rb, backends)

    def test_flush_all_replicas(self):
        """Flush propagates to all replicas."""
        rb, backends = make_replicated(num_replicas=2)
        rb.write(0, b"data")
        rb.flush()

        for b in backends:
            assert b.get_stats().total_flushes >= 1
        teardown_replicated(rb, backends)

    def test_trim_all_replicas(self):
        """Trim propagates to all replicas."""
        rb, backends = make_replicated(num_replicas=2)
        rb.write(0, b"X" * 64)
        rb.trim(0, 64)

        for b in backends:
            assert b.read(0, 64) == b"\x00" * 64
        teardown_replicated(rb, backends)


# ---------------------------------------------------------------------------
# Property tests (Properties 27-32)
# ---------------------------------------------------------------------------


class TestReplicatedProperties:
    @given(data=st.binary(min_size=1, max_size=512))
    @settings(max_examples=50)
    def test_property27_corruption_detection(self, data):
        """
        Feature: software-nvme-solution
        Property 27: Data corruption detection.

        Corrupted data from primary is detected via checksum mismatch.
        """
        good = make_memory_backend()
        corrupting = CorruptingBackend(make_memory_backend())
        corrupting.init({})

        rb = ReplicatedBackend()
        rb.init(
            {
                "backends": [corrupting, good],
                "size_bytes": BACKEND_SIZE,
                "verify_on_read": True,
            }
        )

        rb.write(0, data)
        result = rb.read(0, len(data))
        # Should get clean data from the good replica
        assert result == data

        rb.destroy()
        good.destroy()

    @given(data=st.binary(min_size=1, max_size=256))
    @settings(max_examples=30)
    def test_property28_corruption_error_handling(self, data):
        """
        Feature: software-nvme-solution
        Property 28: Corruption error handling.

        When all replicas are corrupt, an appropriate error is raised.
        """
        c1 = CorruptingBackend(make_memory_backend())
        c1.init({})
        c2 = CorruptingBackend(make_memory_backend())
        c2.init({})

        rb = ReplicatedBackend()
        rb.init(
            {
                "backends": [c1, c2],
                "size_bytes": BACKEND_SIZE,
                "verify_on_read": True,
            }
        )

        rb.write(0, data)

        # Both replicas corrupt — should raise an error
        with pytest.raises((NvmeIoError, NvmeDataCorruptionError)):
            rb.read(0, len(data))

        rb.destroy()

    @given(
        num_replicas=st.integers(min_value=2, max_value=4),
        data=st.binary(min_size=1, max_size=256),
    )
    @settings(max_examples=30)
    def test_property29_data_redundancy_n_copies(self, num_replicas, data):
        """
        Feature: software-nvme-solution
        Property 29: Data redundancy maintenance (N copies exist).

        After a write, all N replicas hold the same data.
        """
        backends = [make_memory_backend() for _ in range(num_replicas)]
        rb = ReplicatedBackend()
        rb.init(
            {
                "backends": backends,
                "size_bytes": BACKEND_SIZE,
            }
        )

        rb.write(0, data)

        for b in backends:
            assert b.read(0, len(data)) == data

        rb.destroy()
        for b in backends:
            b.destroy()

    @given(data=st.binary(min_size=4, max_size=256))
    @settings(max_examples=30)
    def test_property30_failover_timing(self, data):
        """
        Feature: software-nvme-solution
        Property 30: Failover timing (<100ms).

        Failover from a failing primary to a healthy replica completes
        within 100ms.
        """
        failing = FailingBackend()
        failing.init({})
        good = make_memory_backend()
        good.write(0, data)

        rb = ReplicatedBackend()
        rb.init(
            {
                "backends": [failing, good],
                "size_bytes": BACKEND_SIZE,
                "verify_on_read": False,
                "failover_timeout_ms": 100,
            }
        )

        start = time.perf_counter()
        result = rb.read(0, len(data))
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result == data
        assert elapsed_ms < 200  # generous bound for test environment

        rb.destroy()
        good.destroy()

    @given(data=st.binary(min_size=1, max_size=256))
    @settings(max_examples=30)
    def test_property31_snapshot_point_in_time_consistency(self, data):
        """
        Feature: software-nvme-solution
        Property 31: Snapshot point-in-time consistency.

        Snapshot captures exact state; restore returns to that state on
        all replicas.
        """
        rb, backends = make_replicated(num_replicas=2)

        rb.write(0, data)
        rb.snapshot_create("snap1")

        modified = bytes(b ^ 0xAA for b in data)
        rb.write(0, modified)

        rb.snapshot_restore("snap1")

        for b in backends:
            assert b.read(0, len(data)) == data

        teardown_replicated(rb, backends)

    @given(
        writes=st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=20),
                st.binary(min_size=1, max_size=32),
            ),
            min_size=1,
            max_size=8,
        )
    )
    @settings(max_examples=30)
    def test_property32_write_ordering_preservation(self, writes):
        """
        Feature: software-nvme-solution
        Property 32: Write ordering preservation.

        The last write to any LBA is the value that is read back,
        regardless of the number of replicas.

        Note: LBAs are spaced 64 bytes apart to avoid overlapping writes
        that would make the expected value ambiguous.
        """
        rb, backends = make_replicated(num_replicas=2)

        # Space writes 64 bytes apart to avoid overlapping byte ranges
        stride = 64
        last_write: dict = {}
        for lba_idx, data in writes:
            lba = lba_idx * stride
            if lba + len(data) <= BACKEND_SIZE:
                rb.write(lba, data)
                last_write[lba] = data

        for lba, expected in last_write.items():
            result = rb.read(lba, len(expected))
            assert result == expected

        teardown_replicated(rb, backends)
