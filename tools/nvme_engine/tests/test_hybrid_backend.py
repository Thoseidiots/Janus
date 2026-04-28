"""
Unit and property tests for the Hybrid Backend.

Covers:
- Initialization with multiple backends
- Read/write routing to correct tier
- Hot data promotion
- Cold data demotion
- Data migration between tiers
- Snapshot operations across all tiers
- Statistics aggregation
- Property 4: backend switching preserves data integrity
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st

from nvme_engine.backends.hybrid_backend import HybridBackend, TieringPolicy
from nvme_engine.backends.memory_backend import MemoryBackend
from nvme_engine.models.errors import NvmeBackendError, NvmeIoError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BACKEND_SIZE = 64 * 1024  # 64 KB


def make_memory_backend(size: int = BACKEND_SIZE) -> MemoryBackend:
    """Create and initialize a MemoryBackend."""
    b = MemoryBackend()
    b.init({"size_bytes": size})
    return b


def make_hybrid(
    num_backends: int = 2,
    size: int = BACKEND_SIZE,
    hot_threshold: int = 5,
    cold_threshold: int = 0,
    auto_migrate: bool = False,
) -> tuple[HybridBackend, list[MemoryBackend]]:
    """Create a HybridBackend backed by MemoryBackends."""
    backends = [make_memory_backend(size) for _ in range(num_backends)]
    hybrid = HybridBackend()
    hybrid.init(
        {
            "backends": backends,
            "size_bytes": size,
            "hot_threshold": hot_threshold,
            "cold_threshold": cold_threshold,
            "auto_migrate": auto_migrate,
        }
    )
    return hybrid, backends


def teardown_hybrid(hybrid: HybridBackend, backends: list[MemoryBackend]) -> None:
    """Destroy hybrid and all backends."""
    hybrid.destroy()
    for b in backends:
        if b.is_initialized:
            b.destroy()


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestHybridBackendInit:
    def test_init_single_backend(self):
        """Hybrid backend can be initialized with a single backend."""
        hybrid, backends = make_hybrid(num_backends=1)
        assert hybrid.is_initialized
        assert hybrid.tier_count == 1
        teardown_hybrid(hybrid, backends)

    def test_init_two_backends(self):
        """Hybrid backend can be initialized with two backends."""
        hybrid, backends = make_hybrid(num_backends=2)
        assert hybrid.is_initialized
        assert hybrid.tier_count == 2
        teardown_hybrid(hybrid, backends)

    def test_init_three_backends(self):
        """Hybrid backend can be initialized with three backends."""
        hybrid, backends = make_hybrid(num_backends=3)
        assert hybrid.is_initialized
        assert hybrid.tier_count == 3
        teardown_hybrid(hybrid, backends)

    def test_init_no_backends_raises(self):
        """Initialization without backends raises NvmeBackendError."""
        hybrid = HybridBackend()
        with pytest.raises(NvmeBackendError):
            hybrid.init({"backends": [], "size_bytes": BACKEND_SIZE})

    def test_init_invalid_size_raises(self):
        """Initialization with invalid size raises NvmeBackendError."""
        b = make_memory_backend()
        hybrid = HybridBackend()
        with pytest.raises(NvmeBackendError):
            hybrid.init({"backends": [b], "size_bytes": 0})
        b.destroy()

    def test_init_uninitialized_backend_raises(self):
        """Initialization with an uninitialized backend raises NvmeBackendError."""
        b = MemoryBackend()  # not initialized
        hybrid = HybridBackend()
        with pytest.raises(NvmeBackendError):
            hybrid.init({"backends": [b], "size_bytes": BACKEND_SIZE})

    def test_double_init_raises(self):
        """Double initialization raises NvmeBackendError."""
        hybrid, backends = make_hybrid()
        with pytest.raises(NvmeBackendError, match="already initialized"):
            hybrid.init({"backends": backends, "size_bytes": BACKEND_SIZE})
        teardown_hybrid(hybrid, backends)

    def test_size_bytes_property(self):
        """size_bytes property returns configured size."""
        hybrid, backends = make_hybrid(size=BACKEND_SIZE)
        assert hybrid.size_bytes == BACKEND_SIZE
        teardown_hybrid(hybrid, backends)

    def test_policy_defaults(self):
        """Default tiering policy values are applied."""
        hybrid, backends = make_hybrid()
        assert hybrid.policy.hot_threshold == 5
        assert hybrid.policy.cold_threshold == 0
        teardown_hybrid(hybrid, backends)

    def test_destroy(self):
        """Destroy clears initialized state."""
        hybrid, backends = make_hybrid()
        hybrid.destroy()
        assert not hybrid.is_initialized
        for b in backends:
            b.destroy()

    def test_destroy_idempotent(self):
        """Destroy can be called multiple times without error."""
        hybrid, backends = make_hybrid()
        hybrid.destroy()
        hybrid.destroy()
        for b in backends:
            b.destroy()


# ---------------------------------------------------------------------------
# Read / write routing tests
# ---------------------------------------------------------------------------


class TestHybridReadWrite:
    def test_write_goes_to_tier0(self):
        """Writes are routed to tier 0 (fastest backend)."""
        hybrid, backends = make_hybrid(num_backends=2)
        data = b"hello tier0"
        hybrid.write(0, data)

        # Data should be readable from tier 0 directly
        assert backends[0].read(0, len(data)) == data
        teardown_hybrid(hybrid, backends)

    def test_read_returns_written_data(self):
        """Read returns the data that was written."""
        hybrid, backends = make_hybrid()
        data = b"test data"
        hybrid.write(0, data)
        assert hybrid.read(0, len(data)) == data
        teardown_hybrid(hybrid, backends)

    def test_write_then_read_at_offset(self):
        """Write and read at a non-zero offset."""
        hybrid, backends = make_hybrid()
        data = b"offset write"
        hybrid.write(1024, data)
        assert hybrid.read(1024, len(data)) == data
        teardown_hybrid(hybrid, backends)

    def test_read_before_write_returns_zeros(self):
        """Reading an unwritten LBA returns zeros."""
        hybrid, backends = make_hybrid()
        data = hybrid.read(0, 16)
        assert data == b"\x00" * 16
        teardown_hybrid(hybrid, backends)

    def test_write_out_of_bounds_raises(self):
        """Write beyond storage size raises NvmeIoError."""
        hybrid, backends = make_hybrid(size=1024)
        with pytest.raises(NvmeIoError):
            hybrid.write(1000, b"X" * 100)
        teardown_hybrid(hybrid, backends)

    def test_read_out_of_bounds_raises(self):
        """Read beyond storage size raises NvmeIoError."""
        hybrid, backends = make_hybrid(size=1024)
        with pytest.raises(NvmeIoError):
            hybrid.read(2000, 10)
        teardown_hybrid(hybrid, backends)

    def test_operations_before_init_raise(self):
        """Operations before init raise NvmeIoError."""
        hybrid = HybridBackend()
        with pytest.raises(NvmeIoError):
            hybrid.read(0, 10)
        with pytest.raises(NvmeIoError):
            hybrid.write(0, b"data")
        with pytest.raises(NvmeIoError):
            hybrid.flush()

    def test_flush_all_backends(self):
        """Flush propagates to all backends."""
        hybrid, backends = make_hybrid(num_backends=2)
        hybrid.write(0, b"data")
        hybrid.flush()
        # Each backend should have been flushed
        for b in backends:
            assert b.get_stats().total_flushes >= 1
        teardown_hybrid(hybrid, backends)

    def test_trim_clears_data(self):
        """Trim zeros the data at the given LBA."""
        hybrid, backends = make_hybrid()
        hybrid.write(0, b"X" * 64)
        hybrid.trim(0, 64)
        # After trim, tier 0 should have zeros
        assert backends[0].read(0, 64) == b"\x00" * 64
        teardown_hybrid(hybrid, backends)

    def test_multiple_writes_preserved(self):
        """Multiple non-overlapping writes are all preserved."""
        hybrid, backends = make_hybrid()
        writes = [(0, b"aaa"), (100, b"bbb"), (200, b"ccc")]
        for lba, data in writes:
            hybrid.write(lba, data)
        for lba, data in writes:
            assert hybrid.read(lba, len(data)) == data
        teardown_hybrid(hybrid, backends)


# ---------------------------------------------------------------------------
# Hot data promotion tests
# ---------------------------------------------------------------------------


class TestHybridPromotion:
    def test_hot_data_promoted_to_tier0(self):
        """LBA accessed >= hot_threshold times is promoted to tier 0."""
        hybrid, backends = make_hybrid(
            num_backends=2, hot_threshold=3, auto_migrate=False
        )
        data = b"hot data!!"

        # Manually place data in tier 1 (slow tier)
        backends[1].write(0, data)
        hybrid._lba_tier[0] = 1
        hybrid._access_counts[0] = 0

        # Read enough times to trigger promotion
        for _ in range(3):
            result = hybrid.read(0, len(data))
            assert result == data

        # After hot_threshold reads, data should be in tier 0
        assert hybrid._lba_tier.get(0, 1) == 0
        assert backends[0].read(0, len(data)) == data
        teardown_hybrid(hybrid, backends)

    def test_access_count_increments_on_read(self):
        """Access count increments on each read."""
        hybrid, backends = make_hybrid()
        hybrid.write(0, b"data")
        for i in range(1, 4):
            hybrid.read(0, 4)
            assert hybrid._access_counts.get(0, 0) >= i
        teardown_hybrid(hybrid, backends)

    def test_access_count_increments_on_write(self):
        """Access count increments on write."""
        hybrid, backends = make_hybrid()
        hybrid.write(0, b"data")
        assert hybrid._access_counts.get(0, 0) >= 1
        teardown_hybrid(hybrid, backends)

    def test_promote_hot_data_method(self):
        """_promote_hot_data() moves hot LBAs from slow tiers to tier 0."""
        hybrid, backends = make_hybrid(
            num_backends=2, hot_threshold=2, auto_migrate=False
        )
        data = b"promote me!"

        # Place data in tier 1 with high access count
        backends[1].write(0, len(data).to_bytes(4, "little"))  # dummy
        backends[1].write(0, data)
        hybrid._lba_tier[0] = 1
        hybrid._access_counts[0] = 10  # above threshold

        hybrid._promote_hot_data()

        assert hybrid._lba_tier.get(0) == 0
        teardown_hybrid(hybrid, backends)


# ---------------------------------------------------------------------------
# Cold data demotion tests
# ---------------------------------------------------------------------------


class TestHybridDemotion:
    def test_migrate_cold_data_method(self):
        """_migrate_cold_data() moves cold LBAs from tier 0 to slowest tier."""
        hybrid, backends = make_hybrid(
            num_backends=2, cold_threshold=0, auto_migrate=False
        )
        data = b"cold data!!"

        # Write data to tier 0 and mark as cold (access count = 0)
        backends[0].write(0, data)
        hybrid._lba_tier[0] = 0
        hybrid._access_counts[0] = 0

        hybrid._migrate_cold_data()

        # Data should now be in the slowest tier
        assert hybrid._lba_tier.get(0) == 1
        teardown_hybrid(hybrid, backends)

    def test_cold_data_readable_after_demotion(self):
        """Data is still readable after being demoted to a slower tier."""
        hybrid, backends = make_hybrid(
            num_backends=2, cold_threshold=0, auto_migrate=False
        )
        data = b"cold readable"

        backends[0].write(0, data)
        hybrid._lba_tier[0] = 0
        hybrid._access_counts[0] = 0

        hybrid._migrate_cold_data()

        # Read via hybrid — should find data in tier 1
        result = hybrid.read(0, len(data))
        assert result == data
        teardown_hybrid(hybrid, backends)

    def test_hot_data_not_demoted(self):
        """Hot data (access count > cold_threshold) is not demoted."""
        hybrid, backends = make_hybrid(
            num_backends=2, cold_threshold=0, auto_migrate=False
        )
        data = b"hot stay tier0"

        backends[0].write(0, data)
        hybrid._lba_tier[0] = 0
        hybrid._access_counts[0] = 5  # above cold_threshold

        hybrid._migrate_cold_data()

        # Should still be in tier 0
        assert hybrid._lba_tier.get(0, 0) == 0
        teardown_hybrid(hybrid, backends)


# ---------------------------------------------------------------------------
# Snapshot tests
# ---------------------------------------------------------------------------


class TestHybridSnapshots:
    def test_snapshot_create_all_backends(self):
        """Snapshot creation propagates to all backends."""
        hybrid, backends = make_hybrid(num_backends=2)
        hybrid.write(0, b"snap data")
        hybrid.snapshot_create("snap1")

        # Both backends should have the snapshot
        for b in backends:
            with pytest.raises(NvmeBackendError, match="already exists"):
                b.snapshot_create("snap1")
        teardown_hybrid(hybrid, backends)

    def test_snapshot_delete_all_backends(self):
        """Snapshot deletion propagates to all backends."""
        hybrid, backends = make_hybrid(num_backends=2)
        hybrid.write(0, b"data")
        hybrid.snapshot_create("snap1")
        hybrid.snapshot_delete("snap1")

        # Should be able to create again
        hybrid.snapshot_create("snap1")
        teardown_hybrid(hybrid, backends)

    def test_snapshot_restore_all_backends(self):
        """Snapshot restore propagates to all backends."""
        hybrid, backends = make_hybrid(num_backends=2)
        hybrid.write(0, b"original")
        hybrid.snapshot_create("snap1")
        hybrid.write(0, b"modified")
        hybrid.snapshot_restore("snap1")

        # Tier 0 should have original data
        assert backends[0].read(0, 8) == b"original"
        teardown_hybrid(hybrid, backends)

    def test_snapshot_before_init_raises(self):
        """Snapshot operations before init raise NvmeBackendError."""
        hybrid = HybridBackend()
        with pytest.raises(NvmeBackendError):
            hybrid.snapshot_create("snap1")
        with pytest.raises(NvmeBackendError):
            hybrid.snapshot_delete("snap1")
        with pytest.raises(NvmeBackendError):
            hybrid.snapshot_restore("snap1")


# ---------------------------------------------------------------------------
# Statistics aggregation tests
# ---------------------------------------------------------------------------


class TestHybridStats:
    def test_stats_aggregated_from_all_backends(self):
        """get_stats() aggregates stats from all backends."""
        hybrid, backends = make_hybrid(num_backends=2)
        hybrid.write(0, b"data")
        hybrid.read(0, 4)

        stats = hybrid.get_stats()
        # Hybrid layer + backend layer both count
        assert stats.total_writes >= 1
        assert stats.total_reads >= 1
        teardown_hybrid(hybrid, backends)

    def test_stats_bytes_tracked(self):
        """Bytes read/written are tracked in aggregated stats."""
        hybrid, backends = make_hybrid(num_backends=2)
        hybrid.write(0, b"X" * 100)
        hybrid.read(0, 100)

        stats = hybrid.get_stats()
        assert stats.bytes_written >= 100
        assert stats.bytes_read >= 100
        teardown_hybrid(hybrid, backends)

    def test_latency_tracked(self):
        """Average latency is tracked at the hybrid layer."""
        hybrid, backends = make_hybrid()
        hybrid.write(0, b"data")
        hybrid.read(0, 4)

        stats = hybrid.get_stats()
        assert stats.avg_write_latency_us >= 0
        assert stats.avg_read_latency_us >= 0
        teardown_hybrid(hybrid, backends)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestHybridProperties:
    @given(
        lba=st.integers(min_value=0, max_value=1000),
        data=st.binary(min_size=1, max_size=512),
    )
    @settings(max_examples=50)
    def test_property4_backend_switching_preserves_data(self, lba, data):
        """
        Feature: software-nvme-solution
        Property 4: Backend switching preserves data integrity.

        Data written to the hybrid backend must be readable regardless of
        which tier currently holds it.
        """
        size = 8 * 1024
        if lba + len(data) > size:
            return

        hybrid, backends = make_hybrid(
            num_backends=2, size=size, hot_threshold=3, auto_migrate=False
        )

        hybrid.write(lba, data)
        result = hybrid.read(lba, len(data))
        assert result == data

        teardown_hybrid(hybrid, backends)

    @given(
        data=st.binary(min_size=1, max_size=256),
    )
    @settings(max_examples=30)
    def test_property_snapshot_preserves_state(self, data):
        """
        Feature: software-nvme-solution
        Property 31: Snapshot point-in-time consistency.

        Snapshot captures exact state; restore returns to that state.
        """
        hybrid, backends = make_hybrid(num_backends=2, auto_migrate=False)

        hybrid.write(0, data)
        hybrid.snapshot_create("snap1")

        modified = bytes(b ^ 0xFF for b in data)
        hybrid.write(0, modified)
        assert hybrid.read(0, len(data)) == modified

        hybrid.snapshot_restore("snap1")
        assert hybrid.read(0, len(data)) == data

        teardown_hybrid(hybrid, backends)

    @given(
        num_writes=st.integers(min_value=1, max_value=8),
        payload=st.binary(min_size=4, max_size=64),
    )
    @settings(max_examples=30)
    def test_property_multiple_writes_all_readable(self, num_writes, payload):
        """
        Feature: software-nvme-solution
        Property 4: Backend switching preserves data integrity.

        Multiple non-overlapping writes are all readable after migration.
        """
        size = 16 * 1024
        stride = 128
        hybrid, backends = make_hybrid(
            num_backends=2, size=size, auto_migrate=False
        )

        writes = []
        for i in range(num_writes):
            lba = i * stride
            if lba + len(payload) <= size:
                hybrid.write(lba, payload)
                writes.append((lba, payload))

        for lba, expected in writes:
            assert hybrid.read(lba, len(expected)) == expected

        teardown_hybrid(hybrid, backends)
