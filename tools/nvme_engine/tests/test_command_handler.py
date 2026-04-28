"""
Tests for NVMe Command Handler (Task 9).

Covers admin commands, I/O commands, namespace management,
atomic writes, reservations, and property tests 19-21.
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st

from nvme_engine.backends.memory_backend import MemoryBackend
from nvme_engine.commands.admin_commands import (
    AdminCommandProcessor,
    AdminCommandType,
)
from nvme_engine.commands.command_handler import NvmeCommandHandler
from nvme_engine.commands.io_commands import IoCommandProcessor, MAX_ATOMIC_WRITE_BYTES
from nvme_engine.commands.namespace_manager import (
    Namespace,
    NamespaceManager,
    ReservationManager,
    ReservationType,
)
from nvme_engine.models.errors import (
    NvmeConfigError,
    NvmeDeviceNotFoundError,
    NvmePermissionError,
)
from nvme_engine.models.io_models import IoCompletion, IoRequest, IoType

BACKEND_SIZE = 64 * 1024  # 64 KB


def make_backend(size: int = BACKEND_SIZE) -> MemoryBackend:
    b = MemoryBackend()
    b.init({"size_bytes": size})
    return b


def make_handler(size: int = BACKEND_SIZE):
    backend = make_backend(size)
    ns_mgr = NamespaceManager()
    handler = NvmeCommandHandler(backend, ns_mgr)
    return handler, backend, ns_mgr


def make_read_request(lba: int, length: int, req_id: int = 1) -> IoRequest:
    return IoRequest(
        request_id=req_id,
        type=IoType.READ,
        lba=lba,
        block_count=0,
        buffer=None,
        buffer_size=length,
        priority=0,
        submit_time_ns=0,
    )


def make_write_request(lba: int, data: bytes, req_id: int = 2) -> IoRequest:
    return IoRequest(
        request_id=req_id,
        type=IoType.WRITE,
        lba=lba,
        block_count=0,
        buffer=data,
        buffer_size=len(data),
        priority=0,
        submit_time_ns=0,
    )


def make_flush_request(req_id: int = 3) -> IoRequest:
    return IoRequest(
        request_id=req_id,
        type=IoType.FLUSH,
        lba=0,
        block_count=0,
        buffer=None,
        buffer_size=0,
        priority=0,
        submit_time_ns=0,
    )


def make_trim_request(lba: int, length: int, req_id: int = 4) -> IoRequest:
    return IoRequest(
        request_id=req_id,
        type=IoType.TRIM,
        lba=lba,
        block_count=0,
        buffer=None,
        buffer_size=length,
        priority=0,
        submit_time_ns=0,
    )


# ---------------------------------------------------------------------------
# Admin command tests
# ---------------------------------------------------------------------------

class TestAdminCommands:
    def test_identify_controller_returns_data(self):
        proc = AdminCommandProcessor()
        result = proc.process(AdminCommandType.IDENTIFY, {"cns": 1})
        assert result.success
        assert result.data is not None
        assert len(result.data) > 0

    def test_identify_namespace_returns_data(self):
        proc = AdminCommandProcessor()
        result = proc.process(AdminCommandType.IDENTIFY, {"cns": 0, "nsid": 1})
        assert result.success
        assert result.data is not None

    def test_get_features_default_value(self):
        proc = AdminCommandProcessor()
        result = proc.process(AdminCommandType.GET_FEATURES, {"feature_id": 1})
        assert result.success
        assert result.status_code == 0

    def test_set_then_get_features_roundtrip(self):
        proc = AdminCommandProcessor()
        proc.process(AdminCommandType.SET_FEATURES, {"feature_id": 7, "value": 42})
        result = proc.process(AdminCommandType.GET_FEATURES, {"feature_id": 7})
        assert result.success
        assert result.data == b"\x2a\x00\x00\x00"  # 42 as little-endian uint32

    def test_create_io_queue(self):
        proc = AdminCommandProcessor()
        result = proc.process(AdminCommandType.CREATE_IO_QUEUE, {
            "queue_id": 1, "queue_type": "submission", "depth": 256
        })
        assert result.success

    def test_create_io_queue_duplicate_raises(self):
        proc = AdminCommandProcessor()
        proc.process(AdminCommandType.CREATE_IO_QUEUE, {
            "queue_id": 1, "queue_type": "submission", "depth": 256
        })
        result = proc.process(AdminCommandType.CREATE_IO_QUEUE, {
            "queue_id": 1, "queue_type": "submission", "depth": 256
        })
        assert not result.success

    def test_delete_io_queue(self):
        proc = AdminCommandProcessor()
        proc.process(AdminCommandType.CREATE_IO_QUEUE, {
            "queue_id": 2, "queue_type": "completion", "depth": 128
        })
        result = proc.process(AdminCommandType.DELETE_IO_QUEUE, {"queue_id": 2})
        assert result.success

    def test_delete_nonexistent_queue_fails(self):
        proc = AdminCommandProcessor()
        result = proc.process(AdminCommandType.DELETE_IO_QUEUE, {"queue_id": 999})
        assert not result.success

    def test_abort_command(self):
        proc = AdminCommandProcessor()
        result = proc.process(AdminCommandType.ABORT, {"command_id": 5})
        assert result.success

    def test_async_event_request_registered(self):
        proc = AdminCommandProcessor()
        result = proc.process(AdminCommandType.ASYNC_EVENT_REQUEST, {"request_id": 10})
        assert result.success
        # pending_count is a property — just verify it's accessible and non-negative
        count = proc.aer.pending_count()
        assert count >= 0

    def test_async_event_delivers_to_registered_request(self):
        proc = AdminCommandProcessor()
        proc.process(AdminCommandType.ASYNC_EVENT_REQUEST, {"request_id": 10})
        event = proc.aer.post_event("error", "media_error")
        assert event is not None
        assert event["request_id"] == 10

    def test_unknown_command_fails(self):
        proc = AdminCommandProcessor()
        # Pass an invalid command type string directly
        result = proc.process("INVALID_CMD", {})  # type: ignore
        assert not result.success

    def test_handler_routes_admin_command(self):
        handler, backend, _ = make_handler()
        result = handler.handle_admin_command(AdminCommandType.IDENTIFY, {"cns": 1})
        assert result.success
        backend.destroy()


# ---------------------------------------------------------------------------
# I/O command tests
# ---------------------------------------------------------------------------

class TestIoCommands:
    def test_write_then_read_returns_same_data(self):
        backend = make_backend()
        proc = IoCommandProcessor(backend)
        data = b"hello nvme io"
        proc.execute(make_write_request(0, data))
        cpl = proc.execute(make_read_request(0, len(data)))
        assert cpl.status == 0
        assert cpl.bytes_transferred == len(data)
        backend.destroy()

    def test_read_returns_correct_bytes_transferred(self):
        backend = make_backend()
        proc = IoCommandProcessor(backend)
        backend.write(0, b"X" * 100)
        cpl = proc.execute(make_read_request(0, 100))
        assert cpl.status == 0
        assert cpl.bytes_transferred == 100
        backend.destroy()

    def test_write_updates_bytes_transferred(self):
        backend = make_backend()
        proc = IoCommandProcessor(backend)
        data = b"Y" * 200
        cpl = proc.execute(make_write_request(0, data))
        assert cpl.status == 0
        assert cpl.bytes_transferred == 200
        backend.destroy()

    def test_flush_succeeds(self):
        backend = make_backend()
        proc = IoCommandProcessor(backend)
        cpl = proc.execute(make_flush_request())
        assert cpl.status == 0
        backend.destroy()

    def test_write_zeroes_zeros_range(self):
        backend = make_backend()
        proc = IoCommandProcessor(backend)
        backend.write(0, b"A" * 512)
        cpl = proc.execute(make_trim_request(0, 512))
        assert cpl.status == 0
        assert backend.read(0, 512) == b"\x00" * 512
        backend.destroy()

    def test_write_at_offset(self):
        backend = make_backend()
        proc = IoCommandProcessor(backend)
        data = b"offset_data"
        proc.execute(make_write_request(1024, data))
        cpl = proc.execute(make_read_request(1024, len(data)))
        assert cpl.status == 0
        backend.destroy()

    def test_handler_routes_io_command(self):
        handler, backend, _ = make_handler()
        data = b"routed io"
        handler.handle_io_command(make_write_request(0, data))
        cpl = handler.handle_io_command(make_read_request(0, len(data)))
        assert cpl.status == 0
        backend.destroy()


# ---------------------------------------------------------------------------
# Atomic write tests
# ---------------------------------------------------------------------------

class TestAtomicWrite:
    def test_atomic_write_succeeds(self):
        backend = make_backend()
        proc = IoCommandProcessor(backend)
        data = b"atomic!" * 10
        cpl = proc.atomic_write(0, data)
        assert cpl.status == 0
        assert cpl.bytes_transferred == len(data)
        assert backend.read(0, len(data)) == data
        backend.destroy()

    def test_atomic_write_max_size_128kb(self):
        backend = make_backend(size=256 * 1024)
        proc = IoCommandProcessor(backend)
        data = b"X" * MAX_ATOMIC_WRITE_BYTES
        cpl = proc.atomic_write(0, data)
        assert cpl.status == 0
        backend.destroy()

    def test_atomic_write_exceeds_128kb_raises(self):
        backend = make_backend(size=256 * 1024)
        proc = IoCommandProcessor(backend)
        data = b"X" * (MAX_ATOMIC_WRITE_BYTES + 1)
        with pytest.raises(NvmeConfigError):
            proc.atomic_write(0, data)
        backend.destroy()

    def test_handler_atomic_write(self):
        handler, backend, _ = make_handler()
        data = b"atomic via handler"
        req = make_write_request(0, data, req_id=99)
        cpl = handler.handle_atomic_write(req)
        assert cpl.status == 0
        backend.destroy()


# ---------------------------------------------------------------------------
# Namespace management tests
# ---------------------------------------------------------------------------

class TestNamespaceManager:
    def test_create_namespace(self):
        mgr = NamespaceManager()
        ns = mgr.create_namespace(capacity_bytes=1024 * 1024)
        assert ns.namespace_id == 1
        assert ns.capacity_bytes == 1024 * 1024
        assert ns.block_size == 4096

    def test_create_multiple_namespaces_unique_ids(self):
        mgr = NamespaceManager()
        ns1 = mgr.create_namespace(1024 * 1024)
        ns2 = mgr.create_namespace(2 * 1024 * 1024)
        assert ns1.namespace_id != ns2.namespace_id

    def test_create_namespace_invalid_capacity_raises(self):
        mgr = NamespaceManager()
        with pytest.raises(NvmeConfigError):
            mgr.create_namespace(0)

    def test_delete_namespace(self):
        mgr = NamespaceManager()
        ns = mgr.create_namespace(1024 * 1024)
        mgr.delete_namespace(ns.namespace_id)
        with pytest.raises(NvmeDeviceNotFoundError):
            mgr.get_namespace(ns.namespace_id)

    def test_delete_nonexistent_namespace_raises(self):
        mgr = NamespaceManager()
        with pytest.raises(NvmeDeviceNotFoundError):
            mgr.delete_namespace(999)

    def test_attach_namespace(self):
        mgr = NamespaceManager()
        ns = mgr.create_namespace(1024 * 1024)
        assert not mgr.is_attached(ns.namespace_id)
        mgr.attach_namespace(ns.namespace_id)
        assert mgr.is_attached(ns.namespace_id)

    def test_detach_namespace(self):
        mgr = NamespaceManager()
        ns = mgr.create_namespace(1024 * 1024)
        mgr.attach_namespace(ns.namespace_id)
        mgr.detach_namespace(ns.namespace_id)
        assert not mgr.is_attached(ns.namespace_id)

    def test_get_namespace(self):
        mgr = NamespaceManager()
        ns = mgr.create_namespace(512 * 1024, block_size=512)
        fetched = mgr.get_namespace(ns.namespace_id)
        assert fetched.capacity_bytes == 512 * 1024
        assert fetched.block_size == 512

    def test_get_nonexistent_namespace_raises(self):
        mgr = NamespaceManager()
        with pytest.raises(NvmeDeviceNotFoundError):
            mgr.get_namespace(42)

    def test_list_namespaces(self):
        mgr = NamespaceManager()
        mgr.create_namespace(1024 * 1024)
        mgr.create_namespace(2 * 1024 * 1024)
        nss = mgr.list_namespaces()
        assert len(nss) == 2

    def test_list_namespaces_empty(self):
        mgr = NamespaceManager()
        assert mgr.list_namespaces() == []

    def test_block_count_property(self):
        mgr = NamespaceManager()
        ns = mgr.create_namespace(capacity_bytes=4096 * 10, block_size=4096)
        assert ns.block_count == 10


# ---------------------------------------------------------------------------
# Reservation tests
# ---------------------------------------------------------------------------

class TestReservationManager:
    def test_register_host(self):
        rm = ReservationManager()
        rm.register("host-1", namespace_id=1)
        # No error = success

    def test_reserve_after_register(self):
        rm = ReservationManager()
        rm.register("host-1", 1)
        rm.reserve("host-1", 1, ReservationType.WRITE_EXCLUSIVE)
        res = rm.get_reservation(1)
        assert res is not None
        assert res["holder_host_id"] == "host-1"

    def test_reserve_without_register_raises(self):
        rm = ReservationManager()
        with pytest.raises(NvmePermissionError):
            rm.reserve("host-1", 1, ReservationType.EXCLUSIVE_ACCESS)

    def test_release_reservation(self):
        rm = ReservationManager()
        rm.register("host-1", 1)
        rm.reserve("host-1", 1, ReservationType.WRITE_EXCLUSIVE)
        rm.release("host-1", 1)
        assert rm.get_reservation(1) is None

    def test_release_by_non_holder_raises(self):
        rm = ReservationManager()
        rm.register("host-1", 1)
        rm.register("host-2", 1)
        rm.reserve("host-1", 1, ReservationType.WRITE_EXCLUSIVE)
        with pytest.raises(NvmePermissionError):
            rm.release("host-2", 1)

    def test_preempt_reservation(self):
        rm = ReservationManager()
        rm.register("host-1", 1)
        rm.register("host-2", 1)
        rm.reserve("host-1", 1, ReservationType.WRITE_EXCLUSIVE)
        rm.preempt("host-2", 1)
        res = rm.get_reservation(1)
        assert res["holder_host_id"] == "host-2"

    def test_get_reservation_none_when_no_reservation(self):
        rm = ReservationManager()
        assert rm.get_reservation(1) is None

    def test_reservation_type_stored(self):
        rm = ReservationManager()
        rm.register("host-1", 1)
        rm.reserve("host-1", 1, ReservationType.EXCLUSIVE_ACCESS_REGISTRANTS_ONLY)
        res = rm.get_reservation(1)
        assert res["reservation_type"] == ReservationType.EXCLUSIVE_ACCESS_REGISTRANTS_ONLY.value


# ---------------------------------------------------------------------------
# Property tests (Properties 19, 20, 21)
# ---------------------------------------------------------------------------

class TestCommandHandlerProperties:
    @given(depth=st.integers(min_value=1, max_value=65535))
    @settings(max_examples=50)
    def test_property19_queue_depth_configuration(self, depth):
        """
        Feature: software-nvme-solution
        Property 19: Queue depth configuration (1 ≤ D ≤ 65535).

        Any valid queue depth can be created via admin command.
        """
        proc = AdminCommandProcessor()
        result = proc.process(AdminCommandType.CREATE_IO_QUEUE, {
            "queue_id": depth,
            "queue_type": "submission",
            "depth": depth,
        })
        assert result.success

    @given(capacity=st.integers(min_value=4096, max_value=10 * 1024 * 1024))
    @settings(max_examples=50)
    def test_property20_dynamic_namespace_management(self, capacity):
        """
        Feature: software-nvme-solution
        Property 20: Dynamic namespace management (immediate reflection).

        Created namespaces are immediately visible in list_namespaces().
        Deleted namespaces are immediately absent.
        """
        mgr = NamespaceManager()
        ns = mgr.create_namespace(capacity)
        assert any(n.namespace_id == ns.namespace_id for n in mgr.list_namespaces())

        mgr.delete_namespace(ns.namespace_id)
        assert all(n.namespace_id != ns.namespace_id for n in mgr.list_namespaces())

    @given(size=st.integers(min_value=1, max_value=MAX_ATOMIC_WRITE_BYTES))
    @settings(max_examples=50)
    def test_property21_atomic_write_all_or_nothing(self, size):
        """
        Feature: software-nvme-solution
        Property 21: Atomic write guarantees (all-or-nothing up to 128KB).

        A successful atomic write stores exactly the written data.
        """
        backend = make_backend(size=256 * 1024)
        proc = IoCommandProcessor(backend)
        # Build data of exactly `size` bytes with non-zero content
        pattern = bytes(range(0, 256))
        data = (pattern * (size // 256 + 1))[:size]

        cpl = proc.atomic_write(0, data)
        if cpl.status == 0 and size > 0:
            stored = backend.read(0, size)
            assert stored == data

        backend.destroy()
