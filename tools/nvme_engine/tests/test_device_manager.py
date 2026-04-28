"""
Tests for nvme_engine.control.device_manager.

Covers: creation, deletion, modification, hot-plug/unplug, listing,
        thread safety, and property-based tests.
"""

from __future__ import annotations

import threading
import time
from typing import List

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nvme_engine.backends.memory_backend import MemoryBackend
from nvme_engine.control.device_manager import DeviceManager, DeviceState, VirtualDevice
from nvme_engine.models.config import (
    BackendConfig,
    BackendType,
    MemoryBackendConfig,
    NvmeDeviceConfig,
)
from nvme_engine.models.errors import NvmeDeviceNotFoundError, NvmeResourceError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_1MB = 1024 * 1024
_1GB = 1024 * _1MB


def _make_config(
    name: str = "test-device",
    capacity_bytes: int = _1GB,
    namespace_count: int = 1,
    max_queue_pairs: int = 4,
    queue_depth: int = 64,
) -> NvmeDeviceConfig:
    backend = BackendConfig(
        type=BackendType.MEMORY,
        memory=MemoryBackendConfig(size_bytes=capacity_bytes),
    )
    return NvmeDeviceConfig(
        name=name,
        capacity_bytes=capacity_bytes,
        namespace_count=namespace_count,
        max_queue_pairs=max_queue_pairs,
        queue_depth=queue_depth,
        backend=backend,
    )


def _make_manager() -> DeviceManager:
    return DeviceManager()


# ---------------------------------------------------------------------------
# Basic creation tests
# ---------------------------------------------------------------------------


class TestCreateDevice:
    def test_create_returns_virtual_device(self):
        dm = _make_manager()
        cfg = _make_config()
        dev = dm.create_device(cfg)
        assert isinstance(dev, VirtualDevice)

    def test_create_assigns_unique_id(self):
        dm = _make_manager()
        d1 = dm.create_device(_make_config(name="d1"))
        d2 = dm.create_device(_make_config(name="d2"))
        assert d1.device_id != d2.device_id

    def test_create_device_state_is_active(self):
        dm = _make_manager()
        dev = dm.create_device(_make_config())
        assert dev.state == DeviceState.ACTIVE

    def test_create_device_preserves_name(self):
        dm = _make_manager()
        dev = dm.create_device(_make_config(name="my-nvme"))
        assert dev.name == "my-nvme"

    def test_create_device_preserves_config(self):
        dm = _make_manager()
        cfg = _make_config(capacity_bytes=2 * _1MB)
        dev = dm.create_device(cfg)
        assert dev.config.capacity_bytes == 2 * _1MB

    def test_create_device_has_backend(self):
        dm = _make_manager()
        dev = dm.create_device(_make_config())
        assert dev.backend is not None
        assert dev.backend.is_initialized

    def test_create_device_has_namespace_manager(self):
        dm = _make_manager()
        dev = dm.create_device(_make_config(namespace_count=2))
        assert dev.namespace_manager is not None
        assert len(dev.namespace_manager.list_namespaces()) == 2

    def test_create_device_increments_count(self):
        dm = _make_manager()
        assert dm.device_count == 0
        dm.create_device(_make_config())
        assert dm.device_count == 1
        dm.create_device(_make_config(name="d2"))
        assert dm.device_count == 2

    def test_create_device_has_created_at(self):
        dm = _make_manager()
        before = time.time()
        dev = dm.create_device(_make_config())
        after = time.time()
        assert before <= dev.created_at <= after

    def test_create_exceeds_max_raises_resource_error(self):
        dm = _make_manager()
        # Fill up to MAX_DEVICES using tiny 1 MB devices to avoid OOM
        for i in range(DeviceManager.MAX_DEVICES):
            dm.create_device(_make_config(name=f"dev-{i}", capacity_bytes=_1MB))
        with pytest.raises(NvmeResourceError):
            dm.create_device(_make_config(name="overflow", capacity_bytes=_1MB))

    def test_max_devices_is_256(self):
        assert DeviceManager.MAX_DEVICES == 256


# ---------------------------------------------------------------------------
# Deletion tests
# ---------------------------------------------------------------------------


class TestDeleteDevice:
    def test_delete_removes_from_list(self):
        dm = _make_manager()
        dev = dm.create_device(_make_config())
        dm.delete_device(dev.device_id)
        assert dev not in dm.list_devices()

    def test_delete_decrements_count(self):
        dm = _make_manager()
        dev = dm.create_device(_make_config())
        assert dm.device_count == 1
        dm.delete_device(dev.device_id)
        assert dm.device_count == 0

    def test_delete_releases_backend(self):
        dm = _make_manager()
        dev = dm.create_device(_make_config())
        backend = dev.backend
        dm.delete_device(dev.device_id)
        assert dev.backend is None
        assert not backend.is_initialized

    def test_delete_sets_state_deleted(self):
        dm = _make_manager()
        dev = dm.create_device(_make_config())
        dm.delete_device(dev.device_id)
        assert dev.state == DeviceState.DELETED

    def test_delete_nonexistent_raises(self):
        dm = _make_manager()
        with pytest.raises(NvmeDeviceNotFoundError):
            dm.delete_device(9999)

    def test_delete_already_deleted_raises(self):
        dm = _make_manager()
        dev = dm.create_device(_make_config())
        dm.delete_device(dev.device_id)
        with pytest.raises(NvmeDeviceNotFoundError):
            dm.delete_device(dev.device_id)

    def test_delete_within_timeout(self):
        dm = _make_manager()
        dev = dm.create_device(_make_config())
        start = time.monotonic()
        dm.delete_device(dev.device_id, timeout_s=5.0)
        elapsed = time.monotonic() - start
        assert elapsed < 5.0

    def test_delete_allows_new_device_after(self):
        dm = _make_manager()
        dev = dm.create_device(_make_config())
        dm.delete_device(dev.device_id)
        new_dev = dm.create_device(_make_config(name="new"))
        assert new_dev.state == DeviceState.ACTIVE


# ---------------------------------------------------------------------------
# Modification tests
# ---------------------------------------------------------------------------


class TestModifyDevice:
    def test_modify_updates_name(self):
        dm = _make_manager()
        dev = dm.create_device(_make_config(name="original"))
        dm.modify_device(dev.device_id, name="updated")
        assert dev.name == "updated"

    def test_modify_returns_device(self):
        dm = _make_manager()
        dev = dm.create_device(_make_config())
        result = dm.modify_device(dev.device_id, name="x")
        assert result is dev

    def test_modify_state_returns_to_active(self):
        dm = _make_manager()
        dev = dm.create_device(_make_config())
        dm.modify_device(dev.device_id, name="y")
        assert dev.state == DeviceState.ACTIVE

    def test_modify_nonexistent_raises(self):
        dm = _make_manager()
        with pytest.raises(NvmeDeviceNotFoundError):
            dm.modify_device(9999, name="x")

    def test_modify_unknown_kwarg_ignored(self):
        dm = _make_manager()
        dev = dm.create_device(_make_config())
        # Should not raise
        dm.modify_device(dev.device_id, nonexistent_field="value")
        assert dev.state == DeviceState.ACTIVE


# ---------------------------------------------------------------------------
# Hot-plug / hot-unplug tests
# ---------------------------------------------------------------------------


class TestHotPlug:
    def test_hot_plug_device_becomes_active(self):
        dm = _make_manager()
        dev = dm.create_device(_make_config())
        dm.hot_plug(dev.device_id)
        assert dev.state == DeviceState.ACTIVE

    def test_hot_plug_completes_within_2s(self):
        dm = _make_manager()
        dev = dm.create_device(_make_config())
        start = time.monotonic()
        dm.hot_plug(dev.device_id)
        assert time.monotonic() - start < 2.0

    def test_hot_plug_nonexistent_raises(self):
        dm = _make_manager()
        with pytest.raises(NvmeDeviceNotFoundError):
            dm.hot_plug(9999)

    def test_hot_unplug_removes_device(self):
        dm = _make_manager()
        dev = dm.create_device(_make_config())
        dm.hot_unplug(dev.device_id)
        assert dev.state == DeviceState.DELETED
        assert dev not in dm.list_devices()

    def test_hot_unplug_releases_backend(self):
        dm = _make_manager()
        dev = dm.create_device(_make_config())
        backend = dev.backend
        dm.hot_unplug(dev.device_id)
        assert not backend.is_initialized

    def test_hot_unplug_nonexistent_raises(self):
        dm = _make_manager()
        with pytest.raises(NvmeDeviceNotFoundError):
            dm.hot_unplug(9999)


# ---------------------------------------------------------------------------
# List / get tests
# ---------------------------------------------------------------------------


class TestListAndGet:
    def test_list_empty_initially(self):
        dm = _make_manager()
        assert dm.list_devices() == []

    def test_list_returns_all_active(self):
        dm = _make_manager()
        d1 = dm.create_device(_make_config(name="a"))
        d2 = dm.create_device(_make_config(name="b"))
        devices = dm.list_devices()
        assert d1 in devices
        assert d2 in devices

    def test_list_excludes_deleted(self):
        dm = _make_manager()
        d1 = dm.create_device(_make_config(name="a"))
        d2 = dm.create_device(_make_config(name="b"))
        dm.delete_device(d1.device_id)
        devices = dm.list_devices()
        assert d1 not in devices
        assert d2 in devices

    def test_get_device_returns_correct(self):
        dm = _make_manager()
        dev = dm.create_device(_make_config())
        fetched = dm.get_device(dev.device_id)
        assert fetched is dev

    def test_get_device_not_found_raises(self):
        dm = _make_manager()
        with pytest.raises(NvmeDeviceNotFoundError):
            dm.get_device(42)

    def test_get_deleted_device_raises(self):
        dm = _make_manager()
        dev = dm.create_device(_make_config())
        dm.delete_device(dev.device_id)
        with pytest.raises(NvmeDeviceNotFoundError):
            dm.get_device(dev.device_id)


# ---------------------------------------------------------------------------
# Thread-safety smoke test
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_create(self):
        dm = _make_manager()
        errors: List[Exception] = []

        def create_one(idx: int) -> None:
            try:
                dm.create_device(_make_config(name=f"dev-{idx}", capacity_bytes=_1MB))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=create_one, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert dm.device_count == 10


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


@given(
    name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_")),
    capacity=st.integers(min_value=_1MB, max_value=64 * _1MB),
)
@settings(max_examples=30, deadline=5000)
def test_property1_create_preserves_config(name: str, capacity: int):
    """Property 1: Created device preserves its configuration."""
    dm = _make_manager()
    cfg = _make_config(name=name, capacity_bytes=capacity)
    dev = dm.create_device(cfg)
    assert dev.config.name == name
    assert dev.config.capacity_bytes == capacity


@given(count=st.integers(min_value=1, max_value=5))
@settings(max_examples=20, deadline=10000)
def test_property2_delete_releases_resources(count: int):
    """Property 2: Deleting a device releases its backend resources."""
    dm = _make_manager()
    devices = [dm.create_device(_make_config(name=f"d{i}", capacity_bytes=_1MB)) for i in range(count)]
    backends = [d.backend for d in devices]
    for dev in devices:
        dm.delete_device(dev.device_id)
    for backend in backends:
        assert not backend.is_initialized


@given(new_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_")))
@settings(max_examples=30, deadline=5000)
def test_property3_modify_preserves_active_state(new_name: str):
    """Property 3: Modifying a device returns it to ACTIVE state."""
    dm = _make_manager()
    dev = dm.create_device(_make_config())
    dm.modify_device(dev.device_id, name=new_name)
    assert dev.state == DeviceState.ACTIVE


@given(count=st.integers(min_value=1, max_value=3))
@settings(max_examples=10, deadline=10000)
def test_property50_hot_plug_available_within_2s(count: int):
    """Property 50: Hot-plug makes device available within 2 seconds."""
    dm = _make_manager()
    for i in range(count):
        dev = dm.create_device(_make_config(name=f"d{i}", capacity_bytes=_1MB))
        start = time.monotonic()
        dm.hot_plug(dev.device_id)
        assert time.monotonic() - start < 2.0
        assert dev.state == DeviceState.ACTIVE


@given(count=st.integers(min_value=1, max_value=3))
@settings(max_examples=10, deadline=10000)
def test_property51_hot_unplug_graceful(count: int):
    """Property 51: Hot-unplug gracefully completes and releases resources."""
    dm = _make_manager()
    for i in range(count):
        dev = dm.create_device(_make_config(name=f"d{i}", capacity_bytes=_1MB))
        backend = dev.backend
        dm.hot_unplug(dev.device_id)
        assert dev.state == DeviceState.DELETED
        assert not backend.is_initialized
