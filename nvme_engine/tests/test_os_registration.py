"""
Tests for OS block device registration — task 12.2
"""

from __future__ import annotations

import pytest
from nvme_engine.control.os_registration import OsDeviceRegistry, BlockDeviceInfo


@pytest.fixture
def registry():
    return OsDeviceRegistry(force_simulated=True)


class TestSimulatedRegistration:
    def test_register_returns_block_device_info(self, registry):
        info = registry.register(1, "nvme0", 1024 ** 4, 4)
        assert isinstance(info, BlockDeviceInfo)
        assert info.device_id == 1
        assert info.registered is True
        assert info.simulated is True

    def test_block_path_assigned(self, registry):
        info = registry.register(1, "nvme0", 1024 ** 4, 4)
        assert info.block_path.startswith("/dev/nvme")

    def test_multiple_devices_get_unique_paths(self, registry):
        info1 = registry.register(1, "nvme0", 1024 ** 4, 1)
        info2 = registry.register(2, "nvme1", 1024 ** 4, 1)
        assert info1.block_path != info2.block_path

    def test_is_registered_after_register(self, registry):
        registry.register(1, "nvme0", 1024 ** 4, 4)
        assert registry.is_registered(1) is True

    def test_not_registered_before_register(self, registry):
        assert registry.is_registered(99) is False

    def test_unregister_removes_device(self, registry):
        registry.register(1, "nvme0", 1024 ** 4, 4)
        registry.unregister(1)
        assert registry.is_registered(1) is False

    def test_unregister_nonexistent_is_noop(self, registry):
        registry.unregister(999)  # Should not raise

    def test_list_registered(self, registry):
        registry.register(1, "nvme0", 1024 ** 4, 1)
        registry.register(2, "nvme1", 512 * 1024 ** 3, 2)
        devices = registry.list_registered()
        assert len(devices) == 2

    def test_get_info_returns_correct_device(self, registry):
        registry.register(1, "nvme0", 1024 ** 4, 4)
        info = registry.get_info(1)
        assert info is not None
        assert info.name == "nvme0"
        assert info.capacity_bytes == 1024 ** 4
        assert info.namespace_count == 4

    def test_get_info_missing_returns_none(self, registry):
        assert registry.get_info(999) is None


class TestDeviceManagerOsIntegration:
    def test_hot_plug_registers_with_os(self):
        from nvme_engine.control.device_manager import DeviceManager
        from nvme_engine.models.config import (
            NvmeDeviceConfig, BackendConfig, BackendType, MemoryBackendConfig
        )

        dm = DeviceManager(force_simulated_os=True)
        config = NvmeDeviceConfig(
            name="nvme-test",
            capacity_bytes=1024 ** 3,
            namespace_count=1,
            max_queue_pairs=4,
            queue_depth=64,
            backend=BackendConfig(
                type=BackendType.MEMORY,
                memory=MemoryBackendConfig(size_bytes=1024 ** 3),
            ),
        )
        device = dm.create_device(config)
        dm.hot_plug(device.device_id)

        assert dm.os_registry.is_registered(device.device_id)
        info = dm.os_registry.get_info(device.device_id)
        assert info is not None
        assert info.block_path.startswith("/dev/nvme")

    def test_hot_unplug_unregisters_from_os(self):
        from nvme_engine.control.device_manager import DeviceManager
        from nvme_engine.models.config import (
            NvmeDeviceConfig, BackendConfig, BackendType, MemoryBackendConfig
        )

        dm = DeviceManager(force_simulated_os=True)
        config = NvmeDeviceConfig(
            name="nvme-unplug",
            capacity_bytes=1024 ** 3,
            namespace_count=1,
            max_queue_pairs=4,
            queue_depth=64,
            backend=BackendConfig(
                type=BackendType.MEMORY,
                memory=MemoryBackendConfig(size_bytes=1024 ** 3),
            ),
        )
        device = dm.create_device(config)
        dm.hot_plug(device.device_id)
        assert dm.os_registry.is_registered(device.device_id)

        dm.hot_unplug(device.device_id)
        assert not dm.os_registry.is_registered(device.device_id)
