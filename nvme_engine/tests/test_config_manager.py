"""
Unit tests for ConfigManager — tasks 2.9
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from nvme_engine.control.config_manager import ConfigManager, ConfigValidationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mgr():
    return ConfigManager()


VALID_DEVICE = {
    "name": "nvme0",
    "capacity": "1TB",
    "namespaces": 4,
    "backend": {"type": "memory", "numa_node": 0},
    "qos": {"priority": 1, "weight": 100},
    "cache": {"enabled": True, "size": "4GB", "policy": "arc"},
}


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------

class TestLoadFromDict:
    def test_single_device_dict(self, mgr):
        result = mgr.load_from_dict(VALID_DEVICE)
        assert len(result) == 1
        assert result[0]["name"] == "nvme0"

    def test_devices_key(self, mgr):
        data = {"devices": [VALID_DEVICE, {**VALID_DEVICE, "name": "nvme1"}]}
        result = mgr.load_from_dict(data)
        assert len(result) == 2

    def test_list_input(self, mgr):
        result = mgr.load_from_dict([VALID_DEVICE])
        assert len(result) == 1

    def test_invalid_type_raises(self, mgr):
        with pytest.raises(ValueError, match="must be a dict or list"):
            mgr.load_from_dict("not a dict")  # type: ignore


class TestLoadFromFile:
    def test_json_file(self, mgr, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"devices": [VALID_DEVICE]}))
        result = mgr.load_from_file(str(cfg_file))
        assert len(result) == 1

    def test_yaml_file(self, mgr, tmp_path):
        pytest.importorskip("yaml")
        import yaml
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({"devices": [VALID_DEVICE]}))
        result = mgr.load_from_file(str(cfg_file))
        assert len(result) == 1

    def test_missing_file_raises(self, mgr):
        with pytest.raises(FileNotFoundError):
            mgr.load_from_file("/nonexistent/path/config.json")

    def test_invalid_json_raises(self, mgr, tmp_path):
        cfg_file = tmp_path / "bad.json"
        cfg_file.write_text("{not valid json}")
        with pytest.raises(ValueError, match="JSON parse error"):
            mgr.load_from_file(str(cfg_file))


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidate:
    def test_valid_config(self, mgr):
        valid, errors = mgr.validate(VALID_DEVICE)
        assert valid is True
        assert errors == []

    def test_missing_name(self, mgr):
        cfg = {k: v for k, v in VALID_DEVICE.items() if k != "name"}
        valid, errors = mgr.validate(cfg)
        assert valid is False
        assert any("name" in e for e in errors)

    def test_missing_capacity(self, mgr):
        cfg = {k: v for k, v in VALID_DEVICE.items() if k != "capacity"}
        valid, errors = mgr.validate(cfg)
        assert valid is False
        assert any("capacity" in e for e in errors)

    def test_invalid_namespaces_zero(self, mgr):
        cfg = {**VALID_DEVICE, "namespaces": 0}
        valid, errors = mgr.validate(cfg)
        assert valid is False
        assert any("namespaces" in e for e in errors)

    def test_invalid_namespaces_too_large(self, mgr):
        cfg = {**VALID_DEVICE, "namespaces": 257}
        valid, errors = mgr.validate(cfg)
        assert valid is False

    def test_invalid_backend_type(self, mgr):
        cfg = {**VALID_DEVICE, "backend": {"type": "floppy"}}
        valid, errors = mgr.validate(cfg)
        assert valid is False
        assert any("backend.type" in e for e in errors)

    def test_invalid_qos_priority(self, mgr):
        cfg = {**VALID_DEVICE, "qos": {"priority": 5, "weight": 100}}
        valid, errors = mgr.validate(cfg)
        assert valid is False
        assert any("qos.priority" in e for e in errors)

    def test_invalid_qos_weight(self, mgr):
        cfg = {**VALID_DEVICE, "qos": {"priority": 1, "weight": 0}}
        valid, errors = mgr.validate(cfg)
        assert valid is False
        assert any("qos.weight" in e for e in errors)

    def test_invalid_cache_policy(self, mgr):
        cfg = {**VALID_DEVICE, "cache": {"policy": "unknown"}}
        valid, errors = mgr.validate(cfg)
        assert valid is False
        assert any("cache.policy" in e for e in errors)

    def test_error_messages_are_descriptive(self, mgr):
        cfg = {"name": "", "capacity": 123, "namespaces": 999}
        valid, errors = mgr.validate(cfg)
        assert valid is False
        # Each error should mention the field name
        for err in errors:
            assert any(field in err for field in ("name", "capacity", "namespaces"))


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

class TestPersistRestore:
    def test_persist_creates_file(self, mgr, tmp_path):
        out = tmp_path / "state.json"
        mgr.persist([VALID_DEVICE], str(out))
        assert out.exists()

    def test_restore_returns_same_configs(self, mgr, tmp_path):
        out = tmp_path / "state.json"
        mgr.persist([VALID_DEVICE], str(out))
        restored = mgr.restore(str(out))
        assert len(restored) == 1
        assert restored[0]["name"] == VALID_DEVICE["name"]

    def test_restore_missing_file_returns_empty(self, mgr, tmp_path):
        result = mgr.restore(str(tmp_path / "nonexistent.json"))
        assert result == []

    def test_persist_multiple_devices(self, mgr, tmp_path):
        devices = [VALID_DEVICE, {**VALID_DEVICE, "name": "nvme1"}]
        out = tmp_path / "state.json"
        mgr.persist(devices, str(out))
        restored = mgr.restore(str(out))
        assert len(restored) == 2

    def test_persist_is_atomic(self, mgr, tmp_path):
        """Persisted file should be valid JSON (atomic write)."""
        out = tmp_path / "state.json"
        mgr.persist([VALID_DEVICE], str(out))
        data = json.loads(out.read_text())
        assert "devices" in data


# ---------------------------------------------------------------------------
# Template tests
# ---------------------------------------------------------------------------

class TestTemplates:
    def test_list_templates_returns_all(self, mgr):
        templates = mgr.list_templates()
        assert "high-performance" in templates
        assert "persistent" in templates
        assert "network" in templates
        assert "balanced" in templates

    def test_get_template_returns_copy(self, mgr):
        t1 = mgr.get_template("high-performance")
        t2 = mgr.get_template("high-performance")
        t1["name"] = "modified"
        assert t2["name"] != "modified"  # deep copy

    def test_get_unknown_template_raises(self, mgr):
        with pytest.raises(KeyError):
            mgr.get_template("nonexistent")

    def test_all_templates_are_valid(self, mgr):
        for name in mgr.list_templates():
            tmpl = mgr.get_template(name)
            valid, errors = mgr.validate(tmpl)
            assert valid, f"Template '{name}' failed validation: {errors}"


# ---------------------------------------------------------------------------
# to_nvme_device_config tests
# ---------------------------------------------------------------------------

class TestToNvmeDeviceConfig:
    def test_invalid_config_raises(self, mgr):
        with pytest.raises(ConfigValidationError) as exc_info:
            mgr.to_nvme_device_config({"capacity": "1TB"})  # missing name
        assert exc_info.value.errors
