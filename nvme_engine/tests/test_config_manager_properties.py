"""
Property-based tests for ConfigManager — task 2.10
Covers Properties 39, 40, 41.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from nvme_engine.control.config_manager import ConfigManager

mgr = ConfigManager()

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

backend_types = st.sampled_from(["memory", "file", "network", "hybrid"])
cache_policies = st.sampled_from(["arc", "lru", "lfu", "2q"])
priorities = st.integers(min_value=0, max_value=3)
weights = st.integers(min_value=1, max_value=1000)
namespaces = st.integers(min_value=1, max_value=256)
capacity_str = st.sampled_from(["1GB", "10GB", "100GB", "1TB", "512GB", "2TB"])
device_name = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="-_"),
    min_size=1,
    max_size=64,
)


@st.composite
def valid_device_config(draw):
    """Generate a valid device config dict."""
    return {
        "name": draw(device_name),
        "capacity": draw(capacity_str),
        "namespaces": draw(namespaces),
        "backend": {"type": draw(backend_types)},
        "qos": {
            "priority": draw(priorities),
            "weight": draw(weights),
        },
        "cache": {
            "enabled": draw(st.booleans()),
            "policy": draw(cache_policies),
        },
    }


# ---------------------------------------------------------------------------
# Property 39: Any valid config dict passes validation
# ---------------------------------------------------------------------------

@given(cfg=valid_device_config())
@settings(max_examples=100)
def test_property_39_valid_config_passes_validation(cfg):
    """Property 39: For any valid device config dict, validate() returns True with no errors."""
    valid, errors = mgr.validate(cfg)
    assert valid is True, f"Expected valid=True but got errors: {errors}"
    assert errors == []


# ---------------------------------------------------------------------------
# Property 40: persist then restore is identity
# ---------------------------------------------------------------------------

@given(configs=st.lists(valid_device_config(), min_size=1, max_size=5))
@settings(max_examples=100)
def test_property_40_persist_restore_identity(configs):
    """Property 40: For any device config, persist() then restore() returns an equivalent config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "state.json"
        mgr.persist(configs, str(out))
        restored = mgr.restore(str(out))

    assert len(restored) == len(configs)
    for original, restored_cfg in zip(configs, restored):
        assert restored_cfg["name"] == original["name"]
        assert restored_cfg["capacity"] == original["capacity"]
        assert restored_cfg["namespaces"] == original["namespaces"]


# ---------------------------------------------------------------------------
# Property 41: restore after restart returns all persisted devices
# ---------------------------------------------------------------------------

@given(configs=st.lists(valid_device_config(), min_size=1, max_size=10))
@settings(max_examples=100)
def test_property_41_restore_after_restart_returns_all_devices(configs):
    """Property 41: For any config persisted to disk, after simulated restart restore() returns all devices."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "persist.json"
        mgr.persist(configs, str(out))
        # Simulate restart: create a fresh ConfigManager instance
        fresh_mgr = ConfigManager()
        restored = fresh_mgr.restore(str(out))

    assert len(restored) == len(configs), (
        f"Expected {len(configs)} devices after restore, got {len(restored)}"
    )
    restored_names = {c["name"] for c in restored}
    original_names = {c["name"] for c in configs}
    assert restored_names == original_names


# ---------------------------------------------------------------------------
# Additional: invalid configs always fail validation
# ---------------------------------------------------------------------------

@given(
    priority=st.integers().filter(lambda x: not (0 <= x <= 3)),
)
@settings(max_examples=50)
def test_invalid_priority_always_fails(priority):
    """Out-of-range priority values always fail validation."""
    cfg = {
        "name": "test",
        "capacity": "1TB",
        "qos": {"priority": priority, "weight": 100},
    }
    valid, errors = mgr.validate(cfg)
    assert valid is False
    assert any("qos.priority" in e for e in errors)


@given(
    weight=st.integers().filter(lambda x: not (1 <= x <= 1000)),
)
@settings(max_examples=50)
def test_invalid_weight_always_fails(weight):
    """Out-of-range weight values always fail validation."""
    cfg = {
        "name": "test",
        "capacity": "1TB",
        "qos": {"priority": 1, "weight": weight},
    }
    valid, errors = mgr.validate(cfg)
    assert valid is False
    assert any("qos.weight" in e for e in errors)
