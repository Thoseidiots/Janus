"""
Configuration Manager for the Software NVMe Engine.

Handles JSON/YAML config parsing, schema validation, persistence,
restoration, and built-in device profile templates.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

from nvme_engine.models.config import (
    BackendConfig,
    BackendType,
    CacheConfig,
    CachePolicy,
    FeatureFlags,
    NvmeDeviceConfig,
    PerformanceConfig,
    QosConfig,
    SecurityConfig,
)


# ---------------------------------------------------------------------------
# Built-in templates
# ---------------------------------------------------------------------------

_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "high-performance": {
        "name": "nvme-hp",
        "capacity": "64GB",
        "namespaces": 1,
        "backend": {"type": "memory", "numa_node": 0},
        "performance": {"max_iops": 1_000_000, "max_bandwidth": "10GB/s", "queue_depth": 4096},
        "qos": {"priority": 0, "weight": 1000},
        "cache": {"enabled": True, "size": "8GB", "policy": "arc"},
        "features": {"encryption": False, "compression": False, "deduplication": False},
    },
    "persistent": {
        "name": "nvme-persist",
        "capacity": "1TB",
        "namespaces": 4,
        "backend": {"type": "file", "path": "/var/lib/nvme_engine/nvme-persist.img", "sparse": True},
        "performance": {"max_iops": 100_000, "max_bandwidth": "1GB/s", "queue_depth": 1024},
        "qos": {"priority": 2, "weight": 100},
        "cache": {"enabled": True, "size": "4GB", "policy": "arc"},
        "features": {"encryption": True, "compression": False, "deduplication": False},
    },
    "network": {
        "name": "nvme-net",
        "capacity": "10TB",
        "namespaces": 8,
        "backend": {"type": "network", "host": "storage.local", "port": 4420, "transport": "tcp"},
        "performance": {"max_iops": 50_000, "max_bandwidth": "500MB/s", "queue_depth": 512},
        "qos": {"priority": 2, "weight": 100},
        "cache": {"enabled": True, "size": "16GB", "policy": "arc"},
        "features": {"encryption": True, "compression": False, "deduplication": False},
    },
    "balanced": {
        "name": "nvme-balanced",
        "capacity": "2TB",
        "namespaces": 4,
        "backend": {"type": "hybrid"},
        "performance": {"max_iops": 500_000, "max_bandwidth": "5GB/s", "queue_depth": 2048},
        "qos": {"priority": 1, "weight": 500},
        "cache": {"enabled": True, "size": "32GB", "policy": "arc"},
        "features": {"encryption": False, "compression": False, "deduplication": False},
    },
}

# Valid values for enum-like fields
_VALID_BACKEND_TYPES = {"memory", "file", "network", "hybrid"}
_VALID_CACHE_POLICIES = {"arc", "lru", "lfu", "2q"}


# ---------------------------------------------------------------------------
# ConfigManager
# ---------------------------------------------------------------------------

class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        super().__init__("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


class ConfigManager:
    """
    Manages NVMe device configurations: parsing, validation, persistence, and templates.
    """

    def load_from_file(self, path: str) -> List[Dict[str, Any]]:
        """Load device configurations from a JSON or YAML file.

        Returns a list of raw device config dicts.
        Raises FileNotFoundError, ValueError on parse errors.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        text = p.read_text(encoding="utf-8")
        suffix = p.suffix.lower()

        if suffix in (".yaml", ".yml"):
            if not _YAML_AVAILABLE:
                raise ImportError("pyyaml is required for YAML config files. Install with: pip install pyyaml")
            try:
                data = yaml.safe_load(text)
            except yaml.YAMLError as exc:
                raise ValueError(f"YAML parse error in {path}: {exc}") from exc
        elif suffix == ".json":
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON parse error in {path}: {exc}") from exc
        else:
            # Try JSON first, then YAML
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                if _YAML_AVAILABLE:
                    try:
                        data = yaml.safe_load(text)
                    except yaml.YAMLError as exc:
                        raise ValueError(f"Could not parse {path} as JSON or YAML: {exc}") from exc
                else:
                    raise ValueError(f"Unknown file format for {path}. Use .json or .yaml extension.")

        return self._extract_devices(data, path)

    def load_from_dict(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load device configurations from a dict (already parsed).

        Returns a list of raw device config dicts.
        """
        return self._extract_devices(data, "<dict>")

    def _extract_devices(self, data: Any, source: str) -> List[Dict[str, Any]]:
        """Extract the list of device dicts from parsed config data."""
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "devices" in data:
                devices = data["devices"]
                if not isinstance(devices, list):
                    raise ValueError(f"'devices' key in {source} must be a list")
                return devices
            # Single device dict
            return [data]
        raise ValueError(f"Config in {source} must be a dict or list, got {type(data).__name__}")

    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a single device config dict.

        Returns (is_valid, list_of_error_messages).
        Error messages are descriptive: field name, expected, actual.
        """
        errors: List[str] = []

        # Required: name
        if "name" not in config:
            errors.append("Field 'name' is required but missing")
        elif not isinstance(config["name"], str) or not config["name"].strip():
            errors.append(f"Field 'name' must be a non-empty string, got: {config.get('name')!r}")

        # Required: capacity
        if "capacity" not in config:
            errors.append("Field 'capacity' is required but missing")
        elif not isinstance(config["capacity"], str):
            errors.append(f"Field 'capacity' must be a string like '1TB' or '512GB', got: {config.get('capacity')!r}")

        # Optional: namespaces
        if "namespaces" in config:
            ns = config["namespaces"]
            if not isinstance(ns, int) or not (1 <= ns <= 256):
                errors.append(f"Field 'namespaces' must be an integer between 1 and 256, got: {ns!r}")

        # Optional: backend
        if "backend" in config:
            backend = config["backend"]
            if not isinstance(backend, dict):
                errors.append(f"Field 'backend' must be a dict, got: {type(backend).__name__}")
            else:
                btype = backend.get("type")
                if btype is not None and btype not in _VALID_BACKEND_TYPES:
                    errors.append(
                        f"Field 'backend.type' must be one of {sorted(_VALID_BACKEND_TYPES)}, got: {btype!r}"
                    )

        # Optional: qos
        if "qos" in config:
            qos = config["qos"]
            if not isinstance(qos, dict):
                errors.append(f"Field 'qos' must be a dict, got: {type(qos).__name__}")
            else:
                priority = qos.get("priority")
                if priority is not None and (not isinstance(priority, int) or not (0 <= priority <= 3)):
                    errors.append(f"Field 'qos.priority' must be an integer 0-3, got: {priority!r}")
                weight = qos.get("weight")
                if weight is not None and (not isinstance(weight, int) or not (1 <= weight <= 1000)):
                    errors.append(f"Field 'qos.weight' must be an integer 1-1000, got: {weight!r}")

        # Optional: cache
        if "cache" in config:
            cache = config["cache"]
            if not isinstance(cache, dict):
                errors.append(f"Field 'cache' must be a dict, got: {type(cache).__name__}")
            else:
                policy = cache.get("policy")
                if policy is not None and policy not in _VALID_CACHE_POLICIES:
                    errors.append(
                        f"Field 'cache.policy' must be one of {sorted(_VALID_CACHE_POLICIES)}, got: {policy!r}"
                    )

        return (len(errors) == 0, errors)

    def validate_all(self, configs: List[Dict[str, Any]]) -> Tuple[bool, Dict[int, List[str]]]:
        """Validate a list of device configs.

        Returns (all_valid, {index: [errors]}).
        """
        all_valid = True
        error_map: Dict[int, List[str]] = {}
        for i, cfg in enumerate(configs):
            valid, errs = self.validate(cfg)
            if not valid:
                all_valid = False
                error_map[i] = errs
        return all_valid, error_map

    def persist(self, configs: List[Dict[str, Any]], path: str) -> None:
        """Atomically persist a list of device configs to a JSON file.

        Uses write-to-temp-then-rename for crash safety.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        data = {"devices": configs}
        serialized = json.dumps(data, indent=2, ensure_ascii=False)

        # Atomic write: temp file in same directory, then rename
        fd, tmp_path = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(serialized)
            os.replace(tmp_path, str(p))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def restore(self, path: str) -> List[Dict[str, Any]]:
        """Restore device configs from a persisted JSON file.

        Returns list of device config dicts, or empty list if file doesn't exist.
        """
        p = Path(path)
        if not p.exists():
            return []
        try:
            text = p.read_text(encoding="utf-8")
            data = json.loads(text)
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Failed to restore config from {path}: {exc}") from exc
        return self._extract_devices(data, path)

    def get_template(self, name: str) -> Dict[str, Any]:
        """Return a copy of a built-in device profile template by name.

        Raises KeyError if template not found.
        """
        if name not in _TEMPLATES:
            raise KeyError(f"Template {name!r} not found. Available: {self.list_templates()}")
        import copy
        return copy.deepcopy(_TEMPLATES[name])

    def list_templates(self) -> List[str]:
        """Return sorted list of available template names."""
        return sorted(_TEMPLATES.keys())

    def to_nvme_device_config(self, raw: Dict[str, Any]) -> NvmeDeviceConfig:
        """Convert a raw config dict to a typed NvmeDeviceConfig.

        Raises ConfigValidationError if validation fails.
        Normalizes human-readable capacity strings (e.g. "1TB") to bytes.
        """
        valid, errors = self.validate(raw)
        if not valid:
            raise ConfigValidationError(errors)
        normalized = dict(raw)
        if "capacity" in normalized and "capacity_bytes" not in normalized:
            normalized["capacity_bytes"] = _parse_capacity(normalized.pop("capacity"))
        if "namespaces" in normalized and "namespace_count" not in normalized:
            normalized["namespace_count"] = normalized.pop("namespaces")
        normalized.setdefault("max_queue_pairs", 64)
        normalized.setdefault("queue_depth", 1024)
        # Normalize backend type to uppercase for enum compatibility
        if "backend" in normalized and isinstance(normalized["backend"], dict):
            backend = dict(normalized["backend"])
            if "type" in backend:
                backend["type"] = backend["type"].upper()
            normalized["backend"] = backend
        return NvmeDeviceConfig.from_dict(normalized)


def _parse_capacity(capacity_str: str) -> int:
    """Parse a human-readable capacity string to bytes. E.g. '1TB' -> 1_099_511_627_776."""
    s = capacity_str.strip().upper()
    units = {
        "TB": 1024 ** 4,
        "GB": 1024 ** 3,
        "MB": 1024 ** 2,
        "KB": 1024,
        "B": 1,
    }
    for suffix, multiplier in units.items():
        if s.endswith(suffix):
            number = s[: -len(suffix)].strip()
            try:
                return int(float(number) * multiplier)
            except ValueError:
                raise ValueError(f"Cannot parse capacity: {capacity_str!r}")
    raise ValueError(f"Unknown capacity unit in: {capacity_str!r}")
