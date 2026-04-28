"""
Unit tests for ConfigurationPrettyPrinter round-trip integrity.

Verifies that Configuration objects can be formatted to JSON/YAML and
re-parsed back to equivalent Configuration objects (Requirements 6.3, 6.4).
"""

import json
import tempfile
from pathlib import Path

import pytest

from janus_dependency_analyzer.core.configuration import (
    JSONConfigurationParser,
    JSONConfigurationPrettyPrinter,
    YAMLConfigurationParser,
    YAMLConfigurationPrettyPrinter,
)
from janus_dependency_analyzer.core.models import Configuration, PriorityWeights
from janus_dependency_analyzer.config.configuration import ConfigurationManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def configs_equivalent(a: Configuration, b: Configuration) -> bool:
    """Return True when two Configuration objects have identical field values."""
    return (
        a.scan_exclusion_patterns == b.scan_exclusion_patterns
        and a.scan_timeout_seconds == b.scan_timeout_seconds
        and a.max_applications_per_scan == b.max_applications_per_scan
        and a.capability_detection_rules == b.capability_detection_rules
        and a.analysis_timeout_seconds == b.analysis_timeout_seconds
        and a.min_confidence_threshold == b.min_confidence_threshold
        and a.priority_weights.usage == b.priority_weights.usage
        and a.priority_weights.complexity == b.priority_weights.complexity
        and a.priority_weights.security == b.priority_weights.security
        and a.priority_weights.performance == b.priority_weights.performance
        and a.respect_access_controls == b.respect_access_controls
        and a.encrypt_stored_data == b.encrypt_stored_data
        and a.audit_logging_enabled == b.audit_logging_enabled
        and a.max_concurrent_analyses == b.max_concurrent_analyses
        and a.cache_analysis_results == b.cache_analysis_results
        and a.cache_expiry_hours == b.cache_expiry_hours
        and a.report_formats == b.report_formats
        and a.include_charts == b.include_charts
        and a.sanitize_paths == b.sanitize_paths
    )


# ---------------------------------------------------------------------------
# JSON round-trip tests
# ---------------------------------------------------------------------------

class TestJSONRoundTrip:
    """Tests for JSON parse → format → parse round-trip integrity."""

    def test_default_config_json_roundtrip(self):
        """A default Configuration survives a JSON round-trip unchanged."""
        original = Configuration()
        printer = JSONConfigurationPrettyPrinter()
        parser = JSONConfigurationParser()

        formatted = printer.format(original)
        reparsed = parser.parse(formatted)

        assert configs_equivalent(original, reparsed)

    def test_custom_values_json_roundtrip(self):
        """Non-default field values survive a JSON round-trip."""
        original = Configuration(
            scan_exclusion_patterns=["*.tmp", "/proc"],
            scan_timeout_seconds=600,
            max_applications_per_scan=500,
            analysis_timeout_seconds=120,
            min_confidence_threshold=0.75,
            priority_weights=PriorityWeights(
                usage=0.4, complexity=0.3, security=0.2, performance=0.1
            ),
            respect_access_controls=False,
            encrypt_stored_data=False,
            audit_logging_enabled=False,
            max_concurrent_analyses=8,
            cache_analysis_results=False,
            cache_expiry_hours=48,
            report_formats=["csv", "xml"],
            include_charts=False,
            sanitize_paths=False,
        )
        printer = JSONConfigurationPrettyPrinter()
        parser = JSONConfigurationParser()

        formatted = printer.format(original)
        reparsed = parser.parse(formatted)

        assert configs_equivalent(original, reparsed)

    def test_json_roundtrip_from_json_string(self):
        """Parse a JSON string, format it, parse again — both configs are equivalent."""
        json_text = json.dumps({
            "scan_exclusion_patterns": ["/tmp"],
            "scan_timeout_seconds": 120,
            "max_applications_per_scan": 200,
            "capability_detection_rules": {},
            "analysis_timeout_seconds": 30,
            "min_confidence_threshold": 0.6,
            "priority_weights": {
                "usage": 0.35,
                "complexity": 0.25,
                "security": 0.2,
                "performance": 0.2,
            },
            "respect_access_controls": True,
            "encrypt_stored_data": True,
            "audit_logging_enabled": True,
            "max_concurrent_analyses": 2,
            "cache_analysis_results": True,
            "cache_expiry_hours": 12,
            "report_formats": ["json"],
            "include_charts": True,
            "sanitize_paths": True,
        })
        parser = JSONConfigurationParser()
        printer = JSONConfigurationPrettyPrinter()

        first_parse = parser.parse(json_text)
        formatted = printer.format(first_parse)
        second_parse = parser.parse(formatted)

        assert configs_equivalent(first_parse, second_parse)

    def test_json_output_is_valid_json(self):
        """The formatted JSON output must be parseable by the standard json module."""
        config = Configuration()
        printer = JSONConfigurationPrettyPrinter()
        formatted = printer.format(config)

        # Should not raise
        parsed = json.loads(formatted)
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# YAML round-trip tests
# ---------------------------------------------------------------------------

class TestYAMLRoundTrip:
    """Tests for YAML parse → format → parse round-trip integrity."""

    def test_default_config_yaml_roundtrip(self):
        """A default Configuration survives a YAML round-trip unchanged."""
        original = Configuration()
        printer = YAMLConfigurationPrettyPrinter()
        parser = YAMLConfigurationParser()

        formatted = printer.format(original)
        reparsed = parser.parse(formatted)

        assert configs_equivalent(original, reparsed)

    def test_custom_values_yaml_roundtrip(self):
        """Non-default field values survive a YAML round-trip."""
        original = Configuration(
            scan_exclusion_patterns=["*.log", "/dev"],
            scan_timeout_seconds=450,
            max_applications_per_scan=2000,
            analysis_timeout_seconds=90,
            min_confidence_threshold=0.8,
            priority_weights=PriorityWeights(
                usage=0.5, complexity=0.2, security=0.2, performance=0.1
            ),
            respect_access_controls=True,
            encrypt_stored_data=False,
            audit_logging_enabled=True,
            max_concurrent_analyses=6,
            cache_analysis_results=True,
            cache_expiry_hours=72,
            report_formats=["html", "csv"],
            include_charts=True,
            sanitize_paths=False,
        )
        printer = YAMLConfigurationPrettyPrinter()
        parser = YAMLConfigurationParser()

        formatted = printer.format(original)
        reparsed = parser.parse(formatted)

        assert configs_equivalent(original, reparsed)

    def test_yaml_roundtrip_from_yaml_string(self):
        """Parse a YAML string, format it, parse again — both configs are equivalent."""
        yaml_text = """\
scan_exclusion_patterns:
  - /var/log
scan_timeout_seconds: 180
max_applications_per_scan: 300
capability_detection_rules: {}
analysis_timeout_seconds: 45
min_confidence_threshold: 0.55
priority_weights:
  usage: 0.3
  complexity: 0.2
  security: 0.25
  performance: 0.25
respect_access_controls: true
encrypt_stored_data: true
audit_logging_enabled: false
max_concurrent_analyses: 3
cache_analysis_results: true
cache_expiry_hours: 6
report_formats:
  - json
  - html
include_charts: false
sanitize_paths: true
"""
        parser = YAMLConfigurationParser()
        printer = YAMLConfigurationPrettyPrinter()

        first_parse = parser.parse(yaml_text)
        formatted = printer.format(first_parse)
        second_parse = parser.parse(formatted)

        assert configs_equivalent(first_parse, second_parse)


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestInvalidInputErrors:
    """Tests that invalid input raises ValueError with descriptive messages."""

    def test_invalid_json_raises_value_error(self):
        """Malformed JSON raises ValueError with a descriptive message."""
        parser = JSONConfigurationParser()
        with pytest.raises(ValueError) as exc_info:
            parser.parse("{not valid json}")
        assert "Invalid JSON" in str(exc_info.value) or "JSON" in str(exc_info.value)

    def test_invalid_yaml_raises_value_error(self):
        """Malformed YAML raises ValueError with a descriptive message."""
        parser = YAMLConfigurationParser()
        # Tabs are not allowed in YAML indentation
        bad_yaml = "key:\n\t- value"
        with pytest.raises(ValueError) as exc_info:
            parser.parse(bad_yaml)
        assert "YAML" in str(exc_info.value) or "yaml" in str(exc_info.value).lower()

    def test_json_with_invalid_field_values_raises_value_error(self):
        """JSON with out-of-range field values raises ValueError."""
        parser = JSONConfigurationParser()
        bad_json = json.dumps({
            "scan_timeout_seconds": -1,  # must be positive
            "max_applications_per_scan": 10000,
            "analysis_timeout_seconds": 60,
            "min_confidence_threshold": 0.5,
            "priority_weights": {"usage": 0.3, "complexity": 0.2, "security": 0.25, "performance": 0.25},
            "max_concurrent_analyses": 4,
            "cache_expiry_hours": 24,
        })
        with pytest.raises(ValueError):
            parser.parse(bad_json)


# ---------------------------------------------------------------------------
# ConfigurationManager.parse_file() tests
# ---------------------------------------------------------------------------

class TestConfigurationManagerParseFile:
    """Tests for ConfigurationManager.parse_file() extension-based dispatch."""

    def _write_temp_file(self, content: str, suffix: str) -> Path:
        """Write content to a named temp file and return its Path."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, encoding="utf-8"
        )
        tmp.write(content)
        tmp.flush()
        tmp.close()
        return Path(tmp.name)

    def test_parse_file_json_extension(self):
        """parse_file() with .json extension parses successfully."""
        config = Configuration()
        printer = JSONConfigurationPrettyPrinter()
        json_text = printer.format(config)

        path = self._write_temp_file(json_text, ".json")
        try:
            manager = ConfigurationManager()
            result = manager.parse_file(path)
            assert configs_equivalent(config, result)
        finally:
            path.unlink(missing_ok=True)

    def test_parse_file_yaml_extension(self):
        """parse_file() with .yaml extension parses successfully."""
        config = Configuration()
        printer = YAMLConfigurationPrettyPrinter()
        yaml_text = printer.format(config)

        path = self._write_temp_file(yaml_text, ".yaml")
        try:
            manager = ConfigurationManager()
            result = manager.parse_file(path)
            assert configs_equivalent(config, result)
        finally:
            path.unlink(missing_ok=True)

    def test_parse_file_yml_extension(self):
        """parse_file() with .yml extension parses successfully."""
        config = Configuration()
        printer = YAMLConfigurationPrettyPrinter()
        yaml_text = printer.format(config)

        path = self._write_temp_file(yaml_text, ".yml")
        try:
            manager = ConfigurationManager()
            result = manager.parse_file(path)
            assert configs_equivalent(config, result)
        finally:
            path.unlink(missing_ok=True)

    def test_parse_file_unsupported_extension_raises_value_error(self):
        """parse_file() with an unsupported extension raises ValueError."""
        path = self._write_temp_file("{}", ".toml")
        try:
            manager = ConfigurationManager()
            with pytest.raises(ValueError) as exc_info:
                manager.parse_file(path)
            assert ".toml" in str(exc_info.value) or "Unsupported" in str(exc_info.value)
        finally:
            path.unlink(missing_ok=True)

    def test_parse_file_missing_file_raises_file_not_found(self):
        """parse_file() with a non-existent path raises FileNotFoundError."""
        manager = ConfigurationManager()
        with pytest.raises((FileNotFoundError, ValueError)):
            manager.parse_file(Path("/nonexistent/path/config.json"))
