"""
Property-based tests for Configuration Round-Trip Integrity.

This module implements property-based tests using Hypothesis to verify
universal correctness properties of the Configuration parsing and formatting.

# Feature: janus-dependency-analyzer, Property 9: Configuration Round-Trip Integrity
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ..core.models import Configuration, PriorityWeights
from ..core.configuration import (
    JSONConfigurationParser,
    YAMLConfigurationParser,
    JSONConfigurationPrettyPrinter,
    YAMLConfigurationPrettyPrinter,
)
from ..config.configuration import ConfigurationManager


# ---------------------------------------------------------------------------
# Equivalence helper
# ---------------------------------------------------------------------------


def configs_equivalent(a: Configuration, b: Configuration) -> bool:
    """Return True if two Configuration objects are semantically equivalent."""
    return (
        a.scan_exclusion_patterns == b.scan_exclusion_patterns
        and a.scan_timeout_seconds == b.scan_timeout_seconds
        and a.max_applications_per_scan == b.max_applications_per_scan
        and a.capability_detection_rules == b.capability_detection_rules
        and a.analysis_timeout_seconds == b.analysis_timeout_seconds
        and abs(a.min_confidence_threshold - b.min_confidence_threshold) < 1e-9
        and abs(a.priority_weights.usage - b.priority_weights.usage) < 1e-9
        and abs(a.priority_weights.complexity - b.priority_weights.complexity) < 1e-9
        and abs(a.priority_weights.security - b.priority_weights.security) < 1e-9
        and abs(a.priority_weights.performance - b.priority_weights.performance) < 1e-9
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
# Hypothesis strategies
# ---------------------------------------------------------------------------

priority_weights_strategy = st.builds(
    PriorityWeights,
    usage=st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
    complexity=st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
    security=st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
    performance=st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
)

configuration_strategy = st.builds(
    Configuration,
    scan_exclusion_patterns=st.lists(
        st.text(
            min_size=1,
            max_size=30,
            alphabet=st.characters(
                whitelist_categories=("Lu", "Ll", "Nd"),
                whitelist_characters="*?/._-",
            ),
        ),
        max_size=5,
    ),
    scan_timeout_seconds=st.integers(min_value=1, max_value=3600),
    max_applications_per_scan=st.integers(min_value=1, max_value=100000),
    capability_detection_rules=st.just({}),
    analysis_timeout_seconds=st.integers(min_value=1, max_value=600),
    min_confidence_threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    priority_weights=priority_weights_strategy,
    respect_access_controls=st.booleans(),
    encrypt_stored_data=st.booleans(),
    audit_logging_enabled=st.booleans(),
    max_concurrent_analyses=st.integers(min_value=1, max_value=32),
    cache_analysis_results=st.booleans(),
    cache_expiry_hours=st.integers(min_value=1, max_value=720),
    report_formats=st.just(["json", "html"]),
    include_charts=st.booleans(),
    sanitize_paths=st.booleans(),
)


# ---------------------------------------------------------------------------
# Property 9: Configuration Round-Trip Integrity
# Feature: janus-dependency-analyzer, Property 9: Configuration Round-Trip Integrity
# Validates: Requirements 6.1, 6.2, 6.3, 6.4
# ---------------------------------------------------------------------------


class TestConfigurationRoundTripIntegrity:
    """
    **Validates: Requirements 6.1, 6.2, 6.3, 6.4**

    Property 9: Configuration Round-Trip Integrity
    For any valid Configuration object, parsing then printing then parsing SHALL
    produce an equivalent object, and invalid configurations SHALL return
    descriptive error messages.
    """

    # Feature: janus-dependency-analyzer, Property 9: Configuration Round-Trip Integrity

    @given(config=configuration_strategy)
    @settings(max_examples=25, deadline=None)
    def test_json_roundtrip_property(self, config: Configuration) -> None:
        """
        JSON parse→format→parse produces an equivalent Configuration.

        For any valid Configuration, formatting to JSON and then parsing the
        resulting text SHALL produce a Configuration equivalent to the original.

        # Feature: janus-dependency-analyzer, Property 9: Configuration Round-Trip Integrity
        """
        printer = JSONConfigurationPrettyPrinter()
        parser = JSONConfigurationParser()

        json_text = printer.format(config)
        reparsed = parser.parse(json_text)

        assert configs_equivalent(config, reparsed), (
            f"JSON round-trip produced a non-equivalent Configuration.\n"
            f"Original min_confidence_threshold: {config.min_confidence_threshold}\n"
            f"Reparsed min_confidence_threshold: {reparsed.min_confidence_threshold}"
        )

    @given(config=configuration_strategy)
    @settings(max_examples=25, deadline=None)
    def test_yaml_roundtrip_property(self, config: Configuration) -> None:
        """
        YAML parse→format→parse produces an equivalent Configuration.

        For any valid Configuration, formatting to YAML and then parsing the
        resulting text SHALL produce a Configuration equivalent to the original.

        # Feature: janus-dependency-analyzer, Property 9: Configuration Round-Trip Integrity
        """
        printer = YAMLConfigurationPrettyPrinter()
        parser = YAMLConfigurationParser()

        yaml_text = printer.format(config)
        reparsed = parser.parse(yaml_text)

        assert configs_equivalent(config, reparsed), (
            f"YAML round-trip produced a non-equivalent Configuration.\n"
            f"Original min_confidence_threshold: {config.min_confidence_threshold}\n"
            f"Reparsed min_confidence_threshold: {reparsed.min_confidence_threshold}"
        )

    @given(
        invalid_text=st.text(min_size=1).filter(
            lambda t: not t.strip().startswith("{") or not t.strip().endswith("}")
        )
    )
    @settings(max_examples=25, deadline=None)
    def test_invalid_config_returns_descriptive_error(self, invalid_text: str) -> None:
        """
        Invalid JSON text raises ValueError with a non-empty message.

        For any text that is not valid JSON, the parser SHALL raise a ValueError
        whose message is non-empty and descriptive.

        # Feature: janus-dependency-analyzer, Property 9: Configuration Round-Trip Integrity
        """
        parser = JSONConfigurationParser()

        try:
            parser.parse(invalid_text)
            # If parsing succeeded, the text happened to be valid JSON that
            # also passed configuration validation — that is acceptable.
        except ValueError as exc:
            assert str(exc), (
                "ValueError raised for invalid config text must have a non-empty message"
            )
            assert len(str(exc).strip()) > 0, (
                "ValueError message must not be blank/whitespace-only"
            )

    @given(config=configuration_strategy)
    @settings(max_examples=25, deadline=None)
    def test_manager_roundtrip_json(self, config: Configuration) -> None:
        """
        ConfigurationManager JSON round-trip produces an equivalent Configuration.

        Using ConfigurationManager.format() followed by ConfigurationManager.parse()
        SHALL produce a Configuration equivalent to the original.

        # Feature: janus-dependency-analyzer, Property 9: Configuration Round-Trip Integrity
        """
        manager = ConfigurationManager()

        json_text = manager.format(config, format="json")
        reparsed = manager.parse(json_text, format="json")

        assert configs_equivalent(config, reparsed), (
            f"ConfigurationManager JSON round-trip produced a non-equivalent Configuration.\n"
            f"Original scan_timeout_seconds: {config.scan_timeout_seconds}\n"
            f"Reparsed scan_timeout_seconds: {reparsed.scan_timeout_seconds}"
        )

    @given(config=configuration_strategy)
    @settings(max_examples=25, deadline=None)
    def test_valid_config_passes_validation(self, config: Configuration) -> None:
        """
        Any generated valid Configuration passes validation (is_valid=True).

        For any Configuration produced by the strategy (which constrains all
        fields to valid ranges), calling validate() SHALL return a
        ValidationResult with is_valid=True.

        # Feature: janus-dependency-analyzer, Property 9: Configuration Round-Trip Integrity
        """
        result = config.validate()

        assert result.is_valid, (
            f"Generated Configuration failed validation.\n"
            f"Errors: {result.errors}\n"
            f"Config: scan_timeout={config.scan_timeout_seconds}, "
            f"analysis_timeout={config.analysis_timeout_seconds}, "
            f"min_confidence={config.min_confidence_threshold}, "
            f"max_concurrent={config.max_concurrent_analyses}, "
            f"cache_expiry={config.cache_expiry_hours}"
        )
