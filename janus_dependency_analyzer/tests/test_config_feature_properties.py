"""
Property-based tests for Configuration Feature Support.

This module implements property-based tests using Hypothesis to verify
universal correctness properties of configuration features including
exclusion patterns, capability detection rules, and priority weighting.

# Feature: janus-dependency-analyzer, Property 10: Configuration Feature Support
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ..core.models import Configuration, PriorityWeights
from ..config.configuration import (
    ConfigurationManager,
    CapabilityDetectionRule,
    PriorityWeightConfig,
    matches_exclusion_pattern,
)
from ..core.configuration import (
    JSONConfigurationParser,
    JSONConfigurationPrettyPrinter,
)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Strategy for exclusion patterns (valid glob-like strings)
exclusion_pattern_strategy = st.text(
    min_size=1,
    max_size=40,
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters="*?/._-",
    ),
)

# Strategy for CapabilityDetectionRule
capability_rule_strategy = st.builds(
    CapabilityDetectionRule,
    name=st.text(
        min_size=1,
        max_size=20,
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"),
            whitelist_characters="_",
        ),
    ),
    pattern=st.text(
        min_size=1,
        max_size=30,
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"),
            whitelist_characters="*?._-",
        ),
    ),
    category=st.sampled_from(
        [
            "file_processing",
            "network_operations",
            "data_transformation",
            "multimedia",
            "database",
            "security",
        ]
    ),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    description=st.text(max_size=50),
)

# Strategy for PriorityWeightConfig
weight_config_strategy = st.builds(
    PriorityWeightConfig,
    usage=st.floats(min_value=0.01, max_value=2.0, allow_nan=False),
    complexity=st.floats(min_value=0.01, max_value=2.0, allow_nan=False),
    security=st.floats(min_value=0.01, max_value=2.0, allow_nan=False),
    performance=st.floats(min_value=0.01, max_value=2.0, allow_nan=False),
)


# ---------------------------------------------------------------------------
# Property 10: Configuration Feature Support
# Feature: janus-dependency-analyzer, Property 10: Configuration Feature Support
# Validates: Requirements 6.5, 6.6, 6.7
# ---------------------------------------------------------------------------


class TestConfigurationFeatureSupport:
    """
    **Validates: Requirements 6.5, 6.6, 6.7**

    Property 10: Configuration Feature Support
    For any configuration with exclusion patterns, custom capability detection
    rules, or priority weighting factors, the Configuration_Parser SHALL handle
    all features correctly and maintain their semantics.
    """

    # Feature: janus-dependency-analyzer, Property 10: Configuration Feature Support

    @given(patterns=st.lists(exclusion_pattern_strategy, min_size=1, max_size=5))
    @settings(max_examples=25, deadline=None)
    def test_exclusion_patterns_preserved_in_config(self, patterns: list) -> None:
        """
        For any list of exclusion patterns, they are stored and retrievable from Configuration.

        # Feature: janus-dependency-analyzer, Property 10: Configuration Feature Support
        **Validates: Requirements 6.5**
        """
        config = Configuration(scan_exclusion_patterns=patterns)
        assert config.scan_exclusion_patterns == patterns, (
            f"Exclusion patterns not preserved.\n"
            f"Expected: {patterns}\n"
            f"Got: {config.scan_exclusion_patterns}"
        )

    @given(
        patterns=st.lists(exclusion_pattern_strategy, min_size=1, max_size=3),
        path=st.text(min_size=1, max_size=50),
    )
    @settings(max_examples=25, deadline=None)
    def test_path_exclusion_is_consistent(self, patterns: list, path: str) -> None:
        """
        is_path_excluded is consistent with matches_exclusion_pattern.

        ConfigurationManager.is_path_excluded(path, config) must equal
        matches_exclusion_pattern(path, patterns) for any path and patterns.

        # Feature: janus-dependency-analyzer, Property 10: Configuration Feature Support
        **Validates: Requirements 6.5**
        """
        config = Configuration(scan_exclusion_patterns=patterns)
        manager = ConfigurationManager()

        manager_result = manager.is_path_excluded(path, config)
        helper_result = matches_exclusion_pattern(path, patterns)

        assert manager_result == helper_result, (
            f"is_path_excluded and matches_exclusion_pattern disagree.\n"
            f"Path: {path!r}\n"
            f"Patterns: {patterns}\n"
            f"manager.is_path_excluded: {manager_result}\n"
            f"matches_exclusion_pattern: {helper_result}"
        )

    @given(rules=st.lists(capability_rule_strategy, min_size=1, max_size=5))
    @settings(max_examples=25, deadline=None)
    def test_capability_rules_roundtrip(self, rules: list) -> None:
        """
        For any list of CapabilityDetectionRules, apply then get produces equivalent rules.

        apply_capability_rules followed by get_capability_rules SHALL produce rules
        whose names, patterns, categories, and confidences match the originals.
        Rules with duplicate names are deduplicated (last wins).

        # Feature: janus-dependency-analyzer, Property 10: Configuration Feature Support
        **Validates: Requirements 6.6**
        """
        config = Configuration()
        manager = ConfigurationManager()

        # Apply rules
        manager.apply_capability_rules(config, rules)

        # Build expected: last-wins deduplication by name
        expected: dict = {}
        for rule in rules:
            expected[rule.name] = rule

        # Retrieve rules back
        retrieved = manager.get_capability_rules(config)
        retrieved_by_name = {r.name: r for r in retrieved}

        assert set(retrieved_by_name.keys()) == set(expected.keys()), (
            f"Rule names mismatch after round-trip.\n"
            f"Expected names: {set(expected.keys())}\n"
            f"Got names: {set(retrieved_by_name.keys())}"
        )

        for name, orig in expected.items():
            got = retrieved_by_name[name]
            assert got.pattern == orig.pattern, (
                f"Pattern mismatch for rule '{name}': expected {orig.pattern!r}, got {got.pattern!r}"
            )
            assert got.category == orig.category, (
                f"Category mismatch for rule '{name}': expected {orig.category!r}, got {got.category!r}"
            )
            assert abs(got.confidence - orig.confidence) < 1e-9, (
                f"Confidence mismatch for rule '{name}': expected {orig.confidence}, got {got.confidence}"
            )

    @given(weight_config=weight_config_strategy)
    @settings(max_examples=25, deadline=None)
    def test_priority_weights_roundtrip(self, weight_config: PriorityWeightConfig) -> None:
        """
        For any PriorityWeightConfig, update then get produces equivalent weights.

        update_priority_weights followed by get_priority_weight_config SHALL
        produce a PriorityWeightConfig whose values match the original within 1e-9.

        # Feature: janus-dependency-analyzer, Property 10: Configuration Feature Support
        **Validates: Requirements 6.7**
        """
        config = Configuration()
        manager = ConfigurationManager()

        manager.update_priority_weights(config, weight_config)
        retrieved = manager.get_priority_weight_config(config)

        assert abs(retrieved.usage - weight_config.usage) < 1e-9, (
            f"usage mismatch: expected {weight_config.usage}, got {retrieved.usage}"
        )
        assert abs(retrieved.complexity - weight_config.complexity) < 1e-9, (
            f"complexity mismatch: expected {weight_config.complexity}, got {retrieved.complexity}"
        )
        assert abs(retrieved.security - weight_config.security) < 1e-9, (
            f"security mismatch: expected {weight_config.security}, got {retrieved.security}"
        )
        assert abs(retrieved.performance - weight_config.performance) < 1e-9, (
            f"performance mismatch: expected {weight_config.performance}, got {retrieved.performance}"
        )

    @given(weight_config=weight_config_strategy)
    @settings(max_examples=25, deadline=None)
    def test_weight_config_to_priority_weights_consistency(
        self, weight_config: PriorityWeightConfig
    ) -> None:
        """
        PriorityWeightConfig.to_priority_weights() preserves all weight values.

        Converting a PriorityWeightConfig to PriorityWeights SHALL produce an
        object whose usage, complexity, security, and performance values match
        the original within 1e-9.

        # Feature: janus-dependency-analyzer, Property 10: Configuration Feature Support
        **Validates: Requirements 6.7**
        """
        pw = weight_config.to_priority_weights()

        assert abs(pw.usage - weight_config.usage) < 1e-9, (
            f"usage mismatch: expected {weight_config.usage}, got {pw.usage}"
        )
        assert abs(pw.complexity - weight_config.complexity) < 1e-9, (
            f"complexity mismatch: expected {weight_config.complexity}, got {pw.complexity}"
        )
        assert abs(pw.security - weight_config.security) < 1e-9, (
            f"security mismatch: expected {weight_config.security}, got {pw.security}"
        )
        assert abs(pw.performance - weight_config.performance) < 1e-9, (
            f"performance mismatch: expected {weight_config.performance}, got {pw.performance}"
        )

    @given(patterns=st.lists(exclusion_pattern_strategy, max_size=5))
    @settings(max_examples=25, deadline=None)
    def test_exclusion_patterns_survive_json_roundtrip(self, patterns: list) -> None:
        """
        Exclusion patterns are preserved through JSON format→parse round-trip.

        For any list of exclusion patterns, formatting a Configuration to JSON
        and then parsing it back SHALL produce a Configuration with identical
        scan_exclusion_patterns.

        # Feature: janus-dependency-analyzer, Property 10: Configuration Feature Support
        **Validates: Requirements 6.5**
        """
        config = Configuration(scan_exclusion_patterns=patterns)
        printer = JSONConfigurationPrettyPrinter()
        parser = JSONConfigurationParser()

        json_text = printer.format(config)
        reparsed = parser.parse(json_text)

        assert reparsed.scan_exclusion_patterns == patterns, (
            f"Exclusion patterns changed after JSON round-trip.\n"
            f"Original: {patterns}\n"
            f"Reparsed: {reparsed.scan_exclusion_patterns}"
        )
