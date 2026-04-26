"""
Unit tests for advanced configuration features (Requirements 6.5, 6.6, 6.7).

Covers:
- matches_exclusion_pattern() glob helper
- ConfigurationManager.is_path_excluded()
- ConfigurationManager.get_capability_rules() round-trip with apply_capability_rules()
- ConfigurationManager.get_priority_weight_config()
- ConfigurationManager.update_priority_weights()
- PriorityWeightConfig.to_priority_weights() normalization
"""

import pytest

from janus_dependency_analyzer.config import (
    CapabilityDetectionRule,
    Configuration,
    ConfigurationManager,
    PriorityWeightConfig,
    matches_exclusion_pattern,
)
from janus_dependency_analyzer.core.models import PriorityWeights


# ---------------------------------------------------------------------------
# matches_exclusion_pattern
# ---------------------------------------------------------------------------

class TestMatchesExclusionPattern:
    """Tests for the module-level matches_exclusion_pattern() helper."""

    def test_wildcard_extension_matches(self):
        """*.tmp matches a file with .tmp extension."""
        assert matches_exclusion_pattern("cache/file.tmp", ["*.tmp"]) is True

    def test_wildcard_extension_no_match(self):
        """*.tmp does not match a .log file."""
        assert matches_exclusion_pattern("cache/file.log", ["*.tmp"]) is False

    def test_directory_glob_matches(self):
        """/proc/* matches a path inside /proc."""
        assert matches_exclusion_pattern("/proc/1234", ["/proc/*"]) is True

    def test_directory_glob_no_match(self):
        """/proc/* does not match a path outside /proc."""
        assert matches_exclusion_pattern("/usr/bin/python", ["/proc/*"]) is False

    def test_exact_match(self):
        """An exact pattern matches the identical path."""
        assert matches_exclusion_pattern("/etc/passwd", ["/etc/passwd"]) is True

    def test_exact_no_match(self):
        """An exact pattern does not match a different path."""
        assert matches_exclusion_pattern("/etc/shadow", ["/etc/passwd"]) is False

    def test_empty_patterns_never_match(self):
        """An empty pattern list never matches any path."""
        assert matches_exclusion_pattern("/any/path", []) is False

    def test_first_matching_pattern_is_sufficient(self):
        """Returns True as soon as one pattern matches, even if others don't."""
        assert matches_exclusion_pattern("file.bak", ["*.tmp", "*.bak", "*.log"]) is True

    def test_no_pattern_matches_returns_false(self):
        """Returns False when no pattern in the list matches."""
        assert matches_exclusion_pattern("file.py", ["*.tmp", "*.bak"]) is False

    def test_question_mark_wildcard(self):
        """? matches exactly one character."""
        assert matches_exclusion_pattern("file1.txt", ["file?.txt"]) is True
        assert matches_exclusion_pattern("file12.txt", ["file?.txt"]) is False


# ---------------------------------------------------------------------------
# ConfigurationManager.is_path_excluded
# ---------------------------------------------------------------------------

class TestIsPathExcluded:
    """Tests for ConfigurationManager.is_path_excluded()."""

    def _manager(self) -> ConfigurationManager:
        return ConfigurationManager()

    def test_excluded_path_returns_true(self):
        """A path matching a pattern in config is excluded."""
        config = Configuration(scan_exclusion_patterns=["*.tmp", "/proc/*"])
        manager = self._manager()
        assert manager.is_path_excluded("/proc/1234", config) is True

    def test_non_excluded_path_returns_false(self):
        """A path not matching any pattern is not excluded."""
        config = Configuration(scan_exclusion_patterns=["*.tmp"])
        manager = self._manager()
        assert manager.is_path_excluded("/usr/bin/python3", config) is False

    def test_empty_patterns_never_excludes(self):
        """With no exclusion patterns, no path is excluded."""
        config = Configuration(scan_exclusion_patterns=[])
        manager = self._manager()
        assert manager.is_path_excluded("/any/path", config) is False

    def test_multiple_patterns_any_match_excludes(self):
        """A path matching any one of multiple patterns is excluded."""
        config = Configuration(scan_exclusion_patterns=["*.log", "*.bak", "/tmp/*"])
        manager = self._manager()
        assert manager.is_path_excluded("server.log", config) is True
        assert manager.is_path_excluded("backup.bak", config) is True
        assert manager.is_path_excluded("/tmp/scratch", config) is True
        assert manager.is_path_excluded("/usr/bin/ls", config) is False


# ---------------------------------------------------------------------------
# ConfigurationManager.get_capability_rules (round-trip with apply_capability_rules)
# ---------------------------------------------------------------------------

class TestGetCapabilityRules:
    """Tests for the apply_capability_rules → get_capability_rules round-trip."""

    def _manager(self) -> ConfigurationManager:
        return ConfigurationManager()

    def test_roundtrip_single_rule(self):
        """A single rule applied then retrieved is identical."""
        manager = self._manager()
        config = Configuration()
        rule = CapabilityDetectionRule(
            name="image_editor",
            pattern="*image*",
            category="multimedia",
            confidence=0.85,
            description="Detects image editing capabilities",
        )
        manager.apply_capability_rules(config, [rule])
        retrieved = manager.get_capability_rules(config)

        assert len(retrieved) == 1
        r = retrieved[0]
        assert r.name == rule.name
        assert r.pattern == rule.pattern
        assert r.category == rule.category
        assert r.confidence == rule.confidence
        assert r.description == rule.description

    def test_roundtrip_multiple_rules(self):
        """Multiple rules survive the apply → get round-trip."""
        manager = self._manager()
        config = Configuration()
        rules = [
            CapabilityDetectionRule(
                name="video_encoder",
                pattern="*video*",
                category="multimedia",
                confidence=0.9,
                description="Video encoding",
            ),
            CapabilityDetectionRule(
                name="db_client",
                pattern="*sql*",
                category="database",
                confidence=0.75,
                description="Database client",
            ),
        ]
        manager.apply_capability_rules(config, rules)
        retrieved = manager.get_capability_rules(config)

        assert len(retrieved) == 2
        retrieved_by_name = {r.name: r for r in retrieved}
        for original in rules:
            assert original.name in retrieved_by_name
            r = retrieved_by_name[original.name]
            assert r.pattern == original.pattern
            assert r.category == original.category
            assert r.confidence == original.confidence
            assert r.description == original.description

    def test_empty_rules_returns_empty_list(self):
        """get_capability_rules on a config with no rules returns an empty list."""
        manager = self._manager()
        config = Configuration()
        assert manager.get_capability_rules(config) == []

    def test_apply_overwrites_existing_rule(self):
        """Applying a rule with the same name overwrites the previous entry."""
        manager = self._manager()
        config = Configuration()
        original_rule = CapabilityDetectionRule(
            name="my_rule", pattern="old_pattern", category="cat_a", confidence=0.5
        )
        updated_rule = CapabilityDetectionRule(
            name="my_rule", pattern="new_pattern", category="cat_b", confidence=0.9
        )
        manager.apply_capability_rules(config, [original_rule])
        manager.apply_capability_rules(config, [updated_rule])
        retrieved = manager.get_capability_rules(config)

        assert len(retrieved) == 1
        assert retrieved[0].pattern == "new_pattern"
        assert retrieved[0].category == "cat_b"
        assert retrieved[0].confidence == 0.9


# ---------------------------------------------------------------------------
# ConfigurationManager.get_priority_weight_config
# ---------------------------------------------------------------------------

class TestGetPriorityWeightConfig:
    """Tests for ConfigurationManager.get_priority_weight_config()."""

    def _manager(self) -> ConfigurationManager:
        return ConfigurationManager()

    def test_default_weights_extracted_correctly(self):
        """Default PriorityWeights are reflected in the returned PriorityWeightConfig."""
        manager = self._manager()
        config = Configuration()  # uses default PriorityWeights
        weight_config = manager.get_priority_weight_config(config)

        assert weight_config.usage == config.priority_weights.usage
        assert weight_config.complexity == config.priority_weights.complexity
        assert weight_config.security == config.priority_weights.security
        assert weight_config.performance == config.priority_weights.performance

    def test_custom_weights_extracted_correctly(self):
        """Custom PriorityWeights are reflected in the returned PriorityWeightConfig."""
        manager = self._manager()
        config = Configuration(
            priority_weights=PriorityWeights(
                usage=0.5, complexity=0.1, security=0.3, performance=0.1
            )
        )
        weight_config = manager.get_priority_weight_config(config)

        assert weight_config.usage == 0.5
        assert weight_config.complexity == 0.1
        assert weight_config.security == 0.3
        assert weight_config.performance == 0.1

    def test_returns_priority_weight_config_instance(self):
        """The return type is PriorityWeightConfig."""
        manager = self._manager()
        config = Configuration()
        result = manager.get_priority_weight_config(config)
        assert isinstance(result, PriorityWeightConfig)


# ---------------------------------------------------------------------------
# ConfigurationManager.update_priority_weights
# ---------------------------------------------------------------------------

class TestUpdatePriorityWeights:
    """Tests for ConfigurationManager.update_priority_weights()."""

    def _manager(self) -> ConfigurationManager:
        return ConfigurationManager()

    def test_weights_updated_in_config(self):
        """update_priority_weights() changes the config's priority_weights."""
        manager = self._manager()
        config = Configuration()
        new_weights = PriorityWeightConfig(
            usage=0.4, complexity=0.3, security=0.2, performance=0.1
        )
        result = manager.update_priority_weights(config, new_weights)

        assert result.priority_weights.usage == 0.4
        assert result.priority_weights.complexity == 0.3
        assert result.priority_weights.security == 0.2
        assert result.priority_weights.performance == 0.1

    def test_returns_same_config_object(self):
        """update_priority_weights() returns the same Configuration instance."""
        manager = self._manager()
        config = Configuration()
        weight_config = PriorityWeightConfig()
        result = manager.update_priority_weights(config, weight_config)
        assert result is config

    def test_roundtrip_get_then_update(self):
        """get_priority_weight_config → modify → update_priority_weights is consistent."""
        manager = self._manager()
        config = Configuration(
            priority_weights=PriorityWeights(
                usage=0.25, complexity=0.25, security=0.25, performance=0.25
            )
        )
        weight_config = manager.get_priority_weight_config(config)
        # Modify the weight config
        weight_config.usage = 0.6
        weight_config.complexity = 0.1
        weight_config.security = 0.2
        weight_config.performance = 0.1

        manager.update_priority_weights(config, weight_config)

        assert config.priority_weights.usage == 0.6
        assert config.priority_weights.complexity == 0.1
        assert config.priority_weights.security == 0.2
        assert config.priority_weights.performance == 0.1


# ---------------------------------------------------------------------------
# PriorityWeightConfig.to_priority_weights normalization
# ---------------------------------------------------------------------------

class TestPriorityWeightConfigNormalization:
    """Tests for PriorityWeightConfig.to_priority_weights() and PriorityWeights.normalize()."""

    def test_to_priority_weights_preserves_values(self):
        """to_priority_weights() produces a PriorityWeights with the same values."""
        wc = PriorityWeightConfig(usage=0.4, complexity=0.3, security=0.2, performance=0.1)
        pw = wc.to_priority_weights()

        assert isinstance(pw, PriorityWeights)
        assert pw.usage == 0.4
        assert pw.complexity == 0.3
        assert pw.security == 0.2
        assert pw.performance == 0.1

    def test_normalize_sums_to_one(self):
        """PriorityWeights.normalize() produces weights that sum to 1.0."""
        pw = PriorityWeights(usage=1.0, complexity=1.0, security=1.0, performance=1.0)
        normalized = pw.normalize()
        total = (
            normalized.usage
            + normalized.complexity
            + normalized.security
            + normalized.performance
        )
        assert abs(total - 1.0) < 1e-9

    def test_normalize_preserves_ratios(self):
        """normalize() preserves the relative ratios between weights."""
        pw = PriorityWeights(usage=2.0, complexity=1.0, security=1.0, performance=0.0)
        normalized = pw.normalize()
        # usage should be twice complexity
        assert abs(normalized.usage - 2 * normalized.complexity) < 1e-9

    def test_normalize_zero_weights_returns_defaults(self):
        """normalize() on all-zero weights returns the default PriorityWeights."""
        pw = PriorityWeights(usage=0.0, complexity=0.0, security=0.0, performance=0.0)
        normalized = pw.normalize()
        default = PriorityWeights()
        assert normalized.usage == default.usage
        assert normalized.complexity == default.complexity
        assert normalized.security == default.security
        assert normalized.performance == default.performance

    def test_weight_config_to_priority_weights_then_normalize(self):
        """Full pipeline: PriorityWeightConfig → PriorityWeights → normalize."""
        wc = PriorityWeightConfig(usage=3.0, complexity=1.0, security=1.0, performance=1.0)
        pw = wc.to_priority_weights()
        normalized = pw.normalize()
        total = (
            normalized.usage
            + normalized.complexity
            + normalized.security
            + normalized.performance
        )
        assert abs(total - 1.0) < 1e-9
        # usage weight should be 3× each of the others
        assert abs(normalized.usage - 3 * normalized.complexity) < 1e-9
