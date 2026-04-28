"""
Configuration management for the Janus Dependency Analyzer.

This module provides high-level configuration management including
CapabilityDetectionRule, PriorityWeightConfig, and ConfigurationManager
that delegates to the core JSON/YAML parsers and printers.
"""

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any

from ..core.models import Configuration, PriorityWeights, ValidationResult
from ..core.interfaces import ConfigurationParser, ConfigurationPrettyPrinter
from ..core.configuration import (
    JSONConfigurationParser,
    YAMLConfigurationParser,
    JSONConfigurationPrettyPrinter,
    YAMLConfigurationPrettyPrinter,
)


def matches_exclusion_pattern(path: str, patterns: List[str]) -> bool:
    """Return True if path matches any of the exclusion glob patterns.

    Uses :func:`fnmatch.fnmatch` for glob-style matching so that patterns
    such as ``*.tmp`` or ``/proc/*`` work as expected.

    Args:
        path: The file-system path to test.
        patterns: A list of glob patterns to match against.

    Returns:
        bool: ``True`` if *path* matches at least one pattern, ``False``
        otherwise.
    """
    for pattern in patterns:
        if fnmatch.fnmatch(path, pattern):
            return True
    return False


@dataclass
class CapabilityDetectionRule:
    """
    A rule for custom capability detection.

    Defines a pattern-based rule that can be applied to application metadata
    to detect specific capabilities with an associated confidence score.
    """

    name: str
    pattern: str
    category: str
    confidence: float = 0.7
    description: str = ""


@dataclass
class PriorityWeightConfig:
    """
    Wraps priority weights for configuration purposes.

    Provides a convenient dataclass representation of priority weights
    that can be converted to the core PriorityWeights model.
    """

    usage: float = 0.3
    complexity: float = 0.2
    security: float = 0.25
    performance: float = 0.25

    def to_priority_weights(self) -> PriorityWeights:
        """
        Convert to a core PriorityWeights instance.

        Returns:
            PriorityWeights: Core model instance with the same weight values.
        """
        return PriorityWeights(
            usage=self.usage,
            complexity=self.complexity,
            security=self.security,
            performance=self.performance,
        )


class ConfigurationManager:
    """
    High-level configuration manager.

    Provides a unified interface for parsing, formatting, validating, and
    applying capability detection rules to Configuration objects.  Delegates
    the actual parsing/formatting work to the core JSON and YAML
    implementations.
    """

    def __init__(self) -> None:
        self._json_parser = JSONConfigurationParser()
        self._yaml_parser = YAMLConfigurationParser()
        self._json_printer = JSONConfigurationPrettyPrinter()
        self._yaml_printer = YAMLConfigurationPrettyPrinter()

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse(self, config_text: str, format: str = "json") -> Configuration:
        """
        Parse configuration from a text string.

        Args:
            config_text: Configuration text to parse.
            format: Format of the text – ``"json"`` (default) or ``"yaml"``/``"yml"``.

        Returns:
            Configuration: Parsed configuration object.

        Raises:
            ValueError: If the format is unsupported or the text is invalid.
        """
        parser = self._get_parser(format)
        return parser.parse(config_text)

    def parse_file(self, config_path: Path) -> Configuration:
        """
        Parse configuration from a file, auto-detecting format from extension.

        Supported extensions: ``.json``, ``.yaml``, ``.yml``.

        Args:
            config_path: Path to the configuration file.

        Returns:
            Configuration: Parsed configuration object.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the extension is unsupported or the content is invalid.
        """
        suffix = config_path.suffix.lower()
        if suffix == ".json":
            return self._json_parser.parse_file(config_path)
        elif suffix in (".yaml", ".yml"):
            return self._yaml_parser.parse_file(config_path)
        else:
            raise ValueError(
                f"Unsupported configuration file extension '{suffix}'. "
                "Expected .json, .yaml, or .yml."
            )

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format(self, config: Configuration, format: str = "json") -> str:
        """
        Format a Configuration object to a text string.

        Args:
            config: Configuration to format.
            format: Target format – ``"json"`` (default) or ``"yaml"``/``"yml"``.

        Returns:
            str: Formatted configuration text.

        Raises:
            ValueError: If the format is unsupported.
        """
        printer = self._get_printer(format)
        return printer.format(config)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, config: Configuration) -> ValidationResult:
        """
        Validate a Configuration object.

        Delegates to the Configuration's own ``validate()`` method so that
        all validation logic remains in one place.

        Args:
            config: Configuration to validate.

        Returns:
            ValidationResult: Validation result with errors and warnings.
        """
        return config.validate()

    # ------------------------------------------------------------------
    # Capability rule application
    # ------------------------------------------------------------------

    def apply_capability_rules(
        self,
        config: Configuration,
        rules: List[CapabilityDetectionRule],
    ) -> Configuration:
        """
        Merge custom capability detection rules into a configuration.

        Each rule is stored in ``config.capability_detection_rules`` under its
        ``name`` key as a dictionary with the rule's fields.  Existing rules
        with the same name are overwritten.

        Args:
            config: Configuration to update.
            rules: List of capability detection rules to merge.

        Returns:
            Configuration: The same configuration object with rules merged in.
        """
        for rule in rules:
            config.capability_detection_rules[rule.name] = {
                "pattern": rule.pattern,
                "category": rule.category,
                "confidence": rule.confidence,
                "description": rule.description,
            }
        return config

    # ------------------------------------------------------------------
    # Scanning exclusion helpers (Requirement 6.5)
    # ------------------------------------------------------------------

    def is_path_excluded(self, path: str, config: Configuration) -> bool:
        """Check if a path should be excluded based on config exclusion patterns.

        Delegates to the module-level :func:`matches_exclusion_pattern` helper.

        Args:
            path: The file-system path to test.
            config: Configuration whose ``scan_exclusion_patterns`` are used.

        Returns:
            bool: ``True`` if the path matches any exclusion pattern.
        """
        return matches_exclusion_pattern(path, config.scan_exclusion_patterns)

    # ------------------------------------------------------------------
    # Custom capability detection rules (Requirement 6.6)
    # ------------------------------------------------------------------

    def get_capability_rules(self, config: Configuration) -> List[CapabilityDetectionRule]:
        """Extract CapabilityDetectionRule objects from config.capability_detection_rules dict.

        Each entry in ``config.capability_detection_rules`` is expected to be a
        dict with keys ``pattern``, ``category``, ``confidence``, and
        ``description``.  The dict key is used as the rule ``name``.

        Args:
            config: Configuration containing the rules dict.

        Returns:
            List[CapabilityDetectionRule]: Reconstructed rule objects.
        """
        rules: List[CapabilityDetectionRule] = []
        for name, rule_dict in config.capability_detection_rules.items():
            rules.append(
                CapabilityDetectionRule(
                    name=name,
                    pattern=rule_dict.get("pattern", ""),
                    category=rule_dict.get("category", ""),
                    confidence=float(rule_dict.get("confidence", 0.7)),
                    description=rule_dict.get("description", ""),
                )
            )
        return rules

    # ------------------------------------------------------------------
    # Priority weighting factors (Requirement 6.7)
    # ------------------------------------------------------------------

    def get_priority_weight_config(self, config: Configuration) -> PriorityWeightConfig:
        """Extract PriorityWeightConfig from a Configuration object.

        Args:
            config: Configuration whose ``priority_weights`` are read.

        Returns:
            PriorityWeightConfig: Weight configuration mirroring the config values.
        """
        w = config.priority_weights
        return PriorityWeightConfig(
            usage=w.usage,
            complexity=w.complexity,
            security=w.security,
            performance=w.performance,
        )

    def update_priority_weights(
        self, config: Configuration, weight_config: PriorityWeightConfig
    ) -> Configuration:
        """Update a Configuration's priority weights from a PriorityWeightConfig.

        Args:
            config: Configuration to update (mutated in-place).
            weight_config: New weight values to apply.

        Returns:
            Configuration: The same configuration object with updated weights.
        """
        config.priority_weights = weight_config.to_priority_weights()
        return config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_parser(self, format: str) -> ConfigurationParser:
        fmt = format.lower()
        if fmt == "json":
            return self._json_parser
        elif fmt in ("yaml", "yml"):
            return self._yaml_parser
        else:
            raise ValueError(
                f"Unsupported configuration format '{format}'. "
                "Expected 'json' or 'yaml'."
            )

    def _get_printer(self, format: str) -> ConfigurationPrettyPrinter:
        fmt = format.lower()
        if fmt == "json":
            return self._json_printer
        elif fmt in ("yaml", "yml"):
            return self._yaml_printer
        else:
            raise ValueError(
                f"Unsupported configuration format '{format}'. "
                "Expected 'json' or 'yaml'."
            )
