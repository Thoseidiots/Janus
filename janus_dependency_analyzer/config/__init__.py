"""
Configuration management for the Janus Dependency Analyzer.

This package provides configuration parsing, validation, and formatting
for the dependency analyzer system.
"""

from .configuration import (
    CapabilityDetectionRule,
    Configuration,
    ConfigurationManager,
    ConfigurationParser,
    ConfigurationPrettyPrinter,
    PriorityWeightConfig,
    matches_exclusion_pattern,
)

__all__ = [
    "CapabilityDetectionRule",
    "Configuration",
    "ConfigurationManager",
    "ConfigurationParser",
    "ConfigurationPrettyPrinter",
    "PriorityWeightConfig",
    "matches_exclusion_pattern",
]
