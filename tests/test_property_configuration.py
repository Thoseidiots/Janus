"""
Property-based tests for configuration round-trip integrity.

**Property 9: Configuration Round-Trip Integrity**
**Validates: Requirements 6.1, 6.2, 6.3, 6.4**
"""

import pytest
from hypothesis import given, strategies as st, settings
from typing import List, Dict, Any
import json
import yaml

from janus_dependency_analyzer.core.models import Configuration, PriorityWeights
from janus_dependency_analyzer.core.configuration import (
    JSONConfigurationParser,
    YAMLConfigurationParser,
    JSONConfigurationPrettyPrinter,
    YAMLConfigurationPrettyPrinter
)


# Hypothesis strategies for generating test data
@st.composite
def priority_weights_strategy(draw):
    """Generate valid PriorityWeights objects."""
    # Ensure at least one weight is non-zero
    usage = draw(st.floats(min_value=0.0, max_value=1.0))
    complexity = draw(st.floats(min_value=0.0, max_value=1.0))
    security = draw(st.floats(min_value=0.0, max_value=1.0))
    performance = draw(st.floats(min_value=0.0, max_value=1.0))
    
    # If all weights are zero, set at least one to a positive value
    total = usage + complexity + security + performance
    if total == 0.0:
        # Randomly choose one weight to be positive
        choice = draw(st.integers(min_value=0, max_value=3))
        positive_value = draw(st.floats(min_value=0.1, max_value=1.0))
        if choice == 0:
            usage = positive_value
        elif choice == 1:
            complexity = positive_value
        elif choice == 2:
            security = positive_value
        else:
            performance = positive_value
    
    return PriorityWeights(
        usage=usage,
        complexity=complexity,
        security=security,
        performance=performance
    )


@st.composite
def configuration_strategy(draw):
    """Generate valid Configuration objects."""
    # Use printable ASCII characters to avoid YAML encoding issues
    text_strategy = st.text(
        alphabet=st.characters(min_codepoint=32, max_codepoint=126),
        min_size=0, 
        max_size=50
    )
    
    return Configuration(
        # Scanning configuration
        scan_exclusion_patterns=draw(st.lists(text_strategy, max_size=10)),
        scan_timeout_seconds=draw(st.integers(min_value=1, max_value=3600)),
        max_applications_per_scan=draw(st.integers(min_value=1, max_value=100000)),
        
        # Analysis configuration
        capability_detection_rules=draw(st.dictionaries(
            text_strategy,
            st.one_of(text_strategy, st.integers(), st.booleans()),
            max_size=5
        )),
        analysis_timeout_seconds=draw(st.integers(min_value=1, max_value=300)),
        min_confidence_threshold=draw(st.floats(min_value=0.0, max_value=1.0)),
        
        # Priority configuration
        priority_weights=draw(priority_weights_strategy()),
        
        # Security configuration
        respect_access_controls=draw(st.booleans()),
        encrypt_stored_data=draw(st.booleans()),
        audit_logging_enabled=draw(st.booleans()),
        
        # Performance configuration
        max_concurrent_analyses=draw(st.integers(min_value=1, max_value=16)),
        cache_analysis_results=draw(st.booleans()),
        cache_expiry_hours=draw(st.integers(min_value=1, max_value=168)),
        
        # Output configuration
        report_formats=draw(st.lists(
            st.sampled_from(["json", "csv", "html", "xml", "yaml"]),
            min_size=1,
            max_size=5,
            unique=True
        )),
        include_charts=draw(st.booleans()),
        sanitize_paths=draw(st.booleans())
    )


class SimpleConfigurationParser:
    """Simple configuration parser for testing round-trip integrity."""
    
    def parse(self, config_text: str) -> Configuration:
        """Parse configuration from JSON text format."""
        try:
            data = json.loads(config_text)
            return self._dict_to_config(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON configuration: {e}")
    
    def _dict_to_config(self, data: Dict[str, Any]) -> Configuration:
        """Convert dictionary to Configuration object."""
        # Handle priority weights
        priority_weights_data = data.get('priority_weights', {})
        priority_weights = PriorityWeights(
            usage=priority_weights_data.get('usage', 0.3),
            complexity=priority_weights_data.get('complexity', 0.2),
            security=priority_weights_data.get('security', 0.25),
            performance=priority_weights_data.get('performance', 0.25)
        )
        
        return Configuration(
            scan_exclusion_patterns=data.get('scan_exclusion_patterns', []),
            scan_timeout_seconds=data.get('scan_timeout_seconds', 300),
            max_applications_per_scan=data.get('max_applications_per_scan', 10000),
            capability_detection_rules=data.get('capability_detection_rules', {}),
            analysis_timeout_seconds=data.get('analysis_timeout_seconds', 60),
            min_confidence_threshold=data.get('min_confidence_threshold', 0.5),
            priority_weights=priority_weights,
            respect_access_controls=data.get('respect_access_controls', True),
            encrypt_stored_data=data.get('encrypt_stored_data', True),
            audit_logging_enabled=data.get('audit_logging_enabled', True),
            max_concurrent_analyses=data.get('max_concurrent_analyses', 4),
            cache_analysis_results=data.get('cache_analysis_results', True),
            cache_expiry_hours=data.get('cache_expiry_hours', 24),
            report_formats=data.get('report_formats', ["json", "html"]),
            include_charts=data.get('include_charts', True),
            sanitize_paths=data.get('sanitize_paths', True)
        )


class SimpleConfigurationPrettyPrinter:
    """Simple configuration pretty printer for testing round-trip integrity."""
    
    def format(self, config: Configuration) -> str:
        """Format configuration object to JSON text."""
        data = self._config_to_dict(config)
        return json.dumps(data, indent=2, sort_keys=True)
    
    def _config_to_dict(self, config: Configuration) -> Dict[str, Any]:
        """Convert Configuration object to dictionary."""
        return {
            'scan_exclusion_patterns': config.scan_exclusion_patterns,
            'scan_timeout_seconds': config.scan_timeout_seconds,
            'max_applications_per_scan': config.max_applications_per_scan,
            'capability_detection_rules': config.capability_detection_rules,
            'analysis_timeout_seconds': config.analysis_timeout_seconds,
            'min_confidence_threshold': config.min_confidence_threshold,
            'priority_weights': {
                'usage': config.priority_weights.usage,
                'complexity': config.priority_weights.complexity,
                'security': config.priority_weights.security,
                'performance': config.priority_weights.performance
            },
            'respect_access_controls': config.respect_access_controls,
            'encrypt_stored_data': config.encrypt_stored_data,
            'audit_logging_enabled': config.audit_logging_enabled,
            'max_concurrent_analyses': config.max_concurrent_analyses,
            'cache_analysis_results': config.cache_analysis_results,
            'cache_expiry_hours': config.cache_expiry_hours,
            'report_formats': config.report_formats,
            'include_charts': config.include_charts,
            'sanitize_paths': config.sanitize_paths
        }


class TestConfigurationRoundTrip:
    """
    Property-based tests for configuration round-trip integrity.
    
    **Feature: janus-dependency-analyzer, Property 9: Configuration Round-Trip Integrity**
    **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        self.json_parser = JSONConfigurationParser()
        self.json_printer = JSONConfigurationPrettyPrinter()
        self.yaml_parser = YAMLConfigurationParser()
        self.yaml_printer = YAMLConfigurationPrettyPrinter()
    
    @given(configuration_strategy())
    @settings(max_examples=100, deadline=None)
    def test_json_configuration_round_trip_integrity(self, original_config: Configuration):
        """
        **Property 9: Configuration Round-Trip Integrity**
        
        For any valid Configuration object, parsing then printing then parsing
        SHALL produce an equivalent object using JSON format.
        
        **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
        """
        # Step 1: Format the original configuration to JSON text
        formatted_text = self.json_printer.format(original_config)
        
        # Step 2: Parse the formatted text back to a configuration object
        parsed_config = self.json_parser.parse(formatted_text)
        
        # Step 3: Format the parsed configuration to text again
        reformatted_text = self.json_printer.format(parsed_config)
        
        # Step 4: Parse the reformatted text to get the final configuration
        final_config = self.json_parser.parse(reformatted_text)
        
        # Verify round-trip integrity
        self._assert_configurations_equivalent(original_config, final_config)
        
        # Verify text formatting is stable
        assert formatted_text == reformatted_text, "JSON formatting should be stable"
    
    @given(configuration_strategy())
    @settings(max_examples=100, deadline=None)
    def test_yaml_configuration_round_trip_integrity(self, original_config: Configuration):
        """
        **Property 9: Configuration Round-Trip Integrity**
        
        For any valid Configuration object, parsing then printing then parsing
        SHALL produce an equivalent object using YAML format.
        
        **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
        """
        # Step 1: Format the original configuration to YAML text
        formatted_text = self.yaml_printer.format(original_config)
        
        # Step 2: Parse the formatted text back to a configuration object
        parsed_config = self.yaml_parser.parse(formatted_text)
        
        # Step 3: Format the parsed configuration to text again
        reformatted_text = self.yaml_printer.format(parsed_config)
        
        # Step 4: Parse the reformatted text to get the final configuration
        final_config = self.yaml_parser.parse(reformatted_text)
        
        # Verify round-trip integrity
        self._assert_configurations_equivalent(original_config, final_config)
        
        # Verify text formatting is stable
        assert formatted_text == reformatted_text, "YAML formatting should be stable"
    
    @given(st.text())
    def test_invalid_json_configuration_returns_error(self, invalid_text: str):
        """
        **Property 9: Configuration Round-Trip Integrity**
        
        For any invalid JSON configuration text, the parser SHALL return a descriptive
        error message rather than silently failing or producing incorrect results.
        
        **Validates: Requirements 6.1, 6.2**
        """
        # Filter out valid JSON to focus on truly invalid configurations
        try:
            json.loads(invalid_text)
            # If it's valid JSON, skip this test case
            return
        except json.JSONDecodeError:
            pass
        
        # Attempt to parse invalid configuration
        with pytest.raises(ValueError) as exc_info:
            self.json_parser.parse(invalid_text)
        
        # Verify error message is descriptive
        error_message = str(exc_info.value)
        assert len(error_message) > 10, "Error message should be descriptive"
        assert "configuration" in error_message.lower() or "json" in error_message.lower()
    
    @given(st.text())
    def test_invalid_yaml_configuration_returns_error(self, invalid_text: str):
        """
        **Property 9: Configuration Round-Trip Integrity**
        
        For any invalid YAML configuration text, the parser SHALL return a descriptive
        error message rather than silently failing or producing incorrect results.
        
        **Validates: Requirements 6.1, 6.2**
        """
        # Filter out valid YAML to focus on truly invalid configurations
        try:
            yaml.safe_load(invalid_text)
            # If it's valid YAML, skip this test case
            return
        except yaml.YAMLError:
            pass
        
        # Attempt to parse invalid configuration
        with pytest.raises(ValueError) as exc_info:
            self.yaml_parser.parse(invalid_text)
        
        # Verify error message is descriptive
        error_message = str(exc_info.value)
        assert len(error_message) > 10, "Error message should be descriptive"
        assert "configuration" in error_message.lower() or "yaml" in error_message.lower()
    
    @given(configuration_strategy())
    def test_json_configuration_validation_consistency(self, config: Configuration):
        """
        **Property 9: Configuration Round-Trip Integrity**
        
        For any Configuration object that passes validation, the JSON round-trip
        process SHALL preserve the validation status.
        
        **Validates: Requirements 6.4**
        """
        # Check original validation
        original_validation = config.validate()
        
        if original_validation.is_valid:
            # Perform round-trip
            formatted_text = self.json_printer.format(config)
            parsed_config = self.json_parser.parse(formatted_text)
            
            # Check parsed validation
            parsed_validation = parsed_config.validate()
            
            # Both should be valid
            assert parsed_validation.is_valid, f"JSON round-trip should preserve validity: {parsed_validation.errors}"
    
    @given(configuration_strategy())
    def test_yaml_configuration_validation_consistency(self, config: Configuration):
        """
        **Property 9: Configuration Round-Trip Integrity**
        
        For any Configuration object that passes validation, the YAML round-trip
        process SHALL preserve the validation status.
        
        **Validates: Requirements 6.4**
        """
        # Check original validation
        original_validation = config.validate()
        
        if original_validation.is_valid:
            # Perform round-trip
            formatted_text = self.yaml_printer.format(config)
            parsed_config = self.yaml_parser.parse(formatted_text)
            
            # Check parsed validation
            parsed_validation = parsed_config.validate()
            
            # Both should be valid
            assert parsed_validation.is_valid, f"YAML round-trip should preserve validity: {parsed_validation.errors}"
    
    def _assert_configurations_equivalent(self, config1: Configuration, config2: Configuration):
        """Assert that two configurations are equivalent."""
        # Compare basic fields
        assert config1.scan_exclusion_patterns == config2.scan_exclusion_patterns
        assert config1.scan_timeout_seconds == config2.scan_timeout_seconds
        assert config1.max_applications_per_scan == config2.max_applications_per_scan
        assert config1.capability_detection_rules == config2.capability_detection_rules
        assert config1.analysis_timeout_seconds == config2.analysis_timeout_seconds
        assert abs(config1.min_confidence_threshold - config2.min_confidence_threshold) < 1e-10
        
        # Compare priority weights with floating point tolerance
        assert abs(config1.priority_weights.usage - config2.priority_weights.usage) < 1e-10
        assert abs(config1.priority_weights.complexity - config2.priority_weights.complexity) < 1e-10
        assert abs(config1.priority_weights.security - config2.priority_weights.security) < 1e-10
        assert abs(config1.priority_weights.performance - config2.priority_weights.performance) < 1e-10
        
        # Compare boolean fields
        assert config1.respect_access_controls == config2.respect_access_controls
        assert config1.encrypt_stored_data == config2.encrypt_stored_data
        assert config1.audit_logging_enabled == config2.audit_logging_enabled
        assert config1.cache_analysis_results == config2.cache_analysis_results
        assert config1.include_charts == config2.include_charts
        assert config1.sanitize_paths == config2.sanitize_paths
        
        # Compare integer fields
        assert config1.max_concurrent_analyses == config2.max_concurrent_analyses
        assert config1.cache_expiry_hours == config2.cache_expiry_hours
        
        # Compare list fields
        assert sorted(config1.report_formats) == sorted(config2.report_formats)


# Additional unit tests for edge cases
class TestConfigurationEdgeCases:
    """Unit tests for configuration edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.json_parser = JSONConfigurationParser()
        self.json_printer = JSONConfigurationPrettyPrinter()
        self.yaml_parser = YAMLConfigurationParser()
        self.yaml_printer = YAMLConfigurationPrettyPrinter()
    
    def test_empty_json_configuration(self):
        """Test round-trip with minimal JSON configuration."""
        config = Configuration()
        
        formatted_text = self.json_printer.format(config)
        parsed_config = self.json_parser.parse(formatted_text)
        
        # Should preserve default values
        assert parsed_config.scan_timeout_seconds == 300
        assert parsed_config.max_applications_per_scan == 10000
        assert parsed_config.min_confidence_threshold == 0.5
    
    def test_empty_yaml_configuration(self):
        """Test round-trip with minimal YAML configuration."""
        config = Configuration()
        
        formatted_text = self.yaml_printer.format(config)
        parsed_config = self.yaml_parser.parse(formatted_text)
        
        # Should preserve default values
        assert parsed_config.scan_timeout_seconds == 300
        assert parsed_config.max_applications_per_scan == 10000
        assert parsed_config.min_confidence_threshold == 0.5
    
    def test_json_configuration_with_special_characters(self):
        """Test JSON round-trip with special characters in strings."""
        config = Configuration(
            scan_exclusion_patterns=[
                "*/temp/*",
                "C:\\Windows\\System32\\*",
                "/usr/lib/*/cache",
                "~/.cache/*"
            ],
            capability_detection_rules={
                "special_chars": "test with spaces and symbols: !@#$%^&*()",
                "unicode": "测试中文字符",
                "quotes": 'test "quotes" and \'apostrophes\''
            }
        )
        
        formatted_text = self.json_printer.format(config)
        parsed_config = self.json_parser.parse(formatted_text)
        
        assert parsed_config.scan_exclusion_patterns == config.scan_exclusion_patterns
        assert parsed_config.capability_detection_rules == config.capability_detection_rules
    
    def test_yaml_configuration_with_special_characters(self):
        """Test YAML round-trip with special characters in strings."""
        config = Configuration(
            scan_exclusion_patterns=[
                "*/temp/*",
                "C:\\Windows\\System32\\*",
                "/usr/lib/*/cache",
                "~/.cache/*"
            ],
            capability_detection_rules={
                "special_chars": "test with spaces and symbols: !@#$%^&*()",
                "unicode": "测试中文字符",
                "quotes": 'test "quotes" and \'apostrophes\''
            }
        )
        
        formatted_text = self.yaml_printer.format(config)
        parsed_config = self.yaml_parser.parse(formatted_text)
        
        assert parsed_config.scan_exclusion_patterns == config.scan_exclusion_patterns
        assert parsed_config.capability_detection_rules == config.capability_detection_rules
    
    def test_json_configuration_with_extreme_values(self):
        """Test JSON round-trip with extreme but valid values."""
        config = Configuration(
            scan_timeout_seconds=1,  # Minimum
            max_applications_per_scan=1,  # Minimum
            min_confidence_threshold=0.0,  # Minimum
            max_concurrent_analyses=1,  # Minimum
            cache_expiry_hours=1,  # Minimum
            priority_weights=PriorityWeights(
                usage=0.0,
                complexity=0.0,
                security=0.0,
                performance=1.0  # All weight on one factor
            )
        )
        
        formatted_text = self.json_printer.format(config)
        parsed_config = self.json_parser.parse(formatted_text)
        
        assert parsed_config.scan_timeout_seconds == 1
        assert parsed_config.max_applications_per_scan == 1
        assert parsed_config.min_confidence_threshold == 0.0
        assert parsed_config.priority_weights.performance == 1.0
    
    def test_yaml_configuration_with_extreme_values(self):
        """Test YAML round-trip with extreme but valid values."""
        config = Configuration(
            scan_timeout_seconds=1,  # Minimum
            max_applications_per_scan=1,  # Minimum
            min_confidence_threshold=0.0,  # Minimum
            max_concurrent_analyses=1,  # Minimum
            cache_expiry_hours=1,  # Minimum
            priority_weights=PriorityWeights(
                usage=0.0,
                complexity=0.0,
                security=0.0,
                performance=1.0  # All weight on one factor
            )
        )
        
        formatted_text = self.yaml_printer.format(config)
        parsed_config = self.yaml_parser.parse(formatted_text)
        
        assert parsed_config.scan_timeout_seconds == 1
        assert parsed_config.max_applications_per_scan == 1
        assert parsed_config.min_confidence_threshold == 0.0
        assert parsed_config.priority_weights.performance == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])