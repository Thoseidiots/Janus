"""
Configuration parsing and pretty printing implementations.

This module provides concrete implementations of ConfigurationParser and
ConfigurationPrettyPrinter for handling configuration files in JSON and YAML formats.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any

from .interfaces import ConfigurationParser, ConfigurationPrettyPrinter
from .models import Configuration, PriorityWeights, ValidationResult


class JSONConfigurationParser(ConfigurationParser):
    """
    JSON-based configuration parser implementation.
    
    Parses configuration files in JSON format and converts them to
    Configuration objects with comprehensive validation.
    """
    
    def parse(self, config_text: str) -> Configuration:
        """
        Parse configuration from JSON text format.
        
        Args:
            config_text: JSON configuration text to parse
            
        Returns:
            Configuration: Parsed configuration object
            
        Raises:
            ValueError: If configuration is invalid or malformed JSON
        """
        try:
            data = json.loads(config_text)
            return self._dict_to_config(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON configuration: {e}")
        except Exception as e:
            raise ValueError(f"Error parsing configuration: {e}")
    
    def parse_file(self, config_path: Path) -> Configuration:
        """
        Parse configuration from a JSON file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            Configuration: Parsed configuration object
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If configuration is invalid
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_text = f.read()
            return self.parse(config_text)
        except Exception as e:
            raise ValueError(f"Error reading configuration file {config_path}: {e}")
    
    def validate(self, config: Configuration) -> ValidationResult:
        """
        Validate a configuration object.
        
        Args:
            config: Configuration to validate
            
        Returns:
            ValidationResult: Validation results with errors and warnings
        """
        return config.validate()
    
    def _dict_to_config(self, data: Dict[str, Any]) -> Configuration:
        """
        Convert dictionary to Configuration object.
        
        Args:
            data: Dictionary containing configuration data
            
        Returns:
            Configuration: Configuration object
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Handle priority weights
        priority_weights_data = data.get('priority_weights', {})
        priority_weights = PriorityWeights(
            usage=priority_weights_data.get('usage', 0.3),
            complexity=priority_weights_data.get('complexity', 0.2),
            security=priority_weights_data.get('security', 0.25),
            performance=priority_weights_data.get('performance', 0.25)
        )
        
        # Create configuration with validation
        config = Configuration(
            # Scanning configuration
            scan_exclusion_patterns=data.get('scan_exclusion_patterns', []),
            scan_timeout_seconds=data.get('scan_timeout_seconds', 300),
            max_applications_per_scan=data.get('max_applications_per_scan', 10000),
            
            # Analysis configuration
            capability_detection_rules=data.get('capability_detection_rules', {}),
            analysis_timeout_seconds=data.get('analysis_timeout_seconds', 60),
            min_confidence_threshold=data.get('min_confidence_threshold', 0.5),
            
            # Priority configuration
            priority_weights=priority_weights,
            
            # Security configuration
            respect_access_controls=data.get('respect_access_controls', True),
            encrypt_stored_data=data.get('encrypt_stored_data', True),
            audit_logging_enabled=data.get('audit_logging_enabled', True),
            
            # Performance configuration
            max_concurrent_analyses=data.get('max_concurrent_analyses', 4),
            cache_analysis_results=data.get('cache_analysis_results', True),
            cache_expiry_hours=data.get('cache_expiry_hours', 24),
            
            # Output configuration
            report_formats=data.get('report_formats', ["json", "html"]),
            include_charts=data.get('include_charts', True),
            sanitize_paths=data.get('sanitize_paths', True)
        )
        
        # Validate the configuration
        validation_result = config.validate()
        if not validation_result.is_valid:
            error_msg = "Configuration validation failed: " + "; ".join(validation_result.errors)
            raise ValueError(error_msg)
        
        return config


class YAMLConfigurationParser(ConfigurationParser):
    """
    YAML-based configuration parser implementation.
    
    Parses configuration files in YAML format and converts them to
    Configuration objects with comprehensive validation.
    """
    
    def parse(self, config_text: str) -> Configuration:
        """
        Parse configuration from YAML text format.
        
        Args:
            config_text: YAML configuration text to parse
            
        Returns:
            Configuration: Parsed configuration object
            
        Raises:
            ValueError: If configuration is invalid or malformed YAML
        """
        try:
            data = yaml.safe_load(config_text)
            if data is None:
                data = {}
            return self._dict_to_config(data)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
        except Exception as e:
            raise ValueError(f"Error parsing configuration: {e}")
    
    def parse_file(self, config_path: Path) -> Configuration:
        """
        Parse configuration from a YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Configuration: Parsed configuration object
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If configuration is invalid
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_text = f.read()
            return self.parse(config_text)
        except Exception as e:
            raise ValueError(f"Error reading configuration file {config_path}: {e}")
    
    def validate(self, config: Configuration) -> ValidationResult:
        """
        Validate a configuration object.
        
        Args:
            config: Configuration to validate
            
        Returns:
            ValidationResult: Validation results with errors and warnings
        """
        return config.validate()
    
    def _dict_to_config(self, data: Dict[str, Any]) -> Configuration:
        """
        Convert dictionary to Configuration object.
        
        Args:
            data: Dictionary containing configuration data
            
        Returns:
            Configuration: Configuration object
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Handle priority weights
        priority_weights_data = data.get('priority_weights', {})
        priority_weights = PriorityWeights(
            usage=priority_weights_data.get('usage', 0.3),
            complexity=priority_weights_data.get('complexity', 0.2),
            security=priority_weights_data.get('security', 0.25),
            performance=priority_weights_data.get('performance', 0.25)
        )
        
        # Create configuration with validation
        config = Configuration(
            # Scanning configuration
            scan_exclusion_patterns=data.get('scan_exclusion_patterns', []),
            scan_timeout_seconds=data.get('scan_timeout_seconds', 300),
            max_applications_per_scan=data.get('max_applications_per_scan', 10000),
            
            # Analysis configuration
            capability_detection_rules=data.get('capability_detection_rules', {}),
            analysis_timeout_seconds=data.get('analysis_timeout_seconds', 60),
            min_confidence_threshold=data.get('min_confidence_threshold', 0.5),
            
            # Priority configuration
            priority_weights=priority_weights,
            
            # Security configuration
            respect_access_controls=data.get('respect_access_controls', True),
            encrypt_stored_data=data.get('encrypt_stored_data', True),
            audit_logging_enabled=data.get('audit_logging_enabled', True),
            
            # Performance configuration
            max_concurrent_analyses=data.get('max_concurrent_analyses', 4),
            cache_analysis_results=data.get('cache_analysis_results', True),
            cache_expiry_hours=data.get('cache_expiry_hours', 24),
            
            # Output configuration
            report_formats=data.get('report_formats', ["json", "html"]),
            include_charts=data.get('include_charts', True),
            sanitize_paths=data.get('sanitize_paths', True)
        )
        
        # Validate the configuration
        validation_result = config.validate()
        if not validation_result.is_valid:
            error_msg = "Configuration validation failed: " + "; ".join(validation_result.errors)
            raise ValueError(error_msg)
        
        return config


class JSONConfigurationPrettyPrinter(ConfigurationPrettyPrinter):
    """
    JSON-based configuration pretty printer implementation.
    
    Formats Configuration objects back into valid JSON configuration files
    with proper indentation and formatting.
    """
    
    def __init__(self, indent: int = 2, sort_keys: bool = True):
        """
        Initialize the JSON pretty printer.
        
        Args:
            indent: Number of spaces for indentation
            sort_keys: Whether to sort keys alphabetically
        """
        self.indent = indent
        self.sort_keys = sort_keys
    
    def format(self, config: Configuration) -> str:
        """
        Format a configuration object to JSON text.
        
        Args:
            config: Configuration to format
            
        Returns:
            str: Formatted JSON configuration text
        """
        data = self._config_to_dict(config)
        return json.dumps(data, indent=self.indent, sort_keys=self.sort_keys, ensure_ascii=False)
    
    def format_to_file(self, config: Configuration, output_path: Path) -> None:
        """
        Format a configuration object and write to a JSON file.
        
        Args:
            config: Configuration to format
            output_path: Path to output JSON file
        """
        formatted_text = self.format(config)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_text)
    
    def _config_to_dict(self, config: Configuration) -> Dict[str, Any]:
        """
        Convert Configuration object to dictionary.
        
        Args:
            config: Configuration object to convert
            
        Returns:
            Dict[str, Any]: Dictionary representation
        """
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


class YAMLConfigurationPrettyPrinter(ConfigurationPrettyPrinter):
    """
    YAML-based configuration pretty printer implementation.
    
    Formats Configuration objects back into valid YAML configuration files
    with proper indentation and formatting.
    """
    
    def __init__(self, indent: int = 2, default_flow_style: bool = False):
        """
        Initialize the YAML pretty printer.
        
        Args:
            indent: Number of spaces for indentation
            default_flow_style: Whether to use flow style for collections
        """
        self.indent = indent
        self.default_flow_style = default_flow_style
    
    def format(self, config: Configuration) -> str:
        """
        Format a configuration object to YAML text.
        
        Args:
            config: Configuration to format
            
        Returns:
            str: Formatted YAML configuration text
        """
        data = self._config_to_dict(config)
        return yaml.dump(
            data,
            indent=self.indent,
            default_flow_style=self.default_flow_style,
            sort_keys=True,
            allow_unicode=True
        )
    
    def format_to_file(self, config: Configuration, output_path: Path) -> None:
        """
        Format a configuration object and write to a YAML file.
        
        Args:
            config: Configuration to format
            output_path: Path to output YAML file
        """
        formatted_text = self.format(config)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_text)
    
    def _config_to_dict(self, config: Configuration) -> Dict[str, Any]:
        """
        Convert Configuration object to dictionary.
        
        Args:
            config: Configuration object to convert
            
        Returns:
            Dict[str, Any]: Dictionary representation
        """
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


# Default implementations for convenience
DefaultConfigurationParser = JSONConfigurationParser
DefaultConfigurationPrettyPrinter = JSONConfigurationPrettyPrinter