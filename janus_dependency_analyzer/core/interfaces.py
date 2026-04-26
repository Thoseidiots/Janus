"""
Abstract base classes and interfaces for the Janus Dependency Analyzer.

This module defines the contracts that all components must implement,
ensuring a consistent and extensible architecture.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from .models import (
    Application,
    ApplicationMetadata,
    Capability,
    DependencyMapping,
    ImplementationRoadmap,
    PriorityScore,
    RankedCapability,
    Configuration,
    ValidationResult,
    ScanResult,
    AnalysisContext,
    Platform
)


class SystemScanner(ABC):
    """
    Abstract base class for system-wide application scanning.
    
    The SystemScanner orchestrates platform-specific discovery mechanisms
    to build a comprehensive application inventory with support for both
    full and incremental scanning.
    """
    
    @abstractmethod
    def scan_full(self) -> ScanResult:
        """
        Perform a complete system scan for all applications.
        
        Returns:
            ScanResult: Complete scan results with all discovered applications
        """
        pass
    
    @abstractmethod
    def scan_incremental(self, last_scan_time: datetime) -> ScanResult:
        """
        Scan for changes since the last scan time.
        
        Args:
            last_scan_time: Timestamp of the previous scan
            
        Returns:
            ScanResult: Incremental scan results with only changed applications
        """
        pass
    
    @abstractmethod
    def get_platform_scanner(self) -> 'PlatformScanner':
        """
        Get the appropriate platform-specific scanner.
        
        Returns:
            PlatformScanner: Platform-specific scanner implementation
        """
        pass
    
    @abstractmethod
    def detect_platform(self) -> Platform:
        """
        Detect the current operating system platform.
        
        Returns:
            Platform: The detected platform
        """
        pass


class PlatformScanner(ABC):
    """
    Abstract base class for platform-specific application discovery.
    
    Each supported platform (Windows, macOS, Linux) implements this interface
    to provide platform-specific application discovery and metadata extraction.
    """
    
    @abstractmethod
    def discover_applications(self) -> List[Application]:
        """
        Discover all applications on the platform.
        
        Returns:
            List[Application]: List of discovered applications
        """
        pass
    
    @abstractmethod
    def extract_metadata(self, app_path: Path) -> ApplicationMetadata:
        """
        Extract metadata for an application at the given path.
        
        Args:
            app_path: Path to the application
            
        Returns:
            ApplicationMetadata: Extracted metadata information
        """
        pass
    
    @abstractmethod
    def get_platform(self) -> Platform:
        """
        Get the platform this scanner supports.
        
        Returns:
            Platform: The supported platform
        """
        pass
    
    @abstractmethod
    def scan_standard_locations(self) -> List[Application]:
        """
        Scan standard application installation locations.
        
        Returns:
            List[Application]: Applications found in standard locations
        """
        pass
    
    @abstractmethod
    def scan_package_managers(self) -> List[Application]:
        """
        Scan applications installed via package managers.
        
        Returns:
            List[Application]: Applications found via package managers
        """
        pass
    
    @abstractmethod
    def scan_portable_applications(self) -> List[Application]:
        """
        Scan for portable applications not in standard locations.
        
        Returns:
            List[Application]: Portable applications found
        """
        pass


class CapabilityAnalyzer(ABC):
    """
    Abstract base class for application capability analysis.
    
    The CapabilityAnalyzer processes discovered applications through multiple
    analysis strategies to identify their capabilities and assign confidence scores.
    """
    
    @abstractmethod
    def analyze_application(self, app: Application) -> List[Capability]:
        """
        Analyze an application to identify its capabilities.
        
        Args:
            app: Application to analyze
            
        Returns:
            List[Capability]: List of identified capabilities
        """
        pass
    
    @abstractmethod
    def get_analysis_strategies(self) -> List['AnalysisStrategy']:
        """
        Get all available analysis strategies.
        
        Returns:
            List[AnalysisStrategy]: Available analysis strategies
        """
        pass
    
    @abstractmethod
    def merge_capabilities(self, capabilities: List[Capability]) -> List[Capability]:
        """
        Merge overlapping capabilities and calculate confidence scores.
        
        Args:
            capabilities: List of capabilities to merge
            
        Returns:
            List[Capability]: Merged and scored capabilities
        """
        pass


class AnalysisStrategy(ABC):
    """
    Abstract base class for capability analysis strategies.
    
    Each strategy implements a specific method for extracting capabilities
    from applications (documentation, help text, CLI analysis, etc.).
    """
    
    @abstractmethod
    def can_analyze(self, app: Application) -> bool:
        """
        Check if this strategy can analyze the given application.
        
        Args:
            app: Application to check
            
        Returns:
            bool: True if the strategy can analyze this application
        """
        pass
    
    @abstractmethod
    def extract_capabilities(self, app: Application) -> List[Capability]:
        """
        Extract capabilities from the application using this strategy.
        
        Args:
            app: Application to analyze
            
        Returns:
            List[Capability]: Extracted capabilities
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """
        Get the name of this analysis strategy.
        
        Returns:
            str: Strategy name
        """
        pass
    
    @abstractmethod
    def get_confidence_factor(self) -> float:
        """
        Get the base confidence factor for this strategy.
        
        Returns:
            float: Confidence factor between 0.0 and 1.0
        """
        pass


class DependencyMapper(ABC):
    """
    Abstract base class for mapping Janus dependencies to external applications.
    
    The DependencyMapper analyzes the Janus codebase to identify current
    external dependencies and potential beneficial applications.
    """
    
    @abstractmethod
    def scan_janus_codebase(self, codebase_path: Path) -> List[DependencyMapping]:
        """
        Scan the Janus codebase for external application dependencies.
        
        Args:
            codebase_path: Path to the Janus codebase
            
        Returns:
            List[DependencyMapping]: Identified dependency mappings
        """
        pass
    
    @abstractmethod
    def identify_potential_applications(self, capabilities: List[Capability]) -> List[str]:
        """
        Identify potentially beneficial applications not currently used.
        
        Args:
            capabilities: Available capabilities from discovered applications
            
        Returns:
            List[str]: Application IDs of potentially beneficial applications
        """
        pass
    
    @abstractmethod
    def map_capability_gaps(self, dependencies: List[DependencyMapping]) -> Dict[str, List[str]]:
        """
        Map external application capabilities to Janus functionality gaps.
        
        Args:
            dependencies: Current dependency mappings
            
        Returns:
            Dict[str, List[str]]: Mapping of gaps to potential solutions
        """
        pass


class PriorityEngine(ABC):
    """
    Abstract base class for capability implementation priority calculation.
    
    The PriorityEngine implements multi-factor scoring algorithms to rank
    capabilities by implementation priority and value.
    """
    
    @abstractmethod
    def calculate_priority(self, capability: Capability, context: AnalysisContext) -> PriorityScore:
        """
        Calculate implementation priority for a capability.
        
        Args:
            capability: Capability to score
            context: Analysis context with configuration and usage data
            
        Returns:
            PriorityScore: Calculated priority score with justification
        """
        pass
    
    @abstractmethod
    def rank_capabilities(self, capabilities: List[Capability], context: AnalysisContext) -> List[RankedCapability]:
        """
        Rank capabilities by implementation priority.
        
        Args:
            capabilities: Capabilities to rank
            context: Analysis context
            
        Returns:
            List[RankedCapability]: Capabilities ranked by priority
        """
        pass
    
    @abstractmethod
    def calculate_usage_frequency_score(self, capability: Capability, context: AnalysisContext) -> float:
        """
        Calculate usage frequency score for a capability.
        
        Args:
            capability: Capability to score
            context: Analysis context
            
        Returns:
            float: Usage frequency score (0.0 to 1.0)
        """
        pass
    
    @abstractmethod
    def estimate_implementation_complexity(self, capability: Capability) -> float:
        """
        Estimate implementation complexity for a capability.
        
        Args:
            capability: Capability to evaluate
            
        Returns:
            float: Complexity score (0.0 to 1.0, higher = more complex)
        """
        pass


class RoadmapGenerator(ABC):
    """
    Abstract base class for implementation roadmap generation.
    
    The RoadmapGenerator creates detailed implementation plans with effort
    estimates, technical requirements, and risk assessments.
    """
    
    @abstractmethod
    def generate_roadmap(self, capability: Capability, priority_score: PriorityScore) -> ImplementationRoadmap:
        """
        Generate a detailed implementation roadmap for a capability.
        
        Args:
            capability: Capability to create roadmap for
            priority_score: Priority score and analysis
            
        Returns:
            ImplementationRoadmap: Detailed implementation plan
        """
        pass
    
    @abstractmethod
    def estimate_effort(self, capability: Capability) -> int:
        """
        Estimate development effort in person-hours.
        
        Args:
            capability: Capability to estimate
            
        Returns:
            int: Estimated effort in hours
        """
        pass
    
    @abstractmethod
    def identify_risks(self, capability: Capability) -> List[str]:
        """
        Identify potential implementation risks.
        
        Args:
            capability: Capability to analyze
            
        Returns:
            List[str]: Identified risks
        """
        pass


class ConfigurationParser(ABC):
    """
    Abstract base class for configuration parsing.
    
    The ConfigurationParser handles parsing configuration files into
    Configuration objects with validation and error reporting.
    """
    
    @abstractmethod
    def parse(self, config_text: str) -> Configuration:
        """
        Parse configuration from text format.
        
        Args:
            config_text: Configuration text to parse
            
        Returns:
            Configuration: Parsed configuration object
            
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def parse_file(self, config_path: Path) -> Configuration:
        """
        Parse configuration from a file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration: Parsed configuration object
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def validate(self, config: Configuration) -> ValidationResult:
        """
        Validate a configuration object.
        
        Args:
            config: Configuration to validate
            
        Returns:
            ValidationResult: Validation results with errors and warnings
        """
        pass


class ConfigurationPrettyPrinter(ABC):
    """
    Abstract base class for configuration pretty printing.
    
    The ConfigurationPrettyPrinter formats Configuration objects back
    into valid configuration files with proper formatting.
    """
    
    @abstractmethod
    def format(self, config: Configuration) -> str:
        """
        Format a configuration object to text.
        
        Args:
            config: Configuration to format
            
        Returns:
            str: Formatted configuration text
        """
        pass
    
    @abstractmethod
    def format_to_file(self, config: Configuration, output_path: Path) -> None:
        """
        Format configuration to a file.
        
        Args:
            config: Configuration to format
            output_path: Path to write formatted configuration
        """
        pass


class ReportGenerator(ABC):
    """
    Abstract base class for analysis report generation.
    
    The ReportGenerator creates comprehensive reports in multiple formats
    with visual charts and executive summaries.
    """
    
    @abstractmethod
    def generate_summary_report(self, scan_results: List[ScanResult]) -> Dict[str, Any]:
        """
        Generate a summary report of scan results.
        
        Args:
            scan_results: Results from system scans
            
        Returns:
            Dict[str, Any]: Summary report data
        """
        pass
    
    @abstractmethod
    def generate_capability_inventory(self, capabilities: List[Capability]) -> Dict[str, Any]:
        """
        Generate detailed capability inventory report.
        
        Args:
            capabilities: Discovered capabilities
            
        Returns:
            Dict[str, Any]: Capability inventory report
        """
        pass
    
    @abstractmethod
    def generate_dependency_report(self, dependencies: List[DependencyMapping]) -> Dict[str, Any]:
        """
        Generate dependency usage report.
        
        Args:
            dependencies: Current dependency mappings
            
        Returns:
            Dict[str, Any]: Dependency usage report
        """
        pass
    
    @abstractmethod
    def export_report(self, report_data: Dict[str, Any], format: str, output_path: Path) -> None:
        """
        Export report data to specified format and path.
        
        Args:
            report_data: Report data to export
            format: Export format (json, csv, html, etc.)
            output_path: Path to write exported report
        """
        pass