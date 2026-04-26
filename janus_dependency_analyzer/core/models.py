"""
Core data models for the Janus Dependency Analyzer.

This module defines all the fundamental data structures used throughout the system,
including applications, capabilities, dependencies, and analysis results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import uuid


class Platform(Enum):
    """Supported operating system platforms."""
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"


class CapabilityCategory(Enum):
    """Categories of capabilities that applications can provide."""
    FILE_PROCESSING = "file_processing"
    NETWORK_OPERATIONS = "network_operations"
    DATA_TRANSFORMATION = "data_transformation"
    USER_INTERFACE = "user_interface"
    SYSTEM_INTEGRATION = "system_integration"
    DEVELOPMENT_TOOLS = "development_tools"
    MULTIMEDIA = "multimedia"
    SECURITY = "security"
    DATABASE = "database"
    COMMUNICATION = "communication"


class InterfaceType(Enum):
    """Types of interfaces that applications expose."""
    COMMAND_LINE = "command_line"
    REST_API = "rest_api"
    GRAPHQL_API = "graphql_api"
    RPC = "rpc"
    LIBRARY = "library"
    GUI = "gui"
    WEB_INTERFACE = "web_interface"
    PLUGIN = "plugin"


class ComplexityLevel(Enum):
    """Complexity levels for implementation estimation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class Parameter:
    """Represents a parameter for a capability interface."""
    name: str
    type: str
    description: str
    required: bool = True
    default_value: Optional[str] = None


@dataclass
class ApplicationMetadata:
    """Metadata information about an installed application."""
    vendor: Optional[str] = None
    description: Optional[str] = None
    file_size: int = 0
    install_date: Optional[datetime] = None
    digital_signature: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    file_associations: List[str] = field(default_factory=list)
    registry_keys: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)


@dataclass
class Application:
    """Represents an installed application on the system."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    version: str = ""
    installation_path: Path = field(default_factory=Path)
    executable_path: Path = field(default_factory=Path)
    platform: Platform = Platform.LINUX
    metadata: ApplicationMetadata = field(default_factory=ApplicationMetadata)
    discovered_at: datetime = field(default_factory=datetime.now)
    last_analyzed: Optional[datetime] = None
    is_accessible: bool = True
    access_error: Optional[str] = None


@dataclass
class Capability:
    """Represents a capability provided by an application."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    application_id: str = ""
    name: str = ""
    category: CapabilityCategory = CapabilityCategory.SYSTEM_INTEGRATION
    description: str = ""
    interface_type: InterfaceType = InterfaceType.COMMAND_LINE
    parameters: List[Parameter] = field(default_factory=list)
    confidence_score: float = 0.0
    detection_method: str = ""
    examples: List[str] = field(default_factory=list)
    documentation_url: Optional[str] = None
    supported_formats: List[str] = field(default_factory=list)


@dataclass
class UsagePattern:
    """Represents how Janus uses an external application."""
    invocation_method: str  # subprocess, api_call, library_import, etc.
    frequency_per_day: int = 0
    parameters: List[str] = field(default_factory=list)
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    typical_data_size: str = "unknown"  # small, medium, large, very_large
    error_handling: str = "unknown"  # graceful, basic, none


@dataclass
class DependencyMapping:
    """Maps Janus components to external application usage."""
    janus_component: str
    external_application: str
    usage_pattern: UsagePattern
    frequency: int
    last_used: datetime
    context: str
    criticality: str = "medium"  # low, medium, high, critical
    alternatives: List[str] = field(default_factory=list)


@dataclass
class Risk:
    """Represents a risk in implementation roadmap."""
    description: str
    probability: str  # low, medium, high
    impact: str  # low, medium, high
    mitigation_strategy: str
    owner: Optional[str] = None


@dataclass
class TestingRequirements:
    """Testing requirements for implementation roadmap."""
    unit_tests: List[str] = field(default_factory=list)
    integration_tests: List[str] = field(default_factory=list)
    performance_tests: List[str] = field(default_factory=list)
    security_tests: List[str] = field(default_factory=list)
    compatibility_tests: List[str] = field(default_factory=list)
    coverage_target: float = 0.8


@dataclass
class TechnicalComponent:
    """Represents a technical component in an implementation roadmap."""
    name: str
    description: str
    effort_hours: int
    complexity: ComplexityLevel = ComplexityLevel.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    required_skills: List[str] = field(default_factory=list)
    external_dependencies: List[str] = field(default_factory=list)


@dataclass
class Milestone:
    """Represents a milestone in an implementation roadmap."""
    name: str
    description: str
    estimated_completion: int  # hours from start
    deliverables: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)


@dataclass
class ImplementationRoadmap:
    """Detailed implementation plan for replacing external dependency."""
    capability_id: str
    estimated_effort_hours: int
    timeline_weeks: int = 0
    required_team_size: int = 1
    technical_components: List[TechnicalComponent] = field(default_factory=list)
    milestones: List[Milestone] = field(default_factory=list)
    risks: List[Risk] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    testing_requirements: TestingRequirements = field(default_factory=TestingRequirements)
    budget_estimate: Optional[float] = None


@dataclass
class PriorityScore:
    """Priority scoring for capability implementation."""
    usage_frequency: float
    implementation_complexity: float
    security_benefit: float
    performance_impact: float
    total_score: float
    justification: str
    confidence: float = 0.0
    factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class RankedCapability:
    """Capability with priority ranking information."""
    capability: Capability
    priority_score: PriorityScore
    rank: int
    implementation_roadmap: Optional[ImplementationRoadmap] = None


@dataclass
class ChangeRecord:
    """Records a change event for an application in the catalog."""
    app_id: str
    app_name: str
    change_type: str  # "installed", "removed", "updated", "version_changed"
    timestamp: datetime = field(default_factory=datetime.now)
    previous_version: Optional[str] = None
    new_version: Optional[str] = None
    details: str = ""


@dataclass
class ScanResult:
    """Results from a system scan operation."""
    applications: List[Application] = field(default_factory=list)
    scan_start_time: datetime = field(default_factory=datetime.now)
    scan_end_time: Optional[datetime] = None
    platform: Platform = Platform.LINUX
    total_applications: int = 0
    accessible_applications: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    scan_type: str = "full"  # full, incremental
    
    def add_application(self, app: Application) -> None:
        """Add an application to the scan results."""
        self.applications.append(app)
        self.total_applications += 1
        if app.is_accessible:
            self.accessible_applications += 1
    
    def add_error(self, error: str) -> None:
        """Add an error message to the scan results."""
        self.errors.append(error)
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message to the scan results."""
        self.warnings.append(warning)
    
    def finalize(self) -> None:
        """Finalize the scan results by setting end time."""
        self.scan_end_time = datetime.now()


@dataclass
class AnalysisContext:
    """Context information for analysis operations."""
    config: 'Configuration'
    janus_codebase_path: Path
    max_frequency: int = 1000
    analysis_date: datetime = field(default_factory=datetime.now)
    previous_analysis: Optional[datetime] = None
    incremental_mode: bool = False


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, error: str) -> None:
        """Add a validation error."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add a validation warning."""
        self.warnings.append(warning)


@dataclass
class PriorityWeights:
    """Weights for priority calculation factors."""
    usage: float = 0.3
    complexity: float = 0.2
    security: float = 0.25
    performance: float = 0.25
    
    def normalize(self) -> 'PriorityWeights':
        """Normalize weights to sum to 1.0."""
        total = self.usage + self.complexity + self.security + self.performance
        if total == 0:
            return PriorityWeights()
        return PriorityWeights(
            usage=self.usage / total,
            complexity=self.complexity / total,
            security=self.security / total,
            performance=self.performance / total
        )


@dataclass
class Configuration:
    """Configuration for the dependency analyzer."""
    # Scanning configuration
    scan_exclusion_patterns: List[str] = field(default_factory=list)
    scan_timeout_seconds: int = 300
    max_applications_per_scan: int = 10000
    
    # Analysis configuration
    capability_detection_rules: Dict[str, Any] = field(default_factory=dict)
    analysis_timeout_seconds: int = 60
    min_confidence_threshold: float = 0.5
    
    # Priority configuration
    priority_weights: PriorityWeights = field(default_factory=PriorityWeights)
    
    # Security configuration
    respect_access_controls: bool = True
    encrypt_stored_data: bool = True
    audit_logging_enabled: bool = True
    
    # Performance configuration
    max_concurrent_analyses: int = 4
    cache_analysis_results: bool = True
    cache_expiry_hours: int = 24
    
    # Output configuration
    report_formats: List[str] = field(default_factory=lambda: ["json", "html"])
    include_charts: bool = True
    sanitize_paths: bool = True
    
    def validate(self) -> ValidationResult:
        """Validate the configuration."""
        result = ValidationResult(is_valid=True)
        
        if self.scan_timeout_seconds <= 0:
            result.add_error("scan_timeout_seconds must be positive")
        
        if self.max_applications_per_scan <= 0:
            result.add_error("max_applications_per_scan must be positive")
        
        if self.analysis_timeout_seconds <= 0:
            result.add_error("analysis_timeout_seconds must be positive")
        
        if not (0.0 <= self.min_confidence_threshold <= 1.0):
            result.add_error("min_confidence_threshold must be between 0.0 and 1.0")
        
        if self.max_concurrent_analyses <= 0:
            result.add_error("max_concurrent_analyses must be positive")
        
        if self.cache_expiry_hours <= 0:
            result.add_error("cache_expiry_hours must be positive")
        
        # Validate priority weights
        weights = self.priority_weights
        total_weight = weights.usage + weights.complexity + weights.security + weights.performance
        if total_weight == 0:
            result.add_error("Priority weights cannot all be zero")
        
        # Validate report formats
        valid_formats = {"json", "csv", "html", "xml", "yaml"}
        for fmt in self.report_formats:
            if fmt not in valid_formats:
                result.add_warning(f"Unknown report format: {fmt}")
        
        return result