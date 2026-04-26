"""
Janus Dependency Analyzer

A cross-platform system that discovers installed applications, analyzes their capabilities,
maps current Janus dependencies, and generates prioritized implementation roadmaps for
internalizing external dependencies.
"""

__version__ = "0.1.0"
__author__ = "Janus Development Team"

from .core.models import (
    Application,
    ApplicationMetadata,
    Capability,
    CapabilityCategory,
    InterfaceType,
    DependencyMapping,
    UsagePattern,
    ImplementationRoadmap,
    TechnicalComponent,
    Milestone,
    PriorityScore,
    Configuration,
    Platform,
    ComplexityLevel,
    TestingRequirements,
    Risk,
    ScanResult,
    AnalysisContext,
    RankedCapability,
    ValidationResult,
    Parameter
)

from .core.interfaces import (
    SystemScanner,
    PlatformScanner,
    CapabilityAnalyzer,
    AnalysisStrategy,
    PriorityEngine,
    ConfigurationParser,
    ConfigurationPrettyPrinter,
    DependencyMapper,
    RoadmapGenerator,
    ReportGenerator
)

from .core.configuration import (
    JSONConfigurationParser,
    YAMLConfigurationParser,
    JSONConfigurationPrettyPrinter,
    YAMLConfigurationPrettyPrinter,
    DefaultConfigurationParser,
    DefaultConfigurationPrettyPrinter
)

__all__ = [
    # Models
    "Application",
    "ApplicationMetadata", 
    "Capability",
    "CapabilityCategory",
    "InterfaceType",
    "DependencyMapping",
    "UsagePattern",
    "ImplementationRoadmap",
    "TechnicalComponent",
    "Milestone",
    "PriorityScore",
    "Configuration",
    "Platform",
    "ComplexityLevel",
    "TestingRequirements",
    "Risk",
    "ScanResult",
    "AnalysisContext",
    "RankedCapability",
    "ValidationResult",
    "Parameter",
    
    # Interfaces
    "SystemScanner",
    "PlatformScanner",
    "CapabilityAnalyzer",
    "AnalysisStrategy",
    "PriorityEngine",
    "ConfigurationParser",
    "ConfigurationPrettyPrinter",
    "DependencyMapper",
    "RoadmapGenerator",
    "ReportGenerator",
    
    # Configuration implementations
    "JSONConfigurationParser",
    "YAMLConfigurationParser",
    "JSONConfigurationPrettyPrinter",
    "YAMLConfigurationPrettyPrinter",
    "DefaultConfigurationParser",
    "DefaultConfigurationPrettyPrinter"
]