"""
Analysis components for capability detection and dependency mapping.

This package contains the components responsible for analyzing discovered
applications to identify their capabilities and map dependencies.
"""

from .capability_analyzer import CapabilityAnalyzerImpl
from .strategies import (
    DocumentationAnalysisStrategy,
    HelpTextAnalysisStrategy,
    CommandLineInterfaceStrategy,
    APIEndpointStrategy
)

__all__ = [
    "CapabilityAnalyzerImpl",
    "DocumentationAnalysisStrategy",
    "HelpTextAnalysisStrategy", 
    "CommandLineInterfaceStrategy",
    "APIEndpointStrategy"
]