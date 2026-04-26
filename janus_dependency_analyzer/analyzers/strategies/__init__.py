"""
Analysis strategies for capability detection.

This package contains the various strategies used to analyze applications
and extract their capabilities through different methods.
"""

from .documentation_strategy import DocumentationAnalysisStrategy
from .help_text_strategy import HelpTextAnalysisStrategy
from .cli_strategy import CommandLineInterfaceStrategy
from .api_strategy import APIEndpointStrategy
from .appx_manifest_strategy import AppxManifestStrategy
from .oxpecker_strategy import OxpeckerStrategy
from .file_association_strategy import FileAssociationStrategy

__all__ = [
    "DocumentationAnalysisStrategy",
    "HelpTextAnalysisStrategy",
    "CommandLineInterfaceStrategy",
    "APIEndpointStrategy",
    "AppxManifestStrategy",
    "OxpeckerStrategy",
    "FileAssociationStrategy",
]