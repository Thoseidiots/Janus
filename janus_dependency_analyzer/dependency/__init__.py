"""
Dependency mapping module for the Janus Dependency Analyzer.

This module provides tools for scanning the Janus codebase to identify
external application invocations and potential beneficial applications.
"""

from .mapper import (
    ExternalInvocation,
    DependencyMapping,
    DependencyMapper,
    FunctionalityGap,
    PotentialApplicationIdentifier,
)

__all__ = [
    "ExternalInvocation",
    "DependencyMapping",
    "DependencyMapper",
    "FunctionalityGap",
    "PotentialApplicationIdentifier",
]
