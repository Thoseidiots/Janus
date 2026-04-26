"""
Platform-specific scanners for application discovery.

This package contains the platform-specific implementations for discovering
installed applications on Windows, macOS, and Linux systems.
"""

from .base import BasePlatformScanner
from .system_scanner import SystemScannerImpl

__all__ = [
    "BasePlatformScanner",
    "SystemScannerImpl"
]