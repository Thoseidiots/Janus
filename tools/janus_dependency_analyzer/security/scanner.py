"""
Security-first scanning implementation for the Janus Dependency Analyzer.

This module provides security filtering around platform scanners to ensure
sensitive files and restricted areas are never accessed or stored.

Requirements: 9.1, 9.2, 9.6
"""

import fnmatch
import logging
from typing import List, Optional, Tuple

from ..core.models import Application

logger = logging.getLogger(__name__)

# Sensitive path patterns to avoid (cross-platform)
SENSITIVE_PATH_PATTERNS = [
    # Password/credential files
    "*.pem", "*.key", "*.p12", "*.pfx", "*.crt", "*.cer",
    # SSH keys
    ".ssh/*", "id_rsa*", "id_ed25519*", "id_ecdsa*",
    # Password databases
    "*.kdbx", "*.1pif",
    # Personal documents
    "*/Documents/*", "*/Desktop/*", "*/Downloads/*",
    # System sensitive dirs
    "/etc/shadow", "/etc/passwd", "/etc/sudoers",
    "*/AppData/Roaming/Microsoft/Credentials/*",
    "*/Library/Keychains/*",
    # Private keys
    "*.private", "*_private_key*",
]


class SensitivePathFilter:
    """
    Determines whether a path should be excluded from scanning for security reasons.
    Uses fnmatch glob patterns.

    Req 9.1: The System_Scanner SHALL not access or store sensitive files including
    password databases, private keys, or personal documents.
    """

    def __init__(self, patterns: Optional[List[str]] = None):
        """
        Initialise the filter.

        Args:
            patterns: List of fnmatch glob patterns to treat as sensitive.
                      Defaults to SENSITIVE_PATH_PATTERNS if None.
        """
        self._patterns: List[str] = list(patterns) if patterns is not None else list(SENSITIVE_PATH_PATTERNS)

    def is_sensitive(self, path: str) -> bool:
        """
        Return True if *path* matches any sensitive pattern.

        The check is performed against both the full path and the basename
        so that patterns like ``*.pem`` match regardless of directory depth.

        Args:
            path: File system path to evaluate.

        Returns:
            bool: True if the path is considered sensitive.
        """
        import os
        basename = os.path.basename(path)

        for pattern in self._patterns:
            # Match against the full path
            if fnmatch.fnmatch(path, pattern):
                logger.debug("Path '%s' matched sensitive pattern '%s'", path, pattern)
                return True
            # Also match against just the filename component
            if fnmatch.fnmatch(basename, pattern):
                logger.debug("Basename '%s' matched sensitive pattern '%s'", basename, pattern)
                return True

        return False

    def add_pattern(self, pattern: str) -> None:
        """
        Add a custom sensitive path pattern.

        Args:
            pattern: fnmatch glob pattern to add.
        """
        self._patterns.append(pattern)
        logger.debug("Added sensitive path pattern: %s", pattern)

    @property
    def patterns(self) -> List[str]:
        """Return a copy of the current pattern list."""
        return list(self._patterns)


class SecurityScanner:
    """
    Security-first wrapper around any PlatformScanner.

    Enforces:
    - Sensitive path avoidance (Req 9.1)
    - Access control respect — skips paths that raise PermissionError (Req 9.2)
    - Minimal privilege — does not attempt privilege escalation (Req 9.6)

    Wraps a PlatformScanner and filters its results.
    """

    def __init__(self, platform_scanner, sensitive_filter: Optional[SensitivePathFilter] = None):
        """
        Initialise the security scanner.

        Args:
            platform_scanner: Any PlatformScanner implementation whose
                              ``discover_applications()`` method will be called.
            sensitive_filter: Optional SensitivePathFilter instance.
                              Uses a default SensitivePathFilter if None.
        """
        self._platform_scanner = platform_scanner
        self._sensitive_filter: SensitivePathFilter = (
            sensitive_filter if sensitive_filter is not None else SensitivePathFilter()
        )
        self._skipped_paths: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_with_security(self) -> Tuple[List[Application], List[str]]:
        """
        Perform a security-filtered scan.

        Calls ``platform_scanner.discover_applications()``, then filters out
        any application whose ``installation_path`` or ``executable_path``
        is considered sensitive or raises a ``PermissionError`` during the
        check.

        Returns:
            Tuple[List[Application], List[str]]:
                - safe_apps: Applications that passed security filtering.
                - skipped_paths: Paths that were skipped due to security concerns.
        """
        self._skipped_paths = []
        safe_apps: List[Application] = []

        try:
            applications = self._platform_scanner.discover_applications()
        except PermissionError as exc:
            logger.warning("PermissionError during discover_applications(): %s", exc)
            return [], []

        for app in applications:
            skipped = False

            for path_attr in ("installation_path", "executable_path"):
                raw_path = getattr(app, path_attr, None)
                if raw_path is None:
                    continue

                path_str = str(raw_path)

                try:
                    if self._sensitive_filter.is_sensitive(path_str):
                        logger.info(
                            "Skipping app '%s': %s '%s' is sensitive",
                            app.name, path_attr, path_str,
                        )
                        self._skipped_paths.append(path_str)
                        skipped = True
                        break
                except PermissionError as exc:
                    logger.warning(
                        "PermissionError checking path '%s' for app '%s': %s",
                        path_str, app.name, exc,
                    )
                    self._skipped_paths.append(path_str)
                    skipped = True
                    break

            if not skipped:
                safe_apps.append(app)

        logger.info(
            "Security scan complete: %d safe apps, %d skipped paths",
            len(safe_apps), len(self._skipped_paths),
        )
        return safe_apps, list(self._skipped_paths)

    def is_path_safe(self, path: str) -> bool:
        """
        Return True if *path* is safe to scan (not sensitive).

        Args:
            path: File system path to evaluate.

        Returns:
            bool: True if the path is not sensitive.
        """
        return not self._sensitive_filter.is_sensitive(path)

    def get_skipped_paths(self) -> List[str]:
        """
        Return the list of paths skipped during the last call to
        ``scan_with_security()``.

        Returns:
            List[str]: Skipped paths (empty if no scan has been performed yet).
        """
        return list(self._skipped_paths)
