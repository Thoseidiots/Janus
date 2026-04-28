"""
Property-based tests for Security and Privacy Protection.

This module implements property-based tests using Hypothesis to verify
universal correctness properties of the security components.

# Feature: janus-dependency-analyzer, Property 13: Security and Privacy Protection
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ..core.models import Application
from ..reports.generator import ReportGenerator
from ..security.audit import AuditLogger
from ..security.scanner import SecurityScanner, SensitivePathFilter


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Path strategy — mix of sensitive and safe paths
path_strategy = st.one_of(
    # Safe paths
    st.sampled_from(["/usr/bin/python3", "/usr/local/bin/git", "/opt/myapp/bin/app"]),
    # Sensitive paths
    st.sampled_from(["/home/user/.ssh/id_rsa", "/etc/shadow", "/home/user/certs/server.pem"]),
    # Generated paths
    st.text(min_size=1, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz/._-"),
)

# Audit event type strategy
event_type_strategy = st.sampled_from(
    ["scan_attempt", "permission_denied", "sensitive_path_skipped", "encryption"]
)


# ---------------------------------------------------------------------------
# Property 13: Security and Privacy Protection
# Feature: janus-dependency-analyzer, Property 13: Security and Privacy Protection
# Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 9.6
# ---------------------------------------------------------------------------


class TestSecurityAndPrivacyProtection:
    """
    **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 9.6**

    Property 13: Security and Privacy Protection
    For any system scan or report generation, the system SHALL avoid accessing
    sensitive files, respect access controls, encrypt stored metadata, provide
    comprehensive audit logs, sanitize sensitive information in reports, and
    operate with minimal required privileges.
    """

    # Feature: janus-dependency-analyzer, Property 13: Security and Privacy Protection

    @given(path=path_strategy)
    @settings(max_examples=25, deadline=None)
    def test_sensitive_filter_is_consistent(self, path: str) -> None:
        """
        is_sensitive(path) returns the same result on repeated calls.

        # Feature: janus-dependency-analyzer, Property 13: Security and Privacy Protection
        """
        f = SensitivePathFilter()
        result1 = f.is_sensitive(path)
        result2 = f.is_sensitive(path)
        assert result1 == result2, (
            f"is_sensitive('{path}') returned {result1} then {result2} — must be consistent"
        )

    @given(
        path=st.sampled_from(
            [
                "/usr/bin/python3",
                "/usr/local/bin/git",
                "/opt/myapp/bin/app",
                "/usr/lib/python3.11/lib.so",
            ]
        )
    )
    @settings(max_examples=25, deadline=None)
    def test_safe_paths_not_flagged_sensitive(self, path: str) -> None:
        """
        Known safe application paths are not flagged as sensitive.

        # Feature: janus-dependency-analyzer, Property 13: Security and Privacy Protection
        """
        f = SensitivePathFilter()
        assert f.is_sensitive(path) is False, (
            f"Safe path '{path}' was incorrectly flagged as sensitive"
        )

    @given(path=path_strategy, success=st.booleans())
    @settings(max_examples=25, deadline=None)
    def test_audit_entries_have_required_fields(self, path: str, success: bool) -> None:
        """
        Every audit log entry has timestamp, event_type, path, success, details.

        # Feature: janus-dependency-analyzer, Property 13: Security and Privacy Protection
        """
        logger = AuditLogger()
        logger.log_scan_attempt(path, success=success)
        entry = logger.get_audit_log()[0]
        for field in ("timestamp", "event_type", "path", "success", "details"):
            assert field in entry, (
                f"Audit entry missing required field '{field}': {entry}"
            )

    @given(paths=st.lists(path_strategy, min_size=1, max_size=10))
    @settings(max_examples=25, deadline=None)
    def test_audit_log_is_append_only(self, paths: list) -> None:
        """
        Each log_scan_attempt() call adds exactly one entry.

        # Feature: janus-dependency-analyzer, Property 13: Security and Privacy Protection
        """
        logger = AuditLogger()
        for i, path in enumerate(paths):
            logger.log_scan_attempt(path, success=True)
            assert len(logger.get_audit_log()) == i + 1, (
                f"Expected {i + 1} entries after {i + 1} log calls, "
                f"got {len(logger.get_audit_log())}"
            )

    @given(
        suffix=st.text(
            min_size=1,
            max_size=30,
            alphabet="abcdefghijklmnopqrstuvwxyz/._-",
        )
    )
    @settings(max_examples=25, deadline=None)
    def test_sanitize_path_removes_home_dir(self, suffix: str) -> None:
        """
        sanitize_path() replaces home dir prefix with ~ for any path under home.

        # Feature: janus-dependency-analyzer, Property 13: Security and Privacy Protection
        """
        home = "/home/testuser"
        path = home + "/" + suffix.lstrip("/")
        result = ReportGenerator.sanitize_path(path, home_dir=home)
        assert result.startswith("~"), (
            f"Expected ~ prefix, got: {result}"
        )
        assert home not in result, (
            f"Home dir still present in: {result}"
        )

    @given(
        safe_paths=st.lists(
            st.sampled_from(
                ["/usr/bin/python3", "/usr/local/bin/git", "/opt/app/bin/app"]
            ),
            min_size=0,
            max_size=5,
        ),
        sensitive_paths=st.lists(
            st.sampled_from(
                [
                    "/home/user/.ssh/id_rsa",
                    "/etc/shadow",
                    "/home/user/certs/server.pem",
                ]
            ),
            min_size=0,
            max_size=5,
        ),
    )
    @settings(max_examples=25, deadline=None)
    def test_security_scanner_never_returns_sensitive_apps(
        self, safe_paths: list, sensitive_paths: list
    ) -> None:
        """
        SecurityScanner.scan_with_security() never returns apps with sensitive paths.

        # Feature: janus-dependency-analyzer, Property 13: Security and Privacy Protection
        """
        # Build Application objects from safe and sensitive paths
        apps = []
        for p in safe_paths:
            app = Application(
                name="safe_app",
                installation_path=Path(p),
                executable_path=Path(p),
            )
            apps.append(app)
        for p in sensitive_paths:
            app = Application(
                name="sensitive_app",
                installation_path=Path(p),
                executable_path=Path(p),
            )
            apps.append(app)

        # Build mock platform scanner
        mock_scanner = MagicMock()
        mock_scanner.discover_applications.return_value = apps

        scanner = SecurityScanner(mock_scanner)
        safe_apps, skipped = scanner.scan_with_security()

        # Assert no returned app has a sensitive installation_path or executable_path
        sensitive_filter = SensitivePathFilter()
        for app in safe_apps:
            inst_path = str(app.installation_path)
            exec_path = str(app.executable_path)
            assert not sensitive_filter.is_sensitive(inst_path), (
                f"Returned app '{app.name}' has sensitive installation_path: {inst_path}"
            )
            assert not sensitive_filter.is_sensitive(exec_path), (
                f"Returned app '{app.name}' has sensitive executable_path: {exec_path}"
            )
