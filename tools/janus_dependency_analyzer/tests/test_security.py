"""
Unit tests for the security scanning implementation.

Tests cover:
- SensitivePathFilter.is_sensitive() for known sensitive and safe paths
- SensitivePathFilter.add_pattern() for custom patterns
- SecurityScanner.scan_with_security() filtering behaviour
- SecurityScanner.is_path_safe() convenience method
- SecurityScanner.get_skipped_paths() result tracking

Requirements: 9.1, 9.2, 9.6
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ..security.scanner import (
    SENSITIVE_PATH_PATTERNS,
    SecurityScanner,
    SensitivePathFilter,
)
from ..core.models import Application, Platform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app(name: str, install_path: str, exe_path: str) -> Application:
    """Create a minimal Application for testing."""
    return Application(
        name=name,
        installation_path=Path(install_path),
        executable_path=Path(exe_path),
        platform=Platform.LINUX,
    )


def _make_mock_scanner(apps):
    """Return a MagicMock platform scanner that yields *apps*."""
    mock = MagicMock()
    mock.discover_applications.return_value = apps
    return mock


# ---------------------------------------------------------------------------
# SensitivePathFilter tests
# ---------------------------------------------------------------------------

class TestSensitivePathFilter:
    """Tests for SensitivePathFilter."""

    def test_is_sensitive_pem_file(self):
        """is_sensitive() returns True for .pem files."""
        f = SensitivePathFilter()
        assert f.is_sensitive("/home/user/certs/server.pem") is True

    def test_is_sensitive_key_file(self):
        """is_sensitive() returns True for .key files."""
        f = SensitivePathFilter()
        assert f.is_sensitive("/etc/ssl/private/server.key") is True

    def test_is_sensitive_id_rsa(self):
        """is_sensitive() returns True for id_rsa SSH key files."""
        f = SensitivePathFilter()
        assert f.is_sensitive("/home/user/.ssh/id_rsa") is True

    def test_is_sensitive_id_rsa_pub(self):
        """is_sensitive() returns True for id_rsa.pub (matches id_rsa* pattern)."""
        f = SensitivePathFilter()
        assert f.is_sensitive("/home/user/.ssh/id_rsa.pub") is True

    def test_is_sensitive_id_ed25519(self):
        """is_sensitive() returns True for id_ed25519 SSH key files."""
        f = SensitivePathFilter()
        assert f.is_sensitive("/home/user/.ssh/id_ed25519") is True

    def test_is_sensitive_kdbx(self):
        """is_sensitive() returns True for KeePass database files."""
        f = SensitivePathFilter()
        assert f.is_sensitive("/home/user/passwords.kdbx") is True

    def test_is_sensitive_private_key_suffix(self):
        """is_sensitive() returns True for files ending in _private_key."""
        f = SensitivePathFilter()
        assert f.is_sensitive("/home/user/my_private_key") is True

    def test_is_sensitive_etc_shadow(self):
        """is_sensitive() returns True for /etc/shadow."""
        f = SensitivePathFilter()
        assert f.is_sensitive("/etc/shadow") is True

    def test_is_sensitive_etc_passwd(self):
        """is_sensitive() returns True for /etc/passwd."""
        f = SensitivePathFilter()
        assert f.is_sensitive("/etc/passwd") is True

    def test_is_sensitive_documents_dir(self):
        """is_sensitive() returns True for paths inside ~/Documents."""
        f = SensitivePathFilter()
        assert f.is_sensitive("/home/user/Documents/report.pdf") is True

    def test_is_sensitive_desktop_dir(self):
        """is_sensitive() returns True for paths inside ~/Desktop."""
        f = SensitivePathFilter()
        assert f.is_sensitive("/home/user/Desktop/notes.txt") is True

    def test_is_sensitive_downloads_dir(self):
        """is_sensitive() returns True for paths inside ~/Downloads."""
        f = SensitivePathFilter()
        assert f.is_sensitive("/home/user/Downloads/installer.deb") is True

    def test_is_not_sensitive_normal_app(self):
        """is_sensitive() returns False for a normal application path."""
        f = SensitivePathFilter()
        assert f.is_sensitive("/usr/bin/python3") is False

    def test_is_not_sensitive_usr_local(self):
        """is_sensitive() returns False for /usr/local/bin paths."""
        f = SensitivePathFilter()
        assert f.is_sensitive("/usr/local/bin/git") is False

    def test_is_not_sensitive_opt_app(self):
        """is_sensitive() returns False for /opt application paths."""
        f = SensitivePathFilter()
        assert f.is_sensitive("/opt/myapp/bin/myapp") is False

    def test_is_not_sensitive_windows_program_files(self):
        """is_sensitive() returns False for Windows Program Files paths."""
        f = SensitivePathFilter()
        assert f.is_sensitive("C:\\Program Files\\MyApp\\myapp.exe") is False

    def test_add_pattern_custom(self):
        """add_pattern() adds a custom pattern that is then matched."""
        f = SensitivePathFilter()
        f.add_pattern("*.secret")
        assert f.is_sensitive("/home/user/config.secret") is True

    def test_add_pattern_does_not_affect_other_paths(self):
        """add_pattern() only affects paths matching the new pattern."""
        f = SensitivePathFilter()
        f.add_pattern("*.secret")
        assert f.is_sensitive("/usr/bin/python3") is False

    def test_add_pattern_multiple(self):
        """Multiple add_pattern() calls all take effect."""
        f = SensitivePathFilter()
        f.add_pattern("*.secret")
        f.add_pattern("*/vault/*")
        assert f.is_sensitive("/home/user/vault/token") is True
        assert f.is_sensitive("/home/user/config.secret") is True
        assert f.is_sensitive("/usr/bin/python3") is False

    def test_custom_patterns_override_defaults(self):
        """Passing custom patterns replaces the default list."""
        f = SensitivePathFilter(patterns=["*.secret"])
        # Default patterns should NOT apply
        assert f.is_sensitive("/home/user/.ssh/id_rsa") is False
        # Custom pattern should apply
        assert f.is_sensitive("/home/user/config.secret") is True

    def test_empty_patterns_nothing_sensitive(self):
        """An empty pattern list means nothing is sensitive."""
        f = SensitivePathFilter(patterns=[])
        assert f.is_sensitive("/etc/shadow") is False
        assert f.is_sensitive("/home/user/.ssh/id_rsa") is False


# ---------------------------------------------------------------------------
# SecurityScanner tests
# ---------------------------------------------------------------------------

class TestSecurityScanner:
    """Tests for SecurityScanner."""

    # --- scan_with_security ---

    def test_scan_returns_all_apps_when_none_sensitive(self):
        """scan_with_security() returns all apps when no paths are sensitive."""
        apps = [
            _make_app("Python", "/usr/lib/python3", "/usr/bin/python3"),
            _make_app("Git", "/usr/lib/git-core", "/usr/bin/git"),
        ]
        scanner = SecurityScanner(_make_mock_scanner(apps))
        safe_apps, skipped = scanner.scan_with_security()

        assert len(safe_apps) == 2
        assert len(skipped) == 0
        assert {a.name for a in safe_apps} == {"Python", "Git"}

    def test_scan_filters_app_with_sensitive_installation_path(self):
        """scan_with_security() excludes apps whose installation_path is sensitive."""
        apps = [
            _make_app("SafeApp", "/usr/lib/safeapp", "/usr/bin/safeapp"),
            _make_app("KeyApp", "/home/user/.ssh", "/home/user/.ssh/id_rsa"),
        ]
        scanner = SecurityScanner(_make_mock_scanner(apps))
        safe_apps, skipped = scanner.scan_with_security()

        assert len(safe_apps) == 1
        assert safe_apps[0].name == "SafeApp"

    def test_scan_filters_app_with_sensitive_executable_path(self):
        """scan_with_security() excludes apps whose executable_path is sensitive."""
        apps = [
            _make_app("SafeApp", "/usr/lib/safeapp", "/usr/bin/safeapp"),
            _make_app("CertApp", "/usr/lib/certapp", "/etc/ssl/private/server.key"),
        ]
        scanner = SecurityScanner(_make_mock_scanner(apps))
        safe_apps, skipped = scanner.scan_with_security()

        assert len(safe_apps) == 1
        assert safe_apps[0].name == "SafeApp"

    def test_scan_returns_skipped_paths_for_filtered_apps(self):
        """scan_with_security() populates skipped_paths for filtered apps."""
        apps = [
            _make_app("SafeApp", "/usr/lib/safeapp", "/usr/bin/safeapp"),
            _make_app("KeyApp", "/home/user/.ssh", "/home/user/.ssh/id_rsa"),
        ]
        scanner = SecurityScanner(_make_mock_scanner(apps))
        safe_apps, skipped = scanner.scan_with_security()

        assert len(skipped) >= 1
        # At least one of the sensitive paths should appear in skipped
        assert any("id_rsa" in p or ".ssh" in p for p in skipped)

    def test_scan_returns_empty_when_all_sensitive(self):
        """scan_with_security() returns empty safe list when all apps are sensitive."""
        apps = [
            _make_app("KeyApp1", "/home/user/.ssh", "/home/user/.ssh/id_rsa"),
            _make_app("KeyApp2", "/home/user/certs", "/home/user/certs/server.pem"),
        ]
        scanner = SecurityScanner(_make_mock_scanner(apps))
        safe_apps, skipped = scanner.scan_with_security()

        assert safe_apps == []
        assert len(skipped) >= 1

    def test_scan_returns_empty_on_permission_error_from_scanner(self):
        """scan_with_security() returns empty lists if discover_applications raises PermissionError."""
        mock_scanner = MagicMock()
        mock_scanner.discover_applications.side_effect = PermissionError("Access denied")

        scanner = SecurityScanner(mock_scanner)
        safe_apps, skipped = scanner.scan_with_security()

        assert safe_apps == []
        assert skipped == []

    def test_scan_uses_custom_sensitive_filter(self):
        """scan_with_security() respects a custom SensitivePathFilter."""
        custom_filter = SensitivePathFilter(patterns=["*/vault/*"])
        apps = [
            _make_app("VaultApp", "/home/user/vault", "/home/user/vault/app"),
            _make_app("SafeApp", "/usr/bin", "/usr/bin/safeapp"),
        ]
        scanner = SecurityScanner(_make_mock_scanner(apps), sensitive_filter=custom_filter)
        safe_apps, skipped = scanner.scan_with_security()

        assert len(safe_apps) == 1
        assert safe_apps[0].name == "SafeApp"

    def test_scan_resets_skipped_paths_between_calls(self):
        """Consecutive scan_with_security() calls do not accumulate skipped paths."""
        apps_first = [
            _make_app("KeyApp", "/home/user/.ssh", "/home/user/.ssh/id_rsa"),
        ]
        apps_second = [
            _make_app("SafeApp", "/usr/bin", "/usr/bin/safeapp"),
        ]
        mock_scanner = MagicMock()
        mock_scanner.discover_applications.side_effect = [apps_first, apps_second]

        scanner = SecurityScanner(mock_scanner)

        _, skipped_first = scanner.scan_with_security()
        assert len(skipped_first) >= 1

        _, skipped_second = scanner.scan_with_security()
        assert skipped_second == []

    # --- is_path_safe ---

    def test_is_path_safe_returns_true_for_normal_path(self):
        """is_path_safe() returns True for a normal application path."""
        scanner = SecurityScanner(_make_mock_scanner([]))
        assert scanner.is_path_safe("/usr/bin/python3") is True

    def test_is_path_safe_returns_false_for_sensitive_path(self):
        """is_path_safe() returns False for a sensitive path."""
        scanner = SecurityScanner(_make_mock_scanner([]))
        assert scanner.is_path_safe("/home/user/.ssh/id_rsa") is False

    def test_is_path_safe_returns_false_for_pem(self):
        """is_path_safe() returns False for a .pem file."""
        scanner = SecurityScanner(_make_mock_scanner([]))
        assert scanner.is_path_safe("/etc/ssl/certs/server.pem") is False

    # --- get_skipped_paths ---

    def test_get_skipped_paths_empty_before_scan(self):
        """get_skipped_paths() returns empty list before any scan."""
        scanner = SecurityScanner(_make_mock_scanner([]))
        assert scanner.get_skipped_paths() == []

    def test_get_skipped_paths_returns_paths_from_last_scan(self):
        """get_skipped_paths() returns the skipped paths from the last scan."""
        apps = [
            _make_app("KeyApp", "/home/user/.ssh", "/home/user/.ssh/id_rsa"),
            _make_app("SafeApp", "/usr/bin", "/usr/bin/safeapp"),
        ]
        scanner = SecurityScanner(_make_mock_scanner(apps))
        _, skipped = scanner.scan_with_security()

        # get_skipped_paths() should return the same list
        assert scanner.get_skipped_paths() == skipped

    def test_get_skipped_paths_returns_copy(self):
        """get_skipped_paths() returns a copy, not the internal list."""
        apps = [
            _make_app("KeyApp", "/home/user/.ssh", "/home/user/.ssh/id_rsa"),
        ]
        scanner = SecurityScanner(_make_mock_scanner(apps))
        scanner.scan_with_security()

        paths1 = scanner.get_skipped_paths()
        paths1.append("mutated")
        paths2 = scanner.get_skipped_paths()

        assert "mutated" not in paths2
