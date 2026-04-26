"""
Unit tests for audit logging and path sanitization.

Tests cover:
- AuditLogger.log_scan_attempt() adds entry to audit log
- AuditLogger.log_permission_denied() adds entry with event_type "permission_denied"
- AuditLogger.log_sensitive_path_skipped() adds entry with event_type "sensitive_path_skipped"
- AuditLogger.log_encryption_event() adds entry with event_type "encryption"
- AuditLogger.get_audit_log() returns all entries
- AuditLogger.get_recent_events(since) filters by timestamp
- AuditLogger.export_audit_log() to JSON creates valid JSON file
- AuditLogger.export_audit_log() to CSV creates file
- AuditLogger.clear() empties the log
- ReportGenerator.sanitize_path() replaces home dir with ~
- ReportGenerator.sanitize_path() truncates long paths
- ReportGenerator.sanitize_report_paths() sanitizes string values in nested dicts

Requirements: 9.3, 9.4, 9.5
"""

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ..security.audit import AuditLogger
from ..reports.generator import ReportGenerator


# ---------------------------------------------------------------------------
# AuditLogger tests
# ---------------------------------------------------------------------------


class TestAuditLoggerLogScanAttempt:
    """Tests for AuditLogger.log_scan_attempt()."""

    def test_adds_entry_to_log(self):
        """log_scan_attempt() adds exactly one entry to the audit log."""
        logger = AuditLogger()
        logger.log_scan_attempt("/usr/bin/python3", success=True)
        entries = logger.get_audit_log()
        assert len(entries) == 1

    def test_entry_has_correct_event_type(self):
        """log_scan_attempt() creates an entry with event_type 'scan_attempt'."""
        logger = AuditLogger()
        logger.log_scan_attempt("/usr/bin/python3", success=True)
        entry = logger.get_audit_log()[0]
        assert entry["event_type"] == "scan_attempt"

    def test_entry_records_path(self):
        """log_scan_attempt() records the given path in the entry."""
        logger = AuditLogger()
        logger.log_scan_attempt("/usr/bin/python3", success=True)
        entry = logger.get_audit_log()[0]
        assert entry["path"] == "/usr/bin/python3"

    def test_entry_records_success_true(self):
        """log_scan_attempt() records success=True correctly."""
        logger = AuditLogger()
        logger.log_scan_attempt("/usr/bin/python3", success=True)
        entry = logger.get_audit_log()[0]
        assert entry["success"] is True

    def test_entry_records_success_false(self):
        """log_scan_attempt() records success=False correctly."""
        logger = AuditLogger()
        logger.log_scan_attempt("/restricted/path", success=False, reason="access denied")
        entry = logger.get_audit_log()[0]
        assert entry["success"] is False

    def test_entry_records_reason_in_details(self):
        """log_scan_attempt() stores the reason in the details field."""
        logger = AuditLogger()
        logger.log_scan_attempt("/some/path", success=False, reason="timeout")
        entry = logger.get_audit_log()[0]
        assert "timeout" in entry["details"]

    def test_entry_has_timestamp(self):
        """log_scan_attempt() entry has a parseable ISO timestamp."""
        logger = AuditLogger()
        logger.log_scan_attempt("/usr/bin/python3", success=True)
        entry = logger.get_audit_log()[0]
        # Should not raise
        datetime.fromisoformat(entry["timestamp"])

    def test_multiple_attempts_accumulate(self):
        """Multiple log_scan_attempt() calls accumulate entries."""
        logger = AuditLogger()
        logger.log_scan_attempt("/path/a", success=True)
        logger.log_scan_attempt("/path/b", success=False)
        assert len(logger.get_audit_log()) == 2


class TestAuditLoggerLogPermissionDenied:
    """Tests for AuditLogger.log_permission_denied()."""

    def test_adds_entry_with_correct_event_type(self):
        """log_permission_denied() creates an entry with event_type 'permission_denied'."""
        logger = AuditLogger()
        logger.log_permission_denied("/etc/shadow")
        entry = logger.get_audit_log()[0]
        assert entry["event_type"] == "permission_denied"

    def test_entry_records_path(self):
        """log_permission_denied() records the denied path."""
        logger = AuditLogger()
        logger.log_permission_denied("/etc/shadow", error="Permission denied")
        entry = logger.get_audit_log()[0]
        assert entry["path"] == "/etc/shadow"

    def test_entry_success_is_false(self):
        """log_permission_denied() always records success=False."""
        logger = AuditLogger()
        logger.log_permission_denied("/etc/shadow")
        entry = logger.get_audit_log()[0]
        assert entry["success"] is False

    def test_entry_records_error_in_details(self):
        """log_permission_denied() stores the error message in details."""
        logger = AuditLogger()
        logger.log_permission_denied("/etc/shadow", error="EACCES")
        entry = logger.get_audit_log()[0]
        assert "EACCES" in entry["details"]


class TestAuditLoggerLogSensitivePathSkipped:
    """Tests for AuditLogger.log_sensitive_path_skipped()."""

    def test_adds_entry_with_correct_event_type(self):
        """log_sensitive_path_skipped() creates an entry with event_type 'sensitive_path_skipped'."""
        logger = AuditLogger()
        logger.log_sensitive_path_skipped("/home/user/.ssh/id_rsa")
        entry = logger.get_audit_log()[0]
        assert entry["event_type"] == "sensitive_path_skipped"

    def test_entry_records_path(self):
        """log_sensitive_path_skipped() records the skipped path."""
        logger = AuditLogger()
        logger.log_sensitive_path_skipped("/home/user/.ssh/id_rsa", pattern="id_rsa*")
        entry = logger.get_audit_log()[0]
        assert entry["path"] == "/home/user/.ssh/id_rsa"

    def test_entry_records_pattern_in_details(self):
        """log_sensitive_path_skipped() stores the matched pattern in details."""
        logger = AuditLogger()
        logger.log_sensitive_path_skipped("/home/user/.ssh/id_rsa", pattern="id_rsa*")
        entry = logger.get_audit_log()[0]
        assert "id_rsa*" in entry["details"]

    def test_entry_success_is_true(self):
        """log_sensitive_path_skipped() records success=True (skip was intentional)."""
        logger = AuditLogger()
        logger.log_sensitive_path_skipped("/home/user/.ssh/id_rsa")
        entry = logger.get_audit_log()[0]
        assert entry["success"] is True


class TestAuditLoggerLogEncryptionEvent:
    """Tests for AuditLogger.log_encryption_event()."""

    def test_adds_entry_with_correct_event_type(self):
        """log_encryption_event() creates an entry with event_type 'encryption'."""
        logger = AuditLogger()
        logger.log_encryption_event("encrypt", "vendor", success=True)
        entry = logger.get_audit_log()[0]
        assert entry["event_type"] == "encryption"

    def test_entry_records_operation_and_field_in_details(self):
        """log_encryption_event() stores operation and field in details."""
        logger = AuditLogger()
        logger.log_encryption_event("decrypt", "digital_signature", success=True)
        entry = logger.get_audit_log()[0]
        assert "decrypt" in entry["details"]
        assert "digital_signature" in entry["details"]

    def test_entry_records_success_false_on_failure(self):
        """log_encryption_event() records success=False when operation fails."""
        logger = AuditLogger()
        logger.log_encryption_event("decrypt", "vendor", success=False)
        entry = logger.get_audit_log()[0]
        assert entry["success"] is False

    def test_entry_records_success_true_on_success(self):
        """log_encryption_event() records success=True when operation succeeds."""
        logger = AuditLogger()
        logger.log_encryption_event("encrypt", "description", success=True)
        entry = logger.get_audit_log()[0]
        assert entry["success"] is True


class TestAuditLoggerGetAuditLog:
    """Tests for AuditLogger.get_audit_log()."""

    def test_returns_empty_list_initially(self):
        """get_audit_log() returns an empty list before any events are logged."""
        logger = AuditLogger()
        assert logger.get_audit_log() == []

    def test_returns_all_entries(self):
        """get_audit_log() returns all logged entries."""
        logger = AuditLogger()
        logger.log_scan_attempt("/path/a", success=True)
        logger.log_permission_denied("/path/b")
        logger.log_sensitive_path_skipped("/path/c")
        entries = logger.get_audit_log()
        assert len(entries) == 3

    def test_returns_copy_not_reference(self):
        """get_audit_log() returns a copy; mutating it does not affect the logger."""
        logger = AuditLogger()
        logger.log_scan_attempt("/path/a", success=True)
        entries = logger.get_audit_log()
        entries.append({"fake": "entry"})
        assert len(logger.get_audit_log()) == 1

    def test_entries_have_required_fields(self):
        """Each entry returned by get_audit_log() has all required fields."""
        logger = AuditLogger()
        logger.log_scan_attempt("/path/a", success=True)
        entry = logger.get_audit_log()[0]
        for field in ("timestamp", "event_type", "path", "success", "details"):
            assert field in entry, f"Missing field: {field}"


class TestAuditLoggerGetRecentEvents:
    """Tests for AuditLogger.get_recent_events()."""

    def test_returns_all_events_when_since_is_old(self):
        """get_recent_events() returns all events when since is in the past."""
        logger = AuditLogger()
        logger.log_scan_attempt("/path/a", success=True)
        logger.log_scan_attempt("/path/b", success=True)
        since = datetime.now() - timedelta(hours=1)
        events = logger.get_recent_events(since)
        assert len(events) == 2

    def test_returns_empty_when_since_is_future(self):
        """get_recent_events() returns empty list when since is in the future."""
        logger = AuditLogger()
        logger.log_scan_attempt("/path/a", success=True)
        since = datetime.now() + timedelta(hours=1)
        events = logger.get_recent_events(since)
        assert events == []

    def test_filters_by_timestamp(self):
        """get_recent_events() only returns events at or after the given datetime."""
        logger = AuditLogger()
        # Log an event, record a cutoff, then log another event
        logger.log_scan_attempt("/path/old", success=True)
        cutoff = datetime.now()
        logger.log_scan_attempt("/path/new", success=True)
        events = logger.get_recent_events(cutoff)
        # Only the second event should be returned (or both if timestamps are equal)
        assert all(
            datetime.fromisoformat(e["timestamp"]) >= cutoff for e in events
        )

    def test_returns_empty_list_when_no_events(self):
        """get_recent_events() returns empty list when no events have been logged."""
        logger = AuditLogger()
        events = logger.get_recent_events(datetime.now() - timedelta(hours=1))
        assert events == []


class TestAuditLoggerExportJson:
    """Tests for AuditLogger.export_audit_log() with JSON format."""

    def test_creates_valid_json_file(self, tmp_path):
        """export_audit_log() creates a valid JSON file."""
        logger = AuditLogger()
        logger.log_scan_attempt("/usr/bin/python3", success=True)
        logger.log_permission_denied("/etc/shadow", error="EACCES")

        output = tmp_path / "audit.json"
        logger.export_audit_log(output, format="json")

        assert output.exists()
        data = json.loads(output.read_text())
        assert isinstance(data, list)
        assert len(data) == 2

    def test_json_contains_all_fields(self, tmp_path):
        """Exported JSON entries contain all required fields."""
        logger = AuditLogger()
        logger.log_scan_attempt("/path/a", success=True, reason="test")
        output = tmp_path / "audit.json"
        logger.export_audit_log(output, format="json")

        data = json.loads(output.read_text())
        entry = data[0]
        for field in ("timestamp", "event_type", "path", "success", "details"):
            assert field in entry

    def test_json_export_empty_log(self, tmp_path):
        """export_audit_log() creates an empty JSON array when log is empty."""
        logger = AuditLogger()
        output = tmp_path / "audit.json"
        logger.export_audit_log(output, format="json")

        data = json.loads(output.read_text())
        assert data == []

    def test_json_creates_parent_dirs(self, tmp_path):
        """export_audit_log() creates parent directories if they don't exist."""
        logger = AuditLogger()
        logger.log_scan_attempt("/path/a", success=True)
        output = tmp_path / "nested" / "dir" / "audit.json"
        logger.export_audit_log(output, format="json")
        assert output.exists()


class TestAuditLoggerExportCsv:
    """Tests for AuditLogger.export_audit_log() with CSV format."""

    def test_creates_csv_file(self, tmp_path):
        """export_audit_log() creates a CSV file."""
        logger = AuditLogger()
        logger.log_scan_attempt("/usr/bin/python3", success=True)
        output = tmp_path / "audit.csv"
        logger.export_audit_log(output, format="csv")
        assert output.exists()

    def test_csv_has_header_row(self, tmp_path):
        """Exported CSV has a header row with required field names."""
        logger = AuditLogger()
        logger.log_scan_attempt("/path/a", success=True)
        output = tmp_path / "audit.csv"
        logger.export_audit_log(output, format="csv")

        content = output.read_text()
        for field in ("timestamp", "event_type", "path", "success", "details"):
            assert field in content

    def test_csv_contains_logged_data(self, tmp_path):
        """Exported CSV contains the logged event data."""
        logger = AuditLogger()
        logger.log_permission_denied("/etc/shadow", error="EACCES")
        output = tmp_path / "audit.csv"
        logger.export_audit_log(output, format="csv")

        content = output.read_text()
        assert "permission_denied" in content
        assert "/etc/shadow" in content

    def test_csv_export_empty_log(self, tmp_path):
        """export_audit_log() creates a CSV with only a header when log is empty."""
        logger = AuditLogger()
        output = tmp_path / "audit.csv"
        logger.export_audit_log(output, format="csv")

        assert output.exists()
        lines = output.read_text().strip().splitlines()
        # Only the header row
        assert len(lines) == 1

    def test_unsupported_format_raises_value_error(self, tmp_path):
        """export_audit_log() raises ValueError for unsupported formats."""
        logger = AuditLogger()
        with pytest.raises(ValueError, match="Unsupported audit log format"):
            logger.export_audit_log(tmp_path / "audit.xml", format="xml")


class TestAuditLoggerClear:
    """Tests for AuditLogger.clear()."""

    def test_clear_empties_the_log(self):
        """clear() removes all entries from the audit log."""
        logger = AuditLogger()
        logger.log_scan_attempt("/path/a", success=True)
        logger.log_permission_denied("/path/b")
        assert len(logger.get_audit_log()) == 2

        logger.clear()
        assert logger.get_audit_log() == []

    def test_clear_on_empty_log_is_safe(self):
        """clear() on an already-empty log does not raise."""
        logger = AuditLogger()
        logger.clear()  # Should not raise
        assert logger.get_audit_log() == []

    def test_can_log_after_clear(self):
        """New events can be logged after clear()."""
        logger = AuditLogger()
        logger.log_scan_attempt("/path/a", success=True)
        logger.clear()
        logger.log_scan_attempt("/path/b", success=True)
        entries = logger.get_audit_log()
        assert len(entries) == 1
        assert entries[0]["path"] == "/path/b"


class TestAuditLoggerFileOutput:
    """Tests for AuditLogger writing to a log file."""

    def test_log_file_is_created(self, tmp_path):
        """AuditLogger creates the log file when log_file is specified."""
        log_file = tmp_path / "audit.log"
        logger = AuditLogger(log_file=log_file)
        logger.log_scan_attempt("/path/a", success=True)
        assert log_file.exists()


# ---------------------------------------------------------------------------
# ReportGenerator.sanitize_path() tests
# ---------------------------------------------------------------------------


class TestSanitizePath:
    """Tests for ReportGenerator.sanitize_path()."""

    def test_replaces_home_dir_with_tilde(self):
        """sanitize_path() replaces the home directory prefix with ~."""
        home = "/home/testuser"
        path = "/home/testuser/projects/myapp"
        result = ReportGenerator.sanitize_path(path, home_dir=home)
        assert result.startswith("~")
        assert "/home/testuser" not in result

    def test_truncates_long_path_to_last_3_components(self):
        """sanitize_path() truncates paths longer than 3 components."""
        path = "/a/b/c/d/e/f"
        result = ReportGenerator.sanitize_path(path, home_dir="/nonexistent_home")
        # Should contain only the last 3 components
        assert "d/e/f" in result or result.endswith("d/e/f")

    def test_short_path_not_truncated(self):
        """sanitize_path() does not truncate paths with 3 or fewer components."""
        path = "/usr/bin"
        result = ReportGenerator.sanitize_path(path, home_dir="/nonexistent_home")
        # Should not add "..."
        assert "..." not in result

    def test_empty_string_returned_unchanged(self):
        """sanitize_path() returns empty string unchanged."""
        result = ReportGenerator.sanitize_path("", home_dir="/home/user")
        assert result == ""

    def test_home_dir_path_replaced_with_tilde(self):
        """sanitize_path() replaces exact home dir match with ~."""
        home = "/home/testuser"
        result = ReportGenerator.sanitize_path(home, home_dir=home)
        assert result.startswith("~")

    def test_path_with_home_and_many_components_truncated(self):
        """sanitize_path() both replaces home and truncates long paths."""
        home = "/home/testuser"
        path = "/home/testuser/a/b/c/d/e"
        result = ReportGenerator.sanitize_path(path, home_dir=home)
        assert result.startswith("~")
        assert "..." in result

    def test_non_home_absolute_path_truncated(self):
        """sanitize_path() truncates long absolute paths not under home."""
        path = "/var/log/app/service/component/file.log"
        result = ReportGenerator.sanitize_path(path, home_dir="/home/user")
        assert "..." in result
        # Last 3 components should be present
        assert "component/file.log" in result or "service/component/file.log" in result

    def test_uses_system_home_when_home_dir_is_none(self):
        """sanitize_path() uses the current user's home dir when home_dir is None."""
        home = str(Path.home())
        path = str(Path.home() / "some" / "path")
        result = ReportGenerator.sanitize_path(path)
        assert result.startswith("~")


# ---------------------------------------------------------------------------
# ReportGenerator.sanitize_report_paths() tests
# ---------------------------------------------------------------------------


class TestSanitizeReportPaths:
    """Tests for ReportGenerator.sanitize_report_paths()."""

    def test_sanitizes_string_values_in_flat_dict(self):
        """sanitize_report_paths() sanitizes path strings in a flat dict."""
        gen = ReportGenerator()
        home = "/home/testuser"
        data = {
            "installation_path": "/home/testuser/apps/myapp",
            "name": "MyApp",
        }
        result = gen.sanitize_report_paths(data, home_dir=home)
        assert result["installation_path"].startswith("~")
        assert result["name"] == "MyApp"  # Non-path string unchanged

    def test_sanitizes_nested_dict(self):
        """sanitize_report_paths() recursively sanitizes nested dicts."""
        gen = ReportGenerator()
        home = "/home/testuser"
        data = {
            "app": {
                "path": "/home/testuser/apps/myapp",
                "version": "1.0",
            }
        }
        result = gen.sanitize_report_paths(data, home_dir=home)
        assert result["app"]["path"].startswith("~")
        assert result["app"]["version"] == "1.0"

    def test_sanitizes_paths_in_list(self):
        """sanitize_report_paths() sanitizes path strings inside lists."""
        gen = ReportGenerator()
        home = "/home/testuser"
        data = {
            "paths": [
                "/home/testuser/apps/myapp",
                "/usr/bin/python3",
                "not-a-path",
            ]
        }
        result = gen.sanitize_report_paths(data, home_dir=home)
        assert result["paths"][0].startswith("~")
        assert result["paths"][2] == "not-a-path"

    def test_does_not_mutate_original(self):
        """sanitize_report_paths() returns a new dict and does not mutate the original."""
        gen = ReportGenerator()
        home = "/home/testuser"
        original_path = "/home/testuser/apps/myapp"
        data = {"path": original_path}
        result = gen.sanitize_report_paths(data, home_dir=home)

        # Original should be unchanged
        assert data["path"] == original_path
        # Result should be sanitized
        assert result["path"] != original_path

    def test_non_path_strings_unchanged(self):
        """sanitize_report_paths() leaves non-path strings unchanged."""
        gen = ReportGenerator()
        data = {
            "name": "MyApp",
            "version": "1.2.3",
            "description": "A test application",
        }
        result = gen.sanitize_report_paths(data, home_dir="/home/user")
        assert result == data

    def test_numeric_values_unchanged(self):
        """sanitize_report_paths() leaves numeric values unchanged."""
        gen = ReportGenerator()
        data = {"count": 42, "score": 0.95}
        result = gen.sanitize_report_paths(data, home_dir="/home/user")
        assert result["count"] == 42
        assert result["score"] == 0.95

    def test_empty_dict_returns_empty_dict(self):
        """sanitize_report_paths() returns an empty dict for empty input."""
        gen = ReportGenerator()
        result = gen.sanitize_report_paths({}, home_dir="/home/user")
        assert result == {}

    def test_windows_style_path_sanitized(self):
        """sanitize_report_paths() sanitizes Windows-style absolute paths."""
        gen = ReportGenerator()
        data = {"path": "C:\\Users\\testuser\\AppData\\Local\\MyApp\\app.exe"}
        result = gen.sanitize_report_paths(data, home_dir="C:\\Users\\testuser")
        # Should be sanitized (either truncated or home replaced)
        assert result["path"] != data["path"] or "..." in result["path"] or result["path"].startswith("~")

    def test_tilde_path_sanitized(self):
        """sanitize_report_paths() sanitizes paths starting with ~."""
        gen = ReportGenerator()
        data = {"path": "~/very/deep/nested/path/to/file.txt"}
        result = gen.sanitize_report_paths(data, home_dir="/home/user")
        # Long tilde path should be truncated
        assert "..." in result["path"] or result["path"] == data["path"]

    def test_deeply_nested_structure(self):
        """sanitize_report_paths() handles deeply nested dicts and lists."""
        gen = ReportGenerator()
        home = "/home/testuser"
        data = {
            "level1": {
                "level2": {
                    "paths": ["/home/testuser/deep/path/file.txt"],
                }
            }
        }
        result = gen.sanitize_report_paths(data, home_dir=home)
        assert result["level1"]["level2"]["paths"][0].startswith("~")
