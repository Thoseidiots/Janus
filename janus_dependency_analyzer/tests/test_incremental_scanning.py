"""
Unit tests for incremental scanning capabilities.

Tests cover:
- ChangeRecord dataclass creation
- ApplicationCatalog change history methods
- Persistence of change history via save/load
- scan_incremental() integration with a catalog
"""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
import json

import pytest

from ..catalog.catalog import ApplicationCatalog
from ..core.models import Application, ChangeRecord, Platform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app(name: str = "TestApp", version: str = "1.0", app_id: str = None) -> Application:
    """Create a minimal Application for testing."""
    app = Application(
        name=name,
        version=version,
        platform=Platform.LINUX,
    )
    if app_id is not None:
        app.id = app_id
    return app


# ---------------------------------------------------------------------------
# ChangeRecord tests
# ---------------------------------------------------------------------------

class TestChangeRecord:
    def test_creation_with_required_fields(self):
        record = ChangeRecord(app_id="abc", app_name="MyApp", change_type="installed")
        assert record.app_id == "abc"
        assert record.app_name == "MyApp"
        assert record.change_type == "installed"

    def test_default_timestamp_is_recent(self):
        before = datetime.now()
        record = ChangeRecord(app_id="x", app_name="X", change_type="installed")
        after = datetime.now()
        assert before <= record.timestamp <= after

    def test_optional_version_fields_default_to_none(self):
        record = ChangeRecord(app_id="x", app_name="X", change_type="updated")
        assert record.previous_version is None
        assert record.new_version is None

    def test_optional_details_defaults_to_empty_string(self):
        record = ChangeRecord(app_id="x", app_name="X", change_type="removed")
        assert record.details == ""

    def test_explicit_version_fields(self):
        record = ChangeRecord(
            app_id="x",
            app_name="X",
            change_type="version_changed",
            previous_version="1.0",
            new_version="2.0",
        )
        assert record.previous_version == "1.0"
        assert record.new_version == "2.0"


# ---------------------------------------------------------------------------
# ApplicationCatalog.record_change / get_change_history tests
# ---------------------------------------------------------------------------

class TestCatalogChangeHistory:
    def test_record_change_appends_to_history(self):
        catalog = ApplicationCatalog()
        record = ChangeRecord(app_id="a1", app_name="App1", change_type="installed")
        catalog.record_change(record)
        history = catalog.get_change_history()
        assert len(history) == 1
        assert history[0] is record

    def test_multiple_records_are_all_stored(self):
        catalog = ApplicationCatalog()
        for i in range(5):
            catalog.record_change(ChangeRecord(app_id=f"id{i}", app_name=f"App{i}", change_type="installed"))
        assert len(catalog.get_change_history()) == 5

    def test_get_change_history_returns_all_when_no_filter(self):
        catalog = ApplicationCatalog()
        catalog.record_change(ChangeRecord(app_id="a", app_name="A", change_type="installed"))
        catalog.record_change(ChangeRecord(app_id="b", app_name="B", change_type="removed"))
        history = catalog.get_change_history()
        assert len(history) == 2

    def test_get_change_history_filters_by_app_id(self):
        catalog = ApplicationCatalog()
        catalog.record_change(ChangeRecord(app_id="a", app_name="A", change_type="installed"))
        catalog.record_change(ChangeRecord(app_id="b", app_name="B", change_type="installed"))
        catalog.record_change(ChangeRecord(app_id="a", app_name="A", change_type="updated"))

        history_a = catalog.get_change_history(app_id="a")
        assert len(history_a) == 2
        assert all(r.app_id == "a" for r in history_a)

        history_b = catalog.get_change_history(app_id="b")
        assert len(history_b) == 1
        assert history_b[0].app_id == "b"

    def test_get_change_history_returns_empty_for_unknown_app_id(self):
        catalog = ApplicationCatalog()
        catalog.record_change(ChangeRecord(app_id="a", app_name="A", change_type="installed"))
        assert catalog.get_change_history(app_id="nonexistent") == []

    def test_get_change_history_returns_copy(self):
        """Mutating the returned list should not affect the internal state."""
        catalog = ApplicationCatalog()
        catalog.record_change(ChangeRecord(app_id="a", app_name="A", change_type="installed"))
        history = catalog.get_change_history()
        history.clear()
        assert len(catalog.get_change_history()) == 1


# ---------------------------------------------------------------------------
# ApplicationCatalog.get_recent_changes tests
# ---------------------------------------------------------------------------

class TestGetRecentChanges:
    def test_returns_records_at_or_after_since(self):
        catalog = ApplicationCatalog()
        base = datetime(2024, 1, 1, 12, 0, 0)
        old = ChangeRecord(app_id="a", app_name="A", change_type="installed",
                           timestamp=base - timedelta(hours=1))
        new = ChangeRecord(app_id="b", app_name="B", change_type="installed",
                           timestamp=base + timedelta(hours=1))
        exact = ChangeRecord(app_id="c", app_name="C", change_type="installed",
                             timestamp=base)
        catalog.record_change(old)
        catalog.record_change(new)
        catalog.record_change(exact)

        recent = catalog.get_recent_changes(since=base)
        ids = {r.app_id for r in recent}
        assert "b" in ids
        assert "c" in ids
        assert "a" not in ids

    def test_returns_empty_when_no_recent_changes(self):
        catalog = ApplicationCatalog()
        past = datetime(2020, 1, 1)
        catalog.record_change(ChangeRecord(app_id="a", app_name="A", change_type="installed",
                                           timestamp=past))
        recent = catalog.get_recent_changes(since=datetime(2025, 1, 1))
        assert recent == []

    def test_returns_all_when_since_is_very_old(self):
        catalog = ApplicationCatalog()
        for i in range(3):
            catalog.record_change(ChangeRecord(app_id=f"id{i}", app_name=f"App{i}",
                                               change_type="installed"))
        recent = catalog.get_recent_changes(since=datetime(2000, 1, 1))
        assert len(recent) == 3


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

class TestCatalogPersistence:
    def test_save_and_load_persists_change_history(self, tmp_path):
        catalog_file = tmp_path / "catalog.json"
        catalog = ApplicationCatalog(storage_path=catalog_file)

        app = _make_app("Firefox", "120.0")
        catalog.add(app)
        record = ChangeRecord(
            app_id=app.id,
            app_name=app.name,
            change_type="installed",
            previous_version=None,
            new_version="120.0",
            details="initial install",
        )
        catalog.record_change(record)
        catalog.save()

        # Load into a fresh catalog
        catalog2 = ApplicationCatalog(storage_path=catalog_file)
        catalog2.load()

        history = catalog2.get_change_history()
        assert len(history) == 1
        loaded = history[0]
        assert loaded.app_id == app.id
        assert loaded.app_name == "Firefox"
        assert loaded.change_type == "installed"
        assert loaded.new_version == "120.0"
        assert loaded.details == "initial install"

    def test_save_includes_change_history_key(self, tmp_path):
        catalog_file = tmp_path / "catalog.json"
        catalog = ApplicationCatalog(storage_path=catalog_file)
        catalog.record_change(ChangeRecord(app_id="x", app_name="X", change_type="removed"))
        catalog.save()

        with catalog_file.open() as fh:
            data = json.load(fh)
        assert "change_history" in data
        assert len(data["change_history"]) == 1

    def test_load_empty_change_history_when_key_missing(self, tmp_path):
        """Catalog files without change_history key should load cleanly."""
        catalog_file = tmp_path / "catalog.json"
        # Write a catalog file without the change_history key
        catalog_file.write_text(json.dumps({"applications": []}))

        catalog = ApplicationCatalog(storage_path=catalog_file)
        catalog.load()
        assert catalog.get_change_history() == []

    def test_timestamp_round_trips_correctly(self, tmp_path):
        catalog_file = tmp_path / "catalog.json"
        catalog = ApplicationCatalog(storage_path=catalog_file)
        ts = datetime(2024, 6, 15, 10, 30, 45)
        catalog.record_change(ChangeRecord(app_id="a", app_name="A",
                                           change_type="installed", timestamp=ts))
        catalog.save()

        catalog2 = ApplicationCatalog(storage_path=catalog_file)
        catalog2.load()
        loaded_ts = catalog2.get_change_history()[0].timestamp
        assert loaded_ts == ts

    def test_multiple_records_persist_correctly(self, tmp_path):
        catalog_file = tmp_path / "catalog.json"
        catalog = ApplicationCatalog(storage_path=catalog_file)
        for i in range(3):
            catalog.record_change(ChangeRecord(app_id=f"id{i}", app_name=f"App{i}",
                                               change_type="installed"))
        catalog.save()

        catalog2 = ApplicationCatalog(storage_path=catalog_file)
        catalog2.load()
        assert len(catalog2.get_change_history()) == 3


# ---------------------------------------------------------------------------
# scan_incremental with catalog tests
# ---------------------------------------------------------------------------

class TestScanIncrementalWithCatalog:
    """Tests for SystemScannerImpl.scan_incremental() catalog integration."""

    def _make_scanner_with_apps(self, apps):
        """Return a SystemScannerImpl whose platform scanner yields *apps*."""
        from ..scanners.system_scanner import SystemScannerImpl

        scanner = SystemScannerImpl()
        mock_platform_scanner = MagicMock()
        mock_platform_scanner.discover_applications.return_value = apps
        scanner._platform_scanner = mock_platform_scanner
        scanner._detected_platform = Platform.LINUX
        return scanner

    def test_new_app_is_added_to_catalog_and_recorded(self):
        app = _make_app("NewApp", "1.0")
        # discovered_at is set to now by default, so it will be "new"
        scanner = self._make_scanner_with_apps([app])
        catalog = ApplicationCatalog()

        last_scan = datetime.now() - timedelta(hours=1)
        scanner.scan_incremental(last_scan, catalog=catalog)

        assert catalog.get(app.id) is not None
        history = catalog.get_change_history(app_id=app.id)
        assert len(history) == 1
        assert history[0].change_type == "installed"

    def test_unchanged_app_already_in_catalog_not_re_recorded(self):
        """An app already in the catalog with the same version should not generate a new record."""
        app = _make_app("StableApp", "2.0")
        # Put it in the catalog first
        catalog = ApplicationCatalog()
        catalog.add(app)

        # Make discovered_at old so _detect_changes won't flag it as new
        app.discovered_at = datetime.now() - timedelta(days=10)

        scanner = self._make_scanner_with_apps([app])

        # Patch _has_application_changed to return False (no file changes)
        with patch.object(scanner, '_has_application_changed', return_value=False):
            last_scan = datetime.now() - timedelta(hours=1)
            scanner.scan_incremental(last_scan, catalog=catalog)

        # No new change records should have been added
        history = catalog.get_change_history(app_id=app.id)
        assert len(history) == 0

    def test_version_change_records_updated_change(self):
        """When an app's version differs from what's in the catalog, record 'updated'."""
        old_app = _make_app("VersionedApp", "1.0", app_id="fixed-id")
        catalog = ApplicationCatalog()
        catalog.add(old_app)

        # New scan finds a newer version of the same app
        new_app = _make_app("VersionedApp", "2.0", app_id="fixed-id")
        # Make it look recently changed so _detect_changes picks it up
        new_app.discovered_at = datetime.now()

        scanner = self._make_scanner_with_apps([new_app])
        last_scan = datetime.now() - timedelta(hours=1)
        scanner.scan_incremental(last_scan, catalog=catalog)

        history = catalog.get_change_history(app_id="fixed-id")
        assert len(history) == 1
        assert history[0].change_type == "updated"
        assert history[0].previous_version == "1.0"
        assert history[0].new_version == "2.0"

    def test_scan_without_catalog_still_works(self):
        """scan_incremental without a catalog should not raise."""
        app = _make_app("SomeApp", "1.0")
        scanner = self._make_scanner_with_apps([app])
        last_scan = datetime.now() - timedelta(hours=1)
        result = scanner.scan_incremental(last_scan)  # no catalog
        assert result is not None
