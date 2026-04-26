"""
Unit tests for MetadataExtractor and ApplicationCatalog.

Covers:
- MetadataExtractor: ID generation, file size, version info, metadata object
- ApplicationCatalog: CRUD, indexing, persistence, encryption round-trip
"""

import json
import re
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ..catalog.catalog import ApplicationCatalog
from ..core.models import Application, ApplicationMetadata, Platform
from ..metadata.extractor import MetadataExtractor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def extractor():
    return MetadataExtractor()


def _make_app(
    name: str = "TestApp",
    version: str = "1.0",
    platform: Platform = Platform.LINUX,
    app_id: str = "test-id-001",
) -> Application:
    """Create a minimal Application for testing."""
    return Application(
        id=app_id,
        name=name,
        version=version,
        installation_path=Path("/usr/local/bin"),
        executable_path=Path(f"/usr/local/bin/{name.lower()}"),
        platform=platform,
        metadata=ApplicationMetadata(
            vendor="Acme Corp",
            description="A test application",
            file_size=1024,
            digital_signature="Valid",
        ),
        discovered_at=datetime.now(),
    )


# ===========================================================================
# MetadataExtractor tests
# ===========================================================================

class TestGenerateAppId:
    """Tests for MetadataExtractor.generate_app_id."""

    def test_generate_app_id_is_stable(self, extractor):
        """Same inputs always produce the same ID."""
        exe = Path("/usr/bin/myapp")
        id1 = extractor.generate_app_id("MyApp", exe, Platform.LINUX)
        id2 = extractor.generate_app_id("MyApp", exe, Platform.LINUX)
        assert id1 == id2

    def test_generate_app_id_is_unique_for_different_apps(self, extractor):
        """Different apps get different IDs."""
        exe_a = Path("/usr/bin/appA")
        exe_b = Path("/usr/bin/appB")
        id_a = extractor.generate_app_id("AppA", exe_a, Platform.LINUX)
        id_b = extractor.generate_app_id("AppB", exe_b, Platform.LINUX)
        assert id_a != id_b

    def test_generate_app_id_format(self, extractor):
        """ID is exactly 16 lowercase hex characters."""
        app_id = extractor.generate_app_id("SomeApp", Path("/bin/someapp"), Platform.MACOS)
        assert len(app_id) == 16
        assert re.fullmatch(r"[0-9a-f]{16}", app_id), f"Not hex: {app_id!r}"

    def test_generate_app_id_case_insensitive_name(self, extractor):
        """Name comparison is case-insensitive (same ID for 'App' and 'app')."""
        exe = Path("/usr/bin/app")
        id_upper = extractor.generate_app_id("App", exe, Platform.LINUX)
        id_lower = extractor.generate_app_id("app", exe, Platform.LINUX)
        assert id_upper == id_lower

    def test_generate_app_id_different_platforms_differ(self, extractor):
        """Same name/path on different platforms yields different IDs."""
        exe = Path("/usr/bin/app")
        id_linux = extractor.generate_app_id("app", exe, Platform.LINUX)
        id_macos = extractor.generate_app_id("app", exe, Platform.MACOS)
        assert id_linux != id_macos


class TestExtractFileSize:
    """Tests for MetadataExtractor.extract_file_size."""

    def test_extract_file_size_existing_file(self, extractor, tmp_path):
        """Returns the correct byte count for a real file."""
        content = b"hello world"
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(content)
        assert extractor.extract_file_size(test_file) == len(content)

    def test_extract_file_size_missing_file(self, extractor, tmp_path):
        """Returns 0 for a non-existent file."""
        missing = tmp_path / "does_not_exist.exe"
        assert extractor.extract_file_size(missing) == 0

    def test_extract_file_size_empty_file(self, extractor, tmp_path):
        """Returns 0 for an empty file."""
        empty = tmp_path / "empty.bin"
        empty.write_bytes(b"")
        assert extractor.extract_file_size(empty) == 0


class TestExtractVersionInfo:
    """Tests for MetadataExtractor.extract_version_info."""

    def test_extract_version_info_missing_file(self, extractor, tmp_path):
        """Returns empty dict gracefully when the file does not exist."""
        missing = tmp_path / "nonexistent.exe"
        result = extractor.extract_version_info(missing, Platform.WINDOWS)
        assert result == {}

    def test_extract_version_info_non_windows_returns_empty(self, extractor, tmp_path):
        """Non-Windows platforms always return an empty dict."""
        exe = tmp_path / "app"
        exe.write_bytes(b"\x7fELF")
        for plat in (Platform.LINUX, Platform.MACOS):
            assert extractor.extract_version_info(exe, plat) == {}

    @patch("janus_dependency_analyzer.metadata.extractor.subprocess.run")
    def test_extract_version_info_windows_success(self, mock_run, extractor, tmp_path):
        """Parses vendor, description, and version from PowerShell output."""
        exe = tmp_path / "app.exe"
        exe.write_bytes(b"MZ")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Acme Corp|My Application|2.3.1\n"
        mock_run.return_value = mock_result

        info = extractor.extract_version_info(exe, Platform.WINDOWS)
        assert info.get("vendor") == "Acme Corp"
        assert info.get("description") == "My Application"
        assert info.get("version") == "2.3.1"

    @patch("janus_dependency_analyzer.metadata.extractor.subprocess.run")
    def test_extract_version_info_windows_empty_output(self, mock_run, extractor, tmp_path):
        """Returns empty dict when PowerShell returns blank output."""
        exe = tmp_path / "app.exe"
        exe.write_bytes(b"MZ")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_run.return_value = mock_result

        assert extractor.extract_version_info(exe, Platform.WINDOWS) == {}

    @patch("janus_dependency_analyzer.metadata.extractor.subprocess.run")
    def test_extract_version_info_windows_timeout(self, mock_run, extractor, tmp_path):
        """Returns empty dict on subprocess timeout."""
        import subprocess as sp
        exe = tmp_path / "app.exe"
        exe.write_bytes(b"MZ")
        mock_run.side_effect = sp.TimeoutExpired(cmd="powershell", timeout=2)
        assert extractor.extract_version_info(exe, Platform.WINDOWS) == {}


class TestVerifyDigitalSignature:
    """Tests for MetadataExtractor.verify_digital_signature."""

    def test_verify_signature_non_windows_returns_none(self, extractor, tmp_path):
        """Non-Windows platforms always return None."""
        exe = tmp_path / "app"
        exe.write_bytes(b"\x7fELF")
        for plat in (Platform.LINUX, Platform.MACOS):
            assert extractor.verify_digital_signature(exe, plat) is None

    def test_verify_signature_missing_file_returns_none(self, extractor, tmp_path):
        """Returns None when the file does not exist."""
        missing = tmp_path / "ghost.exe"
        assert extractor.verify_digital_signature(missing, Platform.WINDOWS) is None

    @patch("janus_dependency_analyzer.metadata.extractor.subprocess.run")
    def test_verify_signature_valid(self, mock_run, extractor, tmp_path):
        """Returns 'Valid' when PowerShell reports Valid."""
        exe = tmp_path / "signed.exe"
        exe.write_bytes(b"MZ")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Valid\n"
        mock_run.return_value = mock_result

        assert extractor.verify_digital_signature(exe, Platform.WINDOWS) == "Valid"

    @patch("janus_dependency_analyzer.metadata.extractor.subprocess.run")
    def test_verify_signature_invalid(self, mock_run, extractor, tmp_path):
        """Returns 'Invalid' when PowerShell reports HashMismatch."""
        exe = tmp_path / "tampered.exe"
        exe.write_bytes(b"MZ")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "HashMismatch\n"
        mock_run.return_value = mock_result

        assert extractor.verify_digital_signature(exe, Platform.WINDOWS) == "Invalid"


class TestExtractMetadata:
    """Tests for MetadataExtractor.extract."""

    def test_extract_metadata_returns_metadata_object(self, extractor, tmp_path):
        """extract() returns an ApplicationMetadata instance."""
        exe = tmp_path / "app"
        exe.write_bytes(b"data")
        result = extractor.extract(exe, Platform.LINUX)
        assert isinstance(result, ApplicationMetadata)

    def test_extract_metadata_file_size_populated(self, extractor, tmp_path):
        """extract() populates file_size correctly."""
        content = b"x" * 512
        exe = tmp_path / "app"
        exe.write_bytes(content)
        result = extractor.extract(exe, Platform.LINUX)
        assert result.file_size == 512

    def test_extract_metadata_missing_file(self, extractor, tmp_path):
        """extract() returns metadata with file_size=0 for missing file."""
        missing = tmp_path / "ghost"
        result = extractor.extract(missing, Platform.LINUX)
        assert isinstance(result, ApplicationMetadata)
        assert result.file_size == 0

    def test_extract_metadata_install_date_set(self, extractor, tmp_path):
        """extract() sets install_date from filesystem ctime."""
        exe = tmp_path / "app"
        exe.write_bytes(b"data")
        result = extractor.extract(exe, Platform.LINUX)
        assert result.install_date is not None
        assert isinstance(result.install_date, datetime)


# ===========================================================================
# ApplicationCatalog tests
# ===========================================================================

class TestApplicationCatalog:
    """Tests for ApplicationCatalog CRUD and indexing."""

    def test_add_and_get(self, tmp_path):
        """Add an app, retrieve it by ID."""
        catalog = ApplicationCatalog(storage_path=tmp_path / "catalog.json")
        app = _make_app()
        catalog.add(app)
        retrieved = catalog.get(app.id)
        assert retrieved is app

    def test_get_nonexistent_returns_none(self, tmp_path):
        """get() with an unknown ID returns None."""
        catalog = ApplicationCatalog(storage_path=tmp_path / "catalog.json")
        assert catalog.get("no-such-id") is None

    def test_get_by_name_case_insensitive(self, tmp_path):
        """Name lookup is case-insensitive."""
        catalog = ApplicationCatalog(storage_path=tmp_path / "catalog.json")
        app = _make_app(name="MyApp")
        catalog.add(app)
        assert catalog.get_by_name("myapp") == [app]
        assert catalog.get_by_name("MYAPP") == [app]
        assert catalog.get_by_name("MyApp") == [app]

    def test_get_by_platform(self, tmp_path):
        """Platform filter returns only matching apps."""
        catalog = ApplicationCatalog(storage_path=tmp_path / "catalog.json")
        linux_app = _make_app(name="LinuxApp", platform=Platform.LINUX, app_id="id-linux")
        win_app = _make_app(name="WinApp", platform=Platform.WINDOWS, app_id="id-win")
        catalog.add(linux_app)
        catalog.add(win_app)

        linux_results = catalog.get_by_platform(Platform.LINUX)
        assert linux_app in linux_results
        assert win_app not in linux_results

    def test_remove_existing(self, tmp_path):
        """remove() returns True and the app is gone."""
        catalog = ApplicationCatalog(storage_path=tmp_path / "catalog.json")
        app = _make_app()
        catalog.add(app)
        assert catalog.remove(app.id) is True
        assert catalog.get(app.id) is None

    def test_remove_nonexistent(self, tmp_path):
        """remove() returns False for an unknown ID."""
        catalog = ApplicationCatalog(storage_path=tmp_path / "catalog.json")
        assert catalog.remove("ghost-id") is False

    def test_count(self, tmp_path):
        """count() reflects adds and removes."""
        catalog = ApplicationCatalog(storage_path=tmp_path / "catalog.json")
        assert catalog.count() == 0
        app1 = _make_app(app_id="id-1")
        app2 = _make_app(app_id="id-2")
        catalog.add(app1)
        assert catalog.count() == 1
        catalog.add(app2)
        assert catalog.count() == 2
        catalog.remove(app1.id)
        assert catalog.count() == 1

    def test_all_returns_all_apps(self, tmp_path):
        """all() returns every added application."""
        catalog = ApplicationCatalog(storage_path=tmp_path / "catalog.json")
        apps = [_make_app(app_id=f"id-{i}") for i in range(5)]
        for app in apps:
            catalog.add(app)
        result = catalog.all()
        assert len(result) == 5
        for app in apps:
            assert app in result

    def test_clear(self, tmp_path):
        """clear() empties the catalog."""
        catalog = ApplicationCatalog(storage_path=tmp_path / "catalog.json")
        for i in range(3):
            catalog.add(_make_app(app_id=f"id-{i}"))
        catalog.clear()
        assert catalog.count() == 0
        assert catalog.all() == []

    def test_save_and_load(self, tmp_path):
        """Save to a temp file, create a new catalog, load, verify apps present."""
        catalog_path = tmp_path / "catalog.json"
        catalog = ApplicationCatalog(storage_path=catalog_path)
        app = _make_app()
        catalog.add(app)
        catalog.save()

        # New catalog instance loads from the same file
        catalog2 = ApplicationCatalog(storage_path=catalog_path)
        catalog2.load()
        assert catalog2.count() == 1
        loaded = catalog2.get(app.id)
        assert loaded is not None
        assert loaded.name == app.name
        assert loaded.version == app.version
        assert loaded.platform == app.platform

    def test_duplicate_add_updates_existing(self, tmp_path):
        """Adding the same ID twice updates, doesn't duplicate."""
        catalog = ApplicationCatalog(storage_path=tmp_path / "catalog.json")
        app_v1 = _make_app(version="1.0")
        catalog.add(app_v1)

        # Same ID, different version
        app_v2 = Application(
            id=app_v1.id,
            name=app_v1.name,
            version="2.0",
            installation_path=app_v1.installation_path,
            executable_path=app_v1.executable_path,
            platform=app_v1.platform,
            metadata=ApplicationMetadata(),
            discovered_at=datetime.now(),
        )
        catalog.add(app_v2)

        assert catalog.count() == 1
        assert catalog.get(app_v1.id).version == "2.0"

    def test_load_missing_file_leaves_catalog_empty(self, tmp_path):
        """load() on a non-existent file leaves the catalog empty."""
        catalog = ApplicationCatalog(storage_path=tmp_path / "nonexistent.json")
        catalog.load()
        assert catalog.count() == 0

    def test_save_creates_parent_directories(self, tmp_path):
        """save() creates parent directories if they don't exist."""
        deep_path = tmp_path / "a" / "b" / "c" / "catalog.json"
        catalog = ApplicationCatalog(storage_path=deep_path)
        catalog.add(_make_app())
        catalog.save()
        assert deep_path.exists()

    def test_get_by_name_no_match_returns_empty_list(self, tmp_path):
        """get_by_name() returns [] when no app matches."""
        catalog = ApplicationCatalog(storage_path=tmp_path / "catalog.json")
        assert catalog.get_by_name("unknown") == []

    def test_get_by_platform_no_match_returns_empty_list(self, tmp_path):
        """get_by_platform() returns [] when no app matches."""
        catalog = ApplicationCatalog(storage_path=tmp_path / "catalog.json")
        assert catalog.get_by_platform(Platform.WINDOWS) == []


class TestApplicationCatalogEncryption:
    """Tests for ApplicationCatalog encryption support."""

    def test_save_and_load_with_encryption(self, tmp_path):
        """Encrypted catalog round-trips correctly."""
        catalog_path = tmp_path / "catalog.json"
        catalog = ApplicationCatalog(storage_path=catalog_path, encrypt=True)
        app = _make_app()
        catalog.add(app)
        catalog.save()

        # Verify the raw JSON has encrypted (non-plaintext) vendor field
        raw = json.loads(catalog_path.read_text())
        stored_vendor = raw["applications"][0]["metadata"]["vendor"]
        assert stored_vendor != app.metadata.vendor  # should be ciphertext

        # Load into a new catalog and verify decryption
        catalog2 = ApplicationCatalog(storage_path=catalog_path, encrypt=True)
        catalog2.load()
        loaded = catalog2.get(app.id)
        assert loaded is not None
        assert loaded.metadata.vendor == app.metadata.vendor
        assert loaded.metadata.description == app.metadata.description
        assert loaded.metadata.digital_signature == app.metadata.digital_signature

    def test_encryption_key_reused_across_instances(self, tmp_path):
        """The same key file is reused so a second instance can decrypt."""
        catalog_path = tmp_path / "catalog.json"

        c1 = ApplicationCatalog(storage_path=catalog_path, encrypt=True)
        c1.add(_make_app())
        c1.save()

        c2 = ApplicationCatalog(storage_path=catalog_path, encrypt=True)
        c2.load()
        assert c2.count() == 1
