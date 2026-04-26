"""
Property-based tests for the System Scanner.

This module implements property-based tests using Hypothesis to verify
universal correctness properties of the SystemScanner and platform scanners.

Feature: janus-dependency-analyzer
"""

import pytest
from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock
import uuid

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from ..scanners.system_scanner import SystemScannerImpl
from ..core.models import (
    Application,
    ApplicationMetadata,
    Platform,
    ScanResult,
)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Strategy for non-empty strings (application names, versions, etc.)
non_empty_text = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd")),
    min_size=1,
    max_size=64,
)

# Strategy for path-like strings (no null bytes, reasonable length)
path_text = st.text(
    alphabet=st.characters(
        blacklist_characters="\x00",
        whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd", "Po"),
    ),
    min_size=1,
    max_size=128,
)


def application_metadata_strategy() -> st.SearchStrategy:
    """Build a strategy that generates ApplicationMetadata objects."""
    return st.builds(
        ApplicationMetadata,
        vendor=st.one_of(st.none(), non_empty_text),
        description=st.one_of(st.none(), non_empty_text),
        file_size=st.integers(min_value=0, max_value=10_000_000),
        install_date=st.one_of(st.none(), st.just(datetime.now())),
        digital_signature=st.one_of(st.none(), non_empty_text),
    )


def application_strategy(platform: Platform = Platform.LINUX) -> st.SearchStrategy:
    """Build a strategy that generates valid Application objects."""
    return st.builds(
        Application,
        id=st.builds(lambda: str(uuid.uuid4())),
        name=non_empty_text,
        version=st.text(
            alphabet="0123456789.",
            min_size=1,
            max_size=20,
        ),
        installation_path=path_text.map(lambda p: Path(f"/opt/{p}")),
        executable_path=path_text.map(lambda p: Path(f"/usr/bin/{p}")),
        platform=st.just(platform),
        metadata=application_metadata_strategy(),
        discovered_at=st.just(datetime.now()),
        is_accessible=st.booleans(),
        access_error=st.one_of(st.none(), st.just("Permission denied")),
    )


def applications_list_strategy(
    platform: Platform = Platform.LINUX,
    min_size: int = 0,
    max_size: int = 20,
) -> st.SearchStrategy:
    """Build a strategy that generates lists of Application objects."""
    return st.lists(
        application_strategy(platform),
        min_size=min_size,
        max_size=max_size,
    )


def platform_strategy() -> st.SearchStrategy:
    """Strategy that generates one of the three supported platforms."""
    return st.sampled_from([Platform.WINDOWS, Platform.MACOS, Platform.LINUX])


# ---------------------------------------------------------------------------
# Helper: build a SystemScannerImpl whose platform scanner returns `apps`
# ---------------------------------------------------------------------------

def _make_scanner_with_apps(apps: List[Application], platform: Platform) -> SystemScannerImpl:
    """
    Return a SystemScannerImpl that is pre-wired to:
    - report `platform` as the detected platform
    - return `apps` from its platform scanner's discover_applications()
    """
    scanner = SystemScannerImpl()

    mock_platform_scanner = MagicMock()
    mock_platform_scanner.discover_applications.return_value = apps

    scanner._detected_platform = platform
    scanner._platform_scanner = mock_platform_scanner

    return scanner


# ---------------------------------------------------------------------------
# Property 1: Cross-Platform Application Discovery
# Feature: janus-dependency-analyzer, Property 1: Cross-Platform Application Discovery
# Validates: Requirements 1.1, 1.2, 1.3, 1.4
# ---------------------------------------------------------------------------

class TestCrossPlatformDiscovery:
    """
    **Validates: Requirements 1.1, 1.2, 1.3, 1.4**

    Property 1: Cross-Platform Application Discovery
    For any supported platform (Windows, macOS, Linux), the System_Scanner SHALL
    discover all accessible applications and extract complete metadata including
    name, version, installation path, and executable location.
    """

    @given(
        platform=platform_strategy(),
        apps=st.lists(
            st.builds(
                Application,
                id=st.builds(lambda: str(uuid.uuid4())),
                name=non_empty_text,
                version=st.text(alphabet="0123456789.", min_size=1, max_size=20),
                installation_path=path_text.map(lambda p: Path(f"/opt/{p}")),
                executable_path=path_text.map(lambda p: Path(f"/usr/bin/{p}")),
                platform=platform_strategy(),
                metadata=application_metadata_strategy(),
                discovered_at=st.just(datetime.now()),
                is_accessible=st.just(True),
                access_error=st.just(None),
            ),
            min_size=0,
            max_size=20,
        ),
    )
    @settings(max_examples=25)
    def test_every_discovered_app_has_non_empty_name(
        self, platform: Platform, apps: List[Application]
    ) -> None:
        """Every application returned by the scanner has a non-empty name."""
        scanner = _make_scanner_with_apps(apps, platform)
        result = scanner.scan_full()

        for app in result.applications:
            assert app.name, (
                f"Application with id={app.id} has an empty name on platform {platform}"
            )

    @given(
        platform=platform_strategy(),
        apps=st.lists(
            st.builds(
                Application,
                id=st.builds(lambda: str(uuid.uuid4())),
                name=non_empty_text,
                version=st.text(alphabet="0123456789.", min_size=1, max_size=20),
                installation_path=path_text.map(lambda p: Path(f"/opt/{p}")),
                executable_path=path_text.map(lambda p: Path(f"/usr/bin/{p}")),
                platform=platform_strategy(),
                metadata=application_metadata_strategy(),
                discovered_at=st.just(datetime.now()),
                is_accessible=st.just(True),
                access_error=st.just(None),
            ),
            min_size=0,
            max_size=20,
        ),
    )
    @settings(max_examples=25)
    def test_every_discovered_app_has_non_empty_id(
        self, platform: Platform, apps: List[Application]
    ) -> None:
        """Every application returned by the scanner has a non-empty unique id."""
        scanner = _make_scanner_with_apps(apps, platform)
        result = scanner.scan_full()

        for app in result.applications:
            assert app.id, (
                f"Application '{app.name}' has an empty id on platform {platform}"
            )

    @given(
        platform=platform_strategy(),
        apps=st.lists(
            st.builds(
                Application,
                id=st.builds(lambda: str(uuid.uuid4())),
                name=non_empty_text,
                version=st.text(alphabet="0123456789.", min_size=1, max_size=20),
                installation_path=path_text.map(lambda p: Path(f"/opt/{p}")),
                executable_path=path_text.map(lambda p: Path(f"/usr/bin/{p}")),
                platform=platform_strategy(),
                metadata=application_metadata_strategy(),
                discovered_at=st.just(datetime.now()),
                is_accessible=st.just(True),
                access_error=st.just(None),
            ),
            min_size=0,
            max_size=20,
        ),
    )
    @settings(max_examples=25)
    def test_every_discovered_app_has_valid_installation_path(
        self, platform: Platform, apps: List[Application]
    ) -> None:
        """Every application returned by the scanner has a non-None installation_path."""
        scanner = _make_scanner_with_apps(apps, platform)
        result = scanner.scan_full()

        for app in result.applications:
            assert app.installation_path is not None, (
                f"Application '{app.name}' has None installation_path on platform {platform}"
            )

    @given(
        platform=platform_strategy(),
        apps=st.lists(
            st.builds(
                Application,
                id=st.builds(lambda: str(uuid.uuid4())),
                name=non_empty_text,
                version=st.text(alphabet="0123456789.", min_size=1, max_size=20),
                installation_path=path_text.map(lambda p: Path(f"/opt/{p}")),
                executable_path=path_text.map(lambda p: Path(f"/usr/bin/{p}")),
                platform=platform_strategy(),
                metadata=application_metadata_strategy(),
                discovered_at=st.just(datetime.now()),
                is_accessible=st.just(True),
                access_error=st.just(None),
            ),
            min_size=0,
            max_size=20,
        ),
    )
    @settings(max_examples=25)
    def test_every_discovered_app_has_valid_executable_path(
        self, platform: Platform, apps: List[Application]
    ) -> None:
        """Every application returned by the scanner has a non-None executable_path."""
        scanner = _make_scanner_with_apps(apps, platform)
        result = scanner.scan_full()

        for app in result.applications:
            assert app.executable_path is not None, (
                f"Application '{app.name}' has None executable_path on platform {platform}"
            )

    @given(
        platform=platform_strategy(),
        apps=st.lists(
            st.builds(
                Application,
                id=st.builds(lambda: str(uuid.uuid4())),
                name=non_empty_text,
                version=st.text(alphabet="0123456789.", min_size=1, max_size=20),
                installation_path=path_text.map(lambda p: Path(f"/opt/{p}")),
                executable_path=path_text.map(lambda p: Path(f"/usr/bin/{p}")),
                platform=platform_strategy(),
                metadata=application_metadata_strategy(),
                discovered_at=st.just(datetime.now()),
                is_accessible=st.just(True),
                access_error=st.just(None),
            ),
            min_size=0,
            max_size=20,
        ),
    )
    @settings(max_examples=25)
    def test_every_discovered_app_has_platform_set(
        self, platform: Platform, apps: List[Application]
    ) -> None:
        """Every application returned by the scanner has a platform attribute set."""
        scanner = _make_scanner_with_apps(apps, platform)
        result = scanner.scan_full()

        for app in result.applications:
            assert app.platform is not None, (
                f"Application '{app.name}' has no platform set"
            )
            assert isinstance(app.platform, Platform), (
                f"Application '{app.name}' platform is not a Platform enum instance"
            )

    @given(
        platform=platform_strategy(),
        apps=st.lists(
            st.builds(
                Application,
                id=st.builds(lambda: str(uuid.uuid4())),
                name=non_empty_text,
                version=st.text(alphabet="0123456789.", min_size=1, max_size=20),
                installation_path=path_text.map(lambda p: Path(f"/opt/{p}")),
                executable_path=path_text.map(lambda p: Path(f"/usr/bin/{p}")),
                platform=platform_strategy(),
                metadata=application_metadata_strategy(),
                discovered_at=st.just(datetime.now()),
                is_accessible=st.just(True),
                access_error=st.just(None),
            ),
            min_size=0,
            max_size=20,
        ),
    )
    @settings(max_examples=25)
    def test_scanner_is_deterministic(
        self, platform: Platform, apps: List[Application]
    ) -> None:
        """
        Calling scan_full() twice with the same underlying data returns the same
        number of applications (determinism property).
        """
        # Both scanner instances share the same mock data
        scanner1 = _make_scanner_with_apps(apps, platform)
        scanner2 = _make_scanner_with_apps(apps, platform)

        result1 = scanner1.scan_full()
        result2 = scanner2.scan_full()

        assert result1.total_applications == result2.total_applications, (
            "scan_full() returned different application counts on repeated calls "
            f"with the same input on platform {platform}"
        )

        names1 = sorted(a.name for a in result1.applications)
        names2 = sorted(a.name for a in result2.applications)
        assert names1 == names2, (
            "scan_full() returned different application names on repeated calls "
            f"with the same input on platform {platform}"
        )


# ---------------------------------------------------------------------------
# Property 2: Graceful Permission Handling
# Feature: janus-dependency-analyzer, Property 2: Graceful Permission Handling
# Validates: Requirements 1.5, 1.6
# ---------------------------------------------------------------------------

class TestGracefulPermissionHandling:
    """
    **Validates: Requirements 1.5, 1.6**

    Property 2: Graceful Permission Handling
    For any system with restricted access areas, the System_Scanner SHALL log
    access restrictions, continue scanning accessible areas, and store all
    discovered applications with unique identifiers.
    """

    @given(
        accessible_apps=st.lists(
            st.builds(
                Application,
                id=st.builds(lambda: str(uuid.uuid4())),
                name=non_empty_text,
                version=st.text(alphabet="0123456789.", min_size=1, max_size=20),
                installation_path=path_text.map(lambda p: Path(f"/opt/{p}")),
                executable_path=path_text.map(lambda p: Path(f"/usr/bin/{p}")),
                platform=st.just(Platform.LINUX),
                metadata=application_metadata_strategy(),
                discovered_at=st.just(datetime.now()),
                is_accessible=st.just(True),
                access_error=st.just(None),
            ),
            min_size=1,
            max_size=15,
        ),
        inaccessible_count=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=25)
    def test_scanner_continues_after_permission_errors(
        self, accessible_apps: List[Application], inaccessible_count: int
    ) -> None:
        """
        When some paths raise PermissionError, the scanner continues and returns
        the accessible applications.
        """
        # Build inaccessible apps
        inaccessible_apps = [
            Application(
                id=str(uuid.uuid4()),
                name=f"InaccessibleApp{i}",
                version="1.0",
                installation_path=Path(f"/restricted/app{i}"),
                executable_path=Path(f"/restricted/bin/app{i}"),
                platform=Platform.LINUX,
                is_accessible=False,
                access_error="Permission denied",
            )
            for i in range(inaccessible_count)
        ]

        all_apps = accessible_apps + inaccessible_apps

        scanner = SystemScannerImpl()
        mock_platform_scanner = MagicMock()
        mock_platform_scanner.discover_applications.return_value = all_apps
        scanner._detected_platform = Platform.LINUX
        scanner._platform_scanner = mock_platform_scanner

        # Should not raise
        result = scanner.scan_full()

        # All apps (accessible + inaccessible) are returned
        assert result.total_applications == len(all_apps), (
            f"Expected {len(all_apps)} total applications, got {result.total_applications}"
        )

        # Accessible apps are counted correctly
        assert result.accessible_applications == len(accessible_apps), (
            f"Expected {len(accessible_apps)} accessible applications, "
            f"got {result.accessible_applications}"
        )

    @given(
        accessible_apps=st.lists(
            st.builds(
                Application,
                id=st.builds(lambda: str(uuid.uuid4())),
                name=non_empty_text,
                version=st.text(alphabet="0123456789.", min_size=1, max_size=20),
                installation_path=path_text.map(lambda p: Path(f"/opt/{p}")),
                executable_path=path_text.map(lambda p: Path(f"/usr/bin/{p}")),
                platform=st.just(Platform.LINUX),
                metadata=application_metadata_strategy(),
                discovered_at=st.just(datetime.now()),
                is_accessible=st.just(True),
                access_error=st.just(None),
            ),
            min_size=0,
            max_size=15,
        ),
        inaccessible_count=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=25)
    def test_scan_result_contains_warnings_for_inaccessible_paths(
        self, accessible_apps: List[Application], inaccessible_count: int
    ) -> None:
        """
        When the platform scanner raises PermissionError for some paths, the
        ScanResult contains warnings or errors for those inaccessible paths.

        We simulate this by having discover_applications() raise PermissionError
        so the SystemScanner's error-handling path is exercised and an error
        entry is recorded.
        """
        scanner = SystemScannerImpl()
        mock_platform_scanner = MagicMock()
        mock_platform_scanner.discover_applications.side_effect = PermissionError(
            "Access denied to restricted directory"
        )
        scanner._detected_platform = Platform.LINUX
        scanner._platform_scanner = mock_platform_scanner

        result = scanner.scan_full()

        # The scanner must not raise; it must record the error
        assert len(result.errors) > 0, (
            "Expected at least one error entry when PermissionError is raised, "
            "but result.errors is empty"
        )

    @given(
        apps=st.lists(
            st.builds(
                Application,
                id=st.builds(lambda: str(uuid.uuid4())),
                name=non_empty_text,
                version=st.text(alphabet="0123456789.", min_size=1, max_size=20),
                installation_path=path_text.map(lambda p: Path(f"/opt/{p}")),
                executable_path=path_text.map(lambda p: Path(f"/usr/bin/{p}")),
                platform=st.just(Platform.LINUX),
                metadata=application_metadata_strategy(),
                discovered_at=st.just(datetime.now()),
                is_accessible=st.booleans(),
                access_error=st.one_of(st.none(), st.just("Permission denied")),
            ),
            min_size=0,
            max_size=20,
        ),
    )
    @settings(max_examples=25)
    def test_all_returned_applications_have_unique_ids(
        self, apps: List[Application]
    ) -> None:
        """
        All applications stored in the scan result have unique identifiers
        (Requirement 1.6).
        """
        scanner = _make_scanner_with_apps(apps, Platform.LINUX)
        result = scanner.scan_full()

        ids = [app.id for app in result.applications]
        assert len(ids) == len(set(ids)), (
            f"Duplicate application IDs found in scan result: "
            f"{[i for i in ids if ids.count(i) > 1]}"
        )

    @given(
        accessible_apps=st.lists(
            st.builds(
                Application,
                id=st.builds(lambda: str(uuid.uuid4())),
                name=non_empty_text,
                version=st.text(alphabet="0123456789.", min_size=1, max_size=20),
                installation_path=path_text.map(lambda p: Path(f"/opt/{p}")),
                executable_path=path_text.map(lambda p: Path(f"/usr/bin/{p}")),
                platform=st.just(Platform.LINUX),
                metadata=application_metadata_strategy(),
                discovered_at=st.just(datetime.now()),
                is_accessible=st.just(True),
                access_error=st.just(None),
            ),
            min_size=0,
            max_size=15,
        ),
        num_permission_errors=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=25)
    def test_scanner_never_raises_unhandled_exception(
        self, accessible_apps: List[Application], num_permission_errors: int
    ) -> None:
        """
        The scanner never raises an unhandled exception even when many paths
        are inaccessible (i.e., discover_applications raises PermissionError).
        """
        scanner = SystemScannerImpl()
        mock_platform_scanner = MagicMock()

        # Simulate repeated PermissionErrors
        mock_platform_scanner.discover_applications.side_effect = PermissionError(
            f"Access denied ({num_permission_errors} restricted paths)"
        )
        scanner._detected_platform = Platform.LINUX
        scanner._platform_scanner = mock_platform_scanner

        # Must not raise
        try:
            result = scanner.scan_full()
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"scan_full() raised an unhandled exception when PermissionError "
                f"was thrown by the platform scanner: {exc!r}"
            )

        # Result should be a valid ScanResult with errors recorded
        assert isinstance(result, ScanResult)
        assert len(result.errors) > 0
