"""
macOS-specific application scanner.

This module implements application discovery for macOS systems using
Applications folder scanning, Homebrew integration, and LaunchServices queries.

All macOS-specific code is guarded with sys.platform checks so this module
imports cleanly on Windows and Linux as well.
"""

import logging
import os
import sys
import subprocess
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from .base import BasePlatformScanner
from ..core.models import Application, ApplicationMetadata, Platform


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Swarm infrastructure (local copy – mirrors the pattern in windows_scanner)
# ---------------------------------------------------------------------------

@dataclass
class ScanTask:
    """Represents a scanning task for the macOS swarm."""
    task_type: str   # 'app_bundle', 'brew_cli', 'brew_dir', 'lsregister', 'executable_dir'
    target: Any      # Path, string command key, etc.
    priority: int = 1
    max_depth: int = 1


@dataclass
class ScanResult:
    """Result from a macOS swarm scanning task."""
    task_type: str
    applications: List[Application]
    errors: List[str]
    scan_time: float


class SwarmCoordinator:
    """Coordinates parallel scanning tasks across multiple workers (macOS)."""

    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(8, multiprocessing.cpu_count())
        self.errors: List[str] = []
        self.start_time: Optional[float] = None

    def execute_swarm_scan(
        self,
        tasks: List[ScanTask],
        worker_fn,          # callable(task) -> ScanResult
    ) -> List[Application]:
        """Execute scanning tasks in parallel and return deduplicated applications."""
        self.start_time = time.time()
        all_applications: List[Application] = []

        # Higher priority first
        tasks.sort(key=lambda t: t.priority, reverse=True)

        logger.info(
            "Starting macOS swarm scan with %d tasks using %d workers",
            len(tasks), self.max_workers,
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {executor.submit(worker_fn, task): task for task in tasks}

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result: ScanResult = future.result(timeout=60)
                    if result and result.applications:
                        all_applications.extend(result.applications)
                        logger.debug(
                            "Task %s (%s) completed: %d apps found",
                            task.task_type, task.target, len(result.applications),
                        )
                    if result and result.errors:
                        self.errors.extend(result.errors)
                except Exception as exc:
                    msg = f"Task {task.task_type} ({task.target}) failed: {exc}"
                    logger.warning(msg)
                    self.errors.append(msg)

        unique = self._deduplicate(all_applications)
        elapsed = time.time() - self.start_time
        logger.info(
            "macOS swarm scan completed in %.2fs: %d unique applications found",
            elapsed, len(unique),
        )
        return unique

    @staticmethod
    def _deduplicate(applications: List[Application]) -> List[Application]:
        seen: set = set()
        unique: List[Application] = []
        for app in applications:
            key = (str(app.executable_path).lower(), app.name.lower())
            if key not in seen:
                seen.add(key)
                unique.append(app)
        return unique


# ---------------------------------------------------------------------------
# macOS scanner
# ---------------------------------------------------------------------------

class MacOSScanner(BasePlatformScanner):
    """
    macOS-specific application scanner with swarm-based parallel processing.

    Implements application discovery for macOS systems by scanning:
    - /Applications, ~/Applications, /System/Applications (app bundles)
    - Homebrew installations via ``brew list`` and ``brew list --cask``
    - LaunchServices database via ``lsregister -dump``
    - Common portable-executable directories
    """

    def __init__(self):
        """Initialize the macOS scanner."""
        super().__init__(Platform.MACOS)

        # Standard .app bundle directories
        self.app_directories: List[Path] = [
            Path("/Applications"),
            Path.home() / "Applications",
            Path("/System/Applications"),
            Path("/System/Library/CoreServices/Applications"),
        ]

        # Homebrew prefix candidates (Apple Silicon vs Intel)
        self._homebrew_prefixes: List[Path] = [
            Path("/opt/homebrew"),   # Apple Silicon
            Path("/usr/local"),      # Intel
        ]

        self.swarm = SwarmCoordinator()

    # ------------------------------------------------------------------
    # Public interface (required by BasePlatformScanner)
    # ------------------------------------------------------------------

    def scan_standard_locations(self) -> List[Application]:
        """
        Scan standard macOS application installation locations.

        Scans .app bundle directories in parallel and also queries the
        LaunchServices database for any additional registered applications.

        Returns:
            List[Application]: Applications found in standard locations.
        """
        tasks: List[ScanTask] = []

        # One task per app directory (high priority)
        for app_dir in self.app_directories:
            tasks.append(ScanTask(
                task_type="app_bundle",
                target=app_dir,
                priority=3,
            ))

        # LaunchServices database query via lsregister (medium priority)
        tasks.append(ScanTask(
            task_type="lsregister",
            target="lsregister",
            priority=2,
        ))

        # system_profiler SPApplicationsDataType (medium priority, macOS only)
        tasks.append(ScanTask(
            task_type="system_profiler",
            target="SPApplicationsDataType",
            priority=2,
        ))

        return self.swarm.execute_swarm_scan(tasks, self._dispatch_task)

    def scan_package_managers(self) -> List[Application]:
        """
        Scan applications installed via macOS package managers.

        Uses ``brew list --cask`` and ``brew list`` for Homebrew, and also
        falls back to scanning the Cellar/Caskroom directories directly.
        MacPorts is scanned via its ``/opt/local/bin`` directory.

        Returns:
            List[Application]: Applications found via package managers.
        """
        tasks: List[ScanTask] = []

        # Homebrew CLI tasks (highest priority – most accurate)
        tasks.append(ScanTask(task_type="brew_cli", target="cask", priority=3))
        tasks.append(ScanTask(task_type="brew_cli", target="formula", priority=3))

        # Homebrew directory fallback tasks (medium priority)
        for prefix in self._homebrew_prefixes:
            cellar = prefix / "Cellar"
            caskroom = prefix / "Caskroom"
            if cellar.exists():
                tasks.append(ScanTask(task_type="brew_dir", target=cellar, priority=2))
            if caskroom.exists():
                tasks.append(ScanTask(task_type="brew_dir", target=caskroom, priority=2))

        # MacPorts
        macports_bin = Path("/opt/local/bin")
        if macports_bin.exists():
            tasks.append(ScanTask(
                task_type="executable_dir",
                target=macports_bin,
                priority=1,
            ))

        return self.swarm.execute_swarm_scan(tasks, self._dispatch_task)

    def scan_portable_applications(self) -> List[Application]:
        """
        Scan for portable applications in common locations.

        Returns:
            List[Application]: Portable applications found.
        """
        tasks: List[ScanTask] = []

        portable_dirs: List[Path] = [
            Path.home() / "Desktop",
            Path.home() / "Downloads",
            Path("/usr/local/bin"),
            Path("/opt"),
        ]

        for directory in portable_dirs:
            if directory.exists():
                tasks.append(ScanTask(
                    task_type="executable_dir",
                    target=directory,
                    priority=1,
                ))

        return self.swarm.execute_swarm_scan(tasks, self._dispatch_task)

    # ------------------------------------------------------------------
    # Metadata extraction (required by PlatformScanner interface)
    # ------------------------------------------------------------------

    def extract_metadata(self, app_path: Path) -> ApplicationMetadata:
        """
        Extract metadata for an application at the given path.

        For ``.app`` bundles, reads ``Contents/Info.plist`` for rich metadata.
        For plain executables, falls back to filesystem stat information.

        Args:
            app_path: Path to the application (bundle directory or executable).

        Returns:
            ApplicationMetadata: Extracted metadata.
        """
        metadata = ApplicationMetadata()

        # Filesystem stat (always available)
        try:
            stat = app_path.stat()
            metadata.file_size = stat.st_size
            metadata.install_date = datetime.fromtimestamp(stat.st_ctime)
        except (OSError, PermissionError) as exc:
            self.logger.debug("Could not stat %s: %s", app_path, exc)

        # .app bundle: enrich from Info.plist
        if app_path.is_dir() and app_path.suffix == ".app":
            try:
                import plistlib
                info_plist = app_path / "Contents" / "Info.plist"
                if info_plist.exists():
                    with open(info_plist, "rb") as fh:
                        plist_data = plistlib.load(fh)
                    bundle_id: str = plist_data.get("CFBundleIdentifier", "")
                    if "." in bundle_id:
                        metadata.vendor = bundle_id.split(".")[0]
                    metadata.description = plist_data.get("CFBundleGetInfoString", "")
                    for doc_type in plist_data.get("CFBundleDocumentTypes", []):
                        metadata.file_associations.extend(
                            doc_type.get("CFBundleTypeExtensions", [])
                        )
            except Exception as exc:  # noqa: BLE001
                self.logger.debug("Could not read plist for %s: %s", app_path, exc)

        return metadata

    # ------------------------------------------------------------------
    # Swarm task dispatcher
    # ------------------------------------------------------------------

    def _dispatch_task(self, task: ScanTask) -> ScanResult:
        """Route a ScanTask to the appropriate worker method."""
        start = time.time()
        try:
            if task.task_type == "app_bundle":
                apps = self._scan_applications_directory(task.target)
                return ScanResult("app_bundle", apps, [], time.time() - start)

            elif task.task_type == "lsregister":
                apps = self._scan_launch_services()
                return ScanResult("lsregister", apps, [], time.time() - start)

            elif task.task_type == "system_profiler":
                apps = self._scan_system_profiler()
                return ScanResult("system_profiler", apps, [], time.time() - start)

            elif task.task_type == "brew_cli":
                apps = self._scan_homebrew_cli(task.target)
                return ScanResult("brew_cli", apps, [], time.time() - start)

            elif task.task_type == "brew_dir":
                apps = self._scan_homebrew_directory(task.target)
                return ScanResult("brew_dir", apps, [], time.time() - start)

            elif task.task_type == "executable_dir":
                apps = self._scan_directory_for_executables(task.target)
                return ScanResult("executable_dir", apps, [], time.time() - start)

            else:
                return ScanResult(task.task_type, [], [f"Unknown task type: {task.task_type}"], 0.0)

        except Exception as exc:
            return ScanResult(task.task_type, [], [str(exc)], time.time() - start)

    # ------------------------------------------------------------------
    # .app bundle scanning
    # ------------------------------------------------------------------

    def _scan_applications_directory(self, app_dir: Path) -> List[Application]:
        """Scan an Applications directory for .app bundles."""
        applications: List[Application] = []

        if not app_dir.exists():
            return applications

        try:
            for item in app_dir.iterdir():
                if item.is_dir() and item.suffix == ".app":
                    app = self._create_app_from_bundle(item)
                    if app:
                        applications.append(app)
        except (PermissionError, OSError) as exc:
            self.logger.warning("Could not scan applications directory %s: %s", app_dir, exc)

        return applications

    def _create_app_from_bundle(self, bundle_path: Path) -> Optional[Application]:
        """
        Create an Application object from a macOS .app bundle.

        Reads ``Contents/Info.plist`` for metadata.  Returns ``None`` if the
        bundle is missing required fields or the executable does not exist.
        """
        # Guard: plistlib is available on all platforms but the paths only
        # make sense on macOS.  We still allow the code to run for testing.
        try:
            import plistlib  # stdlib – always available
        except ImportError:
            return None

        try:
            info_plist_path = bundle_path / "Contents" / "Info.plist"
            if not info_plist_path.exists():
                return None

            with open(info_plist_path, "rb") as fh:
                plist_data = plistlib.load(fh)

            name: str = (
                plist_data.get("CFBundleDisplayName")
                or plist_data.get("CFBundleName")
                or bundle_path.stem
            )
            version: str = plist_data.get("CFBundleShortVersionString", "")
            bundle_id: str = plist_data.get("CFBundleIdentifier", "")

            executable_name: Optional[str] = plist_data.get("CFBundleExecutable")
            if not executable_name:
                return None

            executable_path = bundle_path / "Contents" / "MacOS" / executable_name
            if not executable_path.exists():
                return None

            # Build metadata
            metadata = ApplicationMetadata()
            if "." in bundle_id:
                metadata.vendor = bundle_id.split(".")[0]
            metadata.description = plist_data.get("CFBundleGetInfoString", "")

            for doc_type in plist_data.get("CFBundleDocumentTypes", []):
                metadata.file_associations.extend(
                    doc_type.get("CFBundleTypeExtensions", [])
                )

            return self._create_application(
                name=name,
                executable_path=executable_path,
                installation_path=bundle_path,
                version=version,
                metadata=metadata,
            )

        except (OSError, PermissionError) as exc:
            self.logger.debug("Could not process app bundle %s: %s", bundle_path, exc)
            return None
        except Exception as exc:
            # Catch plistlib.InvalidFileException and any other parse errors
            self.logger.debug("Unexpected error processing bundle %s: %s", bundle_path, exc)
            return None

    # ------------------------------------------------------------------
    # Homebrew CLI integration
    # ------------------------------------------------------------------

    def _find_brew_executable(self) -> Optional[str]:
        """Return the path to the ``brew`` executable, or None if not found."""
        candidates = [
            "/opt/homebrew/bin/brew",   # Apple Silicon
            "/usr/local/bin/brew",      # Intel
            "brew",                     # PATH fallback
        ]
        for candidate in candidates:
            try:
                result = subprocess.run(
                    [candidate, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    encoding="utf-8",
                    errors="ignore",
                )
                if result.returncode == 0:
                    return candidate
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                continue
        return None

    def _scan_homebrew_cli(self, list_type: str) -> List[Application]:
        """
        Scan Homebrew packages using the ``brew`` CLI.

        Args:
            list_type: ``"cask"`` for GUI apps (``brew list --cask``),
                       ``"formula"`` for CLI tools (``brew list``).

        Returns:
            List[Application]: Discovered Homebrew applications.
        """
        applications: List[Application] = []

        brew = self._find_brew_executable()
        if brew is None:
            self.logger.debug("brew not found; skipping Homebrew CLI scan (%s)", list_type)
            return applications

        if list_type == "cask":
            cmd = [brew, "list", "--cask", "--versions"]
        else:
            cmd = [brew, "list", "--versions"]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                encoding="utf-8",
                errors="ignore",
            )
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as exc:
            self.logger.warning("brew list (%s) failed: %s", list_type, exc)
            return applications

        if result.returncode != 0:
            self.logger.debug(
                "brew list (%s) returned non-zero exit code %d", list_type, result.returncode
            )
            return applications

        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            package_name = parts[0]
            version = parts[1] if len(parts) > 1 else ""

            if list_type == "cask":
                app = self._resolve_brew_cask_app(brew, package_name, version)
            else:
                app = self._resolve_brew_formula_app(brew, package_name, version)

            if app:
                applications.append(app)

        return applications

    def _resolve_brew_cask_app(
        self, brew: str, cask_name: str, version: str
    ) -> Optional[Application]:
        """
        Resolve a Homebrew cask to an Application by finding its .app bundle.

        Tries the Caskroom directory first, then falls back to /Applications.
        """
        # Try Caskroom/<cask_name>/<version>/*.app
        for prefix in self._homebrew_prefixes:
            cask_dir = prefix / "Caskroom" / cask_name
            if cask_dir.exists():
                for item in cask_dir.rglob("*.app"):
                    if item.is_dir():
                        app = self._create_app_from_bundle(item)
                        if app:
                            return app

        # Fallback: look in /Applications for a matching name
        for app_dir in [Path("/Applications"), Path.home() / "Applications"]:
            candidate = app_dir / f"{cask_name}.app"
            if not candidate.exists():
                # Try title-cased name
                candidate = app_dir / f"{cask_name.replace('-', ' ').title()}.app"
            if candidate.exists():
                app = self._create_app_from_bundle(candidate)
                if app:
                    return app

        # Last resort: create a minimal Application entry
        return self._create_application(
            name=cask_name,
            executable_path=Path(f"/opt/homebrew/Caskroom/{cask_name}/{version}/{cask_name}"),
            installation_path=Path(f"/opt/homebrew/Caskroom/{cask_name}/{version}"),
            version=version,
        )

    def _resolve_brew_formula_app(
        self, brew: str, formula_name: str, version: str
    ) -> Optional[Application]:
        """
        Resolve a Homebrew formula to an Application by locating its binary.

        Checks ``<prefix>/bin/<formula_name>`` for each known Homebrew prefix.
        """
        for prefix in self._homebrew_prefixes:
            bin_path = prefix / "bin" / formula_name
            if bin_path.exists() and self._is_executable(bin_path):
                return self._create_application(
                    name=formula_name,
                    executable_path=bin_path,
                    installation_path=bin_path.parent,
                    version=version,
                )

        # Try Cellar
        for prefix in self._homebrew_prefixes:
            cellar_pkg = prefix / "Cellar" / formula_name
            if cellar_pkg.exists():
                version_dirs = sorted(
                    [d for d in cellar_pkg.iterdir() if d.is_dir()],
                    reverse=True,
                )
                for ver_dir in version_dirs:
                    bin_dir = ver_dir / "bin"
                    if bin_dir.exists():
                        for exe in bin_dir.iterdir():
                            if exe.is_file() and self._is_executable(exe):
                                return self._create_application(
                                    name=formula_name,
                                    executable_path=exe,
                                    installation_path=ver_dir,
                                    version=ver_dir.name,
                                )

        # Minimal fallback
        return self._create_application(
            name=formula_name,
            executable_path=Path(f"/usr/local/bin/{formula_name}"),
            installation_path=Path("/usr/local/bin"),
            version=version,
        )

    # ------------------------------------------------------------------
    # Homebrew directory scanning (fallback)
    # ------------------------------------------------------------------

    def _scan_homebrew_directory(self, directory: Path) -> List[Application]:
        """
        Scan a Homebrew Cellar or Caskroom directory for installed packages.

        This is a fallback for when the ``brew`` CLI is unavailable.
        """
        applications: List[Application] = []

        if not directory.exists():
            return applications

        dir_name = directory.name.lower()

        try:
            for package_dir in directory.iterdir():
                if not package_dir.is_dir():
                    continue

                if dir_name == "caskroom":
                    # Caskroom/<name>/<version>/*.app
                    for item in package_dir.rglob("*.app"):
                        if item.is_dir():
                            app = self._create_app_from_bundle(item)
                            if app:
                                applications.append(app)
                                break  # one per cask
                else:
                    # Cellar/<name>/<version>/bin/<exe>
                    version_dirs = sorted(
                        [d for d in package_dir.iterdir() if d.is_dir()],
                        reverse=True,
                    )
                    if not version_dirs:
                        continue
                    latest = version_dirs[0]
                    bin_dir = latest / "bin"
                    if bin_dir.exists():
                        for exe in bin_dir.iterdir():
                            if exe.is_file() and self._is_executable(exe):
                                app = self._create_application(
                                    name=package_dir.name,
                                    executable_path=exe,
                                    installation_path=latest,
                                    version=latest.name,
                                )
                                applications.append(app)
                                break  # one per formula

        except (PermissionError, OSError) as exc:
            self.logger.warning("Could not scan Homebrew directory %s: %s", directory, exc)

        return applications

    # ------------------------------------------------------------------
    # system_profiler integration (LaunchServices-style)
    # ------------------------------------------------------------------

    def _scan_system_profiler(self) -> List[Application]:
        """
        Query macOS system_profiler for installed applications.

        Runs ``system_profiler SPApplicationsDataType -json`` and parses the
        JSON output to discover applications registered with the OS.  This
        provides a LaunchServices-style view of installed apps without
        requiring the lsregister binary.

        Only runs on macOS (sys.platform check).

        Returns:
            List[Application]: Applications discovered via system_profiler.
        """
        if sys.platform != "darwin":
            self.logger.debug("Skipping system_profiler scan on non-macOS platform")
            return []

        try:
            result = subprocess.run(
                ["system_profiler", "SPApplicationsDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=120,
                encoding="utf-8",
                errors="ignore",
            )
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as exc:
            self.logger.warning("system_profiler SPApplicationsDataType failed: %s", exc)
            return []

        if result.returncode != 0:
            self.logger.debug(
                "system_profiler returned non-zero exit code %d", result.returncode
            )
            return []

        return self._parse_system_profiler_output(result.stdout)

    def _parse_system_profiler_output(self, output: str) -> List[Application]:
        """
        Parse ``system_profiler SPApplicationsDataType -json`` output.

        The JSON structure is::

            {
              "SPApplicationsDataType": [
                {
                  "_name": "Safari",
                  "path": "/Applications/Safari.app",
                  "version": "16.0",
                  "obtained_from": "apple",
                  "lastModified": "2023-01-01T00:00:00Z",
                  "signed_by": ["Apple Inc."],
                  ...
                },
                ...
              ]
            }

        Returns:
            List[Application]: Applications parsed from the output.
        """
        import json as _json

        applications: List[Application] = []
        seen_paths: set = set()

        try:
            data = _json.loads(output)
        except (_json.JSONDecodeError, ValueError) as exc:
            self.logger.warning("Could not parse system_profiler JSON output: %s", exc)
            return applications

        app_list = data.get("SPApplicationsDataType", [])
        if not isinstance(app_list, list):
            return applications

        for entry in app_list:
            if not isinstance(entry, dict):
                continue

            app_path_str: Optional[str] = entry.get("path")
            if not app_path_str:
                continue

            app_path = Path(app_path_str)
            path_key = str(app_path).lower()

            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)

            # If it's a .app bundle, use the full bundle parser for rich metadata
            if app_path_str.endswith(".app") and app_path.exists():
                app = self._create_app_from_bundle(app_path)
                if app:
                    # Enrich with system_profiler data if plist didn't provide it
                    if not app.version and entry.get("version"):
                        app.version = str(entry["version"])
                    if not app.metadata.vendor and entry.get("obtained_from"):
                        app.metadata.vendor = str(entry["obtained_from"])
                    if not app.metadata.digital_signature:
                        signed_by = entry.get("signed_by")
                        if isinstance(signed_by, list) and signed_by:
                            app.metadata.digital_signature = ", ".join(signed_by)
                    applications.append(app)
                    continue

            # Non-bundle executable (e.g. command-line tools listed by system_profiler)
            name: str = entry.get("_name") or app_path.stem
            version: str = str(entry.get("version", ""))

            metadata = ApplicationMetadata()
            obtained_from = entry.get("obtained_from")
            if obtained_from:
                metadata.vendor = str(obtained_from)
            signed_by = entry.get("signed_by")
            if isinstance(signed_by, list) and signed_by:
                metadata.digital_signature = ", ".join(signed_by)

            # Parse lastModified as install_date
            last_modified = entry.get("lastModified")
            if last_modified:
                try:
                    metadata.install_date = datetime.fromisoformat(
                        last_modified.replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    pass

            app = self._create_application(
                name=name,
                executable_path=app_path,
                installation_path=app_path.parent,
                version=version,
                metadata=metadata,
            )
            applications.append(app)

        return applications

    # ------------------------------------------------------------------
    # LaunchServices database
    # ------------------------------------------------------------------

    _LSREGISTER_PATH = (
        "/System/Library/Frameworks/CoreServices.framework"
        "/Frameworks/LaunchServices.framework"
        "/Support/lsregister"
    )

    def _scan_launch_services(self) -> List[Application]:
        """
        Query the macOS LaunchServices database via ``lsregister -dump``.

        Parses the output to find registered .app bundles and creates
        Application objects from them.  Only runs on macOS (sys.platform check).

        Returns:
            List[Application]: Applications registered in LaunchServices.
        """
        if sys.platform != "darwin":
            self.logger.debug("Skipping LaunchServices scan on non-macOS platform")
            return []

        lsregister = Path(self._LSREGISTER_PATH)
        if not lsregister.exists():
            self.logger.debug("lsregister not found at %s", lsregister)
            return []

        try:
            result = subprocess.run(
                [str(lsregister), "-dump"],
                capture_output=True,
                text=True,
                timeout=60,
                encoding="utf-8",
                errors="ignore",
            )
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as exc:
            self.logger.warning("lsregister -dump failed: %s", exc)
            return []

        if result.returncode != 0:
            self.logger.debug("lsregister returned non-zero exit code %d", result.returncode)
            return []

        return self._parse_lsregister_output(result.stdout)

    def _parse_lsregister_output(self, output: str) -> List[Application]:
        """
        Parse ``lsregister -dump`` output and return Application objects.

        The output is a series of records separated by ``-`` lines.  Each
        record may contain a ``path:`` field pointing to a ``.app`` bundle.
        We collect unique bundle paths and create Application objects from them.
        """
        applications: List[Application] = []
        seen_paths: set = set()

        current_path: Optional[str] = None

        for raw_line in output.splitlines():
            line = raw_line.strip()

            # A line of dashes marks the start of a new record
            if line.startswith("---") or line.startswith("==="):
                current_path = None
                continue

            # Look for the bundle path
            if line.lower().startswith("path:"):
                path_value = line[5:].strip()
                if path_value:
                    current_path = path_value
                continue

            # Some records use "bundle path:" or "url:" with file:// scheme
            if line.lower().startswith("bundle path:"):
                path_value = line[len("bundle path:"):].strip()
                if path_value:
                    current_path = path_value
                continue

            if line.lower().startswith("url:") and "file://" in line.lower():
                url_value = line[4:].strip()
                # Convert file:///path/to/App.app -> /path/to/App.app
                if url_value.startswith("file://"):
                    current_path = url_value[7:]  # strip "file://"
                continue

            # When we have a path that ends in .app, try to create an Application
            if current_path and current_path.endswith(".app"):
                bundle_path = Path(current_path)
                path_key = str(bundle_path).lower()

                if path_key not in seen_paths and bundle_path.exists():
                    seen_paths.add(path_key)
                    app = self._create_app_from_bundle(bundle_path)
                    if app:
                        applications.append(app)
                current_path = None

        return applications

    # ------------------------------------------------------------------
    # Executable directory scanning (portable apps / PATH tools)
    # ------------------------------------------------------------------

    def _scan_directory_for_executables(self, directory: Path) -> List[Application]:
        """
        Scan a directory for executable files (non-.app portable tools).

        Args:
            directory: Directory to scan.

        Returns:
            List[Application]: Applications found in directory.
        """
        applications: List[Application] = []

        if not directory.exists():
            return applications

        try:
            for item in directory.iterdir():
                if not item.is_file():
                    continue
                if not self._is_executable(item):
                    continue
                # Skip common noise
                if any(
                    skip in item.name.lower()
                    for skip in ("uninstall", "setup", "install")
                ):
                    continue
                app = self._create_application(
                    name=item.name,
                    executable_path=item,
                    installation_path=directory,
                )
                applications.append(app)
        except (PermissionError, OSError) as exc:
            self.logger.debug("Could not scan directory %s: %s", directory, exc)

        return applications

    # ------------------------------------------------------------------
    # Legacy helpers kept for backward compatibility
    # ------------------------------------------------------------------

    def _scan_homebrew(self) -> List[Application]:
        """Legacy method – prefer scan_package_managers()."""
        return self._scan_homebrew_cli("cask") + self._scan_homebrew_cli("formula")

    def _scan_homebrew_cellar(self, cellar_path: Path) -> List[Application]:
        """Legacy method – prefer _scan_homebrew_directory()."""
        return self._scan_homebrew_directory(cellar_path)

    def _scan_homebrew_cask(self, cask_path: Path) -> List[Application]:
        """Legacy method – prefer _scan_homebrew_directory()."""
        return self._scan_homebrew_directory(cask_path)

    def _scan_macports(self) -> List[Application]:
        """Legacy method – prefer scan_package_managers()."""
        return self._scan_directory_for_executables(Path("/opt/local/bin"))

    def _query_launch_services(self) -> List[Dict[str, Any]]:
        """
        Legacy method – returns raw dicts from LaunchServices for backward compat.

        Prefer _scan_launch_services() which returns Application objects.
        """
        if sys.platform != "darwin":
            return []

        lsregister = Path(self._LSREGISTER_PATH)
        if not lsregister.exists():
            return []

        try:
            result = subprocess.run(
                [str(lsregister), "-dump"],
                capture_output=True,
                text=True,
                timeout=60,
                encoding="utf-8",
                errors="ignore",
            )
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as exc:
            self.logger.debug("Could not query LaunchServices: %s", exc)
            return []

        if result.returncode != 0:
            return []

        apps: List[Dict[str, Any]] = []
        current_app: Dict[str, Any] = {}

        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("---") or line.startswith("==="):
                if current_app:
                    apps.append(current_app)
                current_app = {}
            elif line.lower().startswith("path:"):
                current_app["path"] = line[5:].strip()
            elif "CFBundleIdentifier" in line and current_app:
                current_app["bundle_id"] = line.split("=")[-1].strip()
            elif "CFBundleName" in line and current_app:
                current_app["name"] = line.split("=")[-1].strip()

        if current_app:
            apps.append(current_app)

        return apps
