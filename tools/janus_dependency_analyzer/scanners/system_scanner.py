"""
System scanner implementation that orchestrates platform-specific scanners.

This module provides the main SystemScanner implementation that detects the
current platform and delegates to the appropriate platform-specific scanner.
"""

import logging
import platform
import sys
from datetime import datetime
from typing import Optional, List

from ..core.interfaces import SystemScanner, PlatformScanner
from ..core.models import Platform, ScanResult, Application, ChangeRecord


logger = logging.getLogger(__name__)


class SystemScannerImpl(SystemScanner):
    """
    Main system scanner implementation.
    
    This class orchestrates the scanning process by detecting the current
    platform and delegating to the appropriate platform-specific scanner.
    """
    
    def __init__(self):
        """Initialize the system scanner."""
        self.logger = logging.getLogger(__name__)
        self._platform_scanner: Optional[PlatformScanner] = None
        self._detected_platform: Optional[Platform] = None
    
    def detect_platform(self) -> Platform:
        """
        Detect the current operating system platform.
        
        Uses Python's platform.system() to identify the OS and maps it to
        our Platform enum. Caches the result for subsequent calls.
        
        Returns:
            Platform: The detected platform
            
        Raises:
            RuntimeError: If the platform is not supported
        """
        if self._detected_platform is not None:
            return self._detected_platform
        
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        self.logger.debug(f"Detecting platform: system={system}, machine={machine}")
        
        if system == "windows":
            self._detected_platform = Platform.WINDOWS
        elif system == "darwin":
            self._detected_platform = Platform.MACOS
        elif system == "linux":
            self._detected_platform = Platform.LINUX
        else:
            supported_platforms = [p.value for p in Platform]
            raise RuntimeError(
                f"Unsupported platform: {system}. "
                f"Supported platforms: {', '.join(supported_platforms)}"
            )
        
        self.logger.info(f"Detected platform: {self._detected_platform.value} "
                        f"(system: {system}, machine: {machine})")
        return self._detected_platform
    
    def get_platform_scanner(self) -> PlatformScanner:
        """
        Get the appropriate platform-specific scanner.
        
        This method uses dynamic imports to load platform-specific scanners,
        which prevents import errors on platforms where certain dependencies
        might not be available. Falls back to a basic scanner if the
        platform-specific scanner cannot be loaded.
        
        Returns:
            PlatformScanner: Platform-specific scanner implementation
            
        Raises:
            RuntimeError: If no scanner is available for the current platform
        """
        if self._platform_scanner is not None:
            return self._platform_scanner
        
        detected_platform = self.detect_platform()
        
        # Import platform-specific scanners dynamically to avoid import errors
        # on platforms where certain dependencies might not be available
        try:
            if detected_platform == Platform.WINDOWS:
                self.logger.debug("Loading Windows scanner...")
                from .windows_scanner import WindowsScanner
                self._platform_scanner = WindowsScanner()
                self.logger.info("Windows scanner loaded successfully")
                
            elif detected_platform == Platform.MACOS:
                self.logger.debug("Loading macOS scanner...")
                from .macos_scanner import MacOSScanner
                self._platform_scanner = MacOSScanner()
                self.logger.info("macOS scanner loaded successfully")
                
            elif detected_platform == Platform.LINUX:
                self.logger.debug("Loading Linux scanner...")
                from .linux_scanner import LinuxScanner
                self._platform_scanner = LinuxScanner()
                self.logger.info("Linux scanner loaded successfully")
                
            else:
                raise RuntimeError(f"No scanner available for platform: {detected_platform}")
                
        except ImportError as e:
            self.logger.error(f"Failed to import platform scanner for {detected_platform.value}: {e}")
            self.logger.warning("Falling back to basic scanner with limited functionality")
            
            # Fall back to a basic scanner that can at least attempt discovery
            try:
                from .base import BasePlatformScanner
                self._platform_scanner = BasePlatformScanner(detected_platform)
                self.logger.info(f"Basic scanner loaded for {detected_platform.value}")
            except ImportError as base_error:
                raise RuntimeError(
                    f"Could not load any scanner for {detected_platform.value}. "
                    f"Platform scanner error: {e}. Base scanner error: {base_error}"
                )
        
        except Exception as e:
            self.logger.error(f"Unexpected error loading scanner for {detected_platform.value}: {e}")
            raise RuntimeError(f"Failed to initialize scanner for {detected_platform.value}: {e}")
        
        return self._platform_scanner
    
    def scan_full(self) -> ScanResult:
        """
        Perform a complete system scan for all applications.
        
        This method orchestrates a comprehensive scan of the system by:
        1. Detecting the current platform
        2. Getting the appropriate platform scanner
        3. Discovering all applications using multiple strategies
        4. Collecting and reporting scan statistics
        
        Returns:
            ScanResult: Complete scan results with all discovered applications
        """
        self.logger.info("Starting full system scan...")
        
        result = ScanResult(
            scan_type="full",
            platform=self.detect_platform(),
            scan_start_time=datetime.now()
        )
        
        try:
            scanner = self.get_platform_scanner()
            self.logger.info(f"Using scanner: {scanner.__class__.__name__}")
            
            # Perform the discovery
            self.logger.debug("Beginning application discovery...")
            applications = scanner.discover_applications()
            
            # Process discovered applications
            for app in applications:
                result.add_application(app)
            
            # Log scan statistics
            self.logger.info(
                f"Full scan completed successfully. "
                f"Discovered {result.total_applications} applications "
                f"({result.accessible_applications} accessible, "
                f"{result.total_applications - result.accessible_applications} inaccessible)"
            )
            
            if result.errors:
                self.logger.warning(f"Scan completed with {len(result.errors)} errors")
                for error in result.errors:
                    self.logger.error(f"Scan error: {error}")
            
            if result.warnings:
                self.logger.info(f"Scan completed with {len(result.warnings)} warnings")
                for warning in result.warnings:
                    self.logger.warning(f"Scan warning: {warning}")
            
        except Exception as e:
            error_msg = f"Full scan failed with unexpected error: {e}"
            self.logger.error(error_msg, exc_info=True)
            result.add_error(error_msg)
        
        result.finalize()
        return result
    
    def scan_incremental(self, last_scan_time: datetime, catalog=None) -> ScanResult:
        """
        Scan for changes since the last scan time.
        
        This method performs an intelligent incremental scan by:
        1. Performing a full discovery to get current state
        2. Comparing with the last scan time to identify changes
        3. Detecting new, modified, and removed applications
        4. Only including changed applications in the result
        
        Args:
            last_scan_time: Timestamp of the previous scan
            catalog: Optional ApplicationCatalog to update with changes and
                     record change history.
            
        Returns:
            ScanResult: Incremental scan results with only changed applications
        """
        self.logger.info(f"Starting incremental scan since {last_scan_time}")
        
        result = ScanResult(
            scan_type="incremental",
            platform=self.detect_platform(),
            scan_start_time=datetime.now()
        )
        
        try:
            scanner = self.get_platform_scanner()
            current_applications = scanner.discover_applications()
            
            # Filter applications based on change detection
            changed_applications = self._detect_changes(current_applications, last_scan_time)
            
            for app in changed_applications:
                result.add_application(app)
                if catalog is not None:
                    existing = catalog.get(app.id)
                    if existing is None:
                        # Newly discovered application
                        catalog.add(app)
                        catalog.record_change(ChangeRecord(
                            app_id=app.id,
                            app_name=app.name,
                            change_type="installed",
                        ))
                    elif existing.version != app.version:
                        # Version has changed
                        old_version = existing.version
                        catalog.add(app)
                        catalog.record_change(ChangeRecord(
                            app_id=app.id,
                            app_name=app.name,
                            change_type="updated",
                            previous_version=old_version,
                            new_version=app.version,
                        ))
            
            self.logger.info(f"Incremental scan completed. Found {result.total_applications} "
                           f"changed applications ({result.accessible_applications} accessible) "
                           f"since {last_scan_time}")
            
        except Exception as e:
            error_msg = f"Incremental scan failed: {e}"
            self.logger.error(error_msg)
            result.add_error(error_msg)
        
        result.finalize()
        return result
    
    def _detect_changes(self, current_applications: List[Application], last_scan_time: datetime) -> List[Application]:
        """
        Detect changes in applications since the last scan time.
        
        This method identifies applications that are:
        - Newly installed (discovered_at > last_scan_time)
        - Recently modified (installation files modified after last_scan_time)
        - Previously inaccessible but now accessible (or vice versa)
        
        Args:
            current_applications: Currently discovered applications
            last_scan_time: Timestamp of the previous scan
            
        Returns:
            List[Application]: Applications that have changed since last scan
        """
        changed_applications = []
        
        for app in current_applications:
            # Check if application is newly discovered
            if app.discovered_at >= last_scan_time:
                self.logger.debug(f"New application detected: {app.name}")
                changed_applications.append(app)
                continue
            
            # Check if application files have been modified
            if self._has_application_changed(app, last_scan_time):
                self.logger.debug(f"Modified application detected: {app.name}")
                # Update the discovered_at timestamp to reflect the change
                app.discovered_at = datetime.now()
                changed_applications.append(app)
                continue
            
            # Check if accessibility status might have changed
            # This is a heuristic check - we include apps that were previously
            # problematic in case their status has changed
            if not app.is_accessible and app.access_error:
                self.logger.debug(f"Re-checking previously inaccessible application: {app.name}")
                changed_applications.append(app)
        
        return changed_applications
    
    def _has_application_changed(self, app: Application, last_scan_time: datetime) -> bool:
        """
        Check if an application has been modified since the last scan.
        
        This method checks various indicators of application changes:
        - Executable file modification time
        - Installation directory modification time
        - Version information changes (if available)
        
        Args:
            app: Application to check
            last_scan_time: Timestamp of the previous scan
            
        Returns:
            bool: True if the application has changed since last scan
        """
        try:
            # Check executable file modification time
            if app.executable_path.exists():
                exe_mtime = datetime.fromtimestamp(app.executable_path.stat().st_mtime)
                if exe_mtime > last_scan_time:
                    return True
            
            # Check installation directory modification time
            if app.installation_path.exists():
                install_mtime = datetime.fromtimestamp(app.installation_path.stat().st_mtime)
                if install_mtime > last_scan_time:
                    return True
            
            # For package-managed applications, we could check package databases
            # This is a platform-specific optimization that could be added later
            
        except (OSError, PermissionError) as e:
            self.logger.debug(f"Could not check modification time for {app.name}: {e}")
            # If we can't check, assume it might have changed to be safe
            return True
        
        # If files don't exist or haven't been modified, no change detected
        return False