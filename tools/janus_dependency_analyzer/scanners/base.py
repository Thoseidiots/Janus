"""
Base implementation for platform scanners.

This module provides a common base class that implements shared functionality
for all platform-specific scanners.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from ..core.interfaces import PlatformScanner
from ..core.models import Application, ApplicationMetadata, Platform


logger = logging.getLogger(__name__)


class BasePlatformScanner(PlatformScanner):
    """
    Base implementation for platform-specific scanners.
    
    Provides common functionality and error handling that all platform
    scanners can inherit and extend.
    """
    
    def __init__(self, platform: Platform):
        """
        Initialize the base platform scanner.
        
        Args:
            platform: The platform this scanner supports
        """
        self.platform = platform
        self.logger = logging.getLogger(f"{__name__}.{platform.value}")
    
    def get_platform(self) -> Platform:
        """Get the platform this scanner supports."""
        return self.platform
    
    def discover_applications(self) -> List[Application]:
        """
        Discover all applications using all available methods.
        
        This method orchestrates the discovery process by calling all
        specific discovery methods and combining their results.
        """
        applications = []
        
        try:
            # Scan standard locations
            self.logger.info("Scanning standard application locations...")
            standard_apps = self.scan_standard_locations()
            applications.extend(standard_apps)
            self.logger.info(f"Found {len(standard_apps)} applications in standard locations")
        except Exception as e:
            self.logger.error(f"Error scanning standard locations: {e}")
        
        try:
            # Scan package managers
            self.logger.info("Scanning package manager installations...")
            package_apps = self.scan_package_managers()
            applications.extend(package_apps)
            self.logger.info(f"Found {len(package_apps)} applications from package managers")
        except Exception as e:
            self.logger.error(f"Error scanning package managers: {e}")
        
        try:
            # Scan for portable applications
            self.logger.info("Scanning for portable applications...")
            portable_apps = self.scan_portable_applications()
            applications.extend(portable_apps)
            self.logger.info(f"Found {len(portable_apps)} portable applications")
        except Exception as e:
            self.logger.error(f"Error scanning portable applications: {e}")
        
        # Deduplicate applications
        unique_apps = self._deduplicate_applications(applications)
        self.logger.info(f"Total unique applications discovered: {len(unique_apps)}")
        
        return unique_apps
    
    def extract_metadata(self, app_path: Path) -> ApplicationMetadata:
        """
        Extract basic metadata for an application.
        
        This base implementation provides common metadata extraction.
        Platform-specific scanners should override this to add platform-specific details.
        """
        metadata = ApplicationMetadata()
        
        try:
            if app_path.exists():
                stat = app_path.stat()
                metadata.file_size = stat.st_size
                metadata.install_date = datetime.fromtimestamp(stat.st_ctime)
        except (OSError, PermissionError) as e:
            self.logger.warning(f"Could not extract metadata for {app_path}: {e}")
        
        return metadata
    
    def _deduplicate_applications(self, applications: List[Application]) -> List[Application]:
        """
        Remove duplicate applications based on executable path and name.
        
        Args:
            applications: List of applications that may contain duplicates
            
        Returns:
            List[Application]: Deduplicated list of applications
        """
        seen = set()
        unique_apps = []
        
        for app in applications:
            # Create a key based on executable path and name
            key = (str(app.executable_path).lower(), app.name.lower())
            
            if key not in seen:
                seen.add(key)
                unique_apps.append(app)
            else:
                self.logger.debug(f"Skipping duplicate application: {app.name} at {app.executable_path}")
        
        return unique_apps
    
    def _is_executable(self, path: Path) -> bool:
        """
        Check if a path points to an executable file.
        
        Args:
            path: Path to check
            
        Returns:
            bool: True if the path is an executable file
        """
        try:
            return path.is_file() and os.access(path, os.X_OK)
        except (OSError, PermissionError):
            return False
    
    def _create_application(
        self,
        name: str,
        executable_path: Path,
        installation_path: Optional[Path] = None,
        version: str = "",
        metadata: Optional[ApplicationMetadata] = None
    ) -> Application:
        """
        Create an Application object with common fields populated.
        
        Args:
            name: Application name
            executable_path: Path to the executable
            installation_path: Path to the installation directory
            version: Application version
            metadata: Additional metadata
            
        Returns:
            Application: Created application object
        """
        if installation_path is None:
            installation_path = executable_path.parent
        
        if metadata is None:
            metadata = self.extract_metadata(executable_path)
        
        # Check if the application is accessible
        is_accessible = True
        access_error = None
        
        try:
            if not executable_path.exists():
                is_accessible = False
                access_error = "Executable not found"
            elif not os.access(executable_path, os.R_OK):
                is_accessible = False
                access_error = "Permission denied"
        except (OSError, PermissionError) as e:
            is_accessible = False
            access_error = str(e)
        
        return Application(
            name=name,
            version=version,
            installation_path=installation_path,
            executable_path=executable_path,
            platform=self.platform,
            metadata=metadata,
            discovered_at=datetime.now(),
            is_accessible=is_accessible,
            access_error=access_error
        )
    
    def scan_standard_locations(self) -> List[Application]:
        """
        Scan standard application installation locations.
        
        This base implementation returns an empty list.
        Platform-specific scanners must override this method.
        """
        return []
    
    def scan_package_managers(self) -> List[Application]:
        """
        Scan applications installed via package managers.
        
        This base implementation returns an empty list.
        Platform-specific scanners should override this method if applicable.
        """
        return []
    
    def scan_portable_applications(self) -> List[Application]:
        """
        Scan for portable applications not in standard locations.
        
        This base implementation returns an empty list.
        Platform-specific scanners should override this method if needed.
        """
        return []