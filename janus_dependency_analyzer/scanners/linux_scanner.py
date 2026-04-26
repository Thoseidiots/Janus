"""
Linux-specific application scanner.

This module implements application discovery for Linux systems using
package manager integration, /usr/bin scanning, and AppImage/Flatpak detection.
"""

import logging
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
import configparser

from .base import BasePlatformScanner
from ..core.models import Application, ApplicationMetadata, Platform


logger = logging.getLogger(__name__)


class LinuxScanner(BasePlatformScanner):
    """
    Linux-specific application scanner.
    
    Implements application discovery for Linux systems by scanning:
    - Package manager databases (apt, yum, pacman, etc.)
    - Standard binary directories (/usr/bin, /usr/local/bin, etc.)
    - AppImage and Flatpak applications
    - Desktop entry files
    """
    
    def __init__(self):
        """Initialize the Linux scanner."""
        super().__init__(Platform.LINUX)
        
        # Common Linux binary directories
        self.binary_directories = [
            Path("/usr/bin"),
            Path("/usr/local/bin"),
            Path("/bin"),
            Path("/sbin"),
            Path("/usr/sbin"),
            Path("/usr/local/sbin"),
            Path.home() / ".local/bin"
        ]
        
        # Desktop entry directories
        self.desktop_entry_dirs = [
            Path("/usr/share/applications"),
            Path("/usr/local/share/applications"),
            Path.home() / ".local/share/applications"
        ]
    
    def scan_standard_locations(self) -> List[Application]:
        """
        Scan standard Linux application installation locations.
        
        Returns:
            List[Application]: Applications found in standard locations
        """
        applications = []
        
        # Scan binary directories
        for bin_dir in self.binary_directories:
            if bin_dir.exists():
                apps = self._scan_directory_for_executables(bin_dir)
                applications.extend(apps)
        
        # Scan desktop entries for additional metadata
        desktop_apps = self._scan_desktop_entries()
        applications.extend(desktop_apps)
        
        return applications
    
    def scan_package_managers(self) -> List[Application]:
        """
        Scan applications installed via Linux package managers.
        
        Returns:
            List[Application]: Applications found via package managers
        """
        applications = []
        
        # Detect and scan available package managers
        package_managers = self._detect_package_managers()
        
        for pm_name, pm_info in package_managers.items():
            try:
                apps = self._scan_package_manager(pm_name, pm_info)
                applications.extend(apps)
            except Exception as e:
                self.logger.warning(f"Error scanning {pm_name}: {e}")
        
        # Scan Flatpak applications
        flatpak_apps = self._scan_flatpak()
        applications.extend(flatpak_apps)
        
        # Scan Snap applications
        snap_apps = self._scan_snap()
        applications.extend(snap_apps)
        
        return applications
    
    def scan_portable_applications(self) -> List[Application]:
        """
        Scan for portable applications like AppImages.
        
        Returns:
            List[Application]: Portable applications found
        """
        applications = []
        
        # Scan for AppImage files
        appimage_apps = self._scan_appimages()
        applications.extend(appimage_apps)
        
        # Scan common directories for portable executables
        portable_dirs = [
            Path.home() / "Applications",
            Path.home() / "Desktop",
            Path.home() / "Downloads",
            Path("/opt")
        ]
        
        for directory in portable_dirs:
            if directory.exists():
                apps = self._scan_directory_for_executables(directory)
                applications.extend(apps)
        
        return applications
    
    def _detect_package_managers(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect available package managers on the system.
        
        Returns:
            Dict[str, Dict[str, Any]]: Available package managers and their info
        """
        package_managers = {}
        
        # APT (Debian/Ubuntu)
        if shutil.which("apt") or shutil.which("dpkg"):
            package_managers["apt"] = {
                "list_command": ["dpkg", "-l"],
                "info_command": ["dpkg", "-s"],
                "type": "dpkg"
            }
        
        # YUM/DNF (Red Hat/Fedora)
        if shutil.which("dnf"):
            package_managers["dnf"] = {
                "list_command": ["dnf", "list", "installed"],
                "info_command": ["dnf", "info"],
                "type": "rpm"
            }
        elif shutil.which("yum"):
            package_managers["yum"] = {
                "list_command": ["yum", "list", "installed"],
                "info_command": ["yum", "info"],
                "type": "rpm"
            }
        
        # Pacman (Arch Linux)
        if shutil.which("pacman"):
            package_managers["pacman"] = {
                "list_command": ["pacman", "-Q"],
                "info_command": ["pacman", "-Qi"],
                "type": "pacman"
            }
        
        # Zypper (openSUSE)
        if shutil.which("zypper"):
            package_managers["zypper"] = {
                "list_command": ["zypper", "search", "--installed-only"],
                "info_command": ["zypper", "info"],
                "type": "rpm"
            }
        
        return package_managers
    
    def _scan_package_manager(self, pm_name: str, pm_info: Dict[str, Any]) -> List[Application]:
        """
        Scan a specific package manager for installed packages.
        
        Args:
            pm_name: Package manager name
            pm_info: Package manager information
            
        Returns:
            List[Application]: Applications found via package manager
        """
        applications = []
        
        try:
            result = subprocess.run(
                pm_info["list_command"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                if pm_info["type"] == "dpkg":
                    apps = self._parse_dpkg_output(result.stdout)
                elif pm_info["type"] == "rpm":
                    apps = self._parse_rpm_output(result.stdout, pm_name)
                elif pm_info["type"] == "pacman":
                    apps = self._parse_pacman_output(result.stdout)
                else:
                    apps = []
                
                applications.extend(apps)
                
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            self.logger.warning(f"Could not query {pm_name}: {e}")
        
        return applications
    
    def _parse_dpkg_output(self, output: str) -> List[Application]:
        """
        Parse dpkg -l output to extract package information.
        
        Args:
            output: dpkg command output
            
        Returns:
            List[Application]: Parsed applications
        """
        applications = []
        
        for line in output.split('\n'):
            if line.startswith('ii'):  # Installed packages
                parts = line.split()
                if len(parts) >= 4:
                    package_name = parts[1]
                    version = parts[2]
                    description = ' '.join(parts[3:])
                    
                    # Try to find the executable
                    executable_path = self._find_package_executable(package_name)
                    if executable_path:
                        metadata = ApplicationMetadata()
                        metadata.description = description
                        
                        app = self._create_application(
                            name=package_name,
                            executable_path=executable_path,
                            version=version,
                            metadata=metadata
                        )
                        applications.append(app)
        
        return applications
    
    def _parse_rpm_output(self, output: str, pm_name: str) -> List[Application]:
        """
        Parse RPM-based package manager output.
        
        Args:
            output: Package manager command output
            pm_name: Package manager name (dnf, yum, zypper)
            
        Returns:
            List[Application]: Parsed applications
        """
        applications = []
        
        for line in output.split('\n'):
            if line.strip() and not line.startswith('Installed') and not line.startswith('Last'):
                parts = line.split()
                if len(parts) >= 2:
                    package_name = parts[0].split('.')[0]  # Remove architecture
                    version = parts[1] if len(parts) > 1 else ""
                    
                    executable_path = self._find_package_executable(package_name)
                    if executable_path:
                        app = self._create_application(
                            name=package_name,
                            executable_path=executable_path,
                            version=version
                        )
                        applications.append(app)
        
        return applications
    
    def _parse_pacman_output(self, output: str) -> List[Application]:
        """
        Parse pacman -Q output to extract package information.
        
        Args:
            output: pacman command output
            
        Returns:
            List[Application]: Parsed applications
        """
        applications = []
        
        for line in output.split('\n'):
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    package_name = parts[0]
                    version = parts[1]
                    
                    executable_path = self._find_package_executable(package_name)
                    if executable_path:
                        app = self._create_application(
                            name=package_name,
                            executable_path=executable_path,
                            version=version
                        )
                        applications.append(app)
        
        return applications
    
    def _find_package_executable(self, package_name: str) -> Optional[Path]:
        """
        Find the main executable for a package.
        
        Args:
            package_name: Name of the package
            
        Returns:
            Optional[Path]: Path to executable if found
        """
        # Try common locations
        for bin_dir in self.binary_directories:
            executable_path = bin_dir / package_name
            if executable_path.exists() and self._is_executable(executable_path):
                return executable_path
        
        # Try variations of the package name
        variations = [
            package_name.replace('-', ''),
            package_name.replace('_', ''),
            package_name.split('-')[0],
            package_name.split('_')[0]
        ]
        
        for variation in variations:
            for bin_dir in self.binary_directories:
                executable_path = bin_dir / variation
                if executable_path.exists() and self._is_executable(executable_path):
                    return executable_path
        
        return None
    
    def _scan_desktop_entries(self) -> List[Application]:
        """
        Scan desktop entry files for application metadata.
        
        Returns:
            List[Application]: Applications found via desktop entries
        """
        applications = []
        
        for desktop_dir in self.desktop_entry_dirs:
            if desktop_dir.exists():
                try:
                    for desktop_file in desktop_dir.glob("*.desktop"):
                        app = self._parse_desktop_entry(desktop_file)
                        if app:
                            applications.append(app)
                except (PermissionError, OSError) as e:
                    self.logger.debug(f"Could not scan desktop entries in {desktop_dir}: {e}")
        
        return applications
    
    def _parse_desktop_entry(self, desktop_file: Path) -> Optional[Application]:
        """
        Parse a .desktop file to extract application information.
        
        Args:
            desktop_file: Path to .desktop file
            
        Returns:
            Optional[Application]: Application if valid, None otherwise
        """
        try:
            config = configparser.ConfigParser()
            config.read(desktop_file, encoding='utf-8')
            
            if 'Desktop Entry' not in config:
                return None
            
            entry = config['Desktop Entry']
            
            # Skip if not an application
            if entry.get('Type', '') != 'Application':
                return None
            
            # Skip if NoDisplay is true
            if entry.getboolean('NoDisplay', False):
                return None
            
            name = entry.get('Name', desktop_file.stem)
            exec_line = entry.get('Exec', '')
            
            if not exec_line:
                return None
            
            # Parse the Exec line to get the executable path
            exec_parts = exec_line.split()
            if not exec_parts:
                return None
            
            executable_name = exec_parts[0]
            executable_path = None
            
            # If it's an absolute path, use it directly
            if Path(executable_name).is_absolute():
                executable_path = Path(executable_name)
            else:
                # Search in PATH
                executable_path = shutil.which(executable_name)
                if executable_path:
                    executable_path = Path(executable_path)
            
            if not executable_path or not executable_path.exists():
                return None
            
            # Create metadata
            metadata = ApplicationMetadata()
            metadata.description = entry.get('Comment', '')
            metadata.vendor = entry.get('X-GNOME-Bugzilla-Product', '')
            
            # Extract file associations
            mime_types = entry.get('MimeType', '').split(';')
            metadata.file_associations = [mt for mt in mime_types if mt]
            
            version = entry.get('Version', '')
            
            return self._create_application(
                name=name,
                executable_path=executable_path,
                version=version,
                metadata=metadata
            )
            
        except (configparser.Error, OSError, UnicodeDecodeError) as e:
            self.logger.debug(f"Could not parse desktop entry {desktop_file}: {e}")
            return None
    
    def _scan_flatpak(self) -> List[Application]:
        """
        Scan for Flatpak applications.
        
        Returns:
            List[Application]: Flatpak applications found
        """
        applications = []
        
        if not shutil.which("flatpak"):
            return applications
        
        try:
            result = subprocess.run(
                ["flatpak", "list", "--app"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 3:
                            name = parts[0]
                            app_id = parts[1]
                            version = parts[2]
                            
                            # Flatpak apps are executed via flatpak run
                            executable_path = Path(shutil.which("flatpak"))
                            
                            metadata = ApplicationMetadata()
                            metadata.description = f"Flatpak application: {app_id}"
                            
                            app = self._create_application(
                                name=name,
                                executable_path=executable_path,
                                version=version,
                                metadata=metadata
                            )
                            applications.append(app)
                            
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            self.logger.warning(f"Could not query Flatpak: {e}")
        
        return applications
    
    def _scan_snap(self) -> List[Application]:
        """
        Scan for Snap applications.
        
        Returns:
            List[Application]: Snap applications found
        """
        applications = []
        
        if not shutil.which("snap"):
            return applications
        
        try:
            result = subprocess.run(
                ["snap", "list"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 3:
                            name = parts[0]
                            version = parts[1]
                            
                            # Snap apps are in /snap/bin
                            executable_path = Path(f"/snap/bin/{name}")
                            if not executable_path.exists():
                                executable_path = Path(shutil.which("snap"))
                            
                            metadata = ApplicationMetadata()
                            metadata.description = f"Snap application: {name}"
                            
                            app = self._create_application(
                                name=name,
                                executable_path=executable_path,
                                version=version,
                                metadata=metadata
                            )
                            applications.append(app)
                            
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            self.logger.warning(f"Could not query Snap: {e}")
        
        return applications
    
    def _scan_appimages(self) -> List[Application]:
        """
        Scan for AppImage applications.
        
        Returns:
            List[Application]: AppImage applications found
        """
        applications = []
        
        # Common AppImage locations
        appimage_dirs = [
            Path.home() / "Applications",
            Path.home() / "Desktop",
            Path.home() / "Downloads",
            Path("/opt")
        ]
        
        for directory in appimage_dirs:
            if directory.exists():
                try:
                    for appimage in directory.glob("*.AppImage"):
                        if appimage.is_file() and self._is_executable(appimage):
                            app = self._create_application(
                                name=appimage.stem,
                                executable_path=appimage,
                                installation_path=directory
                            )
                            applications.append(app)
                except (PermissionError, OSError) as e:
                    self.logger.debug(f"Could not scan AppImages in {directory}: {e}")
        
        return applications
    
    def _scan_directory_for_executables(self, directory: Path) -> List[Application]:
        """
        Scan a directory for executable files.
        
        Args:
            directory: Directory to scan
            
        Returns:
            List[Application]: Applications found in directory
        """
        applications = []
        
        try:
            for item in directory.iterdir():
                if item.is_file() and self._is_executable(item):
                    # Skip common system utilities and scripts
                    if any(skip in item.name.lower() for skip in [
                        'uninstall', 'setup', 'install', 'config', 'update'
                    ]):
                        continue
                    
                    app = self._create_application(
                        name=item.name,
                        executable_path=item,
                        installation_path=directory
                    )
                    applications.append(app)
        except (PermissionError, OSError) as e:
            self.logger.debug(f"Could not scan directory {directory}: {e}")
        
        return applications