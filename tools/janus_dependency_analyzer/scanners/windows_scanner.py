"""
Windows-specific application scanner.

This module implements application discovery for Windows systems using
registry scanning, Program Files directories, and Windows Store apps.
"""

import json
import logging
import os
import subprocess
import winreg
from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import multiprocessing
import time
from dataclasses import dataclass

from .base import BasePlatformScanner
from ..core.models import Application, ApplicationMetadata, Platform


logger = logging.getLogger(__name__)


@dataclass
class ScanTask:
    """Represents a scanning task for the swarm."""
    task_type: str  # 'registry', 'directory', 'store', 'package_manager'
    target: Any  # Registry key tuple, directory path, etc.
    priority: int = 1  # Higher priority tasks run first
    max_depth: int = 1


@dataclass
class ScanResult:
    """Result from a swarm scanning task."""
    task_type: str
    applications: List[Application]
    errors: List[str]
    scan_time: float


class SwarmCoordinator:
    """Coordinates parallel scanning tasks across multiple workers."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(8, multiprocessing.cpu_count())
        self.results = []
        self.errors = []
        self.start_time = None
        
    def execute_swarm_scan(self, tasks: List[ScanTask]) -> List[Application]:
        """Execute scanning tasks in parallel using swarm tactics."""
        self.start_time = time.time()
        all_applications = []
        
        # Sort tasks by priority (higher priority first)
        tasks.sort(key=lambda x: x.priority, reverse=True)
        
        logger.info(f"Starting swarm scan with {len(tasks)} tasks using {self.max_workers} workers")
        
        # Use ThreadPoolExecutor for I/O bound tasks (registry, file system)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for task in tasks:
                if task.task_type == 'registry':
                    future = executor.submit(self._scan_registry_swarm, task.target)
                elif task.task_type == 'directory':
                    future = executor.submit(self._scan_directory_swarm, task.target, task.max_depth)
                elif task.task_type == 'store':
                    future = executor.submit(self._scan_store_swarm)
                elif task.task_type == 'package_manager':
                    future = executor.submit(self._scan_package_manager_swarm, task.target)
                else:
                    continue
                    
                future_to_task[future] = task
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout per task
                    if result and result.applications:
                        all_applications.extend(result.applications)
                        logger.debug(f"Task {task.task_type} completed: {len(result.applications)} apps found")
                    if result and result.errors:
                        self.errors.extend(result.errors)
                except Exception as e:
                    error_msg = f"Task {task.task_type} failed: {e}"
                    logger.warning(error_msg)
                    self.errors.append(error_msg)
        
        # Deduplicate applications
        unique_apps = self._deduplicate_applications(all_applications)
        
        total_time = time.time() - self.start_time
        logger.info(f"Swarm scan completed in {total_time:.2f}s: {len(unique_apps)} unique applications found")
        
        return unique_apps
    
    def _deduplicate_applications(self, applications: List[Application]) -> List[Application]:
        """Remove duplicate applications based on executable path and name."""
        seen = set()
        unique_apps = []
        
        for app in applications:
            # Create a unique key based on executable path and name
            key = (str(app.executable_path).lower(), app.name.lower())
            if key not in seen:
                seen.add(key)
                unique_apps.append(app)
        
        return unique_apps
    
    def _scan_registry_swarm(self, registry_key: Tuple[int, str]) -> ScanResult:
        """Swarm worker for registry scanning."""
        start_time = time.time()
        applications = []
        errors = []
        
        hkey, subkey_path = registry_key
        
        try:
            with winreg.OpenKey(hkey, subkey_path) as key:
                # Get all subkeys first
                subkeys = []
                i = 0
                while True:
                    try:
                        subkey_name = winreg.EnumKey(key, i)
                        subkeys.append(subkey_name)
                        i += 1
                    except WindowsError:
                        break
                
                # Process subkeys in parallel batches
                batch_size = 20
                for i in range(0, len(subkeys), batch_size):
                    batch = subkeys[i:i + batch_size]
                    
                    with ThreadPoolExecutor(max_workers=4) as batch_executor:
                        batch_futures = {
                            batch_executor.submit(
                                self._extract_app_from_registry_key_fast, 
                                hkey, 
                                f"{subkey_path}\\{subkey_name}"
                            ): subkey_name 
                            for subkey_name in batch
                        }
                        
                        for future in as_completed(batch_futures):
                            try:
                                app = future.result(timeout=5)
                                if app:
                                    applications.append(app)
                            except Exception as e:
                                errors.append(f"Registry key error: {e}")
                                
        except (WindowsError, PermissionError) as e:
            errors.append(f"Could not access registry key {subkey_path}: {e}")
        
        scan_time = time.time() - start_time
        return ScanResult('registry', applications, errors, scan_time)
    
    def _scan_directory_swarm(self, directory: Path, max_depth: int) -> ScanResult:
        """Swarm worker for directory scanning."""
        start_time = time.time()
        applications = []
        errors = []
        
        try:
            if not directory.exists():
                return ScanResult('directory', [], [f"Directory does not exist: {directory}"], 0)
            
            # Collect all executable files first
            exe_files = []
            try:
                if max_depth > 1:
                    exe_files = list(directory.rglob("*.exe"))
                else:
                    exe_files = list(directory.glob("*.exe"))
            except (PermissionError, OSError) as e:
                errors.append(f"Could not scan directory {directory}: {e}")
                return ScanResult('directory', [], errors, time.time() - start_time)
            
            # Filter files quickly
            filtered_files = []
            for exe_file in exe_files:
                if not exe_file.is_file():
                    continue
                
                # Skip obvious non-applications
                if any(skip in exe_file.name.lower() for skip in [
                    'uninstall', 'setup', 'installer', 'temp', 'cache',
                    'update', 'patch', 'helper', 'service', 'launcher',
                    'vcredist', 'dotnet', 'msvc', 'runtime'
                ]):
                    continue
                
                # Quick size check
                try:
                    if exe_file.stat().st_size < 10000:  # Less than 10KB
                        continue
                except (OSError, PermissionError):
                    continue
                
                filtered_files.append(exe_file)
            
            # Process files in parallel batches
            batch_size = 10
            for i in range(0, len(filtered_files), batch_size):
                batch = filtered_files[i:i + batch_size]
                
                with ThreadPoolExecutor(max_workers=4) as batch_executor:
                    batch_futures = {
                        batch_executor.submit(self._process_executable_fast, exe_file): exe_file 
                        for exe_file in batch
                    }
                    
                    for future in as_completed(batch_futures):
                        try:
                            app = future.result(timeout=3)
                            if app:
                                applications.append(app)
                        except Exception as e:
                            errors.append(f"File processing error: {e}")
                            
        except Exception as e:
            errors.append(f"Directory scan error: {e}")
        
        scan_time = time.time() - start_time
        return ScanResult('directory', applications, errors, scan_time)
    
    def _scan_store_swarm(self) -> ScanResult:
        """Swarm worker for Windows Store scanning."""
        start_time = time.time()
        applications = []
        errors = []
        
        try:
            # Use PowerShell to get Windows Store apps with optimized query
            result = subprocess.run([
                'powershell', '-Command',
                '$ErrorActionPreference="SilentlyContinue"; '
                'Get-AppxPackage | Where-Object {$_.IsFramework -eq $false -and $_.InstallLocation -and $_.Name -notlike "*Microsoft*"} | '
                'Select-Object Name, PackageFullName, InstallLocation | ConvertTo-Json -Depth 1 -Compress'
            ], capture_output=True, text=True, timeout=20, encoding='utf-8', errors='ignore')
            
            if result.returncode == 0 and result.stdout.strip():
                try:
                    apps_data = json.loads(result.stdout)
                    
                    if isinstance(apps_data, dict):
                        apps_data = [apps_data]
                    elif not isinstance(apps_data, list):
                        return ScanResult('store', [], ['Invalid store data format'], time.time() - start_time)
                    
                    # Process store apps in parallel
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        futures = {
                            executor.submit(self._process_store_app, app_data): app_data 
                            for app_data in apps_data if isinstance(app_data, dict)
                        }
                        
                        for future in as_completed(futures):
                            try:
                                app = future.result(timeout=5)
                                if app:
                                    applications.append(app)
                            except Exception as e:
                                errors.append(f"Store app processing error: {e}")
                                
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    errors.append(f"Could not parse Windows Store apps JSON: {e}")
                    
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, UnicodeDecodeError) as e:
            errors.append(f"Could not scan Windows Store apps: {e}")
        
        scan_time = time.time() - start_time
        return ScanResult('store', applications, errors, scan_time)
    
    def _scan_package_manager_swarm(self, manager_type: str) -> ScanResult:
        """Swarm worker for package manager scanning."""
        start_time = time.time()
        applications = []
        errors = []
        
        try:
            if manager_type == 'chocolatey':
                choco_dir = Path("C:/ProgramData/chocolatey/lib")
                if choco_dir.exists():
                    result = self._scan_directory_swarm(choco_dir, 3)
                    applications.extend(result.applications)
                    errors.extend(result.errors)
                    
            elif manager_type == 'scoop':
                scoop_dirs = [
                    Path.home() / "scoop/apps",
                    Path("C:/ProgramData/scoop/apps")
                ]
                
                for scoop_dir in scoop_dirs:
                    if scoop_dir.exists():
                        result = self._scan_directory_swarm(scoop_dir, 2)
                        # Add Scoop prefix to app names
                        for app in result.applications:
                            app.name = f"Scoop: {app.name}"
                        applications.extend(result.applications)
                        errors.extend(result.errors)
                        
            elif manager_type == 'winget':
                # Simplified winget scanning - just check if winget is available
                try:
                    result = subprocess.run([
                        'winget', '--version'
                    ], capture_output=True, text=True, timeout=5, encoding='utf-8', errors='ignore')
                    
                    if result.returncode == 0:
                        # Winget is available, but we'll skip the slow listing for now
                        # Could be enhanced later with a more efficient approach
                        pass
                except Exception:
                    pass
                    
        except Exception as e:
            errors.append(f"Package manager {manager_type} scan error: {e}")
        
        scan_time = time.time() - start_time
        return ScanResult('package_manager', applications, errors, scan_time)
    
    def _extract_app_from_registry_key_fast(self, hkey: int, key_path: str) -> Optional[Application]:
        """Fast registry key processing for swarm workers."""
        try:
            with winreg.OpenKey(hkey, key_path) as key:
                # Batch read all values at once
                values = {}
                for value_name in ["DisplayName", "DisplayVersion", "InstallLocation", "DisplayIcon", "Publisher"]:
                    try:
                        values[value_name] = winreg.QueryValueEx(key, value_name)[0]
                    except FileNotFoundError:
                        pass
                
                # Quick validation
                name = values.get("DisplayName")
                if not name:
                    return None
                
                # Skip system components
                name_lower = name.lower()
                if any(skip in name_lower for skip in [
                    'microsoft visual c++', 'microsoft .net', 'windows sdk',
                    'security update', 'hotfix', 'kb', 'redistributable',
                    'runtime', 'framework', 'service pack'
                ]):
                    return None
                
                # Find executable
                executable_path = None
                
                # Try DisplayIcon first
                if "DisplayIcon" in values:
                    icon_path = values["DisplayIcon"]
                    if ',' in icon_path:
                        icon_path = icon_path.split(',')[0]
                    icon_path = icon_path.strip('"').strip()
                    if icon_path:
                        potential_exe = Path(icon_path)
                        if potential_exe.suffix.lower() == '.exe' and potential_exe.exists():
                            executable_path = potential_exe
                
                # Try install location
                if not executable_path and "InstallLocation" in values:
                    install_loc = values["InstallLocation"]
                    if install_loc and install_loc.strip():
                        install_path = Path(install_loc.strip())
                        if install_path.exists():
                            exe_files = list(install_path.glob("*.exe"))
                            if exe_files:
                                executable_path = exe_files[0]  # Take first one for speed
                
                if not executable_path:
                    return None
                
                # Create minimal metadata
                metadata = ApplicationMetadata()
                if "Publisher" in values:
                    metadata.vendor = str(values["Publisher"]).strip()
                
                return Application(
                    id=f"registry_{hash(key_path)}",
                    name=name,
                    version=values.get("DisplayVersion", ""),
                    installation_path=executable_path.parent,
                    executable_path=executable_path,
                    platform=Platform.WINDOWS,
                    metadata=metadata,
                    discovered_at=datetime.now()
                )
                
        except Exception:
            return None
    
    def _process_executable_fast(self, exe_file: Path) -> Optional[Application]:
        """Fast executable processing for swarm workers."""
        try:
            if not exe_file.exists() or not exe_file.is_file():
                return None
            
            # Basic metadata only for speed
            metadata = ApplicationMetadata()
            try:
                metadata.file_size = exe_file.stat().st_size
            except (OSError, PermissionError):
                pass
            
            return Application(
                id=f"file_{hash(str(exe_file))}",
                name=exe_file.stem,
                version="",
                installation_path=exe_file.parent,
                executable_path=exe_file,
                platform=Platform.WINDOWS,
                metadata=metadata,
                discovered_at=datetime.now()
            )
            
        except Exception:
            return None
    
    def _process_store_app(self, app_data: Dict[str, Any]) -> Optional[Application]:
        """Process a single Windows Store app."""
        try:
            install_location = app_data.get('InstallLocation')
            name = app_data.get('Name')
            
            if not install_location or not name:
                return None
            
            install_path = Path(install_location)
            if not install_path.exists():
                return None
            
            # Find executable
            exe_files = list(install_path.glob("*.exe"))
            if not exe_files:
                return None
            
            # Extract version from PackageFullName
            version = ""
            package_full_name = app_data.get('PackageFullName', '')
            if '_' in package_full_name:
                parts = package_full_name.split('_')
                if len(parts) > 1:
                    version = parts[1]
            
            return Application(
                id=f"store_{hash(name)}",
                name=f"Store: {name}",
                version=version,
                installation_path=install_path,
                executable_path=exe_files[0],
                platform=Platform.WINDOWS,
                metadata=ApplicationMetadata(),
                discovered_at=datetime.now()
            )
            
        except Exception:
            return None


class WindowsScanner(BasePlatformScanner):
    """
    Windows-specific application scanner with swarm-based parallel processing.
    
    Implements application discovery for Windows systems by scanning:
    - Windows Registry (HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall)
    - Program Files directories
    - Windows Store applications
    - Package managers (Chocolatey, Scoop, Winget)
    
    Uses swarm tactics for high-performance parallel scanning.
    """
    
    def __init__(self):
        """Initialize the Windows scanner with swarm coordination."""
        super().__init__(Platform.WINDOWS)
        
        # Common Windows application directories
        self.program_files_dirs = [
            Path("C:/Program Files"),
            Path("C:/Program Files (x86)"),
            Path.home() / "AppData/Local/Programs"
        ]
        
        # Registry keys for installed applications
        self.registry_keys = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall")
        ]
        
        # Initialize swarm coordinator
        self.swarm = SwarmCoordinator()
    
    def scan_standard_locations(self) -> List[Application]:
        """
        Scan standard Windows application installation locations using swarm tactics.
        
        Returns:
            List[Application]: Applications found in standard locations
        """
        tasks = []
        
        # Add registry scanning tasks (high priority)
        for registry_key in self.registry_keys:
            tasks.append(ScanTask(
                task_type='registry',
                target=registry_key,
                priority=3  # High priority
            ))
        
        # Add Program Files directory scanning tasks (medium priority)
        for directory in self.program_files_dirs:
            if directory.exists():
                tasks.append(ScanTask(
                    task_type='directory',
                    target=directory,
                    priority=2,  # Medium priority
                    max_depth=2
                ))
        
        return self.swarm.execute_swarm_scan(tasks)
    
    def scan_package_managers(self) -> List[Application]:
        """
        Scan applications installed via Windows package managers using swarm tactics.
        
        Returns:
            List[Application]: Applications found via package managers
        """
        tasks = []
        
        # Add Windows Store scanning task (high priority)
        tasks.append(ScanTask(
            task_type='store',
            target=None,
            priority=3
        ))
        
        # Add package manager scanning tasks (lower priority)
        for manager in ['chocolatey', 'scoop', 'winget']:
            tasks.append(ScanTask(
                task_type='package_manager',
                target=manager,
                priority=1
            ))
        
        return self.swarm.execute_swarm_scan(tasks)
    
    def scan_portable_applications(self) -> List[Application]:
        """
        Scan for portable applications in common locations using swarm tactics.
        
        Returns:
            List[Application]: Portable applications found
        """
        tasks = []
        
        # Common portable application directories
        portable_dirs = [
            Path.home() / "Desktop",
            Path.home() / "Downloads", 
            Path("C:/PortableApps"),
            Path("D:/PortableApps"),
            Path.home() / "Documents/Applications"
        ]
        
        # Add directory scanning tasks for portable apps
        for directory in portable_dirs:
            if directory.exists():
                tasks.append(ScanTask(
                    task_type='directory',
                    target=directory,
                    priority=1,  # Lower priority
                    max_depth=2
                ))
        
        # Add PATH environment scanning
        path_apps = self._detect_environment_applications_fast()
        
        swarm_results = self.swarm.execute_swarm_scan(tasks)
        swarm_results.extend(path_apps)
        
        return swarm_results
    
    def _detect_environment_applications_fast(self) -> List[Application]:
        """
        Fast detection of applications available in system PATH.
        
        Returns:
            List[Application]: Applications found in PATH
        """
        applications = []
        
        try:
            path_env = os.environ.get('PATH', '')
            path_dirs = [Path(p) for p in path_env.split(os.pathsep) if p and Path(p).exists()]
            
            # Limit to first 10 PATH directories to avoid overwhelming scan
            path_dirs = path_dirs[:10]
            
            # Use swarm for PATH scanning too
            tasks = []
            for path_dir in path_dirs:
                if path_dir.is_dir():
                    tasks.append(ScanTask(
                        task_type='directory',
                        target=path_dir,
                        priority=1,
                        max_depth=1
                    ))
            
            if tasks:
                path_results = self.swarm.execute_swarm_scan(tasks)
                # Add PATH prefix to distinguish these apps
                for app in path_results:
                    app.name = f"PATH: {app.name}"
                applications.extend(path_results)
                        
        except Exception as e:
            self.logger.debug(f"Could not scan PATH applications: {e}")
        
        return applications
    
    # Legacy methods kept for compatibility - these are now handled by swarm
    def _scan_registry(self) -> List[Application]:
        """Legacy method - use scan_standard_locations() instead."""
        tasks = [ScanTask('registry', key, 3) for key in self.registry_keys]
        return self.swarm.execute_swarm_scan(tasks)
    
    def _scan_program_files(self) -> List[Application]:
        """Legacy method - use scan_standard_locations() instead."""
        tasks = []
        for directory in self.program_files_dirs:
            if directory.exists():
                tasks.append(ScanTask('directory', directory, 2, max_depth=2))
        return self.swarm.execute_swarm_scan(tasks)
    
    def _scan_windows_store(self) -> List[Application]:
        """Legacy method - use scan_package_managers() instead."""
        tasks = [ScanTask('store', None, 3)]
        return self.swarm.execute_swarm_scan(tasks)
    
