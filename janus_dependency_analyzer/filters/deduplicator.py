"""
Application deduplication to keep only the latest version of each app.

Reduces analysis time by 1.5-2x by removing duplicate versions.
"""

import logging
import re
from typing import List, Dict
from packaging import version as pkg_version

from ..core.models import Application


logger = logging.getLogger(__name__)


class ApplicationDeduplicator:
    """
    Deduplicates applications by keeping only the latest version.
    
    Handles cases like:
    - Python 3.9.13, Python 3.11.6, Python 3.12.10 → Keep Python 3.12.10
    - Git 2.40.0, Git 2.41.0 → Keep Git 2.41.0
    - Multiple Edge instances → Keep one
    """
    
    def __init__(self):
        """Initialize the deduplicator."""
        self.logger = logging.getLogger(__name__)
    
    def extract_base_name(self, app_name: str) -> str:
        """
        Extract the base name without version numbers.
        
        Examples:
            "Python 3.12.10 (64-bit)" → "Python"
            "Git 2.40.0" → "Git"
            "Visual Studio Code 1.85.0" → "Visual Studio Code"
        
        Args:
            app_name: Full application name
            
        Returns:
            Base name without version
        """
        # Remove version patterns
        patterns = [
            r'\s+\d+\.\d+\.\d+.*',  # x.y.z version
            r'\s+\d+\.\d+.*',        # x.y version
            r'\s+v\d+.*',            # v1, v2, etc.
            r'\s+\(\d+-bit\)',       # (64-bit), (32-bit)
            r'\s+\(x\d+\)',          # (x64), (x86)
            r'\s+\(User\)',          # (User)
            r'\s+\(System\)',        # (System)
        ]
        
        base_name = app_name
        for pattern in patterns:
            base_name = re.sub(pattern, '', base_name, flags=re.IGNORECASE)
        
        return base_name.strip()
    
    def extract_version(self, app: Application) -> str:
        """
        Extract version string from application.
        
        Tries multiple sources:
        1. app.version field
        2. Version in app.name
        3. Defaults to "0.0.0"
        
        Args:
            app: Application to extract version from
            
        Returns:
            Version string
        """
        # Try app.version field first
        if app.version:
            return app.version
        
        # Try to extract from name
        version_patterns = [
            r'(\d+\.\d+\.\d+)',  # x.y.z
            r'(\d+\.\d+)',        # x.y
            r'v(\d+)',            # v1, v2
        ]
        
        for pattern in version_patterns:
            match = re.search(pattern, app.name)
            if match:
                return match.group(1)
        
        # Default
        return "0.0.0"
    
    def compare_versions(self, ver1: str, ver2: str) -> int:
        """
        Compare two version strings.
        
        Args:
            ver1: First version
            ver2: Second version
            
        Returns:
            -1 if ver1 < ver2, 0 if equal, 1 if ver1 > ver2
        """
        try:
            v1 = pkg_version.parse(ver1)
            v2 = pkg_version.parse(ver2)
            
            if v1 < v2:
                return -1
            elif v1 > v2:
                return 1
            else:
                return 0
        except Exception:
            # Fallback to string comparison
            if ver1 < ver2:
                return -1
            elif ver1 > ver2:
                return 1
            else:
                return 0
    
    def deduplicate(self, apps: List[Application]) -> List[Application]:
        """
        Deduplicate applications by keeping only the latest version.
        
        Args:
            apps: List of applications
            
        Returns:
            Deduplicated list with only latest versions
        """
        # Group by base name
        app_groups: Dict[str, List[Application]] = {}
        
        for app in apps:
            base_name = self.extract_base_name(app.name)
            if base_name not in app_groups:
                app_groups[base_name] = []
            app_groups[base_name].append(app)
        
        # Keep latest version of each
        result = []
        duplicates_removed = 0
        
        for base_name, versions in app_groups.items():
            if len(versions) == 1:
                # No duplicates
                result.append(versions[0])
            else:
                # Multiple versions - keep latest
                latest = versions[0]
                latest_version = self.extract_version(latest)
                
                for app in versions[1:]:
                    app_version = self.extract_version(app)
                    if self.compare_versions(app_version, latest_version) > 0:
                        latest = app
                        latest_version = app_version
                
                result.append(latest)
                duplicates_removed += len(versions) - 1
                
                self.logger.debug(
                    f"Deduplicated {base_name}: kept {latest.name} "
                    f"(removed {len(versions) - 1} older versions)"
                )
        
        self.logger.info(
            f"Deduplication complete: {len(apps)} apps → {len(result)} apps "
            f"({duplicates_removed} duplicates removed)"
        )
        
        return result
    
    def get_duplicates(self, apps: List[Application]) -> Dict[str, List[Application]]:
        """
        Get all duplicate applications grouped by base name.
        
        Useful for debugging and reporting.
        
        Args:
            apps: List of applications
            
        Returns:
            Dict mapping base name to list of duplicate apps
        """
        app_groups: Dict[str, List[Application]] = {}
        
        for app in apps:
            base_name = self.extract_base_name(app.name)
            if base_name not in app_groups:
                app_groups[base_name] = []
            app_groups[base_name].append(app)
        
        # Return only groups with duplicates
        return {
            name: apps_list
            for name, apps_list in app_groups.items()
            if len(apps_list) > 1
        }
