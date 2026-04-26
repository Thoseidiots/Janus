"""
State manager for tracking scan history and application state.

Enables intelligent incremental analysis by maintaining:
- Last scan timestamp
- Application inventory with versions
- Change history
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..core.models import Application


logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages persistent state for incremental analysis.
    
    Tracks:
    - Last full scan timestamp
    - Application inventory (id -> {name, version, path, last_seen})
    - Scan statistics
    """
    
    def __init__(self, state_dir: Optional[Path] = None):
        """
        Initialize the state manager.
        
        Args:
            state_dir: Directory to store state files (default: ~/.janus_cache)
        """
        self.state_dir = state_dir or Path.home() / '.janus_cache'
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / 'analyzer_state.json'
        
        self.last_scan_time: Optional[datetime] = None
        self.app_inventory: Dict[str, Dict[str, Any]] = {}
        self.scan_count: int = 0
        
        self._load_state()
        logger.info(f"State manager initialized: {self.state_dir}")
    
    def get_last_scan_time(self) -> Optional[datetime]:
        """
        Get the timestamp of the last full scan.
        
        Returns:
            datetime if a previous scan exists, None otherwise
        """
        return self.last_scan_time
    
    def should_use_incremental(self, max_age_hours: int = 24) -> bool:
        """
        Determine if incremental scan should be used.
        
        Args:
            max_age_hours: Maximum age of last scan to use incremental (default: 24)
            
        Returns:
            bool: True if incremental scan is recommended
        """
        if self.last_scan_time is None:
            return False
        
        age = datetime.now() - self.last_scan_time
        age_hours = age.total_seconds() / 3600
        
        return age_hours <= max_age_hours
    
    def update_from_scan(self, applications: List[Application], scan_type: str = "full") -> None:
        """
        Update state after a scan completes.
        
        Args:
            applications: List of applications from the scan
            scan_type: Type of scan ("full" or "incremental")
        """
        now = datetime.now()
        
        # Update last scan time for full scans
        if scan_type == "full":
            self.last_scan_time = now
            self.scan_count += 1
        
        # Update application inventory
        for app in applications:
            self.app_inventory[app.id] = {
                'name': app.name,
                'version': app.version or 'unknown',
                'executable_path': str(app.executable_path),
                'installation_path': str(app.installation_path),
                'last_seen': now.isoformat(),
                'is_accessible': app.is_accessible,
            }
        
        logger.info(f"State updated: {len(applications)} applications, scan_type={scan_type}")
    
    def get_known_applications(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the inventory of known applications.
        
        Returns:
            Dict mapping app_id to application metadata
        """
        return self.app_inventory.copy()
    
    def is_application_known(self, app_id: str) -> bool:
        """
        Check if an application is in the inventory.
        
        Args:
            app_id: Application ID to check
            
        Returns:
            bool: True if application is known
        """
        return app_id in self.app_inventory
    
    def get_application_version(self, app_id: str) -> Optional[str]:
        """
        Get the last known version of an application.
        
        Args:
            app_id: Application ID
            
        Returns:
            str: Version string, or None if not known
        """
        app_data = self.app_inventory.get(app_id)
        return app_data['version'] if app_data else None
    
    def has_version_changed(self, app: Application) -> bool:
        """
        Check if an application's version has changed.
        
        Args:
            app: Application to check
            
        Returns:
            bool: True if version changed or app is new
        """
        if not self.is_application_known(app.id):
            return True  # New application
        
        known_version = self.get_application_version(app.id)
        current_version = app.version or 'unknown'
        
        return known_version != current_version
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get state statistics.
        
        Returns:
            Dict with state statistics
        """
        return {
            'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'known_applications': len(self.app_inventory),
            'scan_count': self.scan_count,
            'state_file': str(self.state_file),
        }
    
    def save(self) -> None:
        """Save state to disk."""
        try:
            state_data = {
                'version': '1.0',
                'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
                'scan_count': self.scan_count,
                'app_inventory': self.app_inventory,
                'saved_at': datetime.now().isoformat(),
            }
            
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2)
            
            logger.info(f"State saved: {len(self.app_inventory)} applications")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _load_state(self) -> None:
        """Load state from disk."""
        if not self.state_file.exists():
            logger.debug("No state file found, starting fresh")
            return
        
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            # Load last scan time
            if state_data.get('last_scan_time'):
                self.last_scan_time = datetime.fromisoformat(state_data['last_scan_time'])
            
            # Load scan count
            self.scan_count = state_data.get('scan_count', 0)
            
            # Load application inventory
            self.app_inventory = state_data.get('app_inventory', {})
            
            logger.info(
                f"State loaded: {len(self.app_inventory)} known applications, "
                f"last scan: {self.last_scan_time.isoformat() if self.last_scan_time else 'never'}"
            )
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    
    def clear(self) -> None:
        """Clear all state."""
        self.last_scan_time = None
        self.app_inventory.clear()
        self.scan_count = 0
        logger.info("State cleared")
    
    def reset(self) -> None:
        """Reset state and delete state file."""
        self.clear()
        if self.state_file.exists():
            self.state_file.unlink()
        logger.info("State reset")
