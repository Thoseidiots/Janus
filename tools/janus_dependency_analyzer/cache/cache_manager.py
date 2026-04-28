"""
Cache manager for storing and retrieving capability analysis results.

Implements disk-based caching with automatic invalidation based on
application version and modification time.
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict

from ..core.models import Application, Capability, CapabilityCategory, InterfaceType, Parameter


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached analysis result."""
    app_id: str
    app_name: str
    app_version: str
    executable_path: str
    analyzed_at: datetime
    capabilities: List[Dict[str, Any]]  # Serialized capabilities
    cache_key: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'app_id': self.app_id,
            'app_name': self.app_name,
            'app_version': self.app_version,
            'executable_path': self.executable_path,
            'analyzed_at': self.analyzed_at.isoformat(),
            'capabilities': self.capabilities,
            'cache_key': self.cache_key,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        return cls(
            app_id=data['app_id'],
            app_name=data['app_name'],
            app_version=data['app_version'],
            executable_path=data['executable_path'],
            analyzed_at=datetime.fromisoformat(data['analyzed_at']),
            capabilities=data['capabilities'],
            cache_key=data['cache_key'],
        )


class CacheManager:
    """
    Manages caching of capability analysis results.
    
    Cache keys are based on:
    - Application ID
    - Application version
    - Executable path (to detect moves/reinstalls)
    - File modification time (when available)
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, max_age_days: int = 30):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache files (default: .janus_cache)
            max_age_days: Maximum age of cache entries in days (default: 30)
        """
        self.cache_dir = cache_dir or Path.home() / '.janus_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = timedelta(days=max_age_days)
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._load_cache()
        logger.info(f"Cache manager initialized: {self.cache_dir}")
    
    def _generate_cache_key(self, app: Application) -> str:
        """
        Generate a unique cache key for an application.
        
        Args:
            app: Application to generate key for
            
        Returns:
            str: Cache key (SHA256 hash)
        """
        # Include app ID, version, and executable path in key
        key_components = [
            app.id,
            app.version or 'unknown',
            str(app.executable_path),
        ]
        
        # Try to include file modification time for better invalidation
        try:
            if app.executable_path.exists():
                mtime = app.executable_path.stat().st_mtime
                key_components.append(str(mtime))
        except Exception:
            pass  # File might not be accessible
        
        key_string = '|'.join(key_components)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(self, app: Application) -> Optional[List[Capability]]:
        """
        Retrieve cached capabilities for an application.
        
        Args:
            app: Application to look up
            
        Returns:
            List[Capability] if cached and valid, None otherwise
        """
        cache_key = self._generate_cache_key(app)
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            entry = self._memory_cache[cache_key]
            
            # Check if cache entry is still valid
            age = datetime.now() - entry.analyzed_at
            if age > self.max_age:
                logger.debug(f"Cache expired for {app.name} (age: {age.days} days)")
                del self._memory_cache[cache_key]
                return None
            
            logger.debug(f"Cache hit for {app.name}")
            return self._deserialize_capabilities(entry.capabilities)
        
        logger.debug(f"Cache miss for {app.name}")
        return None
    
    def put(self, app: Application, capabilities: List[Capability]) -> None:
        """
        Store capabilities in cache for an application.
        
        Args:
            app: Application being cached
            capabilities: Capabilities to cache
        """
        cache_key = self._generate_cache_key(app)
        
        entry = CacheEntry(
            app_id=app.id,
            app_name=app.name,
            app_version=app.version or 'unknown',
            executable_path=str(app.executable_path),
            analyzed_at=datetime.now(),
            capabilities=self._serialize_capabilities(capabilities),
            cache_key=cache_key,
        )
        
        self._memory_cache[cache_key] = entry
        logger.debug(f"Cached {len(capabilities)} capabilities for {app.name}")
    
    def invalidate(self, app: Application) -> None:
        """
        Invalidate cache entry for an application.
        
        Args:
            app: Application to invalidate
        """
        cache_key = self._generate_cache_key(app)
        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]
            logger.debug(f"Invalidated cache for {app.name}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._memory_cache.clear()
        logger.info("Cache cleared")
    
    def save(self) -> None:
        """Persist cache to disk."""
        cache_file = self.cache_dir / 'capability_cache.json'
        
        try:
            cache_data = {
                'version': '1.0',
                'saved_at': datetime.now().isoformat(),
                'entries': [entry.to_dict() for entry in self._memory_cache.values()],
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Cache saved: {len(self._memory_cache)} entries")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        cache_file = self.cache_dir / 'capability_cache.json'
        
        if not cache_file.exists():
            logger.debug("No cache file found")
            return
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Load entries and filter out expired ones
            now = datetime.now()
            loaded_count = 0
            expired_count = 0
            
            for entry_dict in cache_data.get('entries', []):
                entry = CacheEntry.from_dict(entry_dict)
                age = now - entry.analyzed_at
                
                if age <= self.max_age:
                    self._memory_cache[entry.cache_key] = entry
                    loaded_count += 1
                else:
                    expired_count += 1
            
            logger.info(
                f"Cache loaded: {loaded_count} entries "
                f"({expired_count} expired entries discarded)"
            )
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache statistics
        """
        if not self._memory_cache:
            return {
                'total_entries': 0,
                'cache_size_mb': 0,
                'oldest_entry': None,
                'newest_entry': None,
            }
        
        entries = list(self._memory_cache.values())
        analyzed_times = [e.analyzed_at for e in entries]
        
        # Estimate cache size
        cache_file = self.cache_dir / 'capability_cache.json'
        cache_size_mb = cache_file.stat().st_size / (1024 * 1024) if cache_file.exists() else 0
        
        return {
            'total_entries': len(self._memory_cache),
            'cache_size_mb': round(cache_size_mb, 2),
            'oldest_entry': min(analyzed_times).isoformat(),
            'newest_entry': max(analyzed_times).isoformat(),
        }
    
    def _serialize_capabilities(self, capabilities: List[Capability]) -> List[Dict[str, Any]]:
        """Convert capabilities to JSON-serializable format."""
        return [
            {
                'id': cap.id,
                'application_id': cap.application_id,
                'name': cap.name,
                'category': cap.category.value,
                'description': cap.description,
                'interface_type': cap.interface_type.value,
                'parameters': [
                    {
                        'name': p.name,
                        'type': p.type,
                        'description': p.description,
                        'required': p.required,
                        'default_value': p.default_value,
                    }
                    for p in cap.parameters
                ],
                'confidence_score': cap.confidence_score,
                'detection_method': cap.detection_method,
                'examples': cap.examples,
                'documentation_url': cap.documentation_url,
                'supported_formats': cap.supported_formats,
            }
            for cap in capabilities
        ]
    
    def _deserialize_capabilities(self, data: List[Dict[str, Any]]) -> List[Capability]:
        """Convert JSON data back to Capability objects."""
        capabilities = []
        
        for cap_dict in data:
            try:
                parameters = [
                    Parameter(
                        name=p['name'],
                        type=p['type'],
                        description=p.get('description', ''),
                        required=p.get('required', False),
                        default_value=p.get('default_value'),
                    )
                    for p in cap_dict.get('parameters', [])
                ]
                
                capability = Capability(
                    id=cap_dict['id'],
                    application_id=cap_dict['application_id'],
                    name=cap_dict['name'],
                    category=CapabilityCategory(cap_dict['category']),
                    description=cap_dict['description'],
                    interface_type=InterfaceType(cap_dict['interface_type']),
                    parameters=parameters,
                    confidence_score=cap_dict['confidence_score'],
                    detection_method=cap_dict.get('detection_method', ''),
                    examples=cap_dict.get('examples', []),
                    documentation_url=cap_dict.get('documentation_url'),
                    supported_formats=cap_dict.get('supported_formats', []),
                )
                capabilities.append(capability)
            except Exception as e:
                logger.error(f"Failed to deserialize capability: {e}")
                continue
        
        return capabilities
