"""
Subprocess result caching to avoid redundant command executions.

Caches subprocess outputs (help text, version info) to speed up analysis.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Tuple


logger = logging.getLogger(__name__)


class SubprocessCache:
    """
    Caches subprocess execution results to avoid redundant calls.
    
    Cache key: (executable_path, mtime, command_args)
    This ensures cache invalidation when the executable changes.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, max_age_days: int = 7):
        """
        Initialize subprocess cache.
        
        Args:
            cache_dir: Directory for cache storage (default: ~/.janus_cache)
            max_age_days: Maximum age of cache entries in days (default: 7)
        """
        self.cache_dir = cache_dir or Path.home() / '.janus_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / 'subprocess_cache.json'
        self.max_age = timedelta(days=max_age_days)
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()
        logger.info(f"Subprocess cache initialized: {self.cache_file}")
    
    def _generate_cache_key(
        self,
        executable_path: Path,
        command_args: Tuple[str, ...],
        mtime: Optional[float] = None
    ) -> str:
        """
        Generate cache key for a subprocess call.
        
        Args:
            executable_path: Path to executable
            command_args: Command arguments tuple
            mtime: File modification time (auto-detected if None)
            
        Returns:
            Cache key (SHA256 hash)
        """
        # Get file modification time if not provided
        if mtime is None:
            try:
                mtime = executable_path.stat().st_mtime
            except Exception:
                mtime = 0  # Use 0 if file not accessible
        
        # Create key from path, mtime, and args
        key_components = [
            str(executable_path),
            str(mtime),
            '|'.join(command_args)
        ]
        
        key_string = '||'.join(key_components)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(
        self,
        executable_path: Path,
        command_args: Tuple[str, ...]
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached subprocess result.
        
        Args:
            executable_path: Path to executable
            command_args: Command arguments
            
        Returns:
            Dict with 'stdout', 'stderr', 'returncode', 'cached_at' if cached, None otherwise
        """
        cache_key = self._generate_cache_key(executable_path, command_args)
        
        if cache_key in self._memory_cache:
            entry = self._memory_cache[cache_key]
            
            # Check if cache entry is still valid
            cached_at = datetime.fromisoformat(entry['cached_at'])
            age = datetime.now() - cached_at
            
            if age > self.max_age:
                logger.debug(f"Subprocess cache expired for {executable_path.name} {command_args}")
                del self._memory_cache[cache_key]
                return None
            
            logger.debug(f"Subprocess cache hit for {executable_path.name} {command_args}")
            return entry
        
        logger.debug(f"Subprocess cache miss for {executable_path.name} {command_args}")
        return None
    
    def put(
        self,
        executable_path: Path,
        command_args: Tuple[str, ...],
        stdout: str,
        stderr: str,
        returncode: int
    ) -> None:
        """
        Store subprocess result in cache.
        
        Args:
            executable_path: Path to executable
            command_args: Command arguments
            stdout: Standard output
            stderr: Standard error
            returncode: Return code
        """
        cache_key = self._generate_cache_key(executable_path, command_args)
        
        entry = {
            'executable': str(executable_path),
            'command_args': list(command_args),
            'stdout': stdout,
            'stderr': stderr,
            'returncode': returncode,
            'cached_at': datetime.now().isoformat(),
        }
        
        self._memory_cache[cache_key] = entry
        logger.debug(f"Cached subprocess result for {executable_path.name} {command_args}")
    
    def save(self) -> None:
        """Persist cache to disk."""
        try:
            cache_data = {
                'version': '1.0',
                'saved_at': datetime.now().isoformat(),
                'entries': list(self._memory_cache.values()),
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Subprocess cache saved: {len(self._memory_cache)} entries")
        except Exception as e:
            logger.error(f"Failed to save subprocess cache: {e}")
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        if not self.cache_file.exists():
            logger.debug("No subprocess cache file found")
            return
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            now = datetime.now()
            loaded_count = 0
            expired_count = 0
            
            for entry in cache_data.get('entries', []):
                cached_at = datetime.fromisoformat(entry['cached_at'])
                age = now - cached_at
                
                if age <= self.max_age:
                    # Regenerate cache key
                    executable_path = Path(entry['executable'])
                    command_args = tuple(entry['command_args'])
                    cache_key = self._generate_cache_key(executable_path, command_args)
                    
                    self._memory_cache[cache_key] = entry
                    loaded_count += 1
                else:
                    expired_count += 1
            
            logger.info(
                f"Subprocess cache loaded: {loaded_count} entries "
                f"({expired_count} expired entries discarded)"
            )
        except Exception as e:
            logger.error(f"Failed to load subprocess cache: {e}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._memory_cache.clear()
        logger.info("Subprocess cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._memory_cache:
            return {
                'total_entries': 0,
                'cache_size_mb': 0,
                'oldest_entry': None,
                'newest_entry': None,
            }
        
        entries = list(self._memory_cache.values())
        cached_times = [datetime.fromisoformat(e['cached_at']) for e in entries]
        
        # Estimate cache size
        cache_size_mb = self.cache_file.stat().st_size / (1024 * 1024) if self.cache_file.exists() else 0
        
        return {
            'total_entries': len(self._memory_cache),
            'cache_size_mb': round(cache_size_mb, 2),
            'oldest_entry': min(cached_times).isoformat() if cached_times else None,
            'newest_entry': max(cached_times).isoformat() if cached_times else None,
        }


# Global subprocess cache instance
_subprocess_cache: Optional[SubprocessCache] = None


def get_subprocess_cache() -> SubprocessCache:
    """Get or create the global subprocess cache instance."""
    global _subprocess_cache
    if _subprocess_cache is None:
        _subprocess_cache = SubprocessCache()
    return _subprocess_cache
