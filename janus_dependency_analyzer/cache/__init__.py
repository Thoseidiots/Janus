"""
Caching module for Janus Dependency Analyzer.

Provides result caching to avoid re-analyzing unchanged applications.
"""

from .cache_manager import CacheManager, CacheEntry

__all__ = ['CacheManager', 'CacheEntry']
