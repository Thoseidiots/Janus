"""
Cache Manager package for the Software NVMe Engine.

Provides ARC-based caching, sequential prefetching, and write coalescing.
"""

from nvme_engine.cache.arc_cache import ArcCache
from nvme_engine.cache.prefetcher import Prefetcher
from nvme_engine.cache.cache_manager import (
    CacheTier,
    CacheEntryState,
    CacheEntry,
    CacheStats,
    WriteCoalescer,
    CacheManager,
)

__all__ = [
    "ArcCache",
    "Prefetcher",
    "CacheTier",
    "CacheEntryState",
    "CacheEntry",
    "CacheStats",
    "WriteCoalescer",
    "CacheManager",
]
