"""
Smart analyzer that automatically chooses between full and incremental analysis.

This module provides an intelligent wrapper that:
- Tracks scan history
- Automatically uses incremental scans when appropriate
- Falls back to full scans when needed
- Manages caching and state
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from .scanners.system_scanner import SystemScannerImpl
from .analyzers.capability_analyzer import CapabilityAnalyzerImpl
from .state.state_manager import StateManager
from .catalog.catalog import ApplicationCatalog
from .core.models import Application, Capability


logger = logging.getLogger(__name__)


class SmartAnalyzer:
    """
    Intelligent analyzer that automatically optimizes scan strategy.
    
    Features:
    - Automatic incremental vs full scan selection
    - Parallel capability analysis
    - Result caching
    - State management
    """
    
    def __init__(
        self,
        enable_cache: bool = True,
        max_incremental_age_hours: int = 24,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize the smart analyzer.
        
        Args:
            enable_cache: Enable result caching (default: True)
            max_incremental_age_hours: Max age for incremental scans (default: 24)
            max_workers: Number of parallel workers (default: CPU count * 2)
        """
        self.scanner = SystemScannerImpl()
        self.analyzer = CapabilityAnalyzerImpl(enable_cache=enable_cache)
        self.state = StateManager()
        self.catalog = ApplicationCatalog()
        
        self.max_incremental_age_hours = max_incremental_age_hours
        self.max_workers = max_workers or (multiprocessing.cpu_count() * 2)
        
        logger.info(f"Smart analyzer initialized (cache={enable_cache}, workers={self.max_workers})")
    
    def analyze(
        self,
        force_full: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Perform intelligent analysis of system applications.
        
        Automatically chooses between full and incremental scan based on:
        - Time since last scan
        - Force full flag
        - State availability
        
        Args:
            force_full: Force a full scan even if incremental is available
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            Dict with analysis results:
                - scan_type: "full" or "incremental"
                - applications_scanned: int
                - applications_analyzed: int
                - capabilities_found: int
                - scan_time_seconds: float
                - analysis_time_seconds: float
                - cache_hits: int
                - new_applications: List[str]
                - updated_applications: List[str]
        """
        import time
        
        start_time = time.time()
        
        # Determine scan strategy
        use_incremental = (
            not force_full
            and self.state.should_use_incremental(self.max_incremental_age_hours)
        )
        
        scan_type = "incremental" if use_incremental else "full"
        logger.info(f"Starting {scan_type} analysis...")
        
        # Perform scan
        scan_start = time.time()
        if use_incremental:
            last_scan = self.state.get_last_scan_time()
            scan_result = self.scanner.scan_incremental(last_scan, self.catalog)
            logger.info(f"Incremental scan found {scan_result.total_applications} changed applications")
        else:
            scan_result = self.scanner.scan_full()
            logger.info(f"Full scan found {scan_result.total_applications} applications")
        
        scan_time = time.time() - scan_start
        
        # Update state
        self.state.update_from_scan(scan_result.applications, scan_type)
        
        # Analyze capabilities (parallel + cached)
        analysis_start = time.time()
        accessible_apps = [a for a in scan_result.applications if a.is_accessible]
        
        all_capabilities = []
        cache_hits = 0
        
        if accessible_apps:
            logger.info(f"Analyzing {len(accessible_apps)} applications with {self.max_workers} workers...")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_app = {
                    executor.submit(self._analyze_with_tracking, app): app
                    for app in accessible_apps
                }
                
                completed = 0
                for future in as_completed(future_to_app):
                    app = future_to_app[future]
                    try:
                        caps, was_cached = future.result(timeout=60)
                        all_capabilities.extend(caps)
                        if was_cached:
                            cache_hits += 1
                        
                        completed += 1
                        if progress_callback and completed % 10 == 0:
                            progress_callback(completed, len(accessible_apps), f"Analyzed {app.name}")
                    except Exception as e:
                        logger.error(f"Failed to analyze {app.name}: {e}")
        
        analysis_time = time.time() - analysis_start
        
        # Save state and cache
        self.state.save()
        self.analyzer.save_cache()
        
        # Identify new and updated applications
        new_apps = []
        updated_apps = []
        
        for app in scan_result.applications:
            if not self.state.is_application_known(app.id):
                new_apps.append(app.name)
            elif self.state.has_version_changed(app):
                updated_apps.append(app.name)
        
        total_time = time.time() - start_time
        
        result = {
            'scan_type': scan_type,
            'applications_scanned': scan_result.total_applications,
            'applications_analyzed': len(accessible_apps),
            'capabilities_found': len(all_capabilities),
            'scan_time_seconds': round(scan_time, 2),
            'analysis_time_seconds': round(analysis_time, 2),
            'total_time_seconds': round(total_time, 2),
            'cache_hits': cache_hits,
            'cache_hit_rate': round(cache_hits / len(accessible_apps) * 100, 1) if accessible_apps else 0,
            'new_applications': new_apps,
            'updated_applications': updated_apps,
            'capabilities': all_capabilities,
        }
        
        logger.info(
            f"Analysis complete: {scan_type} scan, "
            f"{result['applications_analyzed']} apps analyzed, "
            f"{result['capabilities_found']} capabilities found, "
            f"{result['cache_hit_rate']}% cache hits, "
            f"{total_time:.1f}s total"
        )
        
        return result
    
    def _analyze_with_tracking(self, app: Application) -> tuple[List[Capability], bool]:
        """
        Analyze an application and track if result was cached.
        
        Returns:
            Tuple of (capabilities, was_cached)
        """
        # Check if result will come from cache
        was_cached = False
        if self.analyzer._cache:
            cached = self.analyzer._cache.get(app)
            was_cached = cached is not None
        
        capabilities = self.analyzer.analyze_application(app)
        return capabilities, was_cached
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics.
        
        Returns:
            Dict with analyzer statistics
        """
        state_stats = self.state.get_stats()
        cache_stats = self.analyzer.get_cache_stats()
        
        return {
            'state': state_stats,
            'cache': cache_stats,
            'max_workers': self.max_workers,
        }
    
    def force_full_scan(self) -> Dict[str, Any]:
        """
        Force a full scan and analysis.
        
        Returns:
            Analysis results
        """
        return self.analyze(force_full=True)
    
    def reset(self) -> None:
        """Reset all state and cache."""
        self.state.reset()
        self.analyzer.clear_cache()
        logger.info("Smart analyzer reset")
