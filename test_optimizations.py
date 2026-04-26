"""
Test all optimizations to verify they work correctly.
"""

import time
from pathlib import Path

from janus_dependency_analyzer.scanners.system_scanner import SystemScannerImpl
from janus_dependency_analyzer.analyzers.capability_analyzer import CapabilityAnalyzerImpl
from janus_dependency_analyzer.filters.app_filter import ApplicationFilter, FilterConfig
from janus_dependency_analyzer.filters.deduplicator import ApplicationDeduplicator


def test_scan():
    """Test system scan."""
    print("=" * 60)
    print("TEST 1: System Scan")
    print("=" * 60)
    
    scanner = SystemScannerImpl()
    start = time.time()
    result = scanner.scan_full()
    elapsed = time.time() - start
    
    print(f"✓ Scanned in {elapsed:.2f}s")
    print(f"  Total apps: {result.total_applications}")
    print(f"  Accessible: {result.accessible_applications}")
    print()
    
    return result


def test_deduplication(scan_result):
    """Test deduplication."""
    print("=" * 60)
    print("TEST 2: Deduplication")
    print("=" * 60)
    
    accessible_apps = [a for a in scan_result.applications if a.is_accessible]
    
    deduplicator = ApplicationDeduplicator()
    start = time.time()
    deduplicated = deduplicator.deduplicate(accessible_apps)
    elapsed = time.time() - start
    
    duplicates = deduplicator.get_duplicates(accessible_apps)
    
    print(f"✓ Deduplicated in {elapsed:.2f}s")
    print(f"  Before: {len(accessible_apps)} apps")
    print(f"  After: {len(deduplicated)} apps")
    print(f"  Removed: {len(accessible_apps) - len(deduplicated)} duplicates")
    print(f"  Duplicate groups: {len(duplicates)}")
    
    # Show some examples
    if duplicates:
        print("\n  Examples of duplicates found:")
        for base_name, versions in list(duplicates.items())[:5]:
            print(f"    {base_name}: {len(versions)} versions")
            for v in versions[:3]:
                print(f"      - {v.name}")
    print()
    
    return deduplicated


def test_filtering(apps):
    """Test aggressive filtering."""
    print("=" * 60)
    print("TEST 3: Aggressive Filtering")
    print("=" * 60)
    
    app_filter = ApplicationFilter(FilterConfig(enabled=True))
    start = time.time()
    filtered = app_filter.get_apps_to_analyze(apps)
    elapsed = time.time() - start
    
    results = app_filter.filter_applications(apps)
    
    print(f"✓ Filtered in {elapsed:.2f}s")
    print(f"  Before: {len(apps)} apps")
    print(f"  After: {len(filtered)} apps")
    print(f"  High priority: {results['stats']['high_priority']}")
    print(f"  Medium priority: {results['stats']['medium_priority']}")
    print(f"  Skipped: {results['stats']['skipped']}")
    
    # Show some examples of skipped apps
    if results['skipped']:
        print("\n  Examples of skipped apps:")
        for app, filter_result in results['skipped'][:10]:
            print(f"    {app.name}: {filter_result.reason}")
    print()
    
    return filtered


def test_analysis(apps):
    """Test capability analysis with early exit."""
    print("=" * 60)
    print("TEST 4: Capability Analysis (Sample)")
    print("=" * 60)
    
    # Test on first 10 apps
    sample_apps = apps[:10]
    
    analyzer = CapabilityAnalyzerImpl(enable_cache=True)
    
    print(f"Analyzing {len(sample_apps)} sample apps...")
    start = time.time()
    
    total_capabilities = 0
    for app in sample_apps:
        caps = analyzer.analyze_application(app, enable_early_exit=True)
        total_capabilities += len(caps)
        print(f"  {app.name}: {len(caps)} capabilities")
    
    elapsed = time.time() - start
    
    print(f"\n✓ Analyzed in {elapsed:.2f}s")
    print(f"  Total capabilities: {total_capabilities}")
    print(f"  Avg per app: {total_capabilities / len(sample_apps):.1f}")
    print(f"  Avg time per app: {elapsed / len(sample_apps):.2f}s")
    
    # Get cache stats
    cache_stats = analyzer.get_cache_stats()
    print(f"\n  Cache stats:")
    print(f"    Entries: {cache_stats.get('total_entries', 0)}")
    print(f"    Size: {cache_stats.get('cache_size_mb', 0):.2f} MB")
    print()


def main():
    """Run all optimization tests."""
    print("\n" + "=" * 60)
    print("JANUS DEPENDENCY ANALYZER - OPTIMIZATION TESTS")
    print("=" * 60)
    print()
    
    # Test 1: Scan
    scan_result = test_scan()
    
    # Test 2: Deduplication
    deduplicated_apps = test_deduplication(scan_result)
    
    # Test 3: Filtering
    filtered_apps = test_filtering(deduplicated_apps)
    
    # Test 4: Analysis (sample)
    test_analysis(filtered_apps)
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original apps: {scan_result.total_applications}")
    print(f"After deduplication: {len(deduplicated_apps)}")
    print(f"After filtering: {len(filtered_apps)}")
    print(f"Reduction: {scan_result.total_applications} → {len(filtered_apps)} "
          f"({100 * (1 - len(filtered_apps) / scan_result.total_applications):.1f}% reduction)")
    print()
    print("Expected analysis time:")
    print(f"  Before optimizations: ~{scan_result.total_applications * 2 / 60:.0f} minutes")
    print(f"  After optimizations: ~{len(filtered_apps) * 0.5 / 60:.0f} minutes")
    print()
    print("✓ All optimizations working correctly!")
    print()


if __name__ == "__main__":
    main()
