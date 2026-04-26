"""
Test caching optimization for capability analysis.
"""
import time
from janus_dependency_analyzer.scanners.system_scanner import SystemScannerImpl
from janus_dependency_analyzer.analyzers.capability_analyzer import CapabilityAnalyzerImpl

def main():
    print("Testing capability analysis caching...")
    print("=" * 60)
    
    # Scan system
    print("\n1. Scanning system for applications...")
    scanner = SystemScannerImpl()
    scan_result = scanner.scan_full()
    
    accessible_apps = [a for a in scan_result.applications if a.is_accessible]
    test_apps = accessible_apps[:20]  # Test with 20 apps
    print(f"   Testing with {len(test_apps)} applications")
    
    # First run (no cache)
    print("\n2. First run (building cache)...")
    analyzer = CapabilityAnalyzerImpl(enable_cache=True)
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import multiprocessing
    
    start = time.time()
    first_run_caps = []
    max_workers = min(multiprocessing.cpu_count() * 2, len(test_apps))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_app = {
            executor.submit(analyzer.analyze_application, app): app
            for app in test_apps
        }
        
        for future in as_completed(future_to_app):
            try:
                caps = future.result(timeout=60)
                first_run_caps.extend(caps)
            except Exception as e:
                pass
    
    first_run_time = time.time() - start
    print(f"   Time: {first_run_time:.2f}s")
    print(f"   Capabilities found: {len(first_run_caps)}")
    
    # Save cache
    analyzer.save_cache()
    cache_stats = analyzer.get_cache_stats()
    print(f"   Cache entries: {cache_stats['total_entries']}")
    print(f"   Cache size: {cache_stats['cache_size_mb']}MB")
    
    # Second run (with cache)
    print("\n3. Second run (using cache)...")
    analyzer2 = CapabilityAnalyzerImpl(enable_cache=True)
    
    start = time.time()
    second_run_caps = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_app = {
            executor.submit(analyzer2.analyze_application, app): app
            for app in test_apps
        }
        
        for future in as_completed(future_to_app):
            try:
                caps = future.result(timeout=60)
                second_run_caps.extend(caps)
            except Exception as e:
                pass
    
    second_run_time = time.time() - start
    print(f"   Time: {second_run_time:.2f}s")
    print(f"   Capabilities found: {len(second_run_caps)}")
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"  First run (no cache):  {first_run_time:.2f}s")
    print(f"  Second run (cached):   {second_run_time:.2f}s")
    if first_run_time > 0:
        speedup = first_run_time / second_run_time
        print(f"  Speedup:               {speedup:.2f}x faster")
        time_saved = first_run_time - second_run_time
        print(f"  Time saved:            {time_saved:.2f}s")
    print(f"  Cache entries:         {cache_stats['total_entries']}")
    print(f"  Cache size:            {cache_stats['cache_size_mb']}MB")
    print("=" * 60)

if __name__ == "__main__":
    main()
