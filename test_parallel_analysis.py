"""
Quick test to verify parallel analysis works correctly.
"""
import time
from janus_dependency_analyzer.scanners.system_scanner import SystemScannerImpl
from janus_dependency_analyzer.analyzers.capability_analyzer import CapabilityAnalyzerImpl

def main():
    print("Testing parallel capability analysis...")
    print("=" * 60)
    
    # Scan system
    print("\n1. Scanning system for applications...")
    scanner = SystemScannerImpl()
    scan_result = scanner.scan_full()
    
    accessible_apps = [a for a in scan_result.applications if a.is_accessible]
    print(f"   Found {len(accessible_apps)} accessible applications")
    
    # Test with a small subset first (10 apps)
    test_apps = accessible_apps[:10]
    print(f"\n2. Testing with {len(test_apps)} applications...")
    
    analyzer = CapabilityAnalyzerImpl()
    
    # Sequential analysis (old way)
    print("\n   Sequential analysis:")
    start = time.time()
    sequential_caps = []
    for app in test_apps:
        try:
            caps = analyzer.analyze_application(app)
            sequential_caps.extend(caps)
        except Exception as e:
            print(f"   Error analyzing {app.name}: {e}")
    sequential_time = time.time() - start
    print(f"   Time: {sequential_time:.2f}s")
    print(f"   Capabilities found: {len(sequential_caps)}")
    
    # Parallel analysis (new way)
    print("\n   Parallel analysis:")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import multiprocessing
    
    start = time.time()
    parallel_caps = []
    max_workers = min(multiprocessing.cpu_count() * 2, len(test_apps))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_app = {
            executor.submit(analyzer.analyze_application, app): app
            for app in test_apps
        }
        
        for future in as_completed(future_to_app):
            app = future_to_app[future]
            try:
                caps = future.result(timeout=60)
                parallel_caps.extend(caps)
            except Exception as e:
                print(f"   Error analyzing {app.name}: {e}")
    
    parallel_time = time.time() - start
    print(f"   Time: {parallel_time:.2f}s")
    print(f"   Capabilities found: {len(parallel_caps)}")
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"  Sequential: {sequential_time:.2f}s")
    print(f"  Parallel:   {parallel_time:.2f}s")
    if sequential_time > 0:
        speedup = sequential_time / parallel_time
        print(f"  Speedup:    {speedup:.2f}x faster")
    print(f"  Workers:    {max_workers}")
    print("=" * 60)

if __name__ == "__main__":
    main()
