"""
Quick example showing how to use the optimized Janus Dependency Analyzer.

This demonstrates the performance improvements from parallelization and caching.
"""

from janus_dependency_analyzer.scanners.system_scanner import SystemScannerImpl
from janus_dependency_analyzer.analyzers.capability_analyzer import CapabilityAnalyzerImpl
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import time

def analyze_system_fast():
    """
    Perform a fast system-wide capability analysis using optimizations.
    """
    print("🚀 Janus Dependency Analyzer - Optimized Analysis")
    print("=" * 70)
    
    # Step 1: Scan for applications
    print("\n📡 Step 1: Scanning system for applications...")
    scanner = SystemScannerImpl()
    scan_result = scanner.scan_full()
    
    accessible_apps = [a for a in scan_result.applications if a.is_accessible]
    print(f"   Found {len(accessible_apps)} accessible applications")
    
    # Step 2: Analyze capabilities (with parallelization + caching)
    print("\n🔍 Step 2: Analyzing capabilities (parallel + cached)...")
    analyzer = CapabilityAnalyzerImpl(enable_cache=True)
    
    start_time = time.time()
    all_capabilities = []
    
    # Use parallel processing
    max_workers = min(multiprocessing.cpu_count() * 2, len(accessible_apps))
    print(f"   Using {max_workers} parallel workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_app = {
            executor.submit(analyzer.analyze_application, app): app
            for app in accessible_apps
        }
        
        completed = 0
        for future in as_completed(future_to_app):
            app = future_to_app[future]
            try:
                caps = future.result(timeout=60)
                all_capabilities.extend(caps)
                completed += 1
                
                # Progress indicator every 100 apps
                if completed % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    remaining = (len(accessible_apps) - completed) / rate
                    print(f"   Progress: {completed}/{len(accessible_apps)} "
                          f"({rate:.1f} apps/sec, ~{remaining:.0f}s remaining)")
            except Exception as e:
                print(f"   ⚠️  Error analyzing {app.name}: {e}")
    
    analysis_time = time.time() - start_time
    
    # Step 3: Save cache and show results
    print("\n💾 Step 3: Saving cache...")
    analyzer.save_cache()
    
    cache_stats = analyzer.get_cache_stats()
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"📊 Applications analyzed:  {len(accessible_apps)}")
    print(f"🎯 Capabilities found:     {len(all_capabilities)}")
    print(f"⏱️  Analysis time:          {analysis_time:.1f}s ({analysis_time/60:.1f} min)")
    print(f"⚡ Analysis rate:          {len(accessible_apps)/analysis_time:.1f} apps/sec")
    print(f"💾 Cache entries:          {cache_stats.get('total_entries', 0)}")
    print(f"📦 Cache size:             {cache_stats.get('cache_size_mb', 0):.2f}MB")
    print("=" * 70)
    
    # Show top capabilities by category
    print("\n📈 Top Capabilities by Category:")
    from collections import Counter
    category_counts = Counter(cap.category.value for cap in all_capabilities)
    for category, count in category_counts.most_common(10):
        print(f"   {category:30s}: {count:4d}")
    
    print("\n💡 Tip: Run this again to see caching in action (should be <10 seconds)!")
    
    return all_capabilities

if __name__ == "__main__":
    capabilities = analyze_system_fast()
