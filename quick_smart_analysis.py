"""
Quick smart analysis using automatic incremental detection.

This script demonstrates the full optimization stack:
1. Automatic incremental vs full scan selection
2. Parallel capability analysis
3. Result caching
4. State management
"""

from janus_dependency_analyzer.smart_analyzer import SmartAnalyzer
import time


def progress_callback(current, total, message):
    """Simple progress indicator."""
    percent = (current / total) * 100
    print(f"   Progress: {current}/{total} ({percent:.0f}%) - {message}")


def main():
    print("🚀 Janus Smart Analyzer - Optimized Analysis")
    print("=" * 70)
    
    # Initialize smart analyzer
    analyzer = SmartAnalyzer(
        enable_cache=True,
        max_incremental_age_hours=24,
    )
    
    # Show current state
    stats = analyzer.get_stats()
    print("\n📊 Current State:")
    if stats['state']['last_scan_time']:
        print(f"   Last scan: {stats['state']['last_scan_time']}")
        print(f"   Known applications: {stats['state']['known_applications']}")
        print(f"   Total scans: {stats['state']['scan_count']}")
    else:
        print("   No previous scans found - will perform full scan")
    
    if stats['cache']['total_entries'] > 0:
        print(f"   Cache entries: {stats['cache']['total_entries']}")
        print(f"   Cache size: {stats['cache']['cache_size_mb']}MB")
    
    # Perform smart analysis
    print("\n🔍 Starting smart analysis...")
    print("   (Automatically choosing optimal scan strategy)")
    
    start_time = time.time()
    result = analyzer.analyze(progress_callback=progress_callback)
    total_time = time.time() - start_time
    
    # Display results
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    
    # Scan info
    scan_emoji = "🔄" if result['scan_type'] == "incremental" else "🔍"
    print(f"{scan_emoji} Scan type:              {result['scan_type'].upper()}")
    print(f"📱 Applications scanned:    {result['applications_scanned']}")
    print(f"🔬 Applications analyzed:   {result['applications_analyzed']}")
    print(f"🎯 Capabilities found:      {result['capabilities_found']}")
    
    # Performance
    print(f"\n⏱️  Performance:")
    print(f"   Scan time:              {result['scan_time_seconds']:.1f}s")
    print(f"   Analysis time:          {result['analysis_time_seconds']:.1f}s")
    print(f"   Total time:             {result['total_time_seconds']:.1f}s")
    print(f"   Cache hit rate:         {result['cache_hit_rate']:.1f}%")
    
    # Changes detected
    if result['new_applications']:
        print(f"\n🆕 New applications ({len(result['new_applications'])}):")
        for app_name in result['new_applications'][:10]:
            print(f"   • {app_name}")
        if len(result['new_applications']) > 10:
            print(f"   ... and {len(result['new_applications']) - 10} more")
    
    if result['updated_applications']:
        print(f"\n🔄 Updated applications ({len(result['updated_applications'])}):")
        for app_name in result['updated_applications'][:10]:
            print(f"   • {app_name}")
        if len(result['updated_applications']) > 10:
            print(f"   ... and {len(result['updated_applications']) - 10} more")
    
    # Recommendations
    print("\n💡 Tips:")
    if result['scan_type'] == 'full':
        print("   • Next run will use incremental scan (much faster!)")
        print("   • Run again to see the speed improvement")
    else:
        print("   • Incremental scan detected only changed applications")
        print("   • Use --force-full to rescan everything if needed")
    
    if result['cache_hit_rate'] < 50:
        print("   • Cache is building up - subsequent runs will be faster")
    
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    result = main()
