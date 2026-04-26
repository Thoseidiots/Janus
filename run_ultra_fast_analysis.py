"""
Ultra-fast analysis with aggressive optimizations.
"""

import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

from janus_dependency_analyzer.scanners.system_scanner import SystemScannerImpl
from janus_dependency_analyzer.analyzers.capability_analyzer import CapabilityAnalyzerImpl
from janus_dependency_analyzer.filters.app_filter import ApplicationFilter, FilterConfig, Priority
from janus_dependency_analyzer.filters.deduplicator import ApplicationDeduplicator

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce log noise
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Apps that are known to hang or are low-value - skip them
PROBLEMATIC_APPS = {
    'AppxDebugSysTray', 'Microsoft.WebTools.Languages.LanguageServer.Host',
    'vsgraphics', 'SettingsMigrator', 'vstest.console.arm64', 'VSTestVideoRecorder',
    'inject_dll_x86', 'inject_dll_amd64', 'VSPerfMon', 'VSPerfSrv',
    'adpcmencode3', 'makecat', 'rmlogotest', 'microsoft.windows.softwarelogo.taskengine',
    'makeappx', 'pythonw', 'pip3.9', 'pip3', 't32', 'w32',
    'VSHiveStub', 'CheckHyperVHost', 'BackgroundDownload', 'VSPerfASPNetCmd',
    'VSPerfCmd', 'InstallCleanup', 'vs_layout', 'dump64', 'dump64a',
}

# Timeout per application (seconds)
APP_TIMEOUT = 5

def analyze_app_with_timeout(app, analyzer):
    """Analyze a single app with timeout protection."""
    try:
        capabilities = analyzer.analyze_application(app, enable_early_exit=True)
        if capabilities:
            return {
                'name': app.name,
                'executable_path': str(app.executable_path),
                'installation_path': str(app.installation_path) if app.installation_path else None,
                'version': app.version,
                'capabilities': [
                    {
                        'name': cap.name,
                        'category': cap.category.value,
                        'description': cap.description,
                        'confidence_score': cap.confidence_score,
                        'detection_method': cap.detection_method,
                    }
                    for cap in capabilities
                ]
            }
        return None
    except Exception as e:
        logger.error(f"Error analyzing {app.name}: {e}")
        return None

def main():
    logger.info("Starting ULTRA-FAST analysis...")
    
    # Scan applications
    scanner = SystemScannerImpl()
    logger.info("Scanning for applications...")
    scan_result = scanner.scan_full()
    all_apps = scan_result.applications
    logger.info(f"Found {len(all_apps)} total applications")
    
    # Apply filters - only HIGH priority apps
    filter_config = FilterConfig(
        enabled=True,
        analyze_priorities={Priority.HIGH}
    )
    app_filter = ApplicationFilter(config=filter_config)
    filtered_apps = app_filter.get_apps_to_analyze(all_apps)
    logger.info(f"After filtering (HIGH priority only): {len(filtered_apps)} applications")
    
    # Deduplicate
    deduplicator = ApplicationDeduplicator()
    deduplicated_apps = deduplicator.deduplicate(filtered_apps)
    logger.info(f"After deduplication: {len(deduplicated_apps)} applications")
    
    # Skip problematic apps
    safe_apps = [app for app in deduplicated_apps if app.name not in PROBLEMATIC_APPS]
    skipped_count = len(deduplicated_apps) - len(safe_apps)
    logger.info(f"Skipping {skipped_count} problematic apps, analyzing {len(safe_apps)} apps")
    
    # Analyze with parallel processing
    analyzer = CapabilityAnalyzerImpl(enable_cache=True)
    results = []
    completed = 0
    
    # Process in batches with 4 parallel workers
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        future_to_app = {
            executor.submit(analyze_app_with_timeout, app, analyzer): app
            for app in safe_apps
        }
        
        # Process results as they complete
        for future in as_completed(future_to_app, timeout=APP_TIMEOUT * len(safe_apps)):
            app = future_to_app[future]
            completed += 1
            
            try:
                result = future.result(timeout=APP_TIMEOUT)
                if result:
                    results.append(result)
                    logger.info(f"[{completed}/{len(safe_apps)}] {app.name}: {len(result['capabilities'])} capabilities")
                else:
                    logger.info(f"[{completed}/{len(safe_apps)}] {app.name}: 0 capabilities")
            except TimeoutError:
                logger.warning(f"[{completed}/{len(safe_apps)}] {app.name}: TIMEOUT")
            except Exception as e:
                logger.error(f"[{completed}/{len(safe_apps)}] {app.name}: ERROR - {e}")
    
    # Save results
    output_file = Path('capabilities_report_ultra_fast.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'results': results,
            'stats': {
                'total_scanned': len(all_apps),
                'after_filtering': len(filtered_apps),
                'after_dedup': len(deduplicated_apps),
                'analyzed': len(safe_apps),
                'successful': len(results),
            }
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Analysis complete! Results saved to {output_file}")
    logger.info(f"Successful: {len(results)}/{len(safe_apps)} apps")
    
    # Summary
    total_capabilities = sum(len(r['capabilities']) for r in results)
    logger.info(f"Total capabilities identified: {total_capabilities}")
    logger.info(f"{'='*60}")

if __name__ == '__main__':
    main()
