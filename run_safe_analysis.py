"""
Run analysis with timeout protection and problematic app skipping.
"""

import json
import logging
import signal
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

from janus_dependency_analyzer.scanners.system_scanner import SystemScannerImpl
from janus_dependency_analyzer.analyzers.capability_analyzer import CapabilityAnalyzerImpl
from janus_dependency_analyzer.filters.app_filter import ApplicationFilter
from janus_dependency_analyzer.filters.deduplicator import ApplicationDeduplicator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Apps that are known to hang or are low-value - skip them
PROBLEMATIC_APPS = {
    'AppxDebugSysTray',
    'Microsoft.WebTools.Languages.LanguageServer.Host',
    'vsgraphics',
    'SettingsMigrator',
    'vstest.console.arm64',
    'VSTestVideoRecorder',
    'inject_dll_x86',
    'inject_dll_amd64',
    'VSPerfMon',
    'VSPerfSrv',
    'adpcmencode3',
    'makecat',
    'rmlogotest',
    'microsoft.windows.softwarelogo.taskengine',
    'makeappx',
    'pythonw',  # GUI version, python is enough
    'pip3.9',   # Duplicates
    'pip3',
    't32',
    'w32',
}

# Timeout per application (seconds) - reduced for speed
APP_TIMEOUT = 5

def analyze_app_with_timeout(analyzer, app):
    """Analyze a single app with timeout protection."""
    try:
        capabilities = analyzer.analyze_application(app, enable_early_exit=True)
        return app, capabilities, None
    except Exception as e:
        return app, None, str(e)

def main():
    logger.info("Starting safe analysis with timeout protection...")
    
    # Scan applications
    scanner = SystemScannerImpl()
    logger.info("Scanning for applications...")
    scan_result = scanner.scan_full()
    all_apps = scan_result.applications
    logger.info(f"Found {len(all_apps)} total applications")
    
    # Apply filters - only HIGH priority apps for speed
    from janus_dependency_analyzer.filters.app_filter import FilterConfig, Priority
    filter_config = FilterConfig(
        enabled=True,
        analyze_priorities={Priority.HIGH}  # Only high priority
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
    
    # Analyze with timeout protection (parallel for speed)
    analyzer = CapabilityAnalyzerImpl(enable_cache=True)
    results = []
    failed = []
    timeout_count = 0
    
    # Use 2 workers for parallel analysis
    with ThreadPoolExecutor(max_workers=2) as executor:
        for i, app in enumerate(safe_apps, 1):
            logger.info(f"[{i}/{len(safe_apps)}] Analyzing {app.name}...")
            
            future = executor.submit(analyze_app_with_timeout, analyzer, app)
            try:
                app_result, capabilities, error = future.result(timeout=APP_TIMEOUT)
                
                if error:
                    logger.error(f"Failed to analyze {app.name}: {error}")
                    failed.append({'name': app.name, 'error': error})
                elif capabilities:
                    results.append({
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
                    })
            except TimeoutError:
                timeout_count += 1
                logger.warning(f"Timeout analyzing {app.name} (exceeded {APP_TIMEOUT}s)")
                failed.append({'name': app.name, 'error': f'Timeout after {APP_TIMEOUT}s'})
                # Cancel the future
                future.cancel()
    
    # Save results
    output_file = Path('capabilities_report_safe.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'results': results,
            'failed': failed,
            'stats': {
                'total_scanned': len(all_apps),
                'after_filtering': len(filtered_apps),
                'after_dedup': len(deduplicated_apps),
                'analyzed': len(safe_apps),
                'successful': len(results),
                'failed': len(failed),
                'timeouts': timeout_count,
            }
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Analysis complete! Results saved to {output_file}")
    logger.info(f"Successful: {len(results)}, Failed: {len(failed)}, Timeouts: {timeout_count}")
    
    # Summary
    total_capabilities = sum(len(r['capabilities']) for r in results)
    logger.info(f"Total capabilities identified: {total_capabilities}")

if __name__ == '__main__':
    main()
