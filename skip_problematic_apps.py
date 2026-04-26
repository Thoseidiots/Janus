"""
Quick script to run analysis with problematic apps skipped.
"""

import json
import logging
from pathlib import Path
from janus_dependency_analyzer.scanners.windows_scanner import WindowsScanner
from janus_dependency_analyzer.analyzers.capability_analyzer import CapabilityAnalyzerImpl
from janus_dependency_analyzer.filters.app_filter import ApplicationFilter
from janus_dependency_analyzer.filters.deduplicator import ApplicationDeduplicator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Apps that are known to hang - skip them
PROBLEMATIC_APPS = {
    'AppxDebugSysTray',
    'Microsoft.WebTools.Languages.LanguageServer.Host',
    'vsgraphics',
    'SettingsMigrator',
    'vstest.console.arm64',
    'VSTestVideoRecorder',
}

def main():
    logger.info("Starting optimized analysis with problematic apps skipped...")
    
    # Scan applications
    scanner = WindowsScanner()
    logger.info("Scanning for applications...")
    all_apps = scanner.scan()
    logger.info(f"Found {len(all_apps)} total applications")
    
    # Apply filters
    app_filter = ApplicationFilter()
    filtered_apps = app_filter.filter_applications(all_apps)
    logger.info(f"After filtering: {len(filtered_apps)} applications")
    
    # Deduplicate
    deduplicator = ApplicationDeduplicator()
    deduplicated_apps = deduplicator.deduplicate(filtered_apps)
    logger.info(f"After deduplication: {len(deduplicated_apps)} applications")
    
    # Skip problematic apps
    safe_apps = [app for app in deduplicated_apps if app.name not in PROBLEMATIC_APPS]
    skipped_count = len(deduplicated_apps) - len(safe_apps)
    logger.info(f"Skipping {skipped_count} problematic apps, analyzing {len(safe_apps)} apps")
    
    # Analyze
    analyzer = CapabilityAnalyzerImpl(enable_cache=True)
    results = []
    
    for i, app in enumerate(safe_apps, 1):
        logger.info(f"[{i}/{len(safe_apps)}] Analyzing {app.name}...")
        try:
            capabilities = analyzer.analyze_application(app, enable_early_exit=True)
            if capabilities:
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
        except Exception as e:
            logger.error(f"Failed to analyze {app.name}: {e}")
            continue
    
    # Save results
    output_file = Path('capabilities_report_safe.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Analysis complete! Results saved to {output_file}")
    logger.info(f"Found {len(results)} applications with capabilities")
    
    # Summary
    total_capabilities = sum(len(r['capabilities']) for r in results)
    logger.info(f"Total capabilities identified: {total_capabilities}")

if __name__ == '__main__':
    main()
