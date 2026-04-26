"""
Apply aggressive filtering to reduce analysis time from 30-50 minutes to 5-10 minutes.

This script updates the app_filter.py with more aggressive skip patterns.
"""

import re
from pathlib import Path


def update_filter():
    """Update the filter with more aggressive skip patterns."""
    
    filter_path = Path("janus_dependency_analyzer/filters/app_filter.py")
    content = filter_path.read_text(encoding='utf-8')
    
    # Find the skip_patterns section
    old_skip_patterns = '''        if self.skip_patterns is None:
            self.skip_patterns = [
                # Games
                r".*game.*", r".*steam.*", r".*epic.*", r".*origin.*",
                r".*battle\\.net.*", r".*riot.*", r".*ubisoft.*",
                
                # Media players
                r".*spotify.*", r".*itunes.*", r".*vlc.*", r".*media player.*",
                
                # Browsers (unless analyzing browser capabilities)
                # r"chrome", r"firefox", r"edge", r"safari", r"opera",
                
                # System utilities (usually not dev tools)
                r"uninstall", r"setup", r"installer", r"updater",
                r".*helper.*", r".*service.*", r".*daemon.*",
                
                # Antivirus/Security (can cause issues)
                r".*antivirus.*", r".*defender.*", r".*security.*",
            ]'''
    
    new_skip_patterns = '''        if self.skip_patterns is None:
            self.skip_patterns = [
                # Games
                r".*game.*", r".*steam.*", r".*epic.*", r".*origin.*",
                r".*battle\\.net.*", r".*riot.*", r".*ubisoft.*", r".*gog.*",
                
                # Media players
                r".*spotify.*", r".*itunes.*", r".*vlc.*", r".*media player.*",
                r".*winamp.*", r".*foobar.*",
                
                # Browsers (skip - not dev tools)
                r".*chrome.*", r".*firefox.*", r".*edge.*", r".*safari.*", 
                r".*opera.*", r".*brave.*", r".*webview.*",
                
                # Microsoft Office
                r".*office.*", r".*word.*", r".*excel.*", r".*powerpoint.*",
                r".*outlook.*", r".*onenote.*", r".*access.*", r".*publisher.*",
                
                # Communication apps
                r".*teams.*", r".*skype.*", r".*zoom.*", r".*slack.*",
                r".*discord.*", r".*telegram.*", r".*whatsapp.*",
                
                # Adobe products (unless needed)
                r".*adobe.*", r".*acrobat.*", r".*reader.*", r".*photoshop.*",
                r".*illustrator.*", r".*premiere.*",
                
                # Graphics/Design (unless needed)
                r".*blender.*", r".*gimp.*", r".*inkscape.*",
                
                # System utilities (usually not dev tools)
                r"uninstall", r"setup", r"installer", r"updater", r"launcher",
                r".*helper.*", r".*service.*", r".*daemon.*", r".*agent.*",
                r"feedback", r"diagnostic", r"telemetry",
                
                # Windows system apps
                r"^msedge$", r"^msedge_proxy$", r"^msedgewebview2$",
                r"^old_msedge$", r"^identity_helper$",
                
                # Generic utilities
                r"^cli$", r"^gui$", r"^cli-32$", r"^cli-64$", 
                r"^gui-32$", r"^gui-64$", r"^t64$", r"^w64$",
                r"^t64-arm$", r"^w64-arm$", r"^wininst.*",
                
                # Antivirus/Security (can cause issues)
                r".*antivirus.*", r".*defender.*", r".*security.*",
                r".*denuvo.*", r".*anti-cheat.*",
                
                # Duplicates - keep only base version
                r".*\\s+\\d+\\.\\d+\\.\\d+.*",  # Skip versioned names
            ]'''
    
    # Replace the old patterns with new ones
    if old_skip_patterns in content:
        content = content.replace(old_skip_patterns, new_skip_patterns)
        filter_path.write_text(content, encoding='utf-8')
        print("✓ Updated skip patterns in app_filter.py")
        print("\nNew skip patterns added:")
        print("  - Browsers (Chrome, Edge, Firefox, etc.)")
        print("  - Microsoft Office apps")
        print("  - Communication apps (Teams, Zoom, Slack, etc.)")
        print("  - Adobe products")
        print("  - Generic utilities (cli, gui, wininst, etc.)")
        print("  - Versioned duplicates")
        print("\nExpected impact: 3,211 apps → ~800-1,200 apps")
        print("Expected time: 30-50 minutes → 10-15 minutes")
    else:
        print("✗ Could not find skip_patterns section to update")
        print("  The file may have been modified. Please update manually.")


if __name__ == "__main__":
    print("Janus Dependency Analyzer - Filter Optimization")
    print("=" * 60)
    print()
    
    update_filter()
    
    print()
    print("Next steps:")
    print("1. Run the analysis again:")
    print("   python -m janus_dependency_analyzer.cli report --type capabilities --format json --output capabilities_report.json")
    print()
    print("2. Compare the results:")
    print("   - Before: 3,211 apps analyzed")
    print("   - After: ~800-1,200 apps analyzed")
    print("   - Time saved: 20-35 minutes")
