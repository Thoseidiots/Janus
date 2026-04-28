#!/usr/bin/env python3
"""
Test script to verify Windows scanner functionality.
"""

import sys
import logging
from pathlib import Path

# Add the janus_dependency_analyzer to the path
sys.path.insert(0, str(Path(__file__).parent))

from janus_dependency_analyzer.scanners.windows_scanner import WindowsScanner
from janus_dependency_analyzer.core.models import Platform

def test_windows_scanner():
    """Test the Windows scanner implementation."""
    print("Testing Windows Scanner Implementation...")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create scanner instance
    scanner = WindowsScanner()
    
    print(f"Scanner platform: {scanner.get_platform()}")
    
    # Test registry scanning
    print("\n=== Testing Registry Scanning ===")
    try:
        registry_apps = scanner._scan_registry()
        print(f"Found {len(registry_apps)} applications in registry")
        
        # Show first few applications
        for i, app in enumerate(registry_apps[:5]):
            print(f"  {i+1}. {app.name} v{app.version}")
            print(f"     Path: {app.executable_path}")
            print(f"     Accessible: {app.is_accessible}")
            if app.metadata.vendor:
                print(f"     Vendor: {app.metadata.vendor}")
            print()
    except Exception as e:
        print(f"Registry scanning failed: {e}")
    
    # Test Program Files scanning
    print("\n=== Testing Program Files Scanning ===")
    try:
        program_files_apps = scanner._scan_program_files()
        print(f"Found {len(program_files_apps)} applications in Program Files")
        
        # Show first few applications
        for i, app in enumerate(program_files_apps[:3]):
            print(f"  {i+1}. {app.name}")
            print(f"     Path: {app.executable_path}")
            print(f"     Accessible: {app.is_accessible}")
            print()
    except Exception as e:
        print(f"Program Files scanning failed: {e}")
    
    # Test Windows Store scanning
    print("\n=== Testing Windows Store Scanning ===")
    try:
        store_apps = scanner._scan_windows_store()
        print(f"Found {len(store_apps)} Windows Store applications")
        
        # Show first few applications
        for i, app in enumerate(store_apps[:3]):
            print(f"  {i+1}. {app.name}")
            print(f"     Path: {app.executable_path}")
            print(f"     Accessible: {app.is_accessible}")
            print()
    except Exception as e:
        print(f"Windows Store scanning failed: {e}")
    
    # Test full discovery
    print("\n=== Testing Full Application Discovery ===")
    try:
        all_apps = scanner.discover_applications()
        print(f"Total applications discovered: {len(all_apps)}")
        
        # Show summary by category
        accessible_count = sum(1 for app in all_apps if app.is_accessible)
        inaccessible_count = len(all_apps) - accessible_count
        
        print(f"  Accessible: {accessible_count}")
        print(f"  Inaccessible: {inaccessible_count}")
        
        # Show some examples
        print("\nSample applications:")
        for i, app in enumerate(all_apps[:10]):
            status = "✓" if app.is_accessible else "✗"
            print(f"  {status} {app.name} v{app.version}")
            if not app.is_accessible and app.access_error:
                print(f"    Error: {app.access_error}")
        
        if len(all_apps) > 10:
            print(f"  ... and {len(all_apps) - 10} more applications")
            
    except Exception as e:
        print(f"Full discovery failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_windows_scanner()