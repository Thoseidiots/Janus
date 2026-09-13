#!/usr/bin/env python3
"""
Build script for Universal Oxpecker
===================================
Packages the tool as a standalone Windows executable and prepares Gumroad assets.

Usage:
  python build_oxpecker_distribution.py

Creates:
  - dist/oxpecker.exe (PyInstaller single-file executable)
  - dist/oxpecker-installer.exe (NSIS installer with PATH setup)
  - dist/oxpecker-portable.zip (portable archive)
  - dist/oxpecker-GUMROAD-ready.zip (everything for Gumroad upload)
"""

import os
import sys
import subprocess
import shutil
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"
TOOLS_DIR = PROJECT_ROOT / "tools" / "universal_oxpecker"

def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    colors = {
        "INFO": "\033[36m",
        "SUCCESS": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "RESET": "\033[0m",
    }
    color = colors.get(level, colors["INFO"])
    print(f"[{timestamp}] {color}[{level}]{colors['RESET']} {msg}")

def run(cmd, cwd=None, check=True):
    """Run a command and return success."""
    log(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result.returncode == 0
    except Exception as e:
        log(f"Failed to run command: {e}", "ERROR")
        return False

def ensure_deps():
    """Ensure build dependencies are installed."""
    log("Checking dependencies...")
    
    deps = ["pyinstaller", "nuitka"]  # Optional: nuitka for compiled output
    
    for dep in deps:
        try:
            __import__(dep.replace("-", "_"))
        except ImportError:
            log(f"Installing {dep}...", "WARNING")
            run([sys.executable, "-m", "pip", "install", dep])

def build_executable():
    """Build standalone executable using PyInstaller."""
    log("Building standalone executable...")
    
    # Create spec file for PyInstaller
    spec_content = f'''
# -*- mode: python ; coding: utf-8 -*-
block_cipher = None

a = Analysis(
    [r'{TOOLS_DIR / "cli.py"}'],
    pathex=[],
    binaries=[],
    datas=[
        (r'{TOOLS_DIR / "adapters"}', 'universal_oxpecker/adapters'),
        (r'{TOOLS_DIR / "analysis"}', 'universal_oxpecker/analysis'),
        (r'{TOOLS_DIR / "core"}', 'universal_oxpecker/core'),
    ],
    hiddenimports=['universal_oxpecker.core', 'universal_oxpecker.adapters'],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludedimports=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='oxpecker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='oxpecker',
)
'''
    
    spec_file = BUILD_DIR / "oxpecker.spec"
    spec_file.parent.mkdir(parents=True, exist_ok=True)
    spec_file.write_text(spec_content)
    
    # Run PyInstaller
    if run([sys.executable, "-m", "PyInstaller", str(spec_file), f"--distpath={DIST_DIR}"]):
        log("Executable built successfully", "SUCCESS")
        return True
    else:
        log("PyInstaller failed", "ERROR")
        return False

def create_portable_zip():
    """Create portable ZIP archive."""
    log("Creating portable ZIP archive...")
    
    exe_dir = DIST_DIR / "oxpecker"
    if not exe_dir.exists():
        log(f"Executable directory not found: {exe_dir}", "ERROR")
        return False
    
    portable_zip = DIST_DIR / "oxpecker-portable.zip"
    shutil.make_archive(str(portable_zip.with_suffix('')), 'zip', exe_dir)
    
    log(f"Created {portable_zip.name}", "SUCCESS")
    return True

def create_installer():
    """Create Windows installer (requires NSIS)."""
    log("Creating Windows installer...")
    
    nsi_file = PROJECT_ROOT / "installer" / "oxpecker.nsi"
    if not nsi_file.exists():
        log(f"NSIS script not found: {nsi_file}", "WARNING")
        return False
    
    # Check if NSIS is installed
    nsis_path = Path("C:\\Program Files (x86)\\NSIS\\makensis.exe")
    if not nsis_path.exists():
        log("NSIS not found. Installer will be skipped. Install from: https://nsis.sourceforge.io", "WARNING")
        return False
    
    # Run NSIS
    if run([str(nsis_path), str(nsi_file)]):
        log("Installer created successfully", "SUCCESS")
        return True
    else:
        log("NSIS build failed", "ERROR")
        return False

def create_gumroad_package():
    """Create the final package for Gumroad."""
    log("Creating Gumroad distribution package...")
    
    gumroad_dir = DIST_DIR / "oxpecker-GUMROAD"
    gumroad_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy executables
    if (DIST_DIR / "oxpecker").exists():
        shutil.copytree(DIST_DIR / "oxpecker", gumroad_dir / "oxpecker-windows", dirs_exist_ok=True)
    
    if (DIST_DIR / "oxpecker.exe").exists():
        shutil.copy(DIST_DIR / "oxpecker.exe", gumroad_dir / "oxpecker.exe")
    
    if (DIST_DIR / "oxpecker-portable.zip").exists():
        shutil.copy(DIST_DIR / "oxpecker-portable.zip", gumroad_dir / "oxpecker-portable.zip")
    
    # Copy marketing materials
    files_to_copy = [
        "GUMROAD_LISTING.md",
        "QUICK_START.md",
    ]
    
    for fname in files_to_copy:
        src = DIST_DIR / fname
        if src.exists():
            shutil.copy(src, gumroad_dir / fname)
    
    # Create manifest
    manifest = {
        "name": "Universal Oxpecker",
        "version": "1.2.0",
        "date_built": datetime.now().isoformat(),
        "files": {
            "oxpecker.exe": "Windows standalone executable",
            "oxpecker-portable.zip": "Portable ZIP archive",
            "oxpecker-windows": "Full directory with dependencies",
            "GUMROAD_LISTING.md": "Gumroad product page copy",
            "QUICK_START.md": "User quick start guide",
        }
    }
    
    manifest_file = gumroad_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))
    
    # Create final ZIP
    gumroad_zip = DIST_DIR / "oxpecker-GUMROAD-ready.zip"
    shutil.make_archive(str(gumroad_zip.with_suffix('')), 'zip', gumroad_dir)
    
    log(f"Created {gumroad_zip.name} — Ready to upload to Gumroad!", "SUCCESS")
    return True

def create_readme():
    """Create a distribution README."""
    readme_content = """# Universal Oxpecker Distribution

## Files in this package:

- **oxpecker.exe** — Standalone Windows executable
- **oxpecker-portable.zip** — Portable version (no installation required)
- **oxpecker-windows/** — Full executable directory
- **GUMROAD_LISTING.md** — Product listing for Gumroad
- **QUICK_START.md** — User quick start guide

## To upload to Gumroad:

1. Go to https://gumroad.com
2. Create a new product
3. Add title: "Universal Oxpecker v1.2.0"
4. Add the description from GUMROAD_LISTING.md
5. Upload oxpecker.exe and oxpecker-portable.zip
6. Set price: $0.99 minimum, allow donations
7. Add "I cannot afford this" option
8. Set license key format (optional)
9. Publish!

## Installation for users:

1. Download oxpecker.exe
2. Run the installer
3. Open Command Prompt: oxpecker --help
4. (Optional) Activate license: oxpecker license activate

## Support:

For questions or issues, refer to QUICK_START.md or https://oxpecker.dev
"""
    
    readme_file = DIST_DIR / "README_DISTRIBUTION.md"
    readme_file.write_text(readme_content)
    log(f"Created {readme_file.name}", "SUCCESS")

def main():
    """Main build process."""
    log("=== Universal Oxpecker Distribution Builder ===")
    log(f"Project root: {PROJECT_ROOT}")
    
    # Ensure directories exist
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Check dependencies
    ensure_deps()
    
    # Step 2: Build executable
    if not build_executable():
        log("Build failed at executable stage", "ERROR")
        sys.exit(1)
    
    # Step 3: Create portable ZIP
    if not create_portable_zip():
        log("Portable ZIP creation failed", "WARNING")
    
    # Step 4: Try to create installer (may skip if NSIS not installed)
    create_installer()
    
    # Step 5: Create Gumroad package
    if not create_gumroad_package():
        log("Gumroad package creation failed", "ERROR")
        sys.exit(1)
    
    # Step 6: Create distribution README
    create_readme()
    
    log("\n=== BUILD COMPLETE ===", "SUCCESS")
    log(f"Distribution ready in: {DIST_DIR}")
    log(f"Ready to upload: {DIST_DIR / 'oxpecker-GUMROAD-ready.zip'}")

if __name__ == "__main__":
    main()
