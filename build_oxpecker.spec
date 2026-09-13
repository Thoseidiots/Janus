"""
PyInstaller build configuration for Universal Oxpecker
========================================================
Builds standalone .exe for Windows distribution.
Run: pyinstaller build_oxpecker.spec

Creates:
  - oxpecker.exe (single-file executable)
  - oxpecker-installer.exe (NSIS installer)
"""
# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
import os

block_cipher = None

# Collect all submodules and data
hiddenimports = [
    'universal_oxpecker',
    'universal_oxpecker.core',
    'universal_oxpecker.adapters',
    'universal_oxpecker.analysis',
    'pyperclip',
    'pathspec',
]

a = Analysis(
    ['tools/universal_oxpecker/cli.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('tools/universal_oxpecker/adapters', 'universal_oxpecker/adapters'),
        ('tools/universal_oxpecker/analysis', 'universal_oxpecker/analysis'),
        ('tools/universal_oxpecker/core', 'universal_oxpecker/core'),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
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
    icon='tools/universal_oxpecker/assets/oxpecker.ico',
)

# Collect all analysis plugins
coll = COLLECT(
    exe,
    Tree('tools/universal_oxpecker', prefix='oxpecker_data'),
    name='oxpecker',
)
