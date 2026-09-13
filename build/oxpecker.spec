
# -*- mode: python ; coding: utf-8 -*-
block_cipher = None

a = Analysis(
    [r'C:\Users\legac\.copilot\repos\copilot-worktrees\Janus\thoseidiots-fantastic-waffle\tools\universal_oxpecker\cli.py'],
    pathex=[],
    binaries=[],
    datas=[
        (r'C:\Users\legac\.copilot\repos\copilot-worktrees\Janus\thoseidiots-fantastic-waffle\tools\universal_oxpecker\adapters', 'universal_oxpecker/adapters'),
        (r'C:\Users\legac\.copilot\repos\copilot-worktrees\Janus\thoseidiots-fantastic-waffle\tools\universal_oxpecker\analysis', 'universal_oxpecker/analysis'),
        (r'C:\Users\legac\.copilot\repos\copilot-worktrees\Janus\thoseidiots-fantastic-waffle\tools\universal_oxpecker\core', 'universal_oxpecker/core'),
    ],
    hiddenimports=['universal_oxpecker.core', 'universal_oxpecker.adapters'],
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
