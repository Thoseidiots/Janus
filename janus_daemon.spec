# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for bundling janus_daemon.py into a standalone executable
# Build with: pyinstaller janus_daemon.spec

block_cipher = None

a = Analysis(
    ['janus_daemon.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        # Include all Python modules Janus needs
        ('janus_wallet.py', '.'),
        ('janus_worker_core.py', '.'),
        ('janus_autonomous_worker.py', '.'),
        ('janus_computer_use.py', '.'),
        ('janus_platform_browser.py', '.'),
        ('janus_notify.py', '.'),
        ('janus_selfheal.py', '.'),
    ],
    hiddenimports=[
        'fastapi',
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'fastapi.middleware.cors',
        'starlette',
        'starlette.middleware',
        'starlette.middleware.cors',
        'anyio',
        'anyio._backends._asyncio',
        'cryptography',
        'dotenv',
        'sqlite3',
        'winreg',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
        'torch',
    ],
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
    name='janus_daemon',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico',
)
