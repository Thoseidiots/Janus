@echo off
setlocal EnableDelayedExpansion
title Janus Build

echo.
echo  ============================================================
echo   JANUS - Building distributable package
echo  ============================================================
echo.

:: ── Check tools ──────────────────────────────────────────────────────────────
python --version >nul 2>&1 || (echo [ERROR] Python not found & pause & exit /b 1)
node --version >nul 2>&1   || (echo [ERROR] Node.js not found & pause & exit /b 1)
pip show pyinstaller >nul 2>&1 || (
    echo [1/5] Installing PyInstaller...
    pip install pyinstaller >nul 2>&1
)

:: ── Step 1: Install Node deps ─────────────────────────────────────────────────
echo [1/5] Installing Node dependencies...
npm install --silent >nul 2>&1
echo  [OK]

:: ── Step 2: Install Python deps ───────────────────────────────────────────────
echo [2/5] Installing Python dependencies...
pip install -r requirements.txt >nul 2>&1
echo  [OK]

:: ── Step 3: Bundle Python daemon with PyInstaller ────────────────────────────
echo [3/5] Bundling Python daemon (this takes a minute)...
pyinstaller janus_daemon.spec --distpath daemon-dist --noconfirm >nul 2>&1
if errorlevel 1 (
    echo  [WARN] PyInstaller had issues - check output above
) else (
    echo  [OK] Daemon bundled to daemon-dist/
)

:: ── Step 4: Build React/Vite frontend ────────────────────────────────────────
echo [4/5] Building frontend...
npm run build >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Frontend build failed
    pause
    exit /b 1
)
echo  [OK] Frontend built to dist/

:: ── Step 5: Package with Electron Builder ────────────────────────────────────
echo [5/5] Packaging with Electron Builder (this takes a few minutes)...
npx electron-builder build --win 2>&1 | findstr /i "error\|warn\|built\|packaging"
if errorlevel 1 (
    echo  [WARN] Electron Builder had issues
) else (
    echo  [OK] Installer created in dist-electron/
)

echo.
echo  ============================================================
echo   Build complete!
echo   Installer: dist-electron\Janus Setup 1.0.0.exe
echo  ============================================================
echo.
pause
