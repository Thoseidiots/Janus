@echo off
setlocal EnableDelayedExpansion
title Janus Startup

echo.
echo  ============================================================
echo   JANUS - Autonomous AI Worker
echo   Starting up...
echo  ============================================================
echo.

:: ── Check Python ─────────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Please install Python 3.10+ from https://python.org
    echo  Then re-run this script.
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  [OK] Python %PYVER% found

:: ── Check Node.js ────────────────────────────────────────────────────────────
node --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Node.js not found. Please install Node.js 18+ from https://nodejs.org
    echo  Then re-run this script.
    pause
    exit /b 1
)
for /f %%v in ('node --version 2^>^&1') do set NODEVER=%%v
echo  [OK] Node.js %NODEVER% found

:: ── Install Python dependencies ───────────────────────────────────────────────
echo.
echo  [1/4] Installing Python dependencies...
pip install -r requirements.txt >nul 2>&1
if errorlevel 1 (
    echo  [WARN] Some Python packages may have failed to install. Continuing...
) else (
    echo  [OK] Python dependencies ready
)

:: ── Install Node dependencies ─────────────────────────────────────────────────
echo  [2/4] Installing Node dependencies...
if not exist "node_modules" (
    npm install --silent >nul 2>&1
    echo  [OK] Node modules installed
) else (
    echo  [OK] Node modules already present
)

:: ── Start Python daemon (background) ─────────────────────────────────────────
echo  [3/4] Starting Janus daemon (port 8006)...
start "Janus Daemon" /min cmd /c "python janus_daemon.py --run > janus_daemon_output.log 2>&1"
timeout /t 2 /nobreak >nul

:: Verify daemon started
curl -s http://localhost:8006/health >nul 2>&1
if errorlevel 1 (
    echo  [WARN] Daemon may still be starting up - this is normal
) else (
    echo  [OK] Daemon is running
)

:: ── Build and serve the frontend ─────────────────────────────────────────────
echo  [4/4] Starting Janus UI (port 5173)...
start "Janus UI" /min cmd /c "npm run dev > janus_ui_output.log 2>&1"
timeout /t 3 /nobreak >nul

:: ── Open browser ─────────────────────────────────────────────────────────────
echo.
echo  ============================================================
echo   Janus is starting up!
echo.
echo   UI:     http://localhost:5173
echo   API:    http://localhost:8006
echo.
echo   Logs:
echo     Daemon: janus_daemon_output.log
echo     UI:     janus_ui_output.log
echo.
echo   To stop Janus, close this window or press Ctrl+C
echo  ============================================================
echo.

timeout /t 2 /nobreak >nul
start "" "http://localhost:5173"

:: ── Keep window open ─────────────────────────────────────────────────────────
echo  Press any key to stop all Janus processes...
pause >nul

:: ── Cleanup ──────────────────────────────────────────────────────────────────
echo  Stopping Janus...
taskkill /f /fi "WINDOWTITLE eq Janus Daemon*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq Janus UI*" >nul 2>&1
echo  Done. Goodbye.
timeout /t 1 /nobreak >nul
