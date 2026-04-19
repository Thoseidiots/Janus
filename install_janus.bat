@echo off
setlocal EnableDelayedExpansion
title Janus Installer

echo.
echo ============================================================
echo   JANUS AUTONOMOUS WORKER - INSTALLER
echo ============================================================
echo.

:: ── Check for admin rights ───────────────────────────────────────────────────
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [!] This installer needs administrator rights.
    echo     Right-click install_janus.bat and choose "Run as administrator"
    pause
    exit /b 1
)

set "JANUS_DIR=%~dp0"
cd /d "%JANUS_DIR%"

:: ── Step 0: Add Windows Defender exclusion FIRST ─────────────────────────────
echo [0/8] Adding Windows Defender exclusion for Janus folder...
echo       (This prevents Defender from killing Janus during operation)
powershell -Command "Add-MpPreference -ExclusionPath '%JANUS_DIR%' -ErrorAction SilentlyContinue"
powershell -Command "Add-MpPreference -ExclusionProcess 'python.exe' -ErrorAction SilentlyContinue"
powershell -Command "Add-MpPreference -ExclusionProcess 'pythonw.exe' -ErrorAction SilentlyContinue"
if %errorLevel% equ 0 (
    echo [OK] Defender exclusion added for: %JANUS_DIR%
) else (
    echo [!] Could not add Defender exclusion automatically.
    echo     If Janus gets quarantined, manually add this folder to
    echo     Windows Security ^> Virus ^& threat protection ^> Exclusions:
    echo     %JANUS_DIR%
)
echo.

echo [1/8] Checking Python...
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [!] Python not found. Downloading Python 3.11...
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile '%TEMP%\python_installer.exe'"
    echo     Installing Python 3.11 (this may take a minute)...
    "%TEMP%\python_installer.exe" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    if !errorLevel! neq 0 (
        echo [!] Python installation failed. Please install Python 3.11 manually from python.org
        pause
        exit /b 1
    )
    echo [OK] Python installed.
) else (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
    echo [OK] Python !PYVER! found.
)

echo.
echo [2/8] Installing Python dependencies...
python janus_install_deps.py 2>nul
if %errorLevel% neq 0 (
    echo [!] Some dependencies failed. Trying pip directly...
    pip install fastapi uvicorn requests chardet mutagen pypdf mss pyttsx3 pyzbar schedule pywin32 Pillow opencv-python imagehash pyautogui pygetwindow pyperclip psutil --quiet
)
echo [OK] Python dependencies installed.

echo.
echo [3/8] Checking Tesseract OCR (needed for screen reading)...
tesseract --version >nul 2>&1
if %errorLevel% neq 0 (
    echo     Tesseract not found. Downloading...
    powershell -Command "Invoke-WebRequest -Uri 'https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe' -OutFile '%TEMP%\tesseract_installer.exe'"
    echo     Installing Tesseract OCR...
    "%TEMP%\tesseract_installer.exe" /S
    :: Add to PATH
    setx PATH "%PATH%;C:\Program Files\Tesseract-OCR" /M >nul 2>&1
    echo [OK] Tesseract installed.
) else (
    echo [OK] Tesseract already installed.
)

echo.
echo [4/8] Checking Google Chrome...
if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    echo [OK] Chrome found.
) else if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
    echo [OK] Chrome found.
) else (
    echo [!] Chrome not found. Downloading...
    powershell -Command "Invoke-WebRequest -Uri 'https://dl.google.com/chrome/install/latest/chrome_installer.exe' -OutFile '%TEMP%\chrome_installer.exe'"
    "%TEMP%\chrome_installer.exe" /silent /install
    echo [OK] Chrome installed.
)

echo.
echo [5/8] Running self-test to verify everything works...
python janus_integration_test.py >nul 2>&1
if %errorLevel% equ 0 (
    echo [OK] All integration tests passed.
) else (
    echo [!] Some tests failed. Running auto-repair...
    python janus_selfheal.py >nul 2>&1
    python janus_integration_test.py >nul 2>&1
    if !errorLevel! equ 0 (
        echo [OK] Auto-repair succeeded. All tests now pass.
    ) else (
        echo [!] Some tests still failing after repair.
        echo     Janus will start in degraded mode.
        echo     Check janus_selfheal_report.json for details.
    )
)

echo.
echo [6/8] Registering Janus to start on Windows login...
python janus_daemon.py --install
if %errorLevel% equ 0 (
    echo [OK] Janus will start automatically on login.
) else (
    echo [!] Could not register startup entry. You can start Janus manually.
)

echo.
echo [7/8] Creating desktop shortcut...
set "SHORTCUT=%USERPROFILE%\Desktop\Janus.lnk"
powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT%'); $s.TargetPath = 'python'; $s.Arguments = '\"%JANUS_DIR%janus_daemon.py\" --run'; $s.WorkingDirectory = '%JANUS_DIR%'; $s.Description = 'Start Janus Autonomous Worker'; $s.Save()"
if exist "%SHORTCUT%" (
    echo [OK] Desktop shortcut created.
) else (
    echo [!] Could not create shortcut (non-critical).
)

echo.
echo [8/8] Starting Janus now...
start "Janus Daemon" /min python "%JANUS_DIR%janus_daemon.py" --run
timeout /t 3 /nobreak >nul

:: Check if it started
python -c "import requests; r=requests.get('http://localhost:8006/health',timeout=3); print('[OK] Janus is running! Status:', r.json().get('status'))" 2>nul
if %errorLevel% neq 0 (
    echo [!] Janus is starting in the background.
    echo     Check janus_daemon.log if it doesn't appear within 30 seconds.
)

echo.
echo ============================================================
echo   INSTALLATION COMPLETE
echo ============================================================
echo.
echo   Janus is now running in the background.
echo.
echo   Owner dashboard : open index.html in your browser
echo   User dashboard  : http://localhost:8005
echo   JC API          : http://localhost:8004
echo   Daemon status   : http://localhost:8006/status
echo   Logs            : janus_daemon.log
echo.
echo   To stop Janus   : python janus_daemon.py --status
echo   To uninstall    : python janus_daemon.py --uninstall
echo.
pause
