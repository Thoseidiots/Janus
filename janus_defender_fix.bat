@echo off
:: janus_defender_fix.bat
:: Run this as administrator if Windows Defender quarantines Janus files.
:: It re-adds the exclusion and restores any quarantined files.

net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Right-click this file and choose "Run as administrator"
    pause
    exit /b 1
)

set "JANUS_DIR=%~dp0"

echo Restoring Windows Defender exclusion for Janus...
powershell -Command "Add-MpPreference -ExclusionPath '%JANUS_DIR%' -ErrorAction Stop"
powershell -Command "Add-MpPreference -ExclusionProcess 'python.exe' -ErrorAction SilentlyContinue"
powershell -Command "Add-MpPreference -ExclusionProcess 'pythonw.exe' -ErrorAction SilentlyContinue"
echo Done.

echo.
echo Checking for quarantined files...
powershell -Command "Get-MpThreat | Where-Object {$_.Resources -like '*janus*'} | ForEach-Object { Write-Host 'Found quarantined:' $_.Resources }"

echo.
echo Attempting to restore quarantined items...
powershell -Command "Get-MpThreatDetection | Where-Object {$_.Resources -like '*janus*'} | ForEach-Object { Remove-MpThreat -ThreatID $_.ThreatID -ErrorAction SilentlyContinue }"

echo.
echo If files are still missing, restore them from:
echo   Windows Security ^> Virus ^& threat protection ^> Protection history
echo   Find the Janus items and click "Restore"
echo.
pause
