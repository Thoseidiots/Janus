@echo off
title Janus Shutdown
echo Stopping Janus processes...
taskkill /f /fi "WINDOWTITLE eq Janus Daemon*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq Janus UI*" >nul 2>&1
echo Done.
timeout /t 1 /nobreak >nul
