@echo off
echo ========================================
echo THIET LAP TU DONG DONG BO DU LIEU
echo ========================================
echo.
echo Dang mo PowerShell voi quyen Administrator...
echo.

cd /d "%~dp0"
powershell -Command "Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File \"%~dp0setup_auto_sync.ps1\"'"

pause

