@echo off
echo ========================================
echo THIET LAP TU DONG CRAWL VA DONG BO THOITIET360
echo ========================================
echo.
echo Dang mo PowerShell voi quyen Administrator...
echo.

cd /d "%~dp0"
powershell -Command "Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File \"%~dp0setup_auto_crawl_thoitiet360.ps1\"'"

pause

