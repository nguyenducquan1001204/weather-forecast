@echo off
REM Script to setup complete auto-sync system
REM This will create both startup sync and periodic sync tasks

echo ========================================
echo   SETUP AUTO-SYNC SYSTEM
echo ========================================
echo.

REM Get the current directory (where this script is located)
set SCRIPT_DIR=%~dp0
set PYTHON_SCRIPT=%SCRIPT_DIR%auto_sync.py

echo Setting up auto-sync tasks...
echo.

REM Task 1: Sync on startup
echo [1/2] Creating startup sync task...
schtasks /Create /TN "WeatherForecast_StartupSync" /TR "python \"%PYTHON_SCRIPT%\"" /SC ONSTART /F
if %ERRORLEVEL% == 0 (
    echo   ✅ Startup sync task created successfully!
) else (
    echo   ❌ Failed to create startup sync task
)

echo.

REM Task 2: Sync every 10 minutes
echo [2/2] Creating periodic sync task (every 10 minutes)...
schtasks /Create /TN "WeatherForecast_AutoSync" /TR "python \"%PYTHON_SCRIPT%\"" /SC MINUTE /MO 10 /F
if %ERRORLEVEL% == 0 (
    echo   ✅ Periodic sync task created successfully!
) else (
    echo   ❌ Failed to create periodic sync task
)

echo.
echo ========================================
echo   SETUP COMPLETE
echo ========================================
echo.
echo ✅ Auto-sync system is now active!
echo.
echo Tasks created:
echo   1. WeatherForecast_StartupSync - Syncs when Windows starts
echo   2. WeatherForecast_AutoSync - Syncs every 10 minutes
echo.
echo How it works:
echo   - When you turn on your computer: Files are automatically synced
echo   - While computer is running: Files are synced every 10 minutes
echo   - Result: You always have the latest files!
echo.
echo To view tasks: Task Scheduler ^> Task Scheduler Library
echo To delete tasks:
echo   schtasks /Delete /TN "WeatherForecast_StartupSync"
echo   schtasks /Delete /TN "WeatherForecast_AutoSync"
echo.

pause

