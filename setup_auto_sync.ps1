# PowerShell script to setup auto-sync with admin privileges
# This script will request admin privileges if needed

param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# Check if running as admin
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "Requesting administrator privileges..." -ForegroundColor Yellow
    $scriptPath = $MyInvocation.MyCommand.Path
    Start-Process powershell.exe -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" -Force" -Wait
    exit
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SETUP AUTO-SYNC SYSTEM" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonScript = Join-Path $scriptDir "auto_sync.py"
$wrapperBat = Join-Path $scriptDir "run_auto_sync.bat"

# Find Python
$pythonExe = $null
$possiblePaths = @(
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python39\python.exe"
)

foreach ($path in $possiblePaths) {
    if (Test-Path $path) {
        $pythonExe = $path
        break
    }
}

if (-not $pythonExe) {
    try {
        $pythonExe = (Get-Command python -ErrorAction Stop).Source
    } catch {
        Write-Host "❌ Python not found!" -ForegroundColor Red
        Write-Host "Please install Python or specify path in script." -ForegroundColor Yellow
        pause
        exit 1
    }
}

Write-Host "Found Python: $pythonExe" -ForegroundColor Green
Write-Host ""

# Create wrapper batch file
Write-Host "Creating wrapper script..." -ForegroundColor Yellow
$wrapperContent = @"
@echo off
cd /d "$scriptDir"
"$pythonExe" "$pythonScript"
"@
$wrapperContent | Out-File -FilePath $wrapperBat -Encoding ASCII -Force
Write-Host "  ✅ Wrapper created: $wrapperBat" -ForegroundColor Green
Write-Host ""

# Delete existing tasks
Write-Host "Deleting existing tasks (if any)..." -ForegroundColor Yellow
$tasks = @("WeatherForecast_StartupSync", "WeatherForecast_AutoSync")
foreach ($taskName in $tasks) {
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($task) {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        Write-Host "  ✅ Deleted: $taskName" -ForegroundColor Green
    }
}
Write-Host ""

# Create task actions
$action1 = New-ScheduledTaskAction -Execute $wrapperBat
$action2 = New-ScheduledTaskAction -Execute $wrapperBat

# Create task triggers
$trigger1 = New-ScheduledTaskTrigger -AtStartup
$trigger2 = New-ScheduledTaskTrigger -Once -At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Minutes 10) -RepetitionDuration (New-TimeSpan -Days 365)

# Create task settings
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

# Create principal (run as current user)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest

# Register tasks
Write-Host "Setting up auto-sync tasks..." -ForegroundColor Yellow
Write-Host ""

Write-Host "[1/2] Creating startup sync task..." -ForegroundColor Cyan
try {
    Register-ScheduledTask -TaskName "WeatherForecast_StartupSync" -Action $action1 -Trigger $trigger1 -Settings $settings -Principal $principal -Description "Auto-sync weather data on Windows startup" | Out-Null
    Write-Host "  ✅ Startup sync task created successfully!" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Failed to create startup sync task: $_" -ForegroundColor Red
}

Write-Host ""

Write-Host "[2/2] Creating periodic sync task (every 10 minutes)..." -ForegroundColor Cyan
try {
    Register-ScheduledTask -TaskName "WeatherForecast_AutoSync" -Action $action2 -Trigger $trigger2 -Settings $settings -Principal $principal -Description "Auto-sync weather data every 10 minutes" | Out-Null
    Write-Host "  ✅ Periodic sync task created successfully!" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Failed to create periodic sync task: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SETUP COMPLETE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "✅ Auto-sync system is now active!" -ForegroundColor Green
Write-Host ""
Write-Host "Tasks created:" -ForegroundColor Yellow
Write-Host "  1. WeatherForecast_StartupSync - Syncs when Windows starts"
Write-Host "  2. WeatherForecast_AutoSync - Syncs every 10 minutes"
Write-Host ""
Write-Host "How it works:" -ForegroundColor Yellow
Write-Host "  - When you turn on your computer: Files are automatically synced"
Write-Host "  - While computer is running: Files are synced every 10 minutes"
Write-Host "  - Result: You always have the latest files!"
Write-Host ""
Write-Host "To view tasks: Task Scheduler > Task Scheduler Library" -ForegroundColor Cyan
Write-Host "To test: schtasks /Run /TN `"WeatherForecast_AutoSync`"" -ForegroundColor Cyan
Write-Host ""

pause

