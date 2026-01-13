# Script de thiet lap tu dong tao forecast hang ngay
# Chay script nay voi quyen Administrator

param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# Kiem tra xem co dang chay voi quyen quan tri khong
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "Dang yeu cau quyen quan tri..." -ForegroundColor Yellow
    $scriptPath = $MyInvocation.MyCommand.Path
    Start-Process powershell.exe -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" -Force" -Wait
    exit
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  THIET LAP TU DONG TAO FORECAST" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Lay thu muc chua script
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$forecastScript = Join-Path $scriptDir "auto_generate_forecast.py"

# Tim Python
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
        Write-Host "Khong tim thay Python!" -ForegroundColor Red
        Write-Host "Vui long cai dat Python hoac chi dinh duong dan trong script." -ForegroundColor Yellow
        pause
        exit 1
    }
}

Write-Host "Da tim thay Python: $pythonExe" -ForegroundColor Green
Write-Host ""

# Tao file VBScript wrapper de chay ngam
Write-Host "Dang tao script wrapper (chay ngam)..." -ForegroundColor Yellow
$wrapperVbs = Join-Path $scriptDir "run_auto_generate_forecast.vbs"
$wrapperVbsContent = @"
Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "$scriptDir"
WshShell.Run Chr(34) & "$pythonExe" & Chr(34) & " " & Chr(34) & "$forecastScript" & Chr(34), 0, False
Set WshShell = Nothing
"@
$wrapperVbsContent | Out-File -FilePath $wrapperVbs -Encoding ASCII -Force
Write-Host "  Da tao wrapper VBS (chay ngam): $wrapperVbs" -ForegroundColor Green
Write-Host ""

# Xoa task hien co (neu co)
Write-Host "Dang xoa task hien co (neu co)..." -ForegroundColor Yellow
$taskName = "WeatherForecast_AutoGenerateForecast"
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($task) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "  Da xoa: $taskName" -ForegroundColor Green
}
Write-Host ""

# Tao hanh dong cho task
$action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$wrapperVbs`""

# Tao trigger: chay vao 10h sang moi ngay (sau khi crawl thoitiet360)
$trigger = New-ScheduledTaskTrigger -Daily -At "10:00"

# Cai dat task (chay ngam, khong hien window)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -Hidden

# Principal: chay voi user hien tai
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest

# Dang ky task
Write-Host "Dang thiet lap task tu dong tao forecast..." -ForegroundColor Yellow
Write-Host ""

try {
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description "Tu dong tao forecast hang ngay vao 10h sang" | Out-Null
    Write-Host "  Da tao task tu dong tao forecast thanh cong!" -ForegroundColor Green
} catch {
    Write-Host "  Loi: Khong the tao task: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  HOAN TAT THIET LAP" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "He thong tu dong tao forecast da duoc kich hoat!" -ForegroundColor Green
Write-Host ""
Write-Host "Task da tao:" -ForegroundColor Yellow
Write-Host "  WeatherForecast_AutoGenerateForecast - Tao forecast vao 10h sang moi ngay" -ForegroundColor White
Write-Host ""
Write-Host "Cach hoat dong:" -ForegroundColor Yellow
Write-Host "  - Vao 10h sang moi ngay: Tu dong tao forecast cho tat ca thanh pho va luu vao database - CHAY NGAM" -ForegroundColor White
Write-Host "  - Forecast se duoc luu vao bang system_forecasts" -ForegroundColor White
Write-Host "  - Trang lich su se tu dong co du lieu moi moi ngay" -ForegroundColor White
Write-Host ""
Write-Host "De xem cac task: Task Scheduler > Task Scheduler Library" -ForegroundColor Cyan
Write-Host "De kiem tra: schtasks /Run /TN WeatherForecast_AutoGenerateForecast" -ForegroundColor Cyan
Write-Host ""

pause
