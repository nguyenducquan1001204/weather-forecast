# Script PowerShell de thiet lap tu dong dong bo voi quyen quan tri
# Script nay se yeu cau quyen quan tri neu can

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
Write-Host "  THIET LAP HE THONG TU DONG DONG BO" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Lay thu muc chua script
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonScript = Join-Path $scriptDir "sync_all.py"

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

# Tao file VBScript wrapper de chay ngam (khong hien terminal)
Write-Host "Dang tao script wrapper (chay ngam)..." -ForegroundColor Yellow
$wrapperVbs = Join-Path $scriptDir "run_sync_all.vbs"
$wrapperVbsContent = @"
Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "$scriptDir"
WshShell.Run Chr(34) & "$pythonExe" & Chr(34) & " " & Chr(34) & "$pythonScript" & Chr(34), 0, False
Set WshShell = Nothing
"@
$wrapperVbsContent | Out-File -FilePath $wrapperVbs -Encoding ASCII -Force
    Write-Host "  Da tao wrapper VBS (chay ngam): $wrapperVbs" -ForegroundColor Green
Write-Host ""

# Xoa cac task hien co
Write-Host "Dang xoa cac task hien co (neu co)..." -ForegroundColor Yellow
$tasks = @("WeatherForecast_StartupSync", "WeatherForecast_AutoSync")
foreach ($taskName in $tasks) {
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($task) {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        Write-Host "  Da xoa: $taskName" -ForegroundColor Green
    }
}
Write-Host ""

# Tao cac hanh dong cho task (su dung VBScript de chay ngam)
$action1 = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$wrapperVbs`""
$action2 = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$wrapperVbs`""

# Tao cac trigger cho task
$trigger1 = New-ScheduledTaskTrigger -AtStartup
$trigger2 = New-ScheduledTaskTrigger -Once -At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Minutes 10) -RepetitionDuration (New-TimeSpan -Days 365)

# Tao cai dat cho task (chay ngam, khong hien window)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -Hidden

# Tao principal (chay voi user hien tai)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest

# Dang ky cac task
Write-Host "Dang thiet lap cac task tu dong dong bo..." -ForegroundColor Yellow
Write-Host ""

Write-Host "[1/2] Dang tao task dong bo khi khoi dong..." -ForegroundColor Cyan
try {
    Register-ScheduledTask -TaskName "WeatherForecast_StartupSync" -Action $action1 -Trigger $trigger1 -Settings $settings -Principal $principal -Description "Tu dong dong bo du lieu thoi tiet khi Windows khoi dong" | Out-Null
    Write-Host "  Da tao task dong bo khi khoi dong thanh cong!" -ForegroundColor Green
} catch {
    Write-Host "  Loi: Khong the tao task dong bo khi khoi dong: $_" -ForegroundColor Red
}

Write-Host ""

Write-Host "[2/2] Dang tao task dong bo dinh ky (moi 10 phut)..." -ForegroundColor Cyan
try {
    Register-ScheduledTask -TaskName "WeatherForecast_AutoSync" -Action $action2 -Trigger $trigger2 -Settings $settings -Principal $principal -Description "Tu dong dong bo du lieu thoi tiet moi 10 phut" | Out-Null
    Write-Host "  Da tao task dong bo dinh ky thanh cong!" -ForegroundColor Green
} catch {
    Write-Host "  Loi: Khong the tao task dong bo dinh ky: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  HOAN TAT THIET LAP" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "He thong tu dong dong bo da duoc kich hoat!" -ForegroundColor Green
Write-Host ""
Write-Host "Cac task da tao:" -ForegroundColor Yellow
Write-Host "  1. WeatherForecast_StartupSync - Dong bo khi Windows khoi dong"
Write-Host "  2. WeatherForecast_AutoSync - Dong bo moi 10 phut"
Write-Host ""
Write-Host "Cach hoat dong:" -ForegroundColor Yellow
Write-Host "  - Khi ban bat may tinh: Tu dong pull tat ca file tu GitHub - CHAY NGAM"
Write-Host "  - Khi may tinh dang chay: Tu dong pull moi 10 phut - CHAY NGAM (khong hien terminal)"
Write-Host "  - Tu dong import thoitiet360_data.csv vao database neu co file moi"
Write-Host "  - Ket qua: Ban luon co cac file moi nhat va database duoc cap nhat!"
Write-Host ""
Write-Host "De xem cac task: Task Scheduler > Task Scheduler Library" -ForegroundColor Cyan
Write-Host "De kiem tra: schtasks /Run /TN WeatherForecast_AutoSync" -ForegroundColor Cyan
Write-Host ""

pause

