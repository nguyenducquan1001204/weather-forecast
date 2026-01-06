# Script PowerShell để thiết lập tự động đồng bộ với quyền quản trị
# Script này sẽ yêu cầu quyền quản trị nếu cần

param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# Kiểm tra xem có đang chạy với quyền quản trị không
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "Đang yêu cầu quyền quản trị..." -ForegroundColor Yellow
    $scriptPath = $MyInvocation.MyCommand.Path
    Start-Process powershell.exe -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" -Force" -Wait
    exit
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  THIẾT LẬP HỆ THỐNG TỰ ĐỘNG ĐỒNG BỘ" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Lấy thư mục chứa script
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonScript = Join-Path $scriptDir "auto_sync.py"
$wrapperBat = Join-Path $scriptDir "run_auto_sync.bat"

# Tìm Python
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
        Write-Host "❌ Không tìm thấy Python!" -ForegroundColor Red
        Write-Host "Vui lòng cài đặt Python hoặc chỉ định đường dẫn trong script." -ForegroundColor Yellow
        pause
        exit 1
    }
}

Write-Host "Đã tìm thấy Python: $pythonExe" -ForegroundColor Green
Write-Host ""

# Tạo file batch wrapper
Write-Host "Đang tạo script wrapper..." -ForegroundColor Yellow
$wrapperContent = @"
@echo off
cd /d "$scriptDir"
"$pythonExe" "$pythonScript"
"@
$wrapperContent | Out-File -FilePath $wrapperBat -Encoding ASCII -Force
Write-Host "  ✅ Đã tạo wrapper: $wrapperBat" -ForegroundColor Green
Write-Host ""

# Xóa các task hiện có
Write-Host "Đang xóa các task hiện có (nếu có)..." -ForegroundColor Yellow
$tasks = @("WeatherForecast_StartupSync", "WeatherForecast_AutoSync")
foreach ($taskName in $tasks) {
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($task) {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        Write-Host "  ✅ Đã xóa: $taskName" -ForegroundColor Green
    }
}
Write-Host ""

# Tạo các hành động cho task
$action1 = New-ScheduledTaskAction -Execute $wrapperBat
$action2 = New-ScheduledTaskAction -Execute $wrapperBat

# Tạo các trigger cho task
$trigger1 = New-ScheduledTaskTrigger -AtStartup
$trigger2 = New-ScheduledTaskTrigger -Once -At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Minutes 10) -RepetitionDuration (New-TimeSpan -Days 365)

# Tạo cài đặt cho task
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

# Tạo principal (chạy với user hiện tại)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest

# Đăng ký các task
Write-Host "Đang thiết lập các task tự động đồng bộ..." -ForegroundColor Yellow
Write-Host ""

Write-Host "[1/2] Đang tạo task đồng bộ khi khởi động..." -ForegroundColor Cyan
try {
    Register-ScheduledTask -TaskName "WeatherForecast_StartupSync" -Action $action1 -Trigger $trigger1 -Settings $settings -Principal $principal -Description "Tự động đồng bộ dữ liệu thời tiết khi Windows khởi động" | Out-Null
    Write-Host "  ✅ Đã tạo task đồng bộ khi khởi động thành công!" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Không thể tạo task đồng bộ khi khởi động: $_" -ForegroundColor Red
}

Write-Host ""

Write-Host "[2/2] Đang tạo task đồng bộ định kỳ (mỗi 10 phút)..." -ForegroundColor Cyan
try {
    Register-ScheduledTask -TaskName "WeatherForecast_AutoSync" -Action $action2 -Trigger $trigger2 -Settings $settings -Principal $principal -Description "Tự động đồng bộ dữ liệu thời tiết mỗi 10 phút" | Out-Null
    Write-Host "  ✅ Đã tạo task đồng bộ định kỳ thành công!" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Không thể tạo task đồng bộ định kỳ: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  HOÀN TẤT THIẾT LẬP" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "✅ Hệ thống tự động đồng bộ đã được kích hoạt!" -ForegroundColor Green
Write-Host ""
Write-Host "Các task đã tạo:" -ForegroundColor Yellow
Write-Host "  1. WeatherForecast_StartupSync - Đồng bộ khi Windows khởi động"
Write-Host "  2. WeatherForecast_AutoSync - Đồng bộ mỗi 10 phút"
Write-Host ""
Write-Host "Cách hoạt động:" -ForegroundColor Yellow
Write-Host "  - Khi bạn bật máy tính: Các file được tự động đồng bộ"
Write-Host "  - Khi máy tính đang chạy: Các file được đồng bộ mỗi 10 phút"
Write-Host "  - Kết quả: Bạn luôn có các file mới nhất!"
Write-Host ""
Write-Host "Để xem các task: Task Scheduler > Task Scheduler Library" -ForegroundColor Cyan
Write-Host "Để kiểm tra: schtasks /Run /TN `"WeatherForecast_AutoSync`"" -ForegroundColor Cyan
Write-Host ""

pause

