# Script để thiết lập tự động crawl thoitiet360 vào 9h sáng mỗi ngày
# Chạy script này với quyền Administrator

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "THIET LAP TU DONG CRAWL THOITIET360" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Lấy đường dẫn script hiện tại
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$crawlScript = Join-Path $scriptDir "crawl_thoitiet360.py"
$syncScript = Join-Path $scriptDir "sync_thoitiet360.py"

# Tìm Python
Write-Host "Dang tim Python..." -ForegroundColor Yellow
$pythonExe = $null

# Thử các đường dẫn phổ biến
$pythonPaths = @(
    "python",
    "py",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python39\python.exe",
    "C:\Python311\python.exe",
    "C:\Python310\python.exe",
    "C:\Python39\python.exe"
)

foreach ($path in $pythonPaths) {
    try {
        $result = & $path --version 2>&1
        if ($LASTEXITCODE -eq 0 -or $result -match "Python") {
            $pythonExe = $path
            break
        }
    } catch {
        continue
    }
}

if (-not $pythonExe) {
    Write-Host "❌ Khong tim thay Python!" -ForegroundColor Red
    Write-Host "Vui long cai dat Python hoac them Python vao PATH" -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "✅ Da tim thay Python: $pythonExe" -ForegroundColor Green
Write-Host ""

# Tạo file batch wrapper cho crawl
Write-Host "Dang tao script wrapper..." -ForegroundColor Yellow
$crawlWrapperBat = Join-Path $scriptDir "run_crawl_thoitiet360.bat"
$crawlWrapperContent = @"
@echo off
cd /d "$scriptDir"
"$pythonExe" "$crawlScript"
"@
$crawlWrapperContent | Out-File -FilePath $crawlWrapperBat -Encoding ASCII -Force
Write-Host "  ✅ Da tao wrapper crawl: $crawlWrapperBat" -ForegroundColor Green

# Tạo file batch wrapper cho sync
$syncWrapperBat = Join-Path $scriptDir "run_sync_thoitiet360.bat"
$syncWrapperContent = @"
@echo off
cd /d "$scriptDir"
"$pythonExe" "$syncScript"
"@
$syncWrapperContent | Out-File -FilePath $syncWrapperBat -Encoding ASCII -Force
Write-Host "  ✅ Da tao wrapper sync: $syncWrapperBat" -ForegroundColor Green
Write-Host ""

# Xóa các task hiện có (nếu có)
Write-Host "Dang xoa task hien co (neu co)..." -ForegroundColor Yellow
$tasks = @("WeatherForecast_AutoCrawlThoitiet360", "WeatherForecast_StartupSyncThoitiet360", "WeatherForecast_AutoSyncThoitiet360")
foreach ($taskName in $tasks) {
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($task) {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        Write-Host "  ✅ Da xoa task cu: $taskName" -ForegroundColor Green
    }
}
Write-Host ""

# Tạo các hành động cho task
$crawlAction = New-ScheduledTaskAction -Execute $crawlWrapperBat -WorkingDirectory $scriptDir
$syncAction = New-ScheduledTaskAction -Execute $syncWrapperBat -WorkingDirectory $scriptDir

# Tạo các trigger
$crawlTrigger = New-ScheduledTaskTrigger -Daily -At "09:00"  # 9h sáng mỗi ngày
$startupTrigger = New-ScheduledTaskTrigger -AtStartup  # Khi khởi động máy
$syncTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Minutes 10) -RepetitionDuration (New-TimeSpan -Days 365)  # Mỗi 10 phút

# Cài đặt task
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RunOnlyIfNetworkAvailable

# Principal: chạy với user hiện tại
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest

# Đăng ký các task
Write-Host "Dang thiet lap cac task tu dong..." -ForegroundColor Yellow
Write-Host ""

Write-Host "[1/3] Dang tao task crawl vao 9h sang moi ngay..." -ForegroundColor Cyan
try {
    Register-ScheduledTask -TaskName "WeatherForecast_AutoCrawlThoitiet360" -Action $crawlAction -Trigger $crawlTrigger -Settings $settings -Principal $principal -Description "Tu dong crawl du lieu thoitiet360 vao 9h sang moi ngay" | Out-Null
    Write-Host "  ✅ Da tao task crawl thanh cong!" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Khong the tao task crawl: $_" -ForegroundColor Red
}

Write-Host ""

Write-Host "[2/3] Dang tao task dong bo khi khoi dong may..." -ForegroundColor Cyan
try {
    Register-ScheduledTask -TaskName "WeatherForecast_StartupSyncThoitiet360" -Action $syncAction -Trigger $startupTrigger -Settings $settings -Principal $principal -Description "Tu dong dong bo du lieu thoitiet360 khi Windows khoi dong" | Out-Null
    Write-Host "  ✅ Da tao task dong bo khi khoi dong thanh cong!" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Khong the tao task dong bo khi khoi dong: $_" -ForegroundColor Red
}

Write-Host ""

Write-Host "[3/3] Dang tao task dong bo dinh ky (moi 10 phut)..." -ForegroundColor Cyan
try {
    Register-ScheduledTask -TaskName "WeatherForecast_AutoSyncThoitiet360" -Action $syncAction -Trigger $syncTrigger -Settings $settings -Principal $principal -Description "Tu dong dong bo du lieu thoitiet360 moi 10 phut" | Out-Null
    Write-Host "  ✅ Da tao task dong bo dinh ky thanh cong!" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Khong the tao task dong bo dinh ky: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "THIET LAP THANH CONG!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Cac task da tao:" -ForegroundColor Cyan
Write-Host "  1. WeatherForecast_AutoCrawlThoitiet360 - Crawl vao 9h sang moi ngay" -ForegroundColor White
Write-Host "  2. WeatherForecast_StartupSyncThoitiet360 - Dong bo khi khoi dong may" -ForegroundColor White
Write-Host "  3. WeatherForecast_AutoSyncThoitiet360 - Dong bo moi 10 phut" -ForegroundColor White
Write-Host ""
Write-Host "Cach hoat dong:" -ForegroundColor Yellow
Write-Host "  - Khi ban bat may tinh: Tu dong pull du lieu moi tu GitHub" -ForegroundColor White
Write-Host "  - Khi may tinh dang chay: Tu dong pull du lieu moi moi 10 phut" -ForegroundColor White
Write-Host "  - Vao 9h sang moi ngay: Tu dong crawl du lieu moi tu thoitiet360.edu.vn" -ForegroundColor White
Write-Host ""
Write-Host "De kiem tra task:" -ForegroundColor Yellow
Write-Host "  Get-ScheduledTask -TaskName `"WeatherForecast_*Thoitiet360`"" -ForegroundColor White
Write-Host ""
Write-Host "De xoa task:" -ForegroundColor Yellow
Write-Host "  Unregister-ScheduledTask -TaskName `"WeatherForecast_AutoSyncThoitiet360`" -Confirm:`$false" -ForegroundColor White
Write-Host ""

pause

