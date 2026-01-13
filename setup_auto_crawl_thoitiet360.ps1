# Script de thiet lap tu dong crawl thoitiet360 vao 9h sang moi ngay
# Chay script nay voi quyen Administrator

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "THIET LAP TU DONG CRAWL THOITIET360" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Lay duong dan script hien tai
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$crawlScript = Join-Path $scriptDir "crawl_thoitiet360.py"

# Tim Python
Write-Host "Dang tim Python..." -ForegroundColor Yellow
$pythonExe = $null

# Thu cac duong dan pho bien
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
    Write-Host "Khong tim thay Python!" -ForegroundColor Red
    Write-Host "Vui long cai dat Python hoac them Python vao PATH" -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "Da tim thay Python: $pythonExe" -ForegroundColor Green
Write-Host ""

# Tao file VBScript wrapper de chay ngam (khong hien terminal)
Write-Host "Dang tao script wrapper (chay ngam)..." -ForegroundColor Yellow
$crawlWrapperVbs = Join-Path $scriptDir "run_crawl_thoitiet360.vbs"
$crawlWrapperVbsContent = @"
Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "$scriptDir"
WshShell.Run Chr(34) & "$pythonExe" & Chr(34) & " " & Chr(34) & "$crawlScript" & Chr(34), 0, False
Set WshShell = Nothing
"@
$crawlWrapperVbsContent | Out-File -FilePath $crawlWrapperVbs -Encoding ASCII -Force
    Write-Host "  Da tao wrapper VBS (chay ngam): $crawlWrapperVbs" -ForegroundColor Green
Write-Host ""

# Xoa cac task hien co (neu co)
Write-Host "Dang xoa task hien co (neu co)..." -ForegroundColor Yellow
$tasks = @("WeatherForecast_AutoCrawlThoitiet360")
foreach ($taskName in $tasks) {
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($task) {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        Write-Host "  Da xoa task cu: $taskName" -ForegroundColor Green
    }
}
Write-Host ""
Write-Host "  Cac task dong bo da duoc chuyen sang sync_all.py (chay setup_auto_sync.ps1)" -ForegroundColor Cyan
Write-Host ""

# Tao cac hanh dong cho task (su dung VBScript de chay ngam)
$crawlAction = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$crawlWrapperVbs`"" -WorkingDirectory $scriptDir

# Tao trigger cho crawl
$crawlTrigger = New-ScheduledTaskTrigger -Daily -At "09:00"  # 9h sang moi ngay

# Cai dat task (chay ngam, khong hien window)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RunOnlyIfNetworkAvailable -Hidden

# Principal: chay voi user hien tai
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest

# Dang ky task
Write-Host "Dang thiet lap task tu dong crawl..." -ForegroundColor Yellow
Write-Host ""

Write-Host "[1/1] Dang tao task crawl vao 9h sang moi ngay..." -ForegroundColor Cyan
try {
    Register-ScheduledTask -TaskName "WeatherForecast_AutoCrawlThoitiet360" -Action $crawlAction -Trigger $crawlTrigger -Settings $settings -Principal $principal -Description "Tu dong crawl du lieu thoitiet360 vao 9h sang moi ngay" | Out-Null
    Write-Host "  Da tao task crawl thanh cong!" -ForegroundColor Green
} catch {
    Write-Host "  Loi: Khong the tao task crawl: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "THIET LAP THANH CONG!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Task da tao:" -ForegroundColor Cyan
Write-Host "  1. WeatherForecast_AutoCrawlThoitiet360 - Crawl vao 9h sang moi ngay" -ForegroundColor White
Write-Host ""
Write-Host "Cach hoat dong:" -ForegroundColor Yellow
Write-Host "  - Vao 9h sang moi ngay: Tu dong crawl du lieu moi tu thoitiet360.edu.vn - CHAY NGAM" -ForegroundColor White
Write-Host ""
Write-Host "Luu y:" -ForegroundColor Yellow
Write-Host "  - Dong bo tu GitHub da duoc chuyen sang sync_all.py" -ForegroundColor White
Write-Host "  - Chay setup_auto_sync.ps1 de thiet lap dong bo tu GitHub" -ForegroundColor White
Write-Host ""
Write-Host "De kiem tra task:" -ForegroundColor Yellow
Write-Host "  Get-ScheduledTask -TaskName WeatherForecast_*Thoitiet360" -ForegroundColor White
Write-Host ""
Write-Host "De xoa task:" -ForegroundColor Yellow
Write-Host "  Unregister-ScheduledTask -TaskName WeatherForecast_AutoCrawlThoitiet360 -Confirm:`$false" -ForegroundColor White
Write-Host ""

pause

