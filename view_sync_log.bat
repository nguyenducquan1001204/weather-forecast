@echo off
echo ========================================
echo XEM LOG DONG BO TU GITHUB
echo ========================================
echo.

cd /d "%~dp0"

if exist "sync_log.txt" (
    echo Dang hien thi 50 dong cuoi cung cua log...
    echo.
    powershell -Command "Get-Content 'sync_log.txt' -Tail 50"
    echo.
    echo ========================================
    echo Tong so dong trong log:
    powershell -Command "(Get-Content 'sync_log.txt').Count"
) else (
    echo Chua co file log. Chua co lan dong bo nao duoc thuc hien.
)

echo.
pause

