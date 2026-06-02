# Vizard Clone — PowerShell installer (Unicode-safe alternative to install_windows.bat)
# Run from PowerShell:  Set-ExecutionPolicy -Scope Process Bypass; .\install_windows.ps1

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Vizard Clone — установщик (PowerShell)" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# --- Python ---
$pythonExe = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonExe) {
    Write-Host "[ОШИБКА] Python не найден в PATH." -ForegroundColor Red
    Write-Host "Установи Python 3.10+ с https://www.python.org/downloads/"
    Write-Host "При установке отметь галочку 'Add Python to PATH'."
    Read-Host "Нажми Enter для выхода"
    exit 1
}
& python --version
Write-Host ""

# --- ffmpeg ---
$ffmpegExe = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpegExe) {
    Write-Host "[ВНИМАНИЕ] ffmpeg не найден в PATH." -ForegroundColor Yellow
    $answer = Read-Host "Установить ffmpeg через winget сейчас? [y/n]"
    if ($answer -eq "y") {
        winget install --id Gyan.FFmpeg -e --silent --accept-source-agreements --accept-package-agreements
        Write-Host "Перезапусти PowerShell чтобы PATH обновился, затем запусти install_windows.ps1 снова." -ForegroundColor Yellow
        Read-Host "Нажми Enter"
        exit 0
    }
}

# --- venv ---
$venvPath = Join-Path $PSScriptRoot ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Создаю виртуальное окружение в .venv ..." -ForegroundColor Green
    & python -m venv $venvPath
}

# --- install deps ---
$pip = Join-Path $venvPath "Scripts\pip.exe"
$py  = Join-Path $venvPath "Scripts\python.exe"

Write-Host "Обновляю pip..." -ForegroundColor Green
& $py -m pip install --upgrade pip

Write-Host "Устанавливаю зависимости (3-10 минут)..." -ForegroundColor Green
& $pip install -r (Join-Path $PSScriptRoot "requirements.txt")

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Установка завершена!" -ForegroundColor Green
Write-Host "  Запускай run.bat или: .\.venv\Scripts\python.exe main.py" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Read-Host "Нажми Enter"
