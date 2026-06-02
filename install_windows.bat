@echo off
setlocal enabledelayedexpansion
title Vizard Clone - Installer

echo ========================================
echo   Vizard Clone - Windows installer
echo ========================================
echo.

REM --- Check Python ---
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo Install Python 3.10+ from https://www.python.org/downloads/
    echo and make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

python --version
echo.

REM --- Check ffmpeg ---
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo [WARNING] ffmpeg not found in PATH.
    echo.
    echo Install ffmpeg via one of:
    echo   1. winget install Gyan.FFmpeg
    echo   2. Download from https://www.gyan.dev/ffmpeg/builds/
    echo      and add bin folder to PATH manually.
    echo.
    set /p installff=Try to install via winget now? [y/n]: 
    if /i "!installff!"=="y" (
        winget install --id Gyan.FFmpeg -e --silent --accept-source-agreements --accept-package-agreements
        echo.
        echo Please RESTART cmd so PATH refreshes, then run install_windows.bat again.
        pause
        exit /b 0
    )
)

REM --- Create venv ---
echo Creating virtual environment in .venv ...
if not exist .venv (
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv. Check that Python "venv" module is installed.
        pause
        exit /b 1
    )
)

echo Activating venv and upgrading pip ...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo Installing dependencies (this can take 3-10 minutes) ...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed. Check the error message above.
    pause
    exit /b 1
)

REM --- Detect NVIDIA GPU and offer CUDA support for Whisper ---
echo.
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo ========================================
    echo   NVIDIA GPU detected
    echo ========================================
    nvidia-smi --query-gpu=name --format=csv,noheader
    echo.
    echo To use Whisper on GPU ^(5-15x faster transcription^),
    echo we need to install CUDA runtime libraries ^(cuBLAS + cuDNN^).
    echo Size: ~700 MB download.
    echo.
    set /p installcuda=Install CUDA libs for Whisper GPU? [y/n]: 
    if /i "!installcuda!"=="y" (
        echo Installing nvidia-cublas-cu12 nvidia-cudnn-cu12 ...
        pip install nvidia-cublas-cu12==12.4.5.8 nvidia-cudnn-cu12==9.1.0.70
        if errorlevel 1 (
            echo [WARNING] CUDA libs install failed. Whisper will run on CPU.
        ) else (
            echo [OK] CUDA libs installed. Whisper will auto-detect GPU.
        )
    ) else (
        echo Skipped. Whisper will run on CPU. You can install CUDA libs later by running install_cuda.bat
    )
) else (
    echo No NVIDIA GPU detected ^(nvidia-smi not found^). Whisper will run on CPU.
)

echo.
echo ========================================
echo   Install complete!
echo   Run run.bat to start the app.
echo ========================================
pause
