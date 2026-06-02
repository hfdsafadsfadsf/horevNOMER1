@echo off
setlocal
title Vizard Clone - CUDA libs for Whisper

echo ========================================
echo   Installing CUDA libs for Whisper GPU
echo ========================================
echo.

if not exist .venv (
    echo [ERROR] .venv not found. Run install_windows.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo This installs cuBLAS + cuDNN CUDA runtime libraries (~700 MB)
echo so faster-whisper can use NVIDIA GPU.
echo.

where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARNING] nvidia-smi not found. You may not have NVIDIA GPU.
    echo Installation will proceed but Whisper will still fallback to CPU at runtime.
    echo.
)

pip install --upgrade nvidia-cublas-cu12==12.4.5.8 nvidia-cudnn-cu12==9.1.0.70
if errorlevel 1 (
    echo.
    echo [ERROR] Install failed. Try manually:
    echo   pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
    pause
    exit /b 1
)

echo.
echo ========================================
echo   CUDA libs installed!
echo   Whisper will auto-detect GPU at next run.
echo ========================================
pause
