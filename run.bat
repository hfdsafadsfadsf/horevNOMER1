@echo off
title Vizard Clone
cd /d "%~dp0"

if not exist .venv (
    echo Virtual environment not found. Run install_windows.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python main.py
