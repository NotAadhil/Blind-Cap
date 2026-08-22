@echo off
title Blind Cap - Assistive Vision System
echo ============================================================
echo        BLIND CAP - ACCESSIBLE VISION & 3D AUDIO SYSTEM
echo ============================================================
echo.

:: Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

:: Create Virtual Environment if not present
if not exist ".venv" (
    echo [1/3] Creating virtual environment (.venv)...
    python -m venv .venv
)

:: Activate Virtual Environment
call .venv\Scripts\activate.bat

:: Install Requirements
echo [2/3] Checking dependencies...
pip install -r requirements.txt --quiet --disable-pip-version-check

:: Check and Export OpenVINO model if missing
if not exist "models\yolo26m_480_openvino\yolo26m.xml" (
    echo [3/3] Exporting YOLO26m model for OpenVINO GPU/CPU acceleration...
    python export_model.py
) else (
    echo [3/3] Model ready.
)

echo.
echo ============================================================
echo  Starting Blind Cap in 60 FPS Viewfinder Mode...
echo ============================================================
echo.

python main.py

pause
