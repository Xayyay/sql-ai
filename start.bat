@echo off
title SQL AI Summarizer

echo ============================================
echo   SQL AI Summarizer - Starting...
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Download Python from https://python.org
    pause
    exit /b 1
)

REM Install dependencies if needed
echo Checking dependencies...
pip show fastapi >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install dependencies.
        pause
        exit /b 1
    )
)

echo.
echo Starting server...
echo Open your browser at: http://localhost:8000
echo Press Ctrl+C to stop.
echo.

REM Start the app
python app.py

pause
