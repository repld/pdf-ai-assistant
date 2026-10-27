@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo The virtual environment was not found.
    echo Run these commands once from this project folder:
    echo.
    echo   python -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -e ".[dev,web,ui]"
    echo.
    pause
    exit /b 1
)

if not exist ".env" (
    echo The .env file was not found.
    echo Copy .env.example to .env and add your API key.
    echo.
    pause
    exit /b 1
)

start "AI Document Assistant" powershell.exe -NoLogo -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "%~dp0start-assistant.ps1"
exit /b 0
