@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo %date% %time% - ERROR: .venv\Scripts\python.exe not found. >> launch.log
    exit /b 1
)

echo %date% %time% - Launching server\main.py >> launch.log
".venv\Scripts\python.exe" server\main.py >> launch.log 2>&1
