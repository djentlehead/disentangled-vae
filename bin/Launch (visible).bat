@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo ERROR: Could not find .venv\Scripts\python.exe
    echo This launcher must stay in the same folder as server\main.py and .venv.
    echo If you deleted/moved .venv, recreate it with:
    echo     python -m venv .venv
    echo     .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo Starting Style Transfer Studio...
echo Your browser will open automatically once the model finishes loading.
echo Close this window (or press Ctrl+C) to stop the server.
echo.

".venv\Scripts\python.exe" server\main.py

pause
