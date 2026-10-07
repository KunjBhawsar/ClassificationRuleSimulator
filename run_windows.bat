@echo off
setlocal
cd /d "%~dp0"

python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Dependency installation failed.
    pause
    exit /b 1
)

set "PORT=5000"
echo.
echo Starting Classification Rule Simulator on http://127.0.0.1:%PORT%
echo If port %PORT% is unavailable, close other local servers and run this file again.
echo.
python backend\app.py

if errorlevel 1 (
    echo.
    echo The server could not start. Port %PORT% may be blocked or already in use.
    echo You can try another port by running: set PORT=5001 ^&^& python backend\app.py
)
pause
