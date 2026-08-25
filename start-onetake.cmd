@echo off
title OneTake - Local AI Service
cd /d "%~dp0"

python --version >nul 2>nul
if errorlevel 1 goto :python_missing

if not exist "%~dp0vendor\cv2\__init__.py" goto :runtime_missing
if not exist "%~dp0models\yolov3-tiny.weights" goto :runtime_missing
if defined ONETAKE_CMD_SMOKE_TEST (
  echo [OK] Launcher command file parsed correctly.
  exit /b 0
)

echo Starting the newest OneTake build...
echo If an older build is using port 4173, this launcher will choose a free port.
echo Keep this window open while using the product.
echo Press Ctrl+C to stop the service.
echo.
python "%~dp0scripts\start_onetake.py"
if errorlevel 1 goto :start_failed
exit /b 0

:python_missing
echo.
echo [ERROR] Python was not found.
echo Install Python 3.9-3.12 and enable Add Python to PATH.
echo.
pause
exit /b 1

:runtime_missing
echo.
echo [SETUP REQUIRED] The local AI runtime is incomplete.
echo Run the install CMD file in this folder first.
echo.
pause
exit /b 1

:start_failed
echo.
echo [ERROR] The local AI service could not start.
echo If port 4173 is already in use, close the previous OneTake command window.
echo.
pause
exit /b 1
