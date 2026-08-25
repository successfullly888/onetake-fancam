@echo off
title OneTake - Install AI Runtime
cd /d "%~dp0"

python --version >nul 2>nul
if errorlevel 1 goto :python_missing
if defined ONETAKE_CMD_SMOKE_TEST (
  echo [OK] Installer command file parsed correctly.
  exit /b 0
)

echo [1/2] Installing local AI dependencies...
python -m pip install --disable-pip-version-check --target "%~dp0vendor" -r "%~dp0requirements.txt"
if errorlevel 1 goto :failed

echo [2/2] Preparing the local person detection model...
python "%~dp0scripts\download_models.py"
if errorlevel 1 goto :failed

echo.
echo [OK] AI runtime is ready.
echo You can now close this window and run the start CMD file.
echo.
pause
exit /b 0

:python_missing
echo.
echo [ERROR] Python was not found.
echo Install Python 3.9-3.12 and enable Add Python to PATH.
echo.
pause
exit /b 1

:failed
echo.
echo [ERROR] Installation failed.
echo Check the messages above and your network connection, then run this file again.
echo.
pause
exit /b 1
