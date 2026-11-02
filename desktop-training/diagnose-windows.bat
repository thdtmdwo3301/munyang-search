@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
if not exist ".venv\Scripts\python.exe" (
  echo Run install-windows.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" desktop.py
pause
