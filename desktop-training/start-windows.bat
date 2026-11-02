@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run install-windows.bat first.
  pause
  exit /b 1
)
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
".venv\Scripts\python.exe" -c "from PySide6.QtWidgets import QApplication; from PIL import Image; from core_storage_check import check; check()"
if errorlevel 1 (
  echo Desktop environment is missing or this folder is not local. Run install-windows.bat and retry.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "%~dp0desktop.py"
exit /b 0
