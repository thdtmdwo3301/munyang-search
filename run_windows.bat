@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo 가상환경이 없습니다. 먼저 setup_windows.bat을 실행하세요.
  exit /b 1
)

call ".venv\Scripts\activate.bat"
python inference\prepare_weights.py
if errorlevel 1 exit /b 1
python -m evaluation.generate_report --open %*
exit /b %errorlevel%
