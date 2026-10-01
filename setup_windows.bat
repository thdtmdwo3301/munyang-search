@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Python 3.11이 필요합니다.
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" py -3.11 -m venv .venv
if errorlevel 1 exit /b 1

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
python inference\prepare_weights.py
if errorlevel 1 exit /b 1

echo 설치와 가중치 검증이 완료되었습니다.
exit /b 0
