@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  call setup_windows.bat
  if errorlevel 1 goto :failed
)

call run_windows.bat
if errorlevel 1 goto :failed

echo.
echo 평가가 완료되었습니다.
pause
exit /b 0

:failed
echo.
echo 실행 중 오류가 발생했습니다. 위 메시지를 확인하세요.
pause
exit /b 1
