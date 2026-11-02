@echo off
setlocal
cd /d "%~dp0"
echo Installation is now unified in install-windows.bat.
call "%~dp0install-windows.bat"
exit /b %errorlevel%
