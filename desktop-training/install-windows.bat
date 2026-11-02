@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONDONTWRITEBYTECODE=1
set "PIP_CACHE_DIR=%~dp0workspace\cache\pip"
set "PYTHON_CMD=python"
where python >nul 2>&1
if errorlevel 1 goto try_launcher
python -c "import sys,struct; sys.exit(0 if sys.version_info[:2]==(3,11) and struct.calcsize('P')==8 else 1)" >nul 2>&1
if not errorlevel 1 goto python_ready
:try_launcher
set "PYTHON_CMD=py -3.11"
where py >nul 2>&1
if errorlevel 1 goto python_missing
py -3.11 -c "import sys,struct; sys.exit(0 if sys.version_info[:2]==(3,11) and struct.calcsize('P')==8 else 1)" >nul 2>&1
if errorlevel 1 goto python_missing
:python_ready
%PYTHON_CMD% -c "from core_storage_check import check; check()"
if errorlevel 1 goto fail
if not exist "weights_parts.json" goto weights_ready
%PYTHON_CMD% install_weights.py
if errorlevel 1 goto fail
:weights_ready
if exist ".venv\Scripts\python.exe" goto verify_environment
%PYTHON_CMD% -m venv .venv
if errorlevel 1 goto fail
:verify_environment
".venv\Scripts\python.exe" -c "import sys,struct; sys.exit(0 if sys.version_info[:2]==(3,11) and struct.calcsize('P')==8 else 1)"
if errorlevel 1 goto fail
echo Select the device used for training and inference:
echo 1. CPU - choose this if you do not have an NVIDIA GPU
echo 2. NVIDIA GPU - CUDA 12.6 compatible driver required
choice /c 12 /n /m "Choose 1 or 2: "
if errorlevel 255 goto fail
if not errorlevel 1 goto fail
if errorlevel 2 goto gpu
set "PATTERN_TORCH_VARIANT=cpu"
goto install_packages
:gpu
set "PATTERN_TORCH_VARIANT=cu126"
:install_packages
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check torch==2.8.0+%PATTERN_TORCH_VARIANT% torchvision==0.23.0+%PATTERN_TORCH_VARIANT% --index-url https://download.pytorch.org/whl/%PATTERN_TORCH_VARIANT%
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements-ui.txt -r requirements-engine.txt
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -c "import torch,torchvision,transformers,safetensors,numpy; from PySide6.QtWidgets import QApplication; from PIL import Image; print('Torch:',torch.__version__,'CUDA usable:',torch.cuda.is_available()); assert numpy.__version__.startswith('1.'), 'NumPy 1.x is required'"
if errorlevel 1 goto fail
if "%PATTERN_TORCH_VARIANT%"=="cu126" (
  ".venv\Scripts\python.exe" -c "import torch; assert torch.cuda.is_available(), 'GPU is unavailable. Check the driver or rerun this installer and choose CPU.'; x=torch.ones(2,2,device='cuda:0'); assert (x@x).sum().item()==8; print('GPU:',torch.cuda.get_device_name(0))"
  if errorlevel 1 goto fail
)
echo Screen, training and inference packages are ready.
echo Models are in workspace\models. No manual weight registration is needed for the model packages.
echo Double-click start-windows.bat to open the desktop window.
pause
exit /b 0
:python_missing
echo Python 3.11 64-bit was not found using python or py -3.11.
echo Install Python 3.11 x64 with Add python.exe to PATH, then reopen this installer.
pause
exit /b 1
:fail
echo Setup failed. See the error above. Existing data and weights were not deleted. Rerun this installer after fixing the cause.
pause
exit /b 1
