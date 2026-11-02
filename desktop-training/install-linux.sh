#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mode="${1:---cpu}"
case "$mode" in --cpu|--gpu) ;; *) echo 'Usage: bash install-linux.sh [--cpu|--gpu]' >&2; exit 2;; esac
python_cmd="${PATTERN_PYTHON:-python3}"
"$python_cmd" -c "import sys,struct; assert sys.version_info[:2]==(3,11) and struct.calcsize('P')==8, 'Python 3.11 64-bit is required'; from core_storage_check import check; check()"
if [[ -f weights_parts.json ]]; then "$python_cmd" install_weights.py; fi
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
export PIP_CACHE_DIR="$PWD/workspace/cache/pip"
export TMPDIR="$PWD/workspace/cache/tmp"
export CUDA_CACHE_PATH="$PWD/workspace/cache/cuda"
mkdir -p -- "$PIP_CACHE_DIR" "$TMPDIR" "$CUDA_CACHE_PATH"
if [[ ! -e .venv-linux ]]; then "$python_cmd" -m venv .venv-linux; fi
.venv-linux/bin/python -c "import sys; assert sys.version_info[:2]==(3,11), 'Existing Linux venv is not Python 3.11'"
variant='cpu'
if [[ "$mode" == --gpu ]]; then variant='cu126'; fi
.venv-linux/bin/python -m pip install --disable-pip-version-check --progress-bar off "torch==2.8.0+$variant" "torchvision==0.23.0+$variant" --index-url "https://download.pytorch.org/whl/$variant"
.venv-linux/bin/python -m pip install --disable-pip-version-check --progress-bar off -r requirements-ui.txt -r requirements-engine.txt
.venv-linux/bin/python -c "import torch; from PySide6.QtWidgets import QApplication; import transformers,safetensors; print('Torch:',torch.__version__,'Transformers:',transformers.__version__,'CUDA:',torch.cuda.is_available())"
if [[ "$mode" == --gpu ]]; then
  .venv-linux/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable; inspect driver and container GPU settings'"
fi
echo 'Installation complete. Run: bash start-linux.sh'
