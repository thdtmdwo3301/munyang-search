#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv-linux/bin/python ]]; then echo 'Linux environment missing: bash install-linux.sh --gpu' >&2; exit 1; fi
export PYTHONDONTWRITEBYTECODE=1 CUDA_CACHE_DISABLE=1
.venv-linux/bin/python - <<'PY'
import os,sys,torch,transformers,safetensors,numpy
from core_storage_check import check
from PySide6.QtCore import __version__ as qt_version
check()
print('Python:',sys.version.split()[0])
print('Torch:',torch.__version__,'CUDA build:',torch.version.cuda,'CUDA usable:',torch.cuda.is_available())
print('Transformers:',transformers.__version__,'NumPy:',numpy.__version__,'Qt:',qt_version)
print('Graphical display configured:',bool(os.getenv('DISPLAY') or os.getenv('WAYLAND_DISPLAY')))
if torch.cuda.is_available():
    x=torch.ones((2,2),device='cuda:0');assert (x@x).sum().item()==8
    print('GPU:',torch.cuda.get_device_name(0),'CUDA operation: passed')
PY
