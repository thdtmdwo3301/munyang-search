#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv-linux/bin/python ]]; then echo 'Run bash install-linux.sh --gpu (or --cpu) first.' >&2; exit 1; fi
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$PWD/workspace/cache/huggingface" TORCH_HOME="$PWD/workspace/cache/torch"
export CUDA_CACHE_PATH="$PWD/workspace/cache/cuda" TMPDIR="$PWD/workspace/cache/tmp"
export TRITON_CACHE_DIR="$PWD/workspace/cache/triton" XDG_CACHE_HOME="$PWD/workspace/cache/xdg"
export XDG_RUNTIME_DIR="$PWD/workspace/cache/qt-runtime"
.venv-linux/bin/python core_storage_check.py
mkdir -p -- "$TMPDIR" "$CUDA_CACHE_PATH" "$XDG_RUNTIME_DIR"
chmod 700 -- "$XDG_RUNTIME_DIR"
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" && "${QT_QPA_PLATFORM:-}" != offscreen ]]; then
  echo 'No graphical display is connected. Use a Linux desktop/X11 session; see README for headless tests.' >&2
  exit 1
fi
exec .venv-linux/bin/python desktop.py
