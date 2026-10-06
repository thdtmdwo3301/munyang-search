#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -x venv/bin/python ]]; then PATTERN_PYTHON="$PWD/venv/bin/python"
elif [[ -x .venv/bin/python ]]; then PATTERN_PYTHON="$PWD/.venv/bin/python"
else echo "가상환경을 활성화하세요." >&2; exit 1; fi
exec "$PATTERN_PYTHON" main.py train "$@"
