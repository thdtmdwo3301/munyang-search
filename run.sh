#!/usr/bin/env bash
set -euo pipefail
PATTERN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PATTERN_ROOT"
if [[ -x "$PATTERN_ROOT/venv/bin/python" ]]; then
  PATTERN_PYTHON="$PATTERN_ROOT/venv/bin/python"
elif [[ -x "$PATTERN_ROOT/.venv/bin/python" ]]; then
  PATTERN_PYTHON="$PATTERN_ROOT/.venv/bin/python"
else
  echo "먼저 Python 3.11 가상환경을 만들고 requirements.txt를 설치하세요." >&2
  exit 1
fi
export PYTHONNOUSERSITE=1
exec "$PATTERN_PYTHON" main.py evaluate "$@"
