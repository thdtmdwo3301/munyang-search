#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PATTERN_PYTHON="${PYTHON_BIN:-python3}"
"$PATTERN_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,11), "Python 3.11 is required"'
if [[ ! -d venv ]]; then "$PATTERN_PYTHON" -m venv venv; fi
mkdir -p .cache/pip .cache/tmp
export PIP_CACHE_DIR="$PWD/.cache/pip" TMPDIR="$PWD/.cache/tmp"
venv/bin/python -m pip install -r requirements.txt
venv/bin/python -m pip check
