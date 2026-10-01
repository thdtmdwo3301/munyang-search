#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN=python3.11
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
else
  echo "Python 3.11이 필요합니다."
  exit 1
fi

if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python inference/prepare_weights.py

echo "설치와 가중치 검증이 완료되었습니다. 이제 bash run.sh를 실행하세요."
