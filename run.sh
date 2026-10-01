#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f .venv/bin/activate ]; then
  echo "가상환경이 없습니다. 먼저 bash setup.sh를 실행하세요."
  exit 1
fi

source .venv/bin/activate
python inference/prepare_weights.py
python -m evaluation.generate_report --open "$@"
