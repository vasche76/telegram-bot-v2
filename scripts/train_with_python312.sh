#!/bin/bash
# train_with_python312.sh — Run any training script using pyenv Python 3.12
#
# Usage:
#   scripts/train_with_python312.sh scripts/train_stage_a.py --epochs 100 --device mps
#   scripts/train_with_python312.sh scripts/train_stage_b.py --device mps
#   scripts/train_with_python312.sh scripts/build_dataset.py --train --device mps
#
# Reason: PyTorch 2.x is not available for Python 3.13 on macOS x86_64 (Intel).
# This script switches to Python 3.12 (installed via pyenv) for ML tasks.

set -e

PYTHON312="$HOME/.pyenv/versions/3.12.10/bin/python3"

if [ ! -f "$PYTHON312" ]; then
    echo "[FAIL] Python 3.12 not found at $PYTHON312"
    echo "       Install via pyenv:"
    echo "         brew install pyenv"
    echo "         pyenv install 3.12.10"
    echo "         $PYTHON312 -m pip install -r requirements-ml.txt"
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "Usage: $0 <script_path> [args...]"
    echo "Example: $0 scripts/train_stage_a.py --epochs 100 --device mps"
    exit 1
fi

echo "[INFO] Using Python 3.12: $PYTHON312"
echo "[INFO] Running: $*"
exec "$PYTHON312" "$@"
