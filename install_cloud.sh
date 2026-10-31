#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PWD/.cache/pip}"
python -c 'import sys; assert sys.version_info >= (3,12), "Choose a Python 3.12 or 3.13 image"'
if ! python -c 'import torch, torchvision; assert torch.__version__.split("+")[0] == "2.8.0"; assert torchvision.__version__.split("+")[0] == "0.23.0"; assert torch.cuda.is_available()'; then
  python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
fi
python -m pip install -r requirements-cloud.txt
python -m pip install --ignore-installed --no-deps dill==0.3.8
python scripts/prepare_autodl.py prepare --verify
python scripts/verify_cuda.py
python scripts/train_expanded_smoke.py --dry-run
printf '\nSetup complete. No training has been started.\n'
