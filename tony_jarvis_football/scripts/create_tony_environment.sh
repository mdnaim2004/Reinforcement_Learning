#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -d "tony_env" && ! -e "tony_env/bin/activate" && ! -e "tony_env/conda-meta" ]]; then
  rm -rf tony_env
fi

if [[ -d "tony_env" && -x "tony_env/bin/python" ]]; then
  echo "tony_env already exists. Reusing it."
  source tony_env/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  echo "Tony environment is ready (venv mode)."
  echo "Activate with: source tony_env/bin/activate"
  echo "Run app with: python run.py"
  exit 0
fi

if python3 -m venv tony_env 2>/dev/null; then
  echo "Created tony_env virtual environment."
  source tony_env/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  echo "Tony environment is ready (venv mode)."
  echo "Activate with: source tony_env/bin/activate"
  echo "Run app with: python run.py"
  exit 0
fi

echo "python3 venv is unavailable. Falling back to conda environment setup."
if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is not installed. Install python3-venv or conda and rerun this script."
  exit 1
fi

CONDA_ENV_NAME="tony_env"
if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
  echo "Conda environment '$CONDA_ENV_NAME' already exists. Reusing it."
else
  conda create -y -n "$CONDA_ENV_NAME" python=3.10
fi

conda install -y -n "$CONDA_ENV_NAME" -c defaults numpy opencv ultralytics
conda run -n "$CONDA_ENV_NAME" python -m pip install --upgrade pip
conda run -n "$CONDA_ENV_NAME" python -m pip install mediapipe

echo "Tony environment is ready (conda mode)."
echo "Activate with: conda activate $CONDA_ENV_NAME"
echo "Run app with: python run.py"
