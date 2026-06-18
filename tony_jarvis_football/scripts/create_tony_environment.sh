#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -d "tony_env" ]]; then
  echo "tony_env already exists. Reusing it."
else
  python3 -m venv tony_env
  echo "Created tony_env virtual environment."
fi

source tony_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Tony environment is ready."
echo "Activate with: source tony_env/bin/activate"
echo "Run app with: python run.py"
