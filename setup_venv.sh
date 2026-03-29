#!/usr/bin/env bash
set -euo pipefail

if [ -d ".venv" ]; then
  rm -rf ".venv"
fi

PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "ERROR: python3/python not found on PATH"
  exit 1
fi

"$PYTHON_BIN" -m venv .venv

VENV_PYTHON=".venv/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
  echo "ERROR: venv python not found at $VENV_PYTHON"
  exit 1
fi

"$VENV_PYTHON" -m ensurepip --upgrade
"$VENV_PYTHON" -m pip install --upgrade --force-reinstall pip setuptools wheel
"$VENV_PYTHON" -m pip install -e .

echo "OK: venv created and scaffold installed (pip repaired and upgraded)."
