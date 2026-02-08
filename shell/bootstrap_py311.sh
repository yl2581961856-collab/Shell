#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a local venv on Python 3.11 and start the server.
#
# Env vars:
# - PYTHON_BIN: python executable (default: python3.11)
# - VENV_DIR: venv path (default: .venv)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv}"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt

export PYTHON_BIN="python"
exec "${SCRIPT_DIR}/start_server.sh"

