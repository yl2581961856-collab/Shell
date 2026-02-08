#!/usr/bin/env bash
set -euo pipefail

# Minimal launcher for the ASR FastAPI+WebSocket server on Python 3.11.
#
# Env vars:
# - PYTHON_BIN: python executable (default: python3.11)
# - ASR_HOST / ASR_PORT: bind address (default: 0.0.0.0:6008)
# - UVICORN_WORKERS: uvicorn worker processes (default: 1)
# - UVICORN_LOG_LEVEL: debug/info/warning/error (default: info)
# - ASR_MAX_WORKERS: inference executor threads per process (default: 1)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
ASR_HOST="${ASR_HOST:-0.0.0.0}"
ASR_PORT="${ASR_PORT:-6008}"
UVICORN_WORKERS="${UVICORN_WORKERS:-1}"
UVICORN_LOG_LEVEL="${UVICORN_LOG_LEVEL:-info}"

export ASR_MAX_WORKERS="${ASR_MAX_WORKERS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

exec "${PYTHON_BIN}" -m uvicorn src.server:app \
  --host "${ASR_HOST}" \
  --port "${ASR_PORT}" \
  --log-level "${UVICORN_LOG_LEVEL}" \
  --workers "${UVICORN_WORKERS}"

