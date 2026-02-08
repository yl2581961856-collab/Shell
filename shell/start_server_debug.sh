#!/usr/bin/env bash
set -euo pipefail

# Start server with verbose logs (override via ASR_LOG_LEVEL/UVICORN_LOG_LEVEL).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ASR_LOG_LEVEL="${ASR_LOG_LEVEL:-DEBUG}"
export UVICORN_LOG_LEVEL="${UVICORN_LOG_LEVEL:-debug}"

exec "${SCRIPT_DIR}/start_server.sh"
