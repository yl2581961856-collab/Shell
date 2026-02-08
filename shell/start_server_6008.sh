#!/usr/bin/env bash
set -euo pipefail

# Start server on 0.0.0.0:6008 (override via ASR_HOST/ASR_PORT).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ASR_HOST="${ASR_HOST:-0.0.0.0}"
export ASR_PORT="${ASR_PORT:-6008}"

exec "${SCRIPT_DIR}/start_server.sh"
