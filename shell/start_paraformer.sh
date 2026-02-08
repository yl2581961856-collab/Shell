#!/usr/bin/env bash
set -euo pipefail

# Start server with FunASR/Paraformer backend.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Force FunASR backend
export ASR_BACKEND="funasr"

# Disable pseudo-streaming controls (use FunASR streaming cache)
unset ASR_STREAM_MODE
unset ASR_PSEUDO_STEP_MS
unset ASR_PSEUDO_MAX_MS

# Disable simple VAD (Paraformer can use its own VAD via config if needed)
unset ASR_SIMPLE_VAD
unset ASR_VAD_MIN_RMS
unset ASR_VAD_END_MS
unset ASR_VAD_AUTO_FINAL

# Server bind (override as needed)
export ASR_HOST="${ASR_HOST:-0.0.0.0}"
export ASR_PORT="${ASR_PORT:-6008}"

exec "${SCRIPT_DIR}/start_server.sh"
