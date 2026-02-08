#!/usr/bin/env bash
set -euo pipefail

# Start server with SenseVoice source backend (pseudo-streaming).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ASR_BACKEND="sensevoice"
export SENSEVOICE_CODE_PATH="/app/data/asr/SenseVoice"
export ASR_STREAM_MODE="pseudo"

# Pseudo-streaming tuning (override as needed)
export ASR_PSEUDO_STEP_MS="1200"
export ASR_PSEUDO_MAX_MS="8000"

# Simple VAD for pseudo-streaming (override as needed)
export ASR_SIMPLE_VAD="1"
export ASR_VAD_MIN_RMS="0.003"
export ASR_VAD_END_MS="2000"
export ASR_VAD_AUTO_FINAL="1"

# Server bind (override as needed)
export ASR_HOST="127.0.0.1"
export ASR_PORT="6008"

exec "${SCRIPT_DIR}/start_server.sh"
