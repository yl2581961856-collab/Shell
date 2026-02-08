#!/usr/bin/env bash
set -euo pipefail

# Run pseudo-streaming eval for SenseVoice-small and Fun-ASR-Nano.
#
# Required env:
# - SENSEVOICE_MODEL
# - FUNASR_NANO_MODEL
#
# Optional env:
# - SENSEVOICE_DEVICE (default: npu:0)
# - FUNASR_NANO_DEVICE (default: npu:0)
# - PYTHON_BIN (default: python3)
# - CHUNK_MS (default: 1200)
# - UPDATE_EVERY (default: 1)
# - DEVICE (default: npu:0) used if per-model device env is unset.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CHUNK_MS="${CHUNK_MS:-1200}"
UPDATE_EVERY="${UPDATE_EVERY:-1}"
OUTDIR="./eval_logs"
INPUT=""
INPUT_FORMAT=""
SR=""
CH=""
PLAY=0
REALTIME=0

usage() {
  cat <<'EOF'
Usage:
  ./shell/run_model_eval.sh --input /path/to/audio.wav [--chunk-ms 1200] [--update-every 1] [--outdir ./eval_logs]
  ./shell/run_model_eval.sh --input /path/to/audio.pcm --input-format pcm --sr 16000 --ch 1

Env:
  SENSEVOICE_MODEL (required)
  FUNASR_NANO_MODEL (required)
  SENSEVOICE_DEVICE / FUNASR_NANO_DEVICE (optional, default npu:0)
  PYTHON_BIN (optional, default python3)
  Optional flags: --play --realtime
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT="$2"
      shift 2
      ;;
    --outdir)
      OUTDIR="$2"
      shift 2
      ;;
    --chunk-ms)
      CHUNK_MS="$2"
      shift 2
      ;;
    --update-every)
      UPDATE_EVERY="$2"
      shift 2
      ;;
    --input-format)
      INPUT_FORMAT="$2"
      shift 2
      ;;
    --sr)
      SR="$2"
      shift 2
      ;;
    --ch)
      CH="$2"
      shift 2
      ;;
    --play)
      PLAY=1
      shift 1
      ;;
    --realtime)
      REALTIME=1
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${INPUT}" ]]; then
  echo "Missing --input" >&2
  usage
  exit 2
fi

if [[ -z "${SENSEVOICE_MODEL:-}" ]]; then
  echo "Missing SENSEVOICE_MODEL" >&2
  exit 2
fi

if [[ -z "${FUNASR_NANO_MODEL:-}" ]]; then
  echo "Missing FUNASR_NANO_MODEL" >&2
  exit 2
fi

mkdir -p "${OUTDIR}"

ts="$(date +%Y%m%d_%H%M%S)"
sv_out="${OUTDIR}/sensevoice_small_${ts}.jsonl"
nano_out="${OUTDIR}/funasr_nano_${ts}.jsonl"

extra_args=(--input "${INPUT}" --chunk-ms "${CHUNK_MS}" --update-every "${UPDATE_EVERY}")
if [[ -n "${INPUT_FORMAT}" ]]; then
  extra_args+=(--input-format "${INPUT_FORMAT}")
fi
if [[ -n "${SR}" ]]; then
  extra_args+=(--sr "${SR}")
fi
if [[ -n "${CH}" ]]; then
  extra_args+=(--ch "${CH}")
fi
if [[ "${PLAY}" -eq 1 ]]; then
  extra_args+=(--play)
fi
if [[ "${REALTIME}" -eq 1 ]]; then
  extra_args+=(--realtime)
fi

echo "[SenseVoice] output -> ${sv_out}"
SENSEVOICE_DEVICE="${SENSEVOICE_DEVICE:-${DEVICE:-npu:0}}" \
  "${PYTHON_BIN}" scripts/eval_sensevoice_small.py "${extra_args[@]}" | tee "${sv_out}"

echo "[Nano] output -> ${nano_out}"
FUNASR_NANO_DEVICE="${FUNASR_NANO_DEVICE:-${DEVICE:-npu:0}}" \
  "${PYTHON_BIN}" scripts/eval_fun_asr_nano.py "${extra_args[@]}" | tee "${nano_out}"
