# ASR Ascend Service (Skeleton)

This repository is intentionally minimal. It provides a clean starting point for an ASR service on Ascend hardware with HTTP and WebSocket interfaces.

## Purpose
- Python: ASR service (model loading + inference + HTTP/WebSocket API)
- C++: business client (streaming audio, receiving partial/final results)

## Architecture (Current Plan)
- Service: Python ASR core only (no NLP/LLM/Agent modules)
- Protocols: HTTP for file-based ASR, WebSocket for streaming ASR
- Client: C++ WebSocket client (library TBD)

## Quick Start (Dev)
1. Create a virtualenv and install deps:
   - `python -m venv .venv`
   - `./.venv/Scripts/activate`
   - `pip install -r requirements.txt`
2. Run the server:
   - `python -m src.server`
3. Health check:
   - `GET http://server-ip:6008/health`

## SenseVoice WS (Pseudo-Streaming)
When using SenseVoice source backend, start the server with these env vars:
- `ASR_BACKEND=sensevoice`
- `SENSEVOICE_CODE_PATH=/app/data/asr/SenseVoice`
- `ASR_STREAM_MODE=pseudo`
- Optional: `ASR_PSEUDO_STEP_MS=1200`, `ASR_PSEUDO_MAX_MS=8000`
- Optional VAD: `ASR_SIMPLE_VAD=1`, `ASR_VAD_MIN_RMS=0.01`, `ASR_VAD_END_MS=1200`, `ASR_VAD_AUTO_FINAL=1`

Example:
```bash
bash shell/start_sensevoice.sh
```

## Paraformer WS (FunASR)
Run Paraformer via FunASR backend:
```bash
bash shell/start_paraformer.sh
```

Warmup note: the first inference triggers model load and can be slow. Send a short audio request to warm up before live testing.

## Endpoints (Stub)
- `GET /health`
- `POST /asr/file` (multipart file upload)
- `WS /asr/stream`

## Docs
Frontend request examples and protocol details live in `docs/README.md`.
Handover notes for SenseVoice pseudo-streaming live in `docs/handover.md`.

## Config
Model paths and server settings live in `config/config.yaml`.
Default model path root is `/app/data/models`. Update this if your container mounts a different path.

## Status
See `status.md` for the latest status report and open questions.

## Project Layout
- `src/` Python service code
- `config/` YAML config
- `client/` C++ client placeholder
- `docs/` docs and notes
- `scripts/` helper scripts

