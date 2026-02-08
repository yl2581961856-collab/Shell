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

## Endpoints (Stub)
- `GET /health`
- `POST /asr/file` (multipart file upload)
- `WS /asr/stream`

## Docs
Frontend request examples and protocol details live in `docs/README.md`.

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

