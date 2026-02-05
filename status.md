# Status Report (2026-02-03)

## Confirmed (User-Reported)
- Server: Huawei Atlas 300 (Ascend), Ubuntu, drivers installed.
- Network: SSH reverse tunnel enables outbound access; pip/git proxies are working.
- Models: four core models are downloaded to `/data/models`:
  - ASR (Paraformer)
  - VAD
  - Punctuation
  - Speaker

## Decisions (Current)
- Architecture: Python ASR service + C++ client.
- Scope: ASR only (remove NLP/LLM/Agent modules).
- Interfaces: WebSocket (streaming) and HTTP (file upload).

## Open Questions / Risks
- Python serving framework is not finalized. This skeleton uses FastAPI as a placeholder.
- C++ WebSocket client library is not finalized (websocketpp vs drogon).
- Model path mapping into containers needs verification. Skeleton assumes `/app/data/models`.
- FunASR on Ascend: runtime support and performance need validation in this environment.

## Notes
- The previous Shell-deploy base code was removed to keep the repo clean for a fresh rebuild.
- This repo currently contains only a minimal skeleton and stubs.
