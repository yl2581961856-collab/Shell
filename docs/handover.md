# Handover - SenseVoice WS (Pseudo-Streaming)

This document describes how to run the SenseVoice backend over WebSocket in the current repo, plus
how to integrate with a frontend. The port is configurable; `6008` is only the current default.

## 1) Directory Layout (Server)
- Repo: `/app/data/asr/AscendBridge-asr-for-ascend`
- SenseVoice source: `/app/data/asr/SenseVoice`
- SenseVoice model: `/app/data/models/SenseVoiceSmall`

You can change these, but the env vars must match.

## 2) Config
Update `config/config.yaml` (server side):
- `model_paths.asr: /app/data/models/SenseVoiceSmall`
- `model_paths.vad: ""`
- `model_paths.punc: ""`
- `model_paths.spk: ""`

Note: SenseVoice backend does not use FunASR VAD/PUNC.

## 3) Start (SenseVoice Pseudo-Streaming)
Preferred:
```bash
bash shell/start_sensevoice.sh
```

`shell/start_sensevoice.sh` sets the important env vars and starts the server:
- `ASR_BACKEND=sensevoice`
- `SENSEVOICE_CODE_PATH=/app/data/asr/SenseVoice`
- `ASR_STREAM_MODE=pseudo`
- `ASR_PSEUDO_STEP_MS=1200` (adjust for latency vs stability)
- `ASR_PSEUDO_MAX_MS=8000` (adjust for context size)
- `ASR_SIMPLE_VAD=1`, `ASR_VAD_MIN_RMS=0.01`, `ASR_VAD_END_MS=1200`, `ASR_VAD_AUTO_FINAL=1`
- `ASR_HOST=127.0.0.1`
- `ASR_PORT=6008`

Override any of these by exporting env vars before running the script.

## 4) Port Customization
The port is not fixed. Set it via env:
```bash
ASR_PORT=7008 ASR_HOST=0.0.0.0 bash shell/start_sensevoice.sh
```
or directly with `shell/start_server.sh` and your own env vars.

## 5) Warmup
The first inference is slow because the model loads and compiles. Send a short audio once after start
to warm up before live testing.

## 6) Local Access via SSH Port Forward
On local machine:
```bash
ssh -N -L 6008:127.0.0.1:6008 root@connect.gda1.seetacloud.com -p 21926
```
Then access:
- HTTP: `http://127.0.0.1:6008/health`
- WS: `ws://127.0.0.1:6008/asr/stream`

If the server port changes, adjust the port in `-L`.

## 7) WS Client (Mic)
Run on local (recommended):
```bash
python scripts/ws_mic_client.py --uri ws://127.0.0.1:6008/asr/stream --chunk-ms 1200
```

## 8) WS Message Contract (Frontend)
Server messages:
- `ack`: session accepted
- `partial`: intermediate result
- `final`: end of a segment (pseudo-streaming)
- `error`

Common fields:
- `text`, `session_id`, `seq`, `bytes`, `ts`, `stub`
- `segment_id` (added for pseudo-streaming segmentation)

Frontend behavior:
- Use only the **latest** `partial` for display (do not concatenate).
- On `final`, commit the text and clear the partial buffer.
- `text` includes `<|...|>` tokens; frontend can strip or interpret them.

## 9) Pseudo-Streaming Behavior
SenseVoice is an LLM-style model. We do not use FunASR streaming cache.
Instead, each chunk triggers a full-buffer decode (pseudo-streaming).

To avoid infinite accumulation, the server:
- Auto-finalizes on silence (VAD) and clears buffers.
- Forces a maximum utterance length via `ASR_UTT_MAX_MS` (default `30000` in pseudo mode).

Tune these if needed.

## 10) Troubleshooting
- `... is not registered`: SenseVoice must use source backend, not FunASR AutoModel.
  Ensure `ASR_BACKEND=sensevoice` and `SENSEVOICE_CODE_PATH` is correct.
- Slow first response: expected, warm up after start.
- Too many partials: increase `ASR_PSEUDO_STEP_MS` or reduce `--chunk-ms`.
- Latency too high: lower `ASR_PSEUDO_STEP_MS` and `--chunk-ms`.
- Premature sentence end: check `ASR_VAD_AUTO_FINAL` and `ASR_VAD_END_MS`.
