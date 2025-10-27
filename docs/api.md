## Voice Assistant HTTP API

This document describes the REST and WebSocket endpoints exposed by the MCP/FastAPI service.

### Base URL

Use `http://<host>:9090`. When running locally or via Docker, it is typically `http://localhost:9090`.

### Authentication

No authentication is enabled by default. Place the service behind your own gateway or add middleware (API keys, JWT, etc.) if you deploy it publicly.

---

### `POST /speech_to_text`

Upload an audio file and receive a full transcription.

- **Request**: `multipart/form-data` with field `file` (`.wav`, `.mp3`, etc.).
- **Response**:

```json
{
  "text": "希望你以后能够做得比我还好哟",
  "language": "zh",
  "duration": 3.46,
  "segments": [
    {"text": "希望你以后能够做得比我还好哟", "start": 0.0, "end": 3.44}
  ]
}
```

**Example**

```bash
curl -X POST http://localhost:9090/speech_to_text \
  -F "file=@CosyVoice/asset/zero_shot_prompt.wav"
```

---

### `POST /chat`

Plain text query handled by the deterministic pipeline (RAG + LLM + optional TTS).

- **Request**:

```json
{ "text": "介绍一下这个语音助手的模块划分" }
```

- **Response**:

```json
{
  "text": "系统包含 ASR、NLP、TTS 三个核心模块，并在需要时结合检索结果生成回答。",
  "audio_path": "logs/tts_123456789.wav",
  "citations": [
    {"text": "ASR 模块使用 faster-whisper，并支持 GPU 加速。", "score": 0.73, "metadata": {}}
  ]
}
```

Set `agent.auto_tts=false` in the configuration if you do not want audio files to be produced automatically.

---

### `POST /agent_chat`

LangChain agent endpoint. Only available when `agent.enabled=true` in `config/config.yaml`.

- **Request**:

```json
{ "text": "帮我总结今天的会议", "auto_tts": true }
```

- **Response**: Same structure as `/chat`, plus a `tools` array listing which tools were invoked.

---

### `POST /text_to_speech`

Convert text to speech using the configured TTS provider (CosyVoice or Higgs).

- **Request**:

```json
{ "text": "我们下次会议见。" }
```

- **Response**:

```json
{
  "audio_path": "logs/tts_-709813456.wav",
  "format": "wav",
  "sample_rate": 24000
}
```

---

### Streaming (`/ws/asr`)

The WebSocket interface exposes streaming ASR.

1. Client connects to `ws://<host>:9090/ws/asr`.
2. Server replies with `{"type":"ready","session_id":"<uuid>"}`. Keep the `session_id` for later.
3. Client sends audio chunks (base64-encoded 16-bit PCM) with optional language and sample rate:

```json
{
  "type": "chunk",
  "data": "<base64 bytes>",
  "language": "zh",
  "sample_rate": 16000
}
```

4. When ready to decode, send `{ "type": "flush" }`.
5. Server responds with:
   - `{"type":"metadata","language":"zh","duration":...}`
   - One or more `{"type":"segment","text":"...","start":...,"end":...}`
   - `{"type":"flush_complete"}`
6. Send `{ "type": "reset" }` to clear buffers and delete the session file.

All streaming transcripts are persisted to `logs/asr_sessions/<session_id>.jsonl`. You can retrieve them via

```
GET /asr_session/<session_id>
```

---

### Docker Usage

```bash
docker build -t voice-assistant:latest .
docker run --rm --gpus all \
  -p 9090:9090 \
  -v $(pwd)/config:/app/config \
  voice-assistant:latest
```

Mount additional volumes (e.g., `/app/data/models`) if you need to supply model weights or persistent logs. For GPU deployment, keep the default base image (`pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime`) and pass `--gpus all`.

---

### Memory Subsystem

- Enable memory in `config/config.yaml` (`memory.enabled=true`).
- Short-term memory keeps the latest turns in-process for context injection.
- Long-term memory uses a Mem0 instance. Start it locally (`docker run -p 3030:3030 mem0ai/mem0:latest`) and export `MEM0_API_KEY` so the server can authenticate.
- The assistant summarises each turn via the configured LLM before persisting it to Mem0. Retrieved memories are appended to the chat prompt as `Memory notes`.
