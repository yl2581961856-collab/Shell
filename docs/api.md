## Voice Assistant HTTP API

This service exposes a small REST surface area for integrating ASR, LLM, and TTS
capabilities with external platforms (e.g., Java backends).  All endpoints 
return JSON responses and are designed to be fronted by FastAPI.

### Base URL

The container publishes the API on `http://<host>:9000`. When running locally,
that is typically `http://localhost:9000`.

### Authentication

No authentication is enabled by default. If you deploy the service publicly,
place it behind your existing gateway or add middleware (e.g., API keys, JWT).

---

### `POST /speech_to_text`

Transcribe an audio file to text using the configured Whisper model
(`faster-whisper` under the hood).

- **Request**: `multipart/form-data` with field `file` (`.wav`, `.mp3`, etc.).
- **Response**:

```json
{
  "text": "甯屾湜浣犱互鍚庤兘澶熷仛寰楁瘮鎴戣繕濂藉摕",
  "language": "zh",
  "duration": 3.46,
  "segments": [
    {"text": "甯屾湜浣犱互鍚庤兘澶熷仛寰楁瘮鎴戣繕濂藉摕", "start": 0.0, "end": 3.44}
  ]
}
```

**cURL**

```bash
curl -X POST http://localhost:9000/speech_to_text \
  -F "file=@CosyVoice/asset/zero_shot_prompt.wav"
```

---

### `POST /chat`

Send a text query. The backend will run retrieval-augmented generation and 
optionally synthesize an answer with TTS.

- **Request**:

```json
{ "text": "浠嬬粛涓€涓嬭繖涓闊冲姪鎵嬬殑妯″潡鍒掑垎" }
```

- **Response**:

```json
{
  "text": "绯荤粺鍖呭惈 ASR銆丯LP銆乀TS 涓変釜鏍稿績妯″潡鈥︹€?,
  "audio_path": "logs/tts_123456789.wav",
  "citations": [
    {"text": "ASR 妯″潡浣跨敤 faster-whisper鈥︹€?, "score": 0.73, "metadata": {}}
  ]
}
```

Set `agent.auto_tts=false` in configuration if you do not want audio assets
generated automatically.

---

### `POST /agent_chat`

Optional LangChain agent endpoint. Only available when 
`agent.enabled=true` in `config/config.yaml`.

- **Request**:

```json
{ "text": "甯垜鎬荤粨浠婂ぉ鐨勪細璁?, "auto_tts": true }
```
- **Response**: similar to `/chat`, with extra field `tools` listing any tools
invoked during the agent run.

---

### `POST /text_to_speech`

Synthesize speech using the configured TTS provider (CosyVoice or Higgs).

- **Request**:

```json
{ "text": "鎴戜滑涓嬫浼氳瑙併€? }
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

### Streaming Considerations

The new `ASRModule.transcribe_stream()` API exposes a generator that yields
`ASRSegment` objects incrementally. The WebSocket interface at `/ws/asr`
persists each session to `logs/asr_sessions/<session_id>.jsonl`. The protocol:

- Upon connection, the server responds with `{"type":"ready","session_id":"<uuid>"}`.
- Send audio chunks (base64-encoded PCM/WAV bytes; include sample_rate when not using the default 16 kHz):

  ```json
  { "type": "chunk", "data": "<base64 bytes>", "language": "zh", "sample_rate": 16000 }
  ```

- Trigger decoding for buffered audio:

  ```json
  { "type": "flush" }
  ```

- The server replies with metadata, multiple segments, and a completion notice:
  - `{"type":"metadata","language":"zh","duration":3.4}`
  - `{"type":"segment","text":"...","start":0.0,"end":1.5}`
  - `{"type":"flush_complete"}`

- Send `{"type":"reset"}` to clear buffers and delete the session file.

Session history can be retrieved via `GET /asr_session/{session_id}` which returns all recorded entries:

```json
{
  "session_id": "0f0b05961e0a4ad6b9049b1db4f9a60e",
  "entries": [
    {
      "timestamp": 1734713514.123,
      "language": "zh",
      "duration": 3.4,
      "segments": [
        {"text": "...", "start": 0.0, "end": 1.5},
        {"text": "...", "start": 1.5, "end": 3.4}
      ]
    }
  ]
}
```

For lower latency, reduce `chunk` size or apply a client-side sliding window.
HTTP REST endpoints remain batch-oriented.

---

### Docker Usage

```bash
docker build -t voice-assistant:latest .
docker run --rm --gpus all \
  -p 9000:9000 \
  -v $(pwd)/config:/app/config \
  voice-assistant:latest
```

Mount additional volumes (e.g., `/app/models`) if you need to supply model
weights or persistent logs. For GPU deployment, switch the base image and
install the CUDA-enabled PyTorch wheels, then start the container with
`--gpus all`.锛堥粯璁?Dockerfile 宸插熀浜?`pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime`锛夈€?




