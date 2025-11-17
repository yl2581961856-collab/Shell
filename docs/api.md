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
{
  "mode": "meeting",
  "session_id": "e52702fea91740169e49355119f550b0",
  "text": "请整理本次例会的讨论要点和行动项"
}
```

- `text` *(string, required)* – user input or transcript to summarise.
- `mode` *(string, optional, defaults to `normal`)* – send `meeting` to trigger the note-taking persona (otherwise `normal`).
- `session_id` *(string, required when `mode=meeting`)* – reuse the streaming ASR session id for traceability/log correlation.
- `system_prompt` / `preset` *(strings, optional)* – legacy overrides that still work but are no longer necessary.

- **Response** (meeting mode always returns JSON):

```json
{
  "key_points": [
    "语音采集模块需要补齐降噪配置",
    "Q3 版本主打会议纪要体验"
  ],
  "action_items": [
    {"task": "整理降噪选型报告", "owner": "Alice", "due": "2024-08-15"},
    {"task": "准备会议纪要模板 Demo", "owner": "Bob", "due": null}
  ]
}
```

Set `agent.auto_tts=false` in the configuration if you do not want audio files to be produced automatically.  Meeting-style summaries simply pass `mode`: "meeting" (plus `session_id`) instead of constructing custom prompts.

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

#### LLM Streaming (shared WebSocket)

同一条 `ws://<host>:9090/ws/asr` 连接也承载 LLM 流式输出，所有消息都是 JSON 文本帧并通过 `type` 字段区分。典型流程：

1. 客户端发送 `llm_request`：

```json
{
  "type": "llm_request",
  "session_id": "e52702fea91740169e49355119f550b0",   // meeting 模式必填
  "request_id": "req-001",                              // 前端自生成，唯一
  "mode": "meeting",                                    // normal / meeting
  "text": "请生成本次会议纪要（重点和行动项）",
  "extra": { "temperature": 0.7, "max_tokens": 512 }
}
```

2. 服务端开始推送 `llm_delta`（类似打字机效果）：

```json
{
  "type": "llm_delta",
  "session_id": "e52702fea91740169e49355119f550b0",
  "request_id": "req-001",
  "delta": "\"key_points\": [\"语音采集模块需...",
  "index": 0
}
```

3. 所有增量发送完成后，服务端发送 `llm_done`：

```json
{
  "type": "llm_done",
  "session_id": "e52702fea91740169e49355119f550b0",
  "request_id": "req-001",
  "reason": "completed",             // completed | cancel | error
  "final_text": "{\"key_points\": ... }"
}
```

4. 如果用户点击“停止”，发送：

```json
{ "type": "llm_cancel", "session_id": "e52702fea91740169e49355119f550b0", "request_id": "req-001" }
```

收到后服务端会立刻广播 `llm_done (reason="cancel")`。当请求参数缺失或同一 `session_id` 已有活跃生成时，会返回 `llm_error`（包含 `code`/`message`）。

注意事项：

- `request_id` 完成整个生命周期：`llm_request → llm_delta* → llm_done`。无需 delta 的情况会直接 `llm_done`。
- meeting 模式必须携带 `session_id`，用来关联到对应的语音会话；普通模式可选。
- `extra` 仅接受白名单参数（`temperature`、`top_p`、`max_tokens`）。
- 每个 `session_id` 同时只允许一个活跃的生成；新的请求会收到 `llm_error (code="busy")`，可先 `llm_cancel` 再发起。

---

### Docker Usage

```bash
docker build -t voice-assistant:latest .
docker run --rm --gpus all \
  -p 9090:9090 \
  -v $(pwd)/config:/app/config \
  voice-assistant:latest
```

Mount additional volumes (e.g., `/app/models`) if you need to supply model weights or persistent logs. For GPU deployment, keep the default base image (`pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime`) and pass `--gpus all`.

---

### Memory Subsystem

- Enable memory in `config/config.yaml` (`memory.enabled=true`).
- Short-term memory keeps the latest turns in-process for context injection.
- Long-term memory uses a Mem0 instance. Start it locally (`docker run -p 3030:3030 mem0ai/mem0:latest`) and export `MEM0_API_KEY` so the server can authenticate.
- The assistant summarises each turn via the configured LLM before persisting it to Mem0. Retrieved memories are appended to the chat prompt as `Memory notes`.
