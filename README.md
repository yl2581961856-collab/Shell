# Shell

digital soul making

## Voice Assistant Orchestration

A modular voice assistant stack that strings together ASR, retrieval augmented generation, and TTS behind a FastAPI MCP server and optional LangChain agent.

### Features
- Whisper/faster-whisper automatic speech recognition with auto device selection
- Retrieval-augmented generation via FAISS + sentence-transformers
- Pluggable LLM backend (DeepSeek/SiliconFlow, etc.)
- TTS providers (CosyVoice/HiggsAudio)
- Conversation manager for transcripts, citations, audio
- LangChain agent integrating ASR/RAG/TTS tools
- FastAPI MCP server exposing REST & WebSocket endpoints
- Dockerfile for reproducible deployment

### Streaming Workflow
1. Capture audio -> `ws://localhost:9090/ws/asr` (chunk + flush)
2. Persist session under `logs/asr_sessions/<session_id>.jsonl`
3. Query `http://localhost:9090/asr_session/{session_id}` for accumulated transcripts
4. Use `http://localhost:9090/chat`, `/text_to_speech`, or LangChain agent as needed.

See `docs/api.md` for the full API contract and `tools/ws_asr_client.py` for example streaming clients (file & microphone modes).

### ASR Providers
- `asr.provider` supports `faster_whisper` (default) and `funasr` for fully offline Chinese models.
- `faster_whisper` keeps loading from the local directory (`WHISPER_MODEL_PATH`) and forces `local_files_only=True` to avoid outbound traffic.
- `funasr` requires `pip install funasr` and pre-downloaded FunASR assets. Configure paths under `asr.funasr` (model directory, VAD, punctuation model, device, hotwords, etc.). The Python API currently returns full utterances; the WebSocket endpoint will emit results after each flush.

### Memory & Mem0 Integration
- Short-term memory keeps the last few conversation turns in-process (configurable sliding window).
- Long-term memory can stream conversation summaries into a locally hosted [Mem0](https://github.com/mem0ai/mem0) instance.
  1. Launch Mem0 locally, e.g. `docker run -d --name mem0 -p 3030:3030 -e MEM0_API_KEY=local-dev mem0ai/mem0:latest`.
  2. Export the same key before starting the server: `set MEM0_API_KEY=local-dev` (PowerShell) or `export MEM0_API_KEY=local-dev` (bash).
  3. Adjust `memory.long_term` in `config/config.yaml` if your Mem0 endpoint, tags, or summarisation behaviour differ.
- During a turn the assistant summarises the exchange through the configured LLM before persisting the memory. Retrieved notes are injected back into the prompt via `memory_context`.

### 环境变量存放敏感 Key
- 配置文件中的 `api_key` 字段全部改用 `${VAR}` 形式（例如 `${SILICONFLOW_API_KEY}`、`${MEM0_API_KEY}`），实际值通过环境变量注入。
- Windows PowerShell（临时）：`$env:SILICONFLOW_API_KEY="your-key"`；持久化：`setx SILICONFLOW_API_KEY "your-key"`。
- Linux/macOS（bash/zsh）：`export SILICONFLOW_API_KEY=your-key`（可写入 `~/.bashrc`）。
- Docker 运行时可以用 `-e SILICONFLOW_API_KEY=your-key` 注入变量，应用启动后会自动替换 `${VAR}`，避免密钥进入版本库。

### Docker 构建与离线模型
- 构建镜像前，请把已下载的 faster-whisper 模型目录（例如 Hugging Face 缓存中的 `models--Systran--faster-whisper-medium/snapshots/<hash>`）复制到仓库根目录 `data/models/whisper_model/`。
- Dockerfile 会将该目录复制进镜像 `/app/data/models/whisper_model`，并在运行时设置 `WHISPER_MODEL_PATH` 与 `HF_HUB_OFFLINE=1`。
- 若目录缺失或文件不完整（`config.json`、`model.bin` 等），ASR 模块无法加载，请提前准备好完整模型。
