# Shell

digital soul making

## Voice Assistant Orchestration

A modular voice assistant stack that strings together ASR (OpenAI Whisper/faster-whisper), retrieval augmented generation, and TTS behind a FastAPI MCP server and optional LangChain agent.

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

See `docs/api.md` for full API contract and `tools/ws_asr_client.py` for example streaming clients (file & microphone modes).

### Memory & Mem0 Integration
- Short-term memory keeps the last few conversation turns in-process (configurable sliding window).
- Long-term memory can stream conversation summaries into a locally hosted [Mem0](https://github.com/mem0ai/mem0) instance.  
  1. Launch Mem0 locally (example Docker run):  
     `docker run -d --name mem0 -p 3030:3030 -e MEM0_API_KEY=local-dev mem0ai/mem0:latest`  
     Adjust the port/API key to match your environment or follow the official Mem0 deployment docs.
  2. Export the same key before starting the server: `set MEM0_API_KEY=local-dev` (PowerShell) or `export MEM0_API_KEY=local-dev` (bash).
  3. Tune `memory.long_term` in `config/config.yaml` if your Mem0 endpoint, tags, or summarisation behaviour differ.
- During a turn the assistant summarises the exchange through the configured LLM (SiliconFlow DeepSeek by default) before persisting the memory. Queries against Mem0 are injected back into the LLM via the `memory_context`.

### 环境变量存放敏感 Key
- 配置文件中的 `api_key` 字段全部改用 `${...}` 形式（例如 `${SILICONFLOW_API_KEY}`、`${MEM0_API_KEY}`），实际值请通过环境变量提供。
- Windows PowerShell（当前会话）：`$env:SILICONFLOW_API_KEY="your-key"`；永久写入：`setx SILICONFLOW_API_KEY "your-key"`.
- Linux/macOS（bash/zsh）：`export SILICONFLOW_API_KEY=your-key`，如需长期生效可写入 `~/.bashrc`。
- Docker 运行时可以用 `-e SILICONFLOW_API_KEY=your-key` 注入环境变量。
- 项目启动时会解析配置文件并自动替换 `${VAR}` 为对应环境变量，避免把 Key 写进 Git。

### Docker 构建与离线模型
- 构建镜像前，请将已下载的 faster-whisper 模型目录（例如 Hugging Face 缓存中的 `models--Systran--faster-whisper-medium/snapshots/<hash>`）拷贝到仓库根目录的 `models/whisper_model/`。
- Dockerfile 会把该目录复制到镜像内 `/app/models/whisper_model`，并在运行时通过 `WHISPER_MODEL_PATH` 指向该路径，同时默认设置 `HF_HUB_OFFLINE=1` 以禁用联网访问。
- 若模型目录缺失，ASR 模块将无法加载，请确保该路径含有 `config.json`、`model.bin` 等完整文件。
