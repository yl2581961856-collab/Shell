# Shell

digital soul making

## Voice Assistant Orchestration

A modular voice assistant stack that strings together ASR (FunASR Paraformer streaming), retrieval augmented generation, and TTS behind a FastAPI MCP server and optional LangChain agent.

### Features
- FunASR Paraformer-Large speech recognition with built-in VAD, punctuation, timestamps, optional diarization
- Retrieval-augmented generation via FAISS + sentence-transformers
- Pluggable LLM backend (Qwen3 intranet service, DeepSeek/SiliconFlow, etc.)
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
- During a turn the assistant summarises the exchange through the configured LLM (local Qwen3 endpoint by default) before persisting the memory. Queries against Mem0 are injected back into the LLM via the `memory_context`.

### LLM Configuration
- `config/config.yaml` points `nlp.siliconflow` to the intranet Qwen3 service at `http://192.168.11.151:8091` with `model: qwen3-32b-fp8`. The endpoint exposes an OpenAI-compatible `/v1/chat/completions` interface, so no additional code changes are required.
- Switching back to hosted providers (e.g. SiliconFlow/DeepSeek) simply requires setting `nlp.siliconflow.api_key` to `${SILICONFLOW_API_KEY}` (or another env var) and exporting it before launch. Leaving the field empty skips the `Authorization` header, which is exactly what the Qwen3 cluster expects.


### ASR Configuration (FunASR)
- The default `asr` block targets `iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch`, which already ships with VAD, punctuation, and timestamp heads. 我们在 WebSocket/上传流程里加入了“伪流式”切片（`max_chunk_seconds` + `chunk_overlap_seconds`），每段切成约 12 秒再送入离线模型，避免长音频一次性占满显存。
- `asr.speaker` enables CampPlus diarization。可以通过 `mode`（`punc_segment` / `vad_segment`）以及 `cb_kwargs.merge_thr` 控制聚类阈值；识别出的 `speaker` ID 会出现在 `segments[*].speaker` 中，并写入 `logs/asr_sessions/*.jsonl`。
- 安装 `funasr[torch]` 并使用符合 GPU 的 PyTorch（这里是 `torch 2.5.1+cu121`）。FunASR 会自动下载模型，也可提前放到 `models/` 目录并在配置里指向该路径。
- Legacy Whisper-specific request parameters (`beam_size`, `chunk_length`, etc.) are ignored gracefully; prefer FunASR's `generate_kwargs`（如 hotword）进行细调。

### 环境变量存放敏感 Key
- 配置文件中的 `api_key` 字段全部改用 `${...}` 形式（例如 `${SILICONFLOW_API_KEY}`、`${MEM0_API_KEY}`），实际值请通过环境变量提供。
- Windows PowerShell（当前会话）：`$env:SILICONFLOW_API_KEY="your-key"`；永久写入：`setx SILICONFLOW_API_KEY "your-key"`.
- Linux/macOS（bash/zsh）：`export SILICONFLOW_API_KEY=your-key`，如需长期生效可写入 `~/.bashrc`。
- Docker 运行时可以用 `-e SILICONFLOW_API_KEY=your-key` 注入环境变量。
- 项目启动时会解析配置文件并自动替换 `${VAR}` 为对应环境变量，避免把 Key 写进 Git。
