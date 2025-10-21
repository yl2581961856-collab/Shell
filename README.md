# Voice Assistant Orchestration

A modular voice assistant stack that strings together ASR (OpenAI Whisper), retrieval augmented generation (DeepSeek/Qwen or other LLMs), and TTS (HiggsAudio or compatible) behind a FastAPI MCP server and an optional LangChain agent. The code base is designed for local development, remote GPU servers, and containerized deployment.

## Features
- Whisper-based automatic speech recognition with auto device selection (CPU/CUDA/MPS).
- Retrieval-augmented generation pipeline with FAISS + sentence-transformers.
- Pluggable LLM backend (DeepSeek API by default, Qwen or other local inference servers via HTTP).
- HiggsAudio-driven text-to-speech client (HTTP API) with optional auto-speech output.
- Conversation manager that records transcripts, citations, and generated audio paths.
- LangChain agent mode that dynamically decides when to call ASR/RAG/TTS tools.
- FastAPI MCP server exposing `/speech_to_text`, `/chat`, `/agent_chat`, and `/text_to_speech` endpoints.
- Dockerfile for reproducible builds and server deployment.

## Operational Workflow
1. **Speech Input** - User audio is captured and sent to the /speech_to_text tool or the agent; ASRModule loads Whisper (local weights) to produce transcripts and segment metadata.
2. **Context Gathering** - ConversationManager aggregates previous turns (ConversationState) to form chat history for retrieval/LLM reasoning.
3. **Retrieval Augmentation** - KnowledgeBase.search() embeds the query with sentence-transformers, runs FAISS similarity search against documents stored under models/embeddings, and returns scored citations.
4. **LLM Reasoning** - NLPModule.answer() (or the LangChain agent's rag_answer tool) calls the configured LLM backend (DeepSeek API by default) with system prompt, chat history, and retrieved knowledge to craft a response.
5. **Speech Synthesis** - HiggsAudioTTS.synthesize() converts the answer into audio via the configured TTS HTTP endpoint; files land in runtime.tts_output_dir.
6. **State Persistence** - ConversationManager.add_turn() stores each turn's text, citations, and optional audio path so /agent_chat and CLI modes can reuse context or export a transcript.
7. **Serving** - FastAPI (mcp_server.py) exposes deterministic and agent routes; Uvicorn or Docker hosts the service on mcp.port.

## Project Layout
```
voice_assistant/
├── config/
│   ├── config.yaml        # Main configuration (models, API keys, agent, ports)
│   └── logging.conf       # Logging configuration
├── logs/
│   └── assistant.log      # Rotating log target (empty placeholder)
├── models/                # Place for local model weights / indexes
│   ├── asr/
│   ├── tts/
│   └── embeddings/
├── src/
│   ├── asr_module.py      # Whisper wrapper
│   ├── tts_module.py      # HiggsAudio client
│   ├── nlp_module.py      # Knowledge base + LLM client
│   ├── conversation.py    # Orchestration + state
│   ├── langchain_agent.py # Agent wrapper and tools
│   ├── mcp_server.py      # FastAPI endpoints
│   └── main.py            # CLI entry point
├── Dockerfile             # Container build
└── logs/requirement.txt   # Runtime dependencies
```

## Getting Started (Local)
1. **Clone & create environment**
   ```bash
   git clone https://github.com/<your-account>/voice_assistant.git
   cd voice_assistant
   conda create -n voice-assistant python=3.10
   conda activate voice-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r logs/requirement.txt
   ```
   - `faiss-cpu` wheels are available for Windows/Linux; if installation fails, try conda: `conda install -c conda-forge faiss-cpu`.
   - LangChain is optional but required when the agent mode is enabled.

3. **Prepare model assets**
   - Whisper weights: `python -c "import whisper; whisper.load_model('medium')"` or download manually into `models/asr/`.
   - HiggsAudio weights/service: deploy your TTS server and note the base URL (e.g. `http://tts-host:8002`).
   - Knowledge base index: optionally preload documents into `models/embeddings/knowledge_base.json` using `KnowledgeBase.add_documents()`.

4. **Configuration**
   - Copy `config/config.yaml` and adjust:
     - `asr.model_name`, `asr.device` if you prefer a different Whisper size or fixed device.
    - `nlp.siliconflow.api_key` / `nlp.siliconflow.base_url` (set env vars `SILICONFLOW_API_KEY` or inline values).
     - Switch to Qwen or another LLM: update `base_url`, `model`, and adapt `DeepSeekClient` if the API shape differs.
     - `tts.api_base` and optional `tts.api_key` for your HiggsAudio service.
     - `agent.enabled` to toggle LangChain tools.
     - `runtime.tts_output_dir` for generated audio.
   - Logging tweaks go in `config/logging.conf`.

5. **Environment variables (optional)**
   ```bash
   set SILICONFLOW_API_KEY=sk-...
   set HIGGSAUDIO_API_KEY=...
   ```

6. **Run**
   - Deterministic pipeline CLI: `python -m src.main cli --text "你好"`
   - Agent CLI: `python -m src.main agent --text "介绍一下今天的天气"`
   - API server: `python -m src.main server --host 0.0.0.0 --port 9000`

## API Endpoints
- `POST /speech_to_text` — multipart audio upload → transcript.
- `POST /chat` — plain text query handled by deterministic pipeline.
- `POST /agent_chat` — text or audio handled by LangChain agent (requires `agent.enabled`).
- `POST /text_to_speech` — text → synthesized audio file path.

## Docker
1. Build: `docker build -t voice-assistant .`
2. Run:
   ```bash
   docker run \
     -p 9000:9000 \
    -e SILICONFLOW_API_KEY=sk-... \
     -e HIGGSAUDIO_API_KEY=... \
     -v /path/to/models:/app/voice_assistant/models \
     voice-assistant
   ```
   - Mount the model directory if Whisper/TTS/KB assets are on disk.
   - Adjust `config/config.yaml` or override with `--config` CLI flag if using a different path.

## Remote Server Workflow
- **Model hosting**: place large models under a dedicated path (e.g. `/opt/models/{asr,tts,llm}`) and mount or symlink into `voice_assistant/models`.
- **LLM service**: for on-prem Qwen, expose an internal HTTP endpoint (e.g. `http://qwen.internal:8000`). Update `config.yaml` accordingly or extend `DeepSeekClient` to match the API contract.
- **Deployment**: run the Docker container or native process under a process manager (systemd, supervisor, pm2). Ensure ports and firewalls allow access only from trusted networks.
- **Syncing**: if you cannot run Codex remotely, push changes from local to GitHub (`git push origin main`) and pull on the server (`git pull`). Transfer large models with `rsync`, `scp`, or object storage.

## Development Notes
- The repository is designed to run without network once weights are local (except when using external APIs).
- Knowledge base currently stores vectors in JSON for simplicity; consider switching to FAISS binary indexes for large corpora.
- LangChain agent tool usage is logged via `tool_messages` for easier debugging.
- Add tests or notebooks under `tests/` or `notebooks/` as needed; none are provided by default.

## Roadmap Ideas
- Swap DeepSeek client for a more generic `LLMClient` interface to support Qwen/vLLM/TGI.
- Stream TTS audio responses via WebSocket.
- Add hot-reload for knowledge base updates.
- Provide scripted data ingestion for knowledge base content.

## License
MIT (see `LICENSE`).
