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
1. Capture audio -> `/ws/asr` (chunk + flush)
2. Persist session under `logs/asr_sessions/<session_id>.jsonl`
3. Query `/asr_session/{session_id}` for accumulated transcripts
4. Use `/chat`, `/text_to_speech`, LangChain agent as needed.

See `docs/api.md` for full API contract and `tools/ws_asr_client.py` for example streaming clients (file & microphone modes).
