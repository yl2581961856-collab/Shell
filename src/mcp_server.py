"""FastAPI-based MCP server exposing speech, chat, and streaming tools."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
import numpy as np
import soundfile as sf
from pydantic import BaseModel

from .asr_module import ASRModule, ASRSegment
from .conversation import ConversationManager
from .memory.manager import MemoryManager, build_memory_manager
from .nlp_module import KnowledgeBase, NLPModule, SiliconFlowClient
from .setting import PROJECT_ROOT, configure_logging, load_config
from .tts_module import BaseTTS, create_tts

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .langchain_agent import AgentRunResult, LangChainVoiceAgent


class ChatRequest(BaseModel):
    text: str


class ChatResponse(BaseModel):
    text: str
    audio_path: Optional[str]
    citations: list[Any]


class AgentChatRequest(BaseModel):
    text: Optional[str] = None
    audio_path: Optional[str] = None
    auto_tts: Optional[bool] = None


class AgentChatResponse(BaseModel):
    text: str
    audio_path: Optional[str]
    citations: list[Any]
    tools: list[str]


class ASRSessionStore:
    """Persist streaming ASR segments per session."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: Dict[str, list[Dict[str, Any]]] = {}

    def _file_path(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.jsonl"

    def append(
        self,
        session_id: str,
        segments: list[ASRSegment],
        *,
        language: Optional[str],
        duration: float,
    ) -> None:
        entry = {
            "timestamp": time.time(),
            "language": language,
            "duration": duration,
            "segments": [
                {"text": seg.text, "start": seg.start, "end": seg.end} for seg in segments
            ],
        }
        self._sessions.setdefault(session_id, []).append(entry)
        with self._file_path(session_id).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        file_path = self._file_path(session_id)
        if file_path.exists():
            file_path.unlink()

    def get(self, session_id: str) -> list[Dict[str, Any]]:
        if session_id in self._sessions:
            return self._sessions[session_id]
        file_path = self._file_path(session_id)
        if not file_path.exists():
            return []
        entries: list[Dict[str, Any]] = []
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        self._sessions[session_id] = entries
        return entries


def _run_transcribe_stream(
    asr_module: ASRModule,
    audio_path: Path,
    asr_kwargs: Dict[str, Any],
) -> Tuple[list[ASRSegment], Optional[str], float]:
    """Blocking helper executed in a worker thread."""

    result = asr_module.transcribe_stream(audio_path, **asr_kwargs)
    segments = list(result.segments)
    return segments, result.language, result.duration


def build_conversation_manager(config: Dict[str, Any]) -> ConversationManager:
    asr_cfg = config.get("asr", {})
    asr_module = ASRModule(**asr_cfg)

    nlp_cfg = config.get("nlp", {})
    llm_cfg = nlp_cfg.get("siliconflow") or nlp_cfg.get("deepseek", {})
    if not llm_cfg.get("api_key"):
        raise RuntimeError("SiliconFlow/DeepSeek API key missing in configuration")
    if "deepseek" in nlp_cfg and "siliconflow" not in nlp_cfg:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Configuration key nlp.deepseek is deprecated. Rename it to nlp.siliconflow to silence this message."
        )
    llm_client = SiliconFlowClient(
        api_key=llm_cfg["api_key"],
        model=llm_cfg.get("model", "deepseek-r1"),
        base_url=llm_cfg.get("base_url", "https://api.siliconflow.cn"),
    )

    knowledge_cfg = nlp_cfg.get("knowledge_base", {})
    knowledge_base: Optional[KnowledgeBase] = None
    if knowledge_cfg.get("enabled", True):
        kb_path = knowledge_cfg.get("index_path")
        if not kb_path:
            kb_path = PROJECT_ROOT / "models" / "embeddings" / "knowledge_base.json"
        knowledge_base = KnowledgeBase(
            embedding_model=knowledge_cfg.get(
                "embedding_model", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            ),
            index_path=kb_path,
        )

    nlp_module = NLPModule(
        llm_client=llm_client,
        knowledge_base=knowledge_base,
        system_prompt=nlp_cfg.get("system_prompt"),
    )

    tts_cfg = config.get("tts", {})
    tts_module: BaseTTS = create_tts(tts_cfg)

    memory_manager: Optional[MemoryManager] = build_memory_manager(config)

    return ConversationManager(
        asr=asr_module,
        nlp=nlp_module,
        tts=tts_module,
        tts_output_dir=config.get("runtime", {}).get("tts_output_dir", PROJECT_ROOT / "logs"),
        memory_manager=memory_manager,
    )


def build_agent(manager: ConversationManager, config: Dict[str, Any]):
    agent_cfg = config.get("agent", {})
    if not agent_cfg.get("enabled", False):
        return None

    from .langchain_agent import LangChainVoiceAgent  # Imported lazily for optional dependency

    return LangChainVoiceAgent(
        manager,
        system_prompt=agent_cfg.get("system_prompt"),
        retrieval_top_k=agent_cfg.get("retrieval_top_k", 4),
        auto_tts=agent_cfg.get("auto_tts", True),
        llm_temperature=agent_cfg.get("temperature", 0.6),
        llm_top_p=agent_cfg.get("top_p", 0.9),
    )


def create_app(config_path: Optional[str | Path] = None) -> FastAPI:
    configure_logging()
    config = load_config(config_path)
    manager = build_conversation_manager(config)
    agent = build_agent(manager, config)

    app = FastAPI(title="Voice Assistant MCP Server")
    app.state.manager = manager
    app.state.agent = agent
    app.state.asr_session_store = ASRSessionStore(PROJECT_ROOT / "logs" / "asr_sessions")

    @app.post("/speech_to_text")
    async def speech_to_text(file: UploadFile = File(...)) -> Dict[str, Any]:
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)
        try:
            result = app.state.manager.asr.transcribe(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        return {
            "text": result.text,
            "language": result.language,
            "duration": result.duration,
            "segments": [segment.__dict__ for segment in result.segments],
        }

    @app.post("/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest) -> ChatResponse:
        if not payload.text.strip():
            raise HTTPException(status_code=400, detail="Text must not be empty")
        turn = app.state.manager.handle_text(payload.text)
        citations = [citation.__dict__ for citation in turn.citations]
        return ChatResponse(
            text=turn.assistant_text,
            audio_path=str(turn.audio) if turn.audio else None,
            citations=citations,
        )

    @app.post("/agent_chat", response_model=AgentChatResponse)
    async def agent_chat(payload: AgentChatRequest) -> AgentChatResponse:
        agent_instance = app.state.agent
        if agent_instance is None:
            raise HTTPException(status_code=503, detail="Agent mode is disabled in configuration")

        if not (payload.text and payload.text.strip()) and not payload.audio_path:
            raise HTTPException(status_code=400, detail="Provide text or audio_path for the agent")

        result = agent_instance.run(
            text=payload.text if payload.text and payload.text.strip() else None,
            audio_path=payload.audio_path,
            auto_tts=payload.auto_tts,
        )
        return AgentChatResponse(
            text=result.text,
            audio_path=str(result.audio_path) if result.audio_path else None,
            citations=[citation.__dict__ for citation in result.citations],
            tools=result.tool_messages,
        )

    @app.post("/text_to_speech")
    async def text_to_speech(payload: ChatRequest) -> Dict[str, Any]:
        if not payload.text.strip():
            raise HTTPException(status_code=400, detail="Text must not be empty")
        tts_response = app.state.manager.tts.synthesize(payload.text, app.state.manager.tts_output_dir)
        return {
            "audio_path": str(tts_response.audio_path),
            "format": tts_response.format,
            "sample_rate": tts_response.sample_rate,
        }

    @app.get("/asr_session/{session_id}")
    async def get_asr_session(session_id: str) -> Dict[str, Any]:
        store: ASRSessionStore = app.state.asr_session_store
        entries = store.get(session_id)
        if not entries:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"session_id": session_id, "entries": entries}

    @app.websocket("/ws/asr")
    async def asr_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        session_id = uuid.uuid4().hex
        store: ASRSessionStore = app.state.asr_session_store
        buffer = bytearray()
        cached_language: Optional[str] = None
        current_sample_rate = 16000
        await websocket.send_text(
            json.dumps({"type": "ready", "session_id": session_id}, ensure_ascii=False)
        )

        try:
            while True:
                message = await websocket.receive_text()
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": "Invalid JSON payload"}, ensure_ascii=False)
                    )
                    continue

                msg_type = payload.get("type")
                if msg_type == "chunk":
                    chunk = payload.get("data")
                    if not chunk:
                        continue
                    try:
                        decoded = base64.b64decode(chunk)
                    except (ValueError, TypeError):
                        await websocket.send_text(
                            json.dumps({"type": "error", "message": "Chunk data must be base64-encoded"}, ensure_ascii=False)
                        )
                        continue
                    if payload.get("sample_rate") is not None:
                        try:
                            current_sample_rate = int(payload["sample_rate"])
                        except (TypeError, ValueError):
                            pass
                    buffer.extend(decoded)
                    if payload.get("language"):
                        cached_language = payload["language"]
                elif msg_type == "flush":
                    if not buffer:
                        await websocket.send_text(
                            json.dumps({"type": "warning", "message": "No audio buffered"}, ensure_ascii=False)
                        )
                        continue
                    if payload.get("sample_rate") is not None:
                        try:
                            current_sample_rate = int(payload["sample_rate"])
                        except (TypeError, ValueError):
                            pass
                    raw_bytes = bytes(buffer)
                    buffer.clear()
                    if len(raw_bytes) < 2:
                        await websocket.send_text(
                            json.dumps({"type": "warning", "message": "Audio chunk too short"}, ensure_ascii=False)
                        )
                        continue
                    if len(raw_bytes) % 2 != 0:
                        raw_bytes = raw_bytes[:-1]
                    audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
                    if audio_int16.size == 0:
                        await websocket.send_text(
                            json.dumps({"type": "warning", "message": "Audio chunk decode failed"}, ensure_ascii=False)
                        )
                        continue
                    audio_float = audio_int16.astype(np.float32) / 32768.0
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp_path = Path(tmp.name)
                    sf.write(tmp_path, audio_float, current_sample_rate)

                    asr_kwargs: Dict[str, Any] = {}
                    language = payload.get("language") or cached_language
                    if language:
                        asr_kwargs["language"] = language
                    if payload.get("beam_size") is not None:
                        asr_kwargs["beam_size"] = int(payload["beam_size"])
                    if payload.get("chunk_length") is not None:
                        asr_kwargs["chunk_length"] = float(payload["chunk_length"])

                    loop = asyncio.get_running_loop()
                    segments, language, duration = await loop.run_in_executor(
                        None, _run_transcribe_stream, app.state.manager.asr, tmp_path, asr_kwargs
                    )
                    tmp_path.unlink(missing_ok=True)

                    store.append(session_id, segments, language=language, duration=duration)
                    await websocket.send_text(
                        json.dumps(
                            {"type": "metadata", "language": language, "duration": duration},
                            ensure_ascii=False,
                        )
                    )
                    for segment in segments:
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "segment",
                                    "text": segment.text,
                                    "start": segment.start,
                                    "end": segment.end,
                                },
                                ensure_ascii=False,
                            )
                        )
                    await websocket.send_text(json.dumps({"type": "flush_complete"}))
                elif msg_type == "reset":
                    buffer.clear()
                    cached_language = None
                    store.reset(session_id)
                    await websocket.send_text(json.dumps({"type": "reset_ack"}))
                else:
                    await websocket.send_text(
                        json.dumps(
                            {"type": "error", "message": f"Unsupported message type: {msg_type}"},
                            ensure_ascii=False,
                        )
                    )
        except WebSocketDisconnect:
            logging.getLogger(__name__).info("ASR WebSocket disconnected")
        finally:
            buffer.clear()

    return app
