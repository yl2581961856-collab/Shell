"""FastAPI-based MCP server exposing speech, chat, and streaming tools."""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

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

try:  # optional dependency for zh conversion
    from opencc import OpenCC  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    OpenCC = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .langchain_agent import AgentRunResult, LangChainVoiceAgent


class ChatRequest(BaseModel):
    text: str
    mode: Literal["normal", "meeting"] = "normal"
    session_id: Optional[str] = None
    system_prompt: Optional[str] = None
    preset: Optional[str] = None


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


PRESET_SYSTEM_PROMPTS: dict[str, str] = {
    "meeting": """你是会议纪要助手。输出要求：
1. 不展示推理/思考过程，只输出最终结果。
2. 仅返回合法 JSON 字符串，结构如下：
   {
     "key_points": ["重点1", "重点2"],
     "action_items": [
       {"task": "事项", "owner": "负责人", "due": "截止时间(无则 null)"}
     ]
   }
3. key_points 数组列出会议讨论重点，措辞简洁。
4. action_items 只保留真实行动项，缺少负责人或截止时间时填 null。
5. 原文无关或为噪声时返回 {\"key_points\": [], \"action_items\": []}。
""",
}


def _resolve_system_prompt(mode: str, system_prompt: Optional[str], preset: Optional[str]) -> Optional[str]:
    if system_prompt:
        return system_prompt
    if preset:
        preset_key = preset.strip().lower()
        if preset_key:
            resolved = PRESET_SYSTEM_PROMPTS.get(preset_key)
            if resolved:
                return resolved
    if mode == "meeting":
        return PRESET_SYSTEM_PROMPTS.get("meeting")
    return None


def _extract_llm_generation_kwargs(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not raw:
        return {}
    generation: Dict[str, Any] = {}
    if raw.get("temperature") is not None:
        try:
            generation["temperature"] = float(raw["temperature"])
        except (TypeError, ValueError):
            pass
    if raw.get("top_p") is not None:
        try:
            generation["top_p"] = float(raw["top_p"])
        except (TypeError, ValueError):
            pass
    if raw.get("max_tokens") is not None:
        try:
            generation["max_tokens"] = int(raw["max_tokens"])
        except (TypeError, ValueError):
            pass
    return generation


def _chunk_text(text: str, max_chars: int = 60) -> List[str]:
    text = text or ""
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + max_chars)
        chunks.append(text[start:end])
        start = end
    return chunks


async def _llm_stream_task(
    manager: ConversationManager,
    websocket: WebSocket,
    *,
    request_id: str,
    session_id: Optional[str],
    text: str,
    prompt_override: Optional[str],
    generation_kwargs: Optional[Dict[str, Any]],
) -> None:
    loop = asyncio.get_running_loop()
    final_text = ""

    def run_generation() -> str:
        turn = manager.handle_text(
            text,
            system_prompt=prompt_override,
            generation_kwargs=generation_kwargs,
        )
        return turn.assistant_text

    try:
        assistant_text = await loop.run_in_executor(None, run_generation)
        final_text = assistant_text or ""
        for idx, delta in enumerate(_chunk_text(final_text)):
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "llm_delta",
                        "session_id": session_id,
                        "request_id": request_id,
                        "delta": delta,
                        "index": idx,
                    },
                    ensure_ascii=False,
                )
            )
        await websocket.send_text(
            json.dumps(
                {
                    "type": "llm_done",
                    "session_id": session_id,
                    "request_id": request_id,
                    "reason": "completed",
                    "final_text": final_text,
                },
                ensure_ascii=False,
            )
        )
    except asyncio.CancelledError:
        await websocket.send_text(
            json.dumps(
                {
                    "type": "llm_done",
                    "session_id": session_id,
                    "request_id": request_id,
                    "reason": "cancel",
                    "final_text": final_text,
                },
                ensure_ascii=False,
            )
        )
        raise
    except Exception as exc:  # pragma: no cover - defensive handling
        logging.getLogger(__name__).exception("LLM streaming failed")
        await websocket.send_text(
            json.dumps(
                {
                    "type": "llm_error",
                    "session_id": session_id,
                    "request_id": request_id,
                    "code": "server_error",
                    "message": str(exc),
                },
                ensure_ascii=False,
            )
        )
        await websocket.send_text(
            json.dumps(
                {
                    "type": "llm_done",
                    "session_id": session_id,
                    "request_id": request_id,
                    "reason": "error",
                    "final_text": final_text,
                },
                ensure_ascii=False,
            )
        )


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
                {
                    "text": seg.text,
                    "start": seg.start,
                    "end": seg.end,
                    "speaker": seg.speaker,
                }
                for seg in segments
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


class TranscriptionPostProcessor:
    """Apply lightweight cleanup to ASR outputs (simplified, punctuation, filters)."""

    QUESTION_SUFFIXES = ("吗", "么", "呢", "是不是", "好不好", "对不对", "可不可以", "要不要", "能不能")
    QUESTION_KEYWORDS = ("谁", "什么", "哪", "为何", "为什么", "怎样", "怎么", "多少", "是否", "是不是", "能否")
    SENTENCE_ENDINGS = ("。", "！", "？", ".", "!", "?")

    def __init__(self, config: Dict[str, Any]) -> None:
        self.enable_punct = bool(config.get("basic_punctuation"))
        self.convert_simplified = bool(config.get("convert_to_simplified"))
        self.forbidden_patterns = self._compile_patterns(config.get("forbidden_phrases") or [])
        self._opencc = None
        if self.convert_simplified and OpenCC is not None:
            try:
                self._opencc = OpenCC("t2s")
            except Exception:  # pragma: no cover - defensive guard
                logging.getLogger(__name__).warning("OpenCC initialisation failed; disabling conversion.")
                self.convert_simplified = False

    @staticmethod
    def _compile_patterns(phrases: Any) -> list[re.Pattern[str]]:
        patterns: list[re.Pattern[str]] = []
        for phrase in phrases:
            if not isinstance(phrase, str) or not phrase:
                continue
            escaped = re.escape(phrase.strip())
            patterns.append(re.compile(escaped, re.IGNORECASE))
        return patterns

    def apply_to_text(self, text: str) -> str:
        cleaned = text or ""
        if cleaned and self.convert_simplified and self._opencc is not None:
            try:
                cleaned = self._opencc.convert(cleaned)
            except Exception:  # pragma: no cover
                logging.getLogger(__name__).warning("OpenCC conversion failed; returning original text.")
        for pattern in self.forbidden_patterns:
            cleaned = pattern.sub("", cleaned)
        cleaned = cleaned.strip()
        if not cleaned:
            return cleaned
        if self.enable_punct:
            cleaned = self._ensure_sentence_punctuation(cleaned)
        if cleaned and self.convert_simplified and self._opencc is not None:
            try:
                cleaned = self._opencc.convert(cleaned)
            except Exception:  # pragma: no cover
                logging.getLogger(__name__).warning("OpenCC conversion failed; returning original text.")
        return cleaned

    def apply_to_segments(self, segments: list[ASRSegment]) -> None:
        for segment in segments:
            segment.text = self.apply_to_text(segment.text)

    def apply_to_result(self, result: Any) -> None:
        result.text = self.apply_to_text(getattr(result, "text", "") or "")
        segments = getattr(result, "segments", []) or []
        self.apply_to_segments(segments)

    def _ensure_sentence_punctuation(self, text: str) -> str:
        if not text:
            return text
        last_char = text[-1]
        if last_char in self.SENTENCE_ENDINGS:
            return text
        lowered = text.lower()
        if any(text.endswith(suffix) for suffix in self.QUESTION_SUFFIXES) or any(keyword in lowered for keyword in self.QUESTION_KEYWORDS):
            return f"{text}？"
        return f"{text}。"


def _run_transcribe_stream(
    asr_module: ASRModule,
    audio_path: Path,
    asr_kwargs: Dict[str, Any],
) -> Tuple[list[ASRSegment], Optional[str], float]:
    """Blocking helper executed in a worker thread."""

    result = asr_module.transcribe_stream(audio_path, **asr_kwargs)
    segments = list(result.segments)
    return segments, result.language, result.duration


def build_conversation_manager(config: Dict[str, Any]) -> tuple[ConversationManager, Dict[str, Any]]:
    asr_cfg = dict(config.get("asr", {}) or {})
    postprocess_cfg = asr_cfg.pop("postprocess", {}) or {}
    asr_module = ASRModule(**asr_cfg)

    nlp_cfg = config.get("nlp", {})
    llm_cfg = nlp_cfg.get("siliconflow") or nlp_cfg.get("deepseek", {})
    if not llm_cfg:
        raise RuntimeError("Missing LLM configuration under nlp.siliconflow")
    if "deepseek" in nlp_cfg and "siliconflow" not in nlp_cfg:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Configuration key nlp.deepseek is deprecated. Rename it to nlp.siliconflow to silence this message."
        )
    api_key = llm_cfg.get("api_key")
    base_url = llm_cfg.get("base_url", "https://api.siliconflow.cn")
    if not api_key:
        logging.getLogger(__name__).info(
            "No LLM api_key configured; sending requests to %s without Authorization header", base_url
        )
    llm_client = SiliconFlowClient(
        api_key=api_key,
        model=llm_cfg.get("model", "deepseek-r1"),
        base_url=base_url,
        timeout=llm_cfg.get("timeout", 120),
        headers=llm_cfg.get("headers"),
    )

    knowledge_cfg = nlp_cfg.get("knowledge_base", {})
    knowledge_base: Optional[KnowledgeBase] = None
    if knowledge_cfg.get("enabled", True):
        kb_path = knowledge_cfg.get("index_path")
        if not kb_path:
            kb_path = PROJECT_ROOT / "data" / "models" / "embeddings" / "knowledge_base.json"
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
    tts_module: Optional[BaseTTS] = None
    if tts_cfg.get("enabled", True):
        tts_module = create_tts(tts_cfg)

    memory_manager: Optional[MemoryManager] = build_memory_manager(config, llm_client=llm_client)

    manager = ConversationManager(
        asr=asr_module,
        nlp=nlp_module,
        tts=tts_module,
        tts_output_dir=config.get("runtime", {}).get("tts_output_dir", PROJECT_ROOT / "logs"),
        memory_manager=memory_manager,
    )
    return manager, postprocess_cfg


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
    manager, postprocess_cfg = build_conversation_manager(config)
    agent = build_agent(manager, config)

    app = FastAPI(title="Voice Assistant MCP Server")
    app.state.manager = manager
    app.state.agent = agent
    app.state.asr_session_store = ASRSessionStore(PROJECT_ROOT / "logs" / "asr_sessions")
    app.state.asr_lock = asyncio.Lock()
    app.state.transcription_processor = (
        TranscriptionPostProcessor(postprocess_cfg) if postprocess_cfg else None
    )

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
        processor: Optional[TranscriptionPostProcessor] = app.state.transcription_processor
        if processor:
            processor.apply_to_result(result)
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

        mode = payload.mode or "normal"
        if mode == "meeting" and not payload.session_id:
            raise HTTPException(status_code=400, detail="session_id is required for meeting mode")

        prompt_override = _resolve_system_prompt(mode, payload.system_prompt, payload.preset)

        turn = app.state.manager.handle_text(payload.text, system_prompt=prompt_override)
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
        llm_tasks: Dict[str, asyncio.Task] = {}
        llm_session_active: Dict[str, str] = {}

        async def send_json(payload: Dict[str, Any]) -> None:
            await websocket.send_text(json.dumps(payload, ensure_ascii=False))

        def cleanup_task(task: asyncio.Task, request_ref: str, session_ref: Optional[str]) -> None:
            llm_tasks.pop(request_ref, None)
            if session_ref and llm_session_active.get(session_ref) == request_ref:
                llm_session_active.pop(session_ref, None)
            try:
                task.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                logging.getLogger(__name__).exception("LLM task %s failed", request_ref)

        await send_json({"type": "ready", "session_id": session_id})

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
                    asr_lock: asyncio.Lock = app.state.asr_lock
                    try:
                        async with asr_lock:
                            segments, language, duration = await loop.run_in_executor(
                                None, _run_transcribe_stream, app.state.manager.asr, tmp_path, asr_kwargs
                            )
                    except Exception as exc:  # pragma: no cover - defensive handling
                        logging.getLogger(__name__).exception("ASR transcription failed")
                        tmp_path.unlink(missing_ok=True)
                        await websocket.send_text(
                            json.dumps(
                                {"type": "error", "message": f"Transcription failed: {exc}"},
                                ensure_ascii=False,
                            )
                        )
                        await websocket.close(code=1011, reason="transcription failed")
                        return
                    finally:
                        tmp_path.unlink(missing_ok=True)

                    processor: Optional[TranscriptionPostProcessor] = app.state.transcription_processor
                    if processor:
                        processor.apply_to_segments(segments)
                    segments = [seg for seg in segments if seg.text]
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
                                    "speaker": segment.speaker,
                                },
                                ensure_ascii=False,
                            )
                        )
                    await websocket.send_text(json.dumps({"type": "flush_complete"}))
                elif msg_type == "llm_request":
                    request_id = str(payload.get("request_id") or "").strip()
                    text_input = (payload.get("text") or "").strip()
                    if not request_id:
                        await send_json(
                            {
                                "type": "llm_error",
                                "code": "missing_request_id",
                                "message": "Field request_id is required",
                            }
                        )
                        continue
                    if not text_input:
                        await send_json(
                            {
                                "type": "llm_error",
                                "code": "empty_text",
                                "message": "Text must not be empty",
                                "request_id": request_id,
                            }
                        )
                        continue
                    if request_id in llm_tasks:
                        await send_json(
                            {
                                "type": "llm_error",
                                "code": "duplicate_request",
                                "message": f"Request {request_id} already in progress",
                                "request_id": request_id,
                            }
                        )
                        continue
                    mode = str(payload.get("mode") or "normal").strip().lower()
                    llm_session_id = payload.get("session_id")
                    if mode == "meeting" and not llm_session_id:
                        await send_json(
                            {
                                "type": "llm_error",
                                "code": "missing_session",
                                "message": "meeting mode requires session_id",
                                "request_id": request_id,
                            }
                        )
                        continue
                    if llm_session_id:
                        active_request = llm_session_active.get(llm_session_id)
                        if active_request and active_request != request_id:
                            await send_json(
                                {
                                    "type": "llm_error",
                                    "code": "busy",
                                    "message": "Another generation is running for this session",
                                    "session_id": llm_session_id,
                                    "request_id": request_id,
                                }
                            )
                            continue
                    prompt_override = _resolve_system_prompt(
                        mode,
                        payload.get("system_prompt"),
                        payload.get("preset"),
                    )
                    generation_kwargs = _extract_llm_generation_kwargs(payload.get("extra"))
                    task = asyncio.create_task(
                        _llm_stream_task(
                            app.state.manager,
                            websocket,
                            request_id=request_id,
                            session_id=llm_session_id,
                            text=text_input,
                            prompt_override=prompt_override,
                            generation_kwargs=generation_kwargs or None,
                        )
                    )
                    llm_tasks[request_id] = task
                    if llm_session_id:
                        llm_session_active[llm_session_id] = request_id
                    task.add_done_callback(
                        lambda t, req=request_id, sess=llm_session_id: cleanup_task(t, req, sess)
                    )
                elif msg_type == "llm_cancel":
                    request_id = str(payload.get("request_id") or "").strip()
                    if not request_id:
                        await send_json(
                            {
                                "type": "llm_error",
                                "code": "missing_request_id",
                                "message": "request_id is required for cancellation",
                            }
                        )
                        continue
                    task = llm_tasks.get(request_id)
                    if not task:
                        await send_json(
                            {
                                "type": "llm_error",
                                "code": "unknown_request",
                                "message": f"No active generation for {request_id}",
                                "request_id": request_id,
                            }
                        )
                        continue
                    task.cancel()
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
        except Exception as exc:  # pragma: no cover - defensive handling
            logger = logging.getLogger(__name__)
            logger.exception("Unhandled error in ASR WebSocket loop")
            try:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": f"Internal error: {exc}"}, ensure_ascii=False)
                )
            except Exception:
                pass
            with contextlib.suppress(Exception):
                await websocket.close(code=1011, reason="internal error")
        finally:
            buffer.clear()
            for task in list(llm_tasks.values()):
                task.cancel()
            llm_tasks.clear()
            llm_session_active.clear()

    return app


app = create_app()
