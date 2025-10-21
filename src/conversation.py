"""Conversation orchestration tying ASR, NLP, and TTS together."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .asr_module import ASRModule, ASRResult
from .memory.manager import MemoryManager
from .nlp_module import NLPModule, NLPResult, RetrievalResult
from .tts_module import BaseTTS, TTSResponse

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    user_text: str
    assistant_text: str
    citations: List[RetrievalResult] = field(default_factory=list)
    audio: Optional[Path] = None


@dataclass
class ConversationState:
    session_id: str
    turns: List[ConversationTurn] = field(default_factory=list)

    def history_text(self) -> str:
        parts = []
        for turn in self.turns:
            parts.append(f"User: {turn.user_text}")
            parts.append(f"Assistant: {turn.assistant_text}")
        return "\n".join(parts)


class ConversationManager:
    """Coordinate the ASR → NLP → TTS pipeline."""

    def __init__(
        self,
        asr: ASRModule,
        nlp: NLPModule,
        tts: BaseTTS,
        *,
        session_id: Optional[str] = None,
        tts_output_dir: str | Path = "voice_assistant/output",
        memory_manager: Optional[MemoryManager] = None,
    ) -> None:
        self.asr = asr
        self.nlp = nlp
        self.tts = tts
        self.state = ConversationState(session_id=session_id or str(uuid.uuid4()))
        self.tts_output_dir = Path(tts_output_dir)
        self.memory_manager = memory_manager
        logger.info("Conversation manager initialized session=%s", self.state.session_id)

    def _update_state(
        self,
        user_text: str,
        assistant_text: str,
        citations: List[RetrievalResult],
        audio: Optional[Path],
    ) -> ConversationTurn:
        turn = ConversationTurn(
            user_text=user_text,
            assistant_text=assistant_text,
            citations=citations,
            audio=audio,
        )
        self.state.turns.append(turn)
        return turn

    def add_turn(
        self,
        user_text: str,
        assistant_text: str,
        *,
        citations: Optional[List[RetrievalResult]] = None,
        audio: Optional[Path] = None,
    ) -> ConversationTurn:
        """Persist a turn coming from an external pipeline (e.g. a LangChain agent)."""

        turn = self._update_state(
            user_text=user_text,
            assistant_text=assistant_text,
            citations=citations or [],
            audio=audio,
        )
        if self.memory_manager:
            self.memory_manager.record_turn(turn)
        return turn

    def handle_audio(self, audio_path: str | Path) -> ConversationTurn:
        asr_result = self.asr.transcribe(audio_path)
        return self.handle_text(asr_result)

    def handle_text(self, asr_result: ASRResult | str) -> ConversationTurn:
        if isinstance(asr_result, str):
            user_text = asr_result
            asr_metadata = None
        else:
            user_text = asr_result.text
            asr_metadata = asr_result

        logger.debug("Processing user_text=%s", user_text)
        chat_history = self.state.history_text()
        memory_context: Optional[str] = None
        if self.memory_manager:
            memory_context = self.memory_manager.prepare_context(user_text=user_text)

        nlp_result: NLPResult = self.nlp.answer(
            user_text,
            chat_history=chat_history,
            memory_context=memory_context,
        )

        tts_response: Optional[TTSResponse] = None
        if nlp_result.answer:
            tts_response = self.tts.synthesize(nlp_result.answer, self.tts_output_dir)

        turn = self._update_state(
            user_text=user_text,
            assistant_text=nlp_result.answer,
            citations=nlp_result.citations,
            audio=tts_response.audio_path if tts_response else None,
        )

        if self.memory_manager:
            self.memory_manager.record_turn(turn)

        if asr_metadata:
            logger.debug(
                "ASR language=%s duration=%.2fs text=%s",
                asr_metadata.language,
                asr_metadata.duration,
                asr_metadata.text,
            )

        return turn

    def export_transcript(self, path: str | Path) -> Path:
        transcript = self.state.history_text()
        out_path = Path(path)
        out_path.write_text(transcript, encoding="utf-8")
        logger.info("Transcript exported to %s", out_path)
        return out_path


