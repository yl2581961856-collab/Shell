from __future__ import annotations

import threading
import uuid
from typing import Optional

from .config import AppConfig


class AsrService:
    """ASR service that owns shared model weights and creates per-session recognizers.

    Design intent:
    - Load model weights once per process.
    - Create one AsrSession per WebSocket connection to hold streaming/decoder state.
      This avoids cross-session state poisoning (common with streaming decoders).

    Important: All inference APIs here are synchronous. In an async server, call them via
    `loop.run_in_executor(...)` to avoid blocking the event loop.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._ready = False
        self._init_lock = threading.Lock()

        # TODO: store shared/immutable model handles here after load()
        self._models = None

    def load(self) -> None:
        # Thread-safe lazy init: model loading can be expensive and must not run twice.
        if self._ready:
            return
        with self._init_lock:
            if self._ready:
                return

            # TODO: load models from config.model_paths
            # Keep only shared weights here. Do NOT keep per-session decoder state here.
            self._models = {}
            self._ready = True

    def create_session(self, session_id: Optional[str] = None) -> "AsrSession":
        self.load()
        sid = (session_id or "").strip() or uuid.uuid4().hex
        return AsrSession(service=self, session_id=sid)

    # Convenience one-shot API (non-streaming). Prefer create_session() for streaming.
    def transcribe_bytes(self, audio_bytes: bytes) -> str:
        session = self.create_session(session_id=f"oneshot-{uuid.uuid4().hex[:12]}")
        try:
            return session.transcribe_bytes(audio_bytes, stream=False)
        finally:
            session.close()


class AsrSession:
    """Per-connection ASR session.

    Put streaming/decoder state here.

    Threading: This object will typically be used from an executor thread. We keep a
    session-level lock so accidental concurrent calls (e.g. if the server changes to
    pipeline requests) won't corrupt internal state.
    """

    def __init__(self, service: AsrService, session_id: str) -> None:
        self._service = service
        self.session_id = session_id

        self._lock = threading.Lock()
        self._closed = False
        self._bytes_total = 0

        # TODO: create streaming recognizer/decoder here, e.g.
        # self._decoder = FunASRStreamingDecoder(models=service._models, ...)

    @property
    def bytes_total(self) -> int:
        return self._bytes_total

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            # TODO: release per-session resources (decoder, caches, etc)

    def transcribe_bytes(self, audio_bytes: bytes, stream: bool = False) -> str:
        # Keep this method synchronous. The server should call it via run_in_executor.
        self._service.load()

        with self._lock:
            if self._closed:
                return ""

            self._bytes_total += len(audio_bytes)

            # TODO: implement real ASR inference.
            # Return empty string to keep stub behavior.
            return ""
