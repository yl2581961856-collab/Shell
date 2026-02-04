from __future__ import annotations

from .config import AppConfig


class AsrService:
    """Stub ASR service.

    Replace this with FunASR integration and actual model loading.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._ready = False

    def load(self) -> None:
        # TODO: load models from config.model_paths
        self._ready = True

    def transcribe_bytes(self, audio_bytes: bytes, stream: bool = False) -> str:
        if not self._ready:
            self.load()

        # TODO: implement real ASR inference
        # Return empty string to signal stub behavior.
        return ""

    def transcribe_file(self, file_path: str) -> str:
        if not self._ready:
            self.load()

        # TODO: implement real ASR inference
        return ""
