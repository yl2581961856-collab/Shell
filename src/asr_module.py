"""Speech recognition module powered by faster-whisper."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - torch is optional at import time
    torch = None  # type: ignore

try:
    from faster_whisper import WhisperModel  # type: ignore
except ImportError:  # pragma: no cover - make dependency error explicit at runtime
    WhisperModel = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class ASRSegment:
    """Transcribed segment with time alignment."""

    text: str
    start: float
    end: float


@dataclass
class ASRResult:
    """Container for a transcription result."""

    text: str
    segments: List[ASRSegment]
    language: Optional[str]
    duration: float


@dataclass
class ASRStreamingResult:
    """Streaming response container for incremental consumers."""

    segments: Iterator[ASRSegment]
    language: Optional[str]
    duration: float


class ASRModule:
    """Thin wrapper around faster-whisper for easier dependency injection and testing."""

    def __init__(
        self,
        model_name: str = "medium",
        device: str = "auto",
        suppress_silence: bool = True,
        compute_type: str = "auto",
        beam_size: int = 5,
        chunk_length: Optional[float] = None,
        num_workers: int | None = None,
        **load_kwargs,
    ) -> None:
        self.model_name = model_name
        self.requested_device = device
        self.suppress_silence = suppress_silence
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.chunk_length = chunk_length
        self.num_workers = num_workers
        self.load_kwargs = load_kwargs
        self._model = None
        logger.debug(
            "ASRModule configured model=%s device=%s compute_type=%s",
            model_name,
            device,
            compute_type,
        )

    def _resolve_device(self) -> str:
        if self.requested_device != "auto":
            return self.requested_device
        if torch is not None and torch.cuda.is_available():  # pragma: no cover - depends on hardware
            return "cuda"
        if torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_compute_type(self, device: str) -> str:
        if self.compute_type != "auto":
            return self.compute_type
        if device == "cuda":
            return "float16"
        # int8_float16 keeps good accuracy on CPU while using less memory
        return "int8_float16"

    def load(self) -> None:
        if self._model is not None:
            return
        if WhisperModel is None:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "`faster-whisper` is required for ASRModule. Install with `pip install faster-whisper`."
            )

        device = self._resolve_device()
        fw_device = device if device in ("cuda", "cpu") else "cpu"
        compute_type = self._resolve_compute_type(fw_device)
        model_kwargs = dict(self.load_kwargs)
        if self.num_workers is not None:
            model_kwargs.setdefault("num_workers", self.num_workers)
        logger.info(
            "Loading faster-whisper model %s on %s (compute_type=%s)",
            self.model_name,
            fw_device,
            compute_type,
        )
        self._model = WhisperModel(
            self.model_name,
            device=fw_device,
            compute_type=compute_type,
            **model_kwargs,
        )

    def transcribe(self, audio_path: str | Path, **kwargs) -> ASRResult:
        audio = Path(audio_path)
        if not audio.exists():
            raise FileNotFoundError(f"Audio file not found: {audio}")

        if self._model is None:
            self.load()

        start_ts = time.perf_counter()
        assert self._model is not None  # mypy friendliness
        logger.debug("Starting transcription for %s", audio.name)
        transcribe_kwargs = self._build_transcribe_kwargs(kwargs)
        segments_iter, info = self._model.transcribe(str(audio), **transcribe_kwargs)
        segments_list: List[ASRSegment] = []
        text_parts: List[str] = []
        for seg in segments_iter:
            cleaned = seg.text.strip()
            segment = ASRSegment(text=cleaned, start=float(seg.start), end=float(seg.end))
            segments_list.append(segment)
            if cleaned:
                text_parts.append(cleaned)
        latency = time.perf_counter() - start_ts

        text = " ".join(text_parts).strip()
        language = getattr(info, "language", None) if info is not None else None
        duration = float(getattr(info, "duration", 0.0) or latency)

        logger.info("Transcribed %s in %.2fs", audio.name, latency)
        return ASRResult(text=text, segments=segments_list, language=language, duration=duration)

    def transcribe_stream(self, audio_path: str | Path, **kwargs) -> ASRStreamingResult:
        """Return a streaming generator over segments for near real-time use-cases."""

        audio = Path(audio_path)
        if not audio.exists():
            raise FileNotFoundError(f"Audio file not found: {audio}")

        if self._model is None:
            self.load()

        assert self._model is not None
        transcribe_kwargs = self._build_transcribe_kwargs(kwargs)
        segments_iter, info = self._model.transcribe(str(audio), **transcribe_kwargs)

        def iterator() -> Iterator[ASRSegment]:
            for seg in segments_iter:
                yield ASRSegment(text=seg.text.strip(), start=float(seg.start), end=float(seg.end))

        language = getattr(info, "language", None) if info is not None else None
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        return ASRStreamingResult(segments=iterator(), language=language, duration=duration)

    def _build_transcribe_kwargs(self, overrides: dict) -> dict:
        """Merge call-time overrides with module defaults."""

        kwargs = {
            "beam_size": overrides.pop("beam_size", self.beam_size),
            "vad_filter": overrides.pop("vad_filter", self.suppress_silence),
        }
        chunk_length: Optional[float] = overrides.pop("chunk_length", self.chunk_length)
        if chunk_length is not None:
            kwargs["chunk_length"] = float(chunk_length)
        if "language" in overrides:
            kwargs["language"] = overrides.pop("language")
        if "initial_prompt" in overrides:
            kwargs["initial_prompt"] = overrides.pop("initial_prompt")
        kwargs.update(overrides)
        return kwargs
