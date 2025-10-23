import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Optional

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - torch is optional at import time
    torch = None  # type: ignore

try:
    from faster_whisper import WhisperModel  # type: ignore
except ImportError:  # pragma: no cover - make dependency error explicit at runtime
    WhisperModel = None  # type: ignore

try:
    import soundfile as sf  # type: ignore
except ImportError:  # pragma: no cover - already required by the project
    sf = None  # type: ignore

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
    """Pluggable ASR implementation with faster-whisper and FunASR backends."""

    def __init__(
        self,
        model_name: str = "medium",
        provider: str = "faster_whisper",
        device: str = "auto",
        suppress_silence: bool = True,
        compute_type: str = "auto",
        beam_size: int = 5,
        chunk_length: Optional[float] = None,
        num_workers: int | None = None,
        funasr: Optional[dict] = None,
        **load_kwargs,
    ) -> None:
        self.model_name = model_name
        self.provider = (provider or "faster_whisper").lower()
        self.requested_device = device
        self.suppress_silence = suppress_silence
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.chunk_length = chunk_length
        self.num_workers = num_workers
        self.load_kwargs = load_kwargs
        self.funasr_config = funasr or {}
        self._model = None
        logger.debug(
            "ASRModule configured provider=%s model=%s device=%s compute_type=%s",
            self.provider,
            model_name,
            device,
            compute_type,
        )

    # --------------------------------------------------------------------- helpers

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
        return "int8_float16"

    # ---------------------------------------------------------------------- public

    def load(self) -> None:
        if self._model is not None:
            return
        if self.provider == "funasr":
            self._load_funasr()
        else:
            self._load_faster_whisper()

    def transcribe(self, audio_path: str | Path, **kwargs) -> ASRResult:
        audio = Path(audio_path)
        if not audio.exists():
            raise FileNotFoundError(f"Audio file not found: {audio}")

        if self._model is None:
            self.load()

        start_ts = time.perf_counter()
        assert self._model is not None
        logger.debug("Starting transcription for %s via provider=%s", audio.name, self.provider)

        if self.provider == "funasr":
            result = self._transcribe_with_funasr(audio)
        else:
            transcribe_kwargs = self._build_transcribe_kwargs(kwargs)
            segments_iter, info = self._model.transcribe(str(audio), **transcribe_kwargs)
            segments_list, text, duration, language = self._collect_segments(segments_iter, info, start_ts)
            result = ASRResult(text=text, segments=segments_list, language=language, duration=duration)

        logger.info("Transcribed %s in %.2fs", audio.name, time.perf_counter() - start_ts)
        return result

    def transcribe_stream(self, audio_path: str | Path, **kwargs) -> ASRStreamingResult:
        """Return a streaming generator over segments for near real-time use-cases."""

        audio = Path(audio_path)
        if not audio.exists():
            raise FileNotFoundError(f"Audio file not found: {audio}")

        if self._model is None:
            self.load()

        assert self._model is not None
        if self.provider == "funasr":
            # The FunASR python API does not currently expose incremental streaming.
            result = self.transcribe(audio_path, **kwargs)
            return ASRStreamingResult(iter(result.segments), result.language, result.duration)

        transcribe_kwargs = self._build_transcribe_kwargs(kwargs)
        segments_iter, info = self._model.transcribe(str(audio), **transcribe_kwargs)

        def iterator() -> Iterator[ASRSegment]:
            for seg in segments_iter:
                yield ASRSegment(text=seg.text.strip(), start=float(seg.start), end=float(seg.end))

        language = getattr(info, "language", None) if info is not None else None
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        return ASRStreamingResult(iterator(), language, duration)

    # ----------------------------------------------------------- backend loaders

    def _load_faster_whisper(self) -> None:
        if WhisperModel is None:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "`faster-whisper` is required for the default ASR provider. Install with `pip install faster-whisper`."
            )

        model_path = os.getenv("WHISPER_MODEL_PATH", "/app/models/whisper_model")
        model_path = os.path.expandvars(model_path)
        if not Path(model_path).exists():
            raise RuntimeError(f"Model path {model_path} does not exist.")

        device = self._resolve_device()
        fw_device = device if device in ("cuda", "cpu") else "cpu"
        compute_type = self._resolve_compute_type(fw_device)
        model_kwargs = dict(self.load_kwargs)
        if self.num_workers is not None:
            model_kwargs.setdefault("num_workers", self.num_workers)

        logger.info(
            "Loading faster-whisper model %s from %s on %s (compute_type=%s)",
            self.model_name,
            model_path,
            fw_device,
            compute_type,
        )

        self._model = WhisperModel(
            model_path,
            device=fw_device,
            compute_type=compute_type,
            local_files_only=True,
            **model_kwargs,
        )

    def _load_funasr(self) -> None:
        try:
            from funasr import AutoModel  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "`funasr` is required when ASR provider is set to 'funasr'. Install with `pip install funasr`."
            ) from exc

        model_root = self.funasr_config.get("model")
        if not model_root:
            raise RuntimeError("FunASR configuration requires 'model' pointing to the local model directory.")
        model_root = os.path.expandvars(str(model_root))
        if not Path(model_root).exists():
            raise RuntimeError(f"FunASR model directory does not exist: {model_root}")

        init_kwargs: dict[str, Any] = {}
        for key in ("vad_model", "punc_model", "lm_model", "lm_dict"):
            value = self.funasr_config.get(key)
            if value:
                init_kwargs[key] = os.path.expandvars(str(value))
        for key in ("vad_config", "punc_config", "lm_config"):
            value = self.funasr_config.get(key)
            if value:
                init_kwargs[key] = os.path.expandvars(str(value))

        init_kwargs["device"] = self.funasr_config.get("device", self._resolve_device())
        init_kwargs["batch_size"] = int(self.funasr_config.get("batch_size", 1) or 1)

        logger.info("Loading FunASR model from %s (device=%s)", model_root, init_kwargs["device"])
        self._model = AutoModel(model=model_root, **init_kwargs)

    # ------------------------------------------------------------- misc helpers

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

    def _collect_segments(self, segments_iter, info, start_ts: float) -> tuple[List[ASRSegment], str, float, Optional[str]]:
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
        return segments_list, text, duration, language

    def _transcribe_with_funasr(self, audio: Path) -> ASRResult:
        fun_kwargs = {}
        hotword = self.funasr_config.get("hotword")
        if hotword:
            fun_kwargs["hotword"] = hotword
        batch_size = int(self.funasr_config.get("batch_size", 1) or 1)

        outputs = self._model.generate(input=str(audio), batch_size=batch_size, **fun_kwargs)
        if not outputs:
            raise RuntimeError("FunASR returned no transcription output.")

        primary = outputs[0]
        text = (primary.get("text") or primary.get("sentence") or "").strip()

        duration = 0.0
        if sf is not None:
            try:
                with sf.SoundFile(str(audio)) as snd:
                    duration = len(snd) / float(snd.samplerate)
            except Exception:  # pragma: no cover - duration best effort
                duration = 0.0

        segment = ASRSegment(text=text, start=0.0, end=duration if duration > 0 else 0.0)
        return ASRResult(text=text, segments=[segment], language=primary.get("language"), duration=duration)
