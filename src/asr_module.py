"""Speech recognition module powered by FunASR Paraformer."""
from __future__ import annotations

import logging
import time
import re
import tempfile
from numbers import Integral
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

try:  # pragma: no cover - optional dependency resolved at runtime
    from funasr import AutoModel  # type: ignore
except ImportError:  # pragma: no cover
    AutoModel = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class ASRSegment:
    """Transcribed segment with time alignment."""

    text: str
    start: float
    end: float
    speaker: Optional[str] = None


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
    """Thin wrapper around FunASR AutoModel for easier dependency injection and testing."""

    _CJK_SPACE_RE = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")

    def __init__(
        self,
        model: str,
        *,
        vad_model: Optional[str] = None,
        punc_model: Optional[str] = None,
        device: str = "cuda",
        chunk_size: Optional[Sequence[int]] = None,
        encoder_chunk_look_back: int = 4,
        decoder_chunk_look_back: int = 1,
        sentence_timestamp: bool = False,
        return_raw_text: bool = False,
        language: str = "zh",
        max_chunk_seconds: Optional[float] = None,
        chunk_overlap_seconds: float = 0.5,
        speaker: Optional[Dict[str, Any]] = None,
        generate_kwargs: Optional[Dict[str, Any]] = None,
        **model_kwargs: Any,
    ) -> None:
        if not model:
            raise ValueError("ASRModule requires 'model' to point to a FunASR model id or directory.")
        self.model_identifier = model
        self.language = language or None
        self._model: Optional[AutoModel] = None
        self.max_chunk_seconds = float(max_chunk_seconds or 0.0)
        self.chunk_overlap_seconds = max(float(chunk_overlap_seconds or 0.0), 0.0)

        self.model_kwargs: Dict[str, Any] = {
            "model": model,
            "device": device,
            "disable_update": model_kwargs.pop("disable_update", True),
        }
        if vad_model:
            self.model_kwargs["vad_model"] = vad_model
        if punc_model:
            self.model_kwargs["punc_model"] = punc_model
        self.model_kwargs.update(model_kwargs)

        chunk_value: Optional[List[int]] = None
        if chunk_size is not None:
            chunk_value = [int(value) for value in chunk_size]
            self.model_kwargs.setdefault("chunk_size", chunk_value)

        if speaker:
            spk_model_id = speaker.get("model")
            if spk_model_id:
                self.model_kwargs["spk_model"] = spk_model_id
                if speaker.get("mode"):
                    self.model_kwargs["spk_mode"] = speaker.get("mode")
                if speaker.get("preset_spk_num") is not None:
                    self.model_kwargs["preset_spk_num"] = speaker.get("preset_spk_num")
                spk_kwargs = dict(speaker.get("spk_kwargs", {}) or {})
                cb_kwargs = speaker.get("cb_kwargs")
                if cb_kwargs:
                    spk_kwargs["cb_kwargs"] = cb_kwargs
                if spk_kwargs:
                    self.model_kwargs["spk_kwargs"] = spk_kwargs
                if "return_spk_res" in speaker:
                    self.model_kwargs["return_spk_res"] = bool(speaker["return_spk_res"])

        self.generate_defaults: Dict[str, Any] = {
            "encoder_chunk_look_back": encoder_chunk_look_back,
            "decoder_chunk_look_back": decoder_chunk_look_back,
            "sentence_timestamp": sentence_timestamp,
            "return_raw_text": return_raw_text,
        }
        if generate_kwargs:
            for key, value in generate_kwargs.items():
                if value is not None:
                    self.generate_defaults[key] = value


    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        if self._model is not None:
            return
        if AutoModel is None:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "`funasr` is required for ASRModule. Install with `pip install funasr[torch]`."
            )
        logger.info("Loading FunASR model %s (device=%s)", self.model_identifier, self.model_kwargs.get("device"))
        self._model = AutoModel(**self.model_kwargs)

    def transcribe(self, audio_path: str | Path, **kwargs: Any) -> ASRResult:
        path = Path(audio_path)
        self._validate_audio(path)
        overrides = dict(kwargs)
        duration = self._safe_audio_duration(path)
        if self.max_chunk_seconds and duration > self.max_chunk_seconds + 1e-3:
            result = self._transcribe_in_chunks(path, duration, overrides)
        else:
            result = self._run_inference(path, overrides)
        return result

    def transcribe_stream(self, audio_path: str | Path, **kwargs: Any) -> ASRStreamingResult:
        path = Path(audio_path)
        self._validate_audio(path)
        result = self._run_inference(path, kwargs)

        def iterator() -> Iterator[ASRSegment]:
            yield from result.segments

        return ASRStreamingResult(segments=iterator(), language=result.language, duration=result.duration)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _validate_audio(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

    def _run_inference(self, audio_path: Path, overrides: Dict[str, Any]) -> ASRResult:
        if self._model is None:
            self.load()
        assert self._model is not None

        generate_kwargs, language_override = self._build_generate_kwargs(overrides)
        logger.debug("Starting FunASR transcription for %s", audio_path.name)
        start_ts = time.perf_counter()
        outputs = self._model.generate(str(audio_path), **generate_kwargs)
        latency = time.perf_counter() - start_ts
        if not outputs:
            raise RuntimeError("FunASR returned no transcription result")
        raw = outputs[0]
        audio_duration = self._safe_audio_duration(audio_path) or latency
        result = self._convert_raw_result(raw, language_override, audio_duration)
        logger.info("Transcribed %s in %.2fs (duration %.2fs)", audio_path.name, latency, result.duration)
        return result

    def _transcribe_in_chunks(
        self,
        audio_path: Path,
        duration: float,
        overrides: Dict[str, Any],
    ) -> ASRResult:
        data, sample_rate = sf.read(str(audio_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        total_samples = data.shape[0]
        if total_samples == 0:
            return self._run_inference(audio_path, overrides)

        chunk_samples = int(self.max_chunk_seconds * sample_rate)
        chunk_samples = max(chunk_samples, 1)
        overlap_samples = int(self.chunk_overlap_seconds * sample_rate)
        if overlap_samples >= chunk_samples:
            overlap_samples = chunk_samples - 1 if chunk_samples > 1 else 0

        start_sample = 0
        text_parts: List[str] = []
        segments: List[ASRSegment] = []
        language: Optional[str] = None

        while start_sample < total_samples:
            end_sample = min(total_samples, start_sample + chunk_samples)
            chunk = data[start_sample:end_sample]
            if chunk.size == 0:
                break
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                chunk_path = Path(tmp.name)
            try:
                sf.write(chunk_path, chunk, sample_rate)
                chunk_result = self._run_inference(chunk_path, overrides)
                offset = start_sample / sample_rate
                language = language or chunk_result.language or self.language
                if chunk_result.text:
                    text_parts.append(chunk_result.text)
                for seg in chunk_result.segments:
                    segments.append(
                        ASRSegment(
                            text=self._clean_spacing(seg.text),
                            start=seg.start + offset,
                            end=seg.end + offset,
                            speaker=seg.speaker,
                        )
                    )
            finally:
                chunk_path.unlink(missing_ok=True)

            if end_sample >= total_samples:
                break
            start_sample = max(end_sample - overlap_samples, 0)
            if start_sample >= total_samples:
                break

        text = self._clean_spacing(" ".join(text_parts))
        segments.sort(key=lambda seg: seg.start)
        return ASRResult(
            text=text,
            segments=segments,
            language=language or self.language,
            duration=duration,
        )

    def _build_generate_kwargs(self, overrides: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], Optional[str]]:
        overrides = dict(overrides or {})
        language_override = overrides.pop("language", None)
        overrides.pop("beam_size", None)
        overrides.pop("chunk_length", None)
        overrides.pop("vad_filter", None)
        overrides.pop("chunk_size", None)

        generate_kwargs = dict(self.generate_defaults)
        for key, value in overrides.items():
            if value is not None:
                generate_kwargs[key] = value

        return generate_kwargs, language_override

    def _convert_raw_result(
        self,
        raw: Dict[str, Any],
        language_override: Optional[str],
        fallback_duration: float,
    ) -> ASRResult:
        text = self._normalise_text(raw.get("text", ""))
        sentence_info = raw.get("sentence_info") or []
        segments: List[ASRSegment] = []
        for sentence in sentence_info:
            seg_text = self._clean_spacing(self._normalise_text(sentence.get("text", "")))
            if not seg_text:
                continue
            start = self._seconds(sentence.get("start"))
            end = self._seconds(sentence.get("end"))
            if start is None or end is None:
                timestamps = sentence.get("timestamp") or []
                if timestamps:
                    start = self._seconds(timestamps[0][0])
                    end = self._seconds(timestamps[-1][1])
            speaker = sentence.get("spk")
            if speaker is None:
                speaker = sentence.get("speaker")
            if speaker is not None:
                speaker = str(speaker)
            segments.append(
                ASRSegment(
                    text=seg_text,
                    start=float(start or 0.0),
                    end=float(end or (start or 0.0)),
                    speaker=speaker or None,
                )
            )

        if not segments and text:
            end_time = self._infer_duration_from_raw(raw) or fallback_duration
            segments.append(ASRSegment(text=text, start=0.0, end=end_time))

        duration = segments[-1].end if segments else (self._infer_duration_from_raw(raw) or fallback_duration)
        language = language_override or raw.get("language") or self.language
        return ASRResult(
            text=self._clean_spacing(text),
            segments=segments,
            language=language,
            duration=float(duration or 0.0),
        )

    def _safe_audio_duration(self, path: Path) -> float:
        try:
            with sf.SoundFile(str(path)) as descriptor:
                if descriptor.samplerate == 0:
                    return 0.0
                return float(len(descriptor) / descriptor.samplerate)
        except Exception:  # pragma: no cover - best effort
            return 0.0

    @staticmethod
    def _normalise_text(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return "".join(str(item) for item in value).strip()
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _seconds(value: Any) -> Optional[float]:
        if isinstance(value, Integral):
            return float(value) / 1000.0
        return ASRModule._to_float(value)

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _infer_duration_from_raw(raw: Dict[str, Any]) -> Optional[float]:
        timestamp_lists = raw.get("timestamp") or raw.get("timestamp_list")
        if isinstance(timestamp_lists, list) and timestamp_lists:
            last_entry = timestamp_lists[-1]
            if isinstance(last_entry, (list, tuple)) and last_entry:
                candidate = ASRModule._seconds(last_entry[-1])
            else:
                candidate = ASRModule._seconds(last_entry)
            if candidate is not None:
                return candidate
        if isinstance(raw.get("duration"), (int, float)):
            return float(raw["duration"])
        return None

    def _clean_spacing(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        text = self._CJK_SPACE_RE.sub("", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
