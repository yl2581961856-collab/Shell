from __future__ import annotations

import logging
import os
import time
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import AppConfig

logger = logging.getLogger("asr.service")


@dataclass
class AsrResult:
    text: str
    timestamps: Optional[Any] = None


def _parse_chunk_size(value: str) -> Optional[Tuple[int, int, int]]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        return None
    nums = [int(p) for p in parts]
    if len(nums) != 3:
        raise ValueError("ASR_CHUNK_SIZE must have 3 comma-separated integers, e.g. 5,10,5")
    return nums[0], nums[1], nums[2]


def _pcm16le_to_float32(audio_bytes: bytes) -> np.ndarray:
    if not audio_bytes:
        return np.zeros((0,), dtype=np.float32)
    if len(audio_bytes) % 2 == 1:
        audio_bytes = audio_bytes[:-1]
    pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)
    return pcm16.astype(np.float32) / 32768.0


def _extract_asr_result(raw: Any) -> AsrResult:
    if isinstance(raw, AsrResult):
        return raw
    if isinstance(raw, str):
        return AsrResult(text=raw, timestamps=None)
    if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], str):
        return AsrResult(text=raw[0], timestamps=raw[1])

    item = None
    if isinstance(raw, dict):
        item = raw
    elif isinstance(raw, (list, tuple)) and raw:
        if isinstance(raw[0], dict):
            item = raw[0]

    if not item:
        return AsrResult(text="", timestamps=None)

    text = item.get("text") or item.get("sentence") or ""
    timestamps = item.get("timestamps") or item.get("timestamp") or item.get("time_stamp")
    return AsrResult(text=text, timestamps=timestamps)


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
        self._stub = True

        self._model = None
        self._backend = None
        self._backend_type = os.getenv("ASR_BACKEND", "funasr").strip().lower()
        self._torch = None
        self._streaming_kwargs: Dict[str, Any] = {}
        self._stream_mode = os.getenv("ASR_STREAM_MODE", "native").strip().lower()
        self._pseudo_max_bytes = self._parse_pseudo_max_bytes(os.getenv("ASR_PSEUDO_MAX_MS", ""))
        self._pseudo_step_bytes = self._parse_pseudo_step_bytes(os.getenv("ASR_PSEUDO_STEP_MS", ""))

    @staticmethod
    def _parse_pseudo_max_bytes(value: str) -> Optional[int]:
        if not value:
            return None
        try:
            max_ms = int(float(value))
        except ValueError:
            return None
        if max_ms <= 0:
            return None
        # 16kHz * 16-bit * mono => 32000 bytes/sec => 32 bytes/ms
        return max_ms * 32

    @staticmethod
    def _parse_pseudo_step_bytes(value: str) -> Optional[int]:
        if not value:
            return None
        try:
            step_ms = int(float(value))
        except ValueError:
            return None
        if step_ms <= 0:
            return None
        return step_ms * 32

    def is_stub(self) -> bool:
        return self._stub

    def load(self) -> None:
        # Thread-safe lazy init: model loading can be expensive and must not run twice.
        if self._ready:
            return
        with self._init_lock:
            if self._ready:
                return

            backend = self._backend_type
            if backend in {"sensevoice", "sensevoice_small", "sv"}:
                try:
                    import torch
                except Exception as exc:  # pragma: no cover - runtime dependency
                    raise RuntimeError("PyTorch is required but not installed") from exc

                device_id = int(self.config.runtime.device_id)
                device = "cpu"
                if self.config.runtime.device.lower() in {"ascend", "npu"}:
                    device = f"npu:{device_id}"
                    try:
                        import torch_npu  # noqa: F401

                        torch_npu.npu.set_device(device_id)
                    except Exception as exc:  # pragma: no cover - runtime dependency
                        raise RuntimeError("torch_npu is required for Ascend NPU execution") from exc
                elif torch.cuda.is_available():
                    device = f"cuda:{device_id}"

                self._torch = torch

                from .backends.sensevoice import SenseVoiceBackend

                self._backend = SenseVoiceBackend(
                    model_path=self.config.model_paths.asr,
                    device=device,
                )
                self._stub = False
                self._ready = True
                return

            try:
                from funasr import AutoModel
            except Exception as exc:  # pragma: no cover - runtime dependency
                raise RuntimeError("FunASR is required but not installed") from exc

            try:
                import torch
            except Exception as exc:  # pragma: no cover - runtime dependency
                raise RuntimeError("PyTorch is required but not installed") from exc

            device_id = int(self.config.runtime.device_id)
            device = "cpu"
            if self.config.runtime.device.lower() in {"ascend", "npu"}:
                device = f"npu:{device_id}"
                try:
                    import torch_npu  # noqa: F401

                    torch_npu.npu.set_device(device_id)
                except Exception as exc:  # pragma: no cover - runtime dependency
                    raise RuntimeError("torch_npu is required for Ascend NPU execution") from exc

            self._torch = torch

            mp = self.config.model_paths
            model_kwargs: Dict[str, Any] = {
                "model": mp.asr,
                "device": device,
                "disable_update": True,
                "model_hub": "local",
                "trust_remote_code": True
            }
            if mp.vad:
                model_kwargs["vad_model"] = mp.vad
            if mp.punc:
                model_kwargs["punc_model"] = mp.punc
            if mp.spk:
                model_kwargs["spk_model"] = mp.spk

            self._model = AutoModel(**model_kwargs)

            # Optional streaming tuning parameters
            chunk_env = os.getenv("ASR_CHUNK_SIZE", "")
            if chunk_env:
                self._streaming_kwargs["chunk_size"] = _parse_chunk_size(chunk_env)

            enc_lookback = os.getenv("ASR_ENCODER_CHUNK_LOOK_BACK", "")
            if enc_lookback:
                self._streaming_kwargs["encoder_chunk_look_back"] = int(enc_lookback)

            dec_lookback = os.getenv("ASR_DECODER_CHUNK_LOOK_BACK", "")
            if dec_lookback:
                self._streaming_kwargs["decoder_chunk_look_back"] = int(dec_lookback)

            if "chunk_size" not in self._streaming_kwargs and not mp.vad:
                # Default streaming chunk_size works for ASR-only. VAD expects an int (ms),
                # so we skip the tuple default when VAD is enabled to avoid type errors.
                self._streaming_kwargs["chunk_size"] = (5, 10, 5)
            if "encoder_chunk_look_back" not in self._streaming_kwargs:
                self._streaming_kwargs["encoder_chunk_look_back"] = 4
            if "decoder_chunk_look_back" not in self._streaming_kwargs:
                self._streaming_kwargs["decoder_chunk_look_back"] = 1

            self._stub = False
            self._ready = True

    def create_session(self, session_id: Optional[str] = None) -> "AsrSession":
        self.load()
        sid = (session_id or "").strip() or uuid.uuid4().hex
        return AsrSession(service=self, session_id=sid)

    # Convenience one-shot API (non-streaming). Prefer create_session() for streaming.
    def transcribe_bytes(self, audio_bytes: bytes) -> AsrResult:
        session = self.create_session(session_id=f"oneshot-{uuid.uuid4().hex[:12]}")
        try:
            return session.transcribe_bytes(audio_bytes, stream=False)
        finally:
            session.close()


class AsrSession:
    """Per-connection ASR session.

    Put streaming/decoder state here.

    Threading: This object will typically be used from an executor thread. We keep a
    session-level lock so accidental concurrent calls won't corrupt internal state.
    """

    def __init__(self, service: AsrService, session_id: str) -> None:
        self._service = service
        self.session_id = session_id

        self._lock = threading.Lock()
        self._closed = False
        self._bytes_total = 0
        self._cache: Dict[str, Any] = {}
        self._audio_buffer = bytearray()
        self._pseudo_last_emit_bytes = 0

    @property
    def bytes_total(self) -> int:
        return self._bytes_total

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cache.clear()
            self._audio_buffer.clear()

    def reset_stream_state(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._cache.clear()
            self._audio_buffer.clear()
            self._pseudo_last_emit_bytes = 0

    def transcribe_bytes(self, audio_bytes: bytes, stream: bool = False) -> AsrResult:
        # Keep this method synchronous. The server should call it via run_in_executor.
        self._service.load()

        with self._lock:
            if self._closed:
                return AsrResult(text="", timestamps=None)

            start_ts = time.perf_counter()
            if audio_bytes:
                self._bytes_total += len(audio_bytes)
                self._audio_buffer.extend(audio_bytes)

            is_final = not stream

            model = self._service._model
            backend = self._service._backend
            if model is None and backend is None:
                return AsrResult(text="", timestamps=None)

            torch = self._service._torch
            if torch is None:
                return AsrResult(text="", timestamps=None)

            with torch.no_grad():
                if stream:
                    use_pseudo = self._service._stream_mode in {"pseudo", "llm"} or backend is not None
                    if use_pseudo:
                        if self._service._pseudo_step_bytes:
                            if (self._bytes_total - self._pseudo_last_emit_bytes) < self._service._pseudo_step_bytes:
                                return AsrResult(text="", timestamps=None)
                            self._pseudo_last_emit_bytes = self._bytes_total
                        if self._service._pseudo_max_bytes and len(self._audio_buffer) > self._service._pseudo_max_bytes:
                            self._audio_buffer = self._audio_buffer[-self._service._pseudo_max_bytes :]
                        audio = _pcm16le_to_float32(bytes(self._audio_buffer))
                        if backend is not None:
                            result = backend.transcribe_segment(audio)
                        else:
                            if hasattr(model, "generate"):
                                result = model.generate(input=audio)
                            else:
                                result = model(audio)
                    else:
                        audio = _pcm16le_to_float32(audio_bytes)
                        if hasattr(model, "generate"):
                            result = model.generate(
                                input=audio,
                                cache=self._cache,
                                is_final=False,
                                **self._service._streaming_kwargs,
                            )
                        else:
                            result = model(
                                audio,
                                cache=self._cache,
                                is_final=False,
                                **self._service._streaming_kwargs,
                            )
                else:
                    if not audio_bytes and self._audio_buffer:
                        audio_bytes = bytes(self._audio_buffer)
                    if not audio_bytes:
                        return AsrResult(text="", timestamps=None)
                    audio = _pcm16le_to_float32(audio_bytes)
                    if backend is not None:
                        result = backend.transcribe_segment(audio)
                    else:
                        if hasattr(model, "generate"):
                            result = model.generate(input=audio)
                        else:
                            result = model(audio)

            parsed = _extract_asr_result(result)
            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            logger.debug(
                "asr transcribe stream=%s bytes=%s text_len=%s elapsed_ms=%.2f",
                stream,
                len(audio_bytes),
                len(parsed.text or ""),
                elapsed_ms,
            )
            return parsed
