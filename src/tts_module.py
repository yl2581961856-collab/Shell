"""Text-to-speech providers and factory helpers."""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol
import wave

try:
    import requests
except ImportError as exc:  # pragma: no cover - requests should be installed with requirements
    raise RuntimeError("`requests` is required for the TTS client. Install with `pip install requests`.") from exc

try:  # Optional dependency for CosyVoice runtime
    import torch  # type: ignore
except ImportError:  # pragma: no cover - torch might be absent in lightweight envs
    torch = None  # type: ignore

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TTSResponse:
    audio_path: Path
    format: str
    sample_rate: int


class BaseTTS(Protocol):
    """Protocol implemented by all TTS backends."""

    def synthesize(
        self,
        text: str,
        output_dir: str | Path,
        *,
        voice: Optional[str] = None,
        stream: bool = False,
        **kwargs,
    ) -> TTSResponse:
        ...


class HiggsAudioTTS:
    """Minimal client for a HiggsAudio text-to-speech server."""

    def __init__(
        self,
        api_base: str,
        api_key: Optional[str] = None,
        default_voice: str = "default",
        sample_rate: int = 22050,
        audio_format: str = "wav",
        timeout: int = 60,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.default_voice = default_voice
        self.sample_rate = sample_rate
        self.audio_format = audio_format
        self.timeout = timeout
        logger.debug("Initialized HiggsAudioTTS with base=%s", self.api_base)

    def synthesize(
        self,
        text: str,
        output_dir: str | Path,
        *,
        voice: Optional[str] = None,
        stream: bool = False,
    ) -> TTSResponse:
        if not text.strip():
            raise ValueError("Text-to-speech input must not be empty")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"tts_{abs(hash(text))}.{self.audio_format}"

        payload = {
            "text": text,
            "voice": voice or self.default_voice,
            "sample_rate": self.sample_rate,
            "format": self.audio_format,
            "stream": stream,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.api_base}/tts"
        logger.debug("Requesting TTS voice=%s stream=%s", payload["voice"], stream)
        response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        if response.status_code >= 400:
            raise RuntimeError(f"HiggsAudio TTS request failed with {response.status_code}: {response.text}")

        data = response.json()
        audio_b64 = data.get("audio_base64")
        if not audio_b64:
            raise RuntimeError("TTS response missing `audio_base64` field")

        audio_bytes = base64.b64decode(audio_b64)
        output_path.write_bytes(audio_bytes)
        logger.info("Generated speech audio at %s", output_path)
        return TTSResponse(audio_path=output_path, format=self.audio_format, sample_rate=self.sample_rate)


class CosyVoiceTTS:
    """Local CosyVoice text-to-speech backend."""

    def __init__(
        self,
        model_dir: str | Path,
        speaker: Optional[str] = None,
        sample_rate: int = 24000,
        audio_format: str = "wav",
        device: Optional[str] = None,
        text_language: Optional[str] = None,
        prompt_language: Optional[str] = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.speaker = speaker
        self.sample_rate = sample_rate
        self.audio_format = audio_format
        self.device = device
        self.text_language = text_language
        self.prompt_language = prompt_language
        if not self.model_dir.exists():
            logger.warning("CosyVoice model directory %s does not exist yet", self.model_dir)
        self._engine = None

    def _load_engine(self):
        if self._engine is not None:
            return self._engine
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "CosyVoice library is not installed. Install it from the official repository to use the cosyvoice TTS provider."
            ) from exc

        kwargs = {}
        if self.device:
            kwargs["device"] = self.device
        logger.info("Loading CosyVoice model from %s", self.model_dir)
        self._engine = CosyVoice(str(self.model_dir), **kwargs)
        return self._engine

    @staticmethod
    def _to_numpy(audio: object) -> np.ndarray:
        if audio is None:
            raise RuntimeError("CosyVoice produced no audio output.")
        if torch is not None and isinstance(audio, torch.Tensor):
            return audio.detach().cpu().numpy()
        if isinstance(audio, np.ndarray):
            return audio
        if isinstance(audio, (list, tuple)):
            return np.asarray(audio)
        raise RuntimeError(f"Unsupported CosyVoice audio type: {type(audio)!r}")

    def _write_wav(self, audio: np.ndarray, sample_rate: int, output_path: Path) -> None:
        if audio.ndim == 1:
            channels = 1
            samples = audio
        elif audio.ndim == 2:
            if audio.shape[0] == 1:
                channels = 1
                samples = audio.squeeze(axis=0)
            elif audio.shape[0] <= audio.shape[1]:
                channels = audio.shape[0]
                samples = audio.T  # (samples, channels)
            else:
                channels = audio.shape[1]
                samples = audio
        else:
            raise RuntimeError(f"Unexpected audio shape from CosyVoice: {audio.shape}")

        samples = np.clip(samples.astype(np.float32), -1.0, 1.0)
        pcm = (samples * 32767.0).astype(np.int16)
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())

    def synthesize(
        self,
        text: str,
        output_dir: str | Path,
        *,
        voice: Optional[str] = None,
        stream: bool = False,
        **kwargs,
    ) -> TTSResponse:
        if stream:
            raise NotImplementedError("CosyVoice streaming TTS is not implemented.")
        if not text.strip():
            raise ValueError("Text-to-speech input must not be empty")

        engine = self._load_engine()

        speaker_id = voice or self.speaker
        if speaker_id is None:
            raise RuntimeError("CosyVoice requires a `speaker` configuration or the `voice` parameter.")

        inference_kwargs = {key: value for key, value in kwargs.items() if value is not None}
        inference_kwargs.setdefault("spk_id", speaker_id)
        if self.text_language:
            inference_kwargs.setdefault("text_language", self.text_language)
        if self.prompt_language:
            inference_kwargs.setdefault("prompt_language", self.prompt_language)
        if "prompt" in inference_kwargs and "prompt_text" not in inference_kwargs:
            inference_kwargs["prompt_text"] = inference_kwargs["prompt"]

        audio_output = None
        inferred_sample_rate = self.sample_rate

        if hasattr(engine, "tts"):
            audio_output = engine.tts(text=text, **inference_kwargs)
        elif hasattr(engine, "infer"):
            try:
                audio_output = engine.infer(text=text, **inference_kwargs)
            except TypeError:
                payload_keys = {
                    "spk_id",
                    "text_language",
                    "prompt_language",
                    "prompt",
                    "prompt_text",
                    "prompt_audio",
                    "prompt_audio_path",
                    "style",
                    "emotion",
                }
                remaining_kwargs = dict(inference_kwargs)
                payload = {"text": text}
                for key in list(payload_keys):
                    if key in remaining_kwargs:
                        value = remaining_kwargs.pop(key)
                        if value is None:
                            continue
                        if key == "prompt":
                            payload["prompt_text"] = value
                        else:
                            payload[key] = value
                audio_output = engine.infer([payload], **remaining_kwargs)
        else:
            raise RuntimeError("CosyVoice backend must expose `tts` or `infer` method.")

        audio_np, inferred_sample_rate = self._extract_audio(audio_output, inferred_sample_rate)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"tts_{abs(hash((text, speaker_id)))}.{self.audio_format}"

        self._write_wav(audio_np, inferred_sample_rate, output_path)
        logger.info("Generated CosyVoice speech audio at %s", output_path)

        return TTSResponse(audio_path=output_path, format=self.audio_format, sample_rate=inferred_sample_rate)

    def _extract_audio(self, audio_output: object, default_sample_rate: int) -> tuple[np.ndarray, int]:
        def _resolve(obj: object, current_rate: int) -> tuple[np.ndarray, int]:
            if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], (int, float)):
                return _resolve(obj[0], int(obj[1]))
            if isinstance(obj, list):
                if not obj:
                    raise RuntimeError("CosyVoice returned empty audio output.")
                if all(isinstance(item, dict) for item in obj):
                    return _resolve(obj[0], current_rate)
                if len(obj) == 1:
                    return _resolve(obj[0], current_rate)
                return self._to_numpy(obj[0]), current_rate
            if isinstance(obj, dict):
                sample_rate = obj.get("tts_sample_rate")
                if isinstance(sample_rate, (list, tuple)):
                    sample_rate = sample_rate[0] if sample_rate else None
                if not sample_rate:
                    sample_rate = obj.get("sample_rate") or obj.get("sample_rate_hz")
                if isinstance(sample_rate, (int, float)):
                    current_rate = int(sample_rate)
                for key in ("tts_speech", "tts_audio", "audio", "tts_wav", "wav"):
                    if key in obj and obj[key] is not None:
                        value = obj[key]
                        if isinstance(value, (list, tuple)) and len(value) == 1:
                            value = value[0]
                        return _resolve(value, current_rate)
                raise RuntimeError(f"CosyVoice response missing audio content. Available keys: {list(obj.keys())}")
            return self._to_numpy(obj), current_rate

        audio_np, sample_rate = _resolve(audio_output, default_sample_rate)
        return audio_np, sample_rate


def create_tts(provider_config: dict) -> BaseTTS:
    """Factory that returns a configured TTS backend based on ``provider_config``."""

    provider = (provider_config or {}).get("provider", "higgs").lower()
    if provider == "cosyvoice":
        model_dir = provider_config.get("model_dir")
        if not model_dir:
            raise RuntimeError("CosyVoice provider requires `model_dir` in configuration.")
        return CosyVoiceTTS(
            model_dir=model_dir,
            speaker=provider_config.get("speaker"),
            sample_rate=provider_config.get("sample_rate", 24000),
            audio_format=provider_config.get("format", "wav"),
            device=provider_config.get("device"),
            text_language=provider_config.get("text_language"),
            prompt_language=provider_config.get("prompt_language"),
        )

    if provider == "higgs":
        api_base = provider_config.get("api_base")
        if not api_base:
            raise RuntimeError("HiggsAudio provider requires `api_base` in configuration.")
        return HiggsAudioTTS(
            api_base=api_base,
            api_key=provider_config.get("api_key"),
            default_voice=provider_config.get("default_voice", "default"),
            sample_rate=provider_config.get("sample_rate", 22050),
            audio_format=provider_config.get("format", "wav"),
            timeout=provider_config.get("timeout", 60),
        )

    raise RuntimeError(f"Unsupported TTS provider: {provider}")

