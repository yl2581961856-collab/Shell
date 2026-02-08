from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import sys
from typing import Any, Dict

import numpy as np
import torch

logger = logging.getLogger("asr.sensevoice")

class SenseVoiceBackend:
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        code_path = os.getenv("SENSEVOICE_CODE_PATH", "/app/data/SenseVoice")
        if code_path not in sys.path:
            sys.path.insert(0, code_path)

        try:
            from model import SenseVoiceSmall  # type: ignore
        except Exception as exc:
            raise ImportError(f"Failed to import SenseVoice code from {code_path}") from exc

        config_path = os.path.join(model_path, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing SenseVoice config: {config_path}")

        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        tokenizer_conf = config.get("tokenizer_conf") or {}
        model_conf = self._normalize_model_conf(config, model_path, tokenizer_conf)

        self.device = device
        self.model, api_tokenizer = self._build_model(SenseVoiceSmall, config, model_conf, code_path, model_path, device)

        ckpt = os.path.join(model_path, "model.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Model weights not found: {ckpt}")

        if hasattr(self.model, "load_state_dict"):
            state_dict = torch.load(ckpt, map_location="cpu")
            self.model.load_state_dict(state_dict)
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

        self.tokenizer = api_tokenizer
        if self.tokenizer is None:
            try:
                tokenizer_cls = self._load_tokenizer_class(code_path)
                if tokenizer_cls is not None:
                    self.tokenizer = tokenizer_cls(tokenizer_conf)
            except Exception as exc:
                logger.warning("SenseVoice tokenizer init failed: %s", exc)

    def transcribe_segment(self, audio_data: np.ndarray, language: str = "auto") -> Dict[str, Any]:
        if audio_data.size == 0:
            return {"text": ""}

        speech = torch.from_numpy(audio_data).to(self.device)
        speech_lengths = torch.tensor([speech.shape[0]], device=self.device).int()

        try:
            res = self.model.inference(
                data_in=speech.unsqueeze(0),
                data_lengths=speech_lengths,
                language=language,
                use_itn=True,
            )
            text = res[0][0].get("text", "")
        except Exception as exc:
            return {"text": "", "error": str(exc)}

        return {"text": self._clean_text(text), "raw": str(res)}

    @staticmethod
    def _normalize_model_conf(
        config: Dict[str, Any],
        model_path: str,
        tokenizer_conf: Dict[str, Any],
    ) -> Dict[str, Any]:
        model_conf = config.get("model_conf") or {}

        # Some SenseVoice configs keep sub-confs at top-level.
        for key in (
            "encoder_conf",
            "decoder_conf",
            "predictor_conf",
            "frontend_conf",
            "specaug_conf",
            "cmvn_conf",
        ):
            if model_conf.get(key) is None and config.get(key) is not None:
                model_conf[key] = config.get(key)

        # Some configs keep component names at top-level.
        for key in (
            "encoder",
            "decoder",
            "predictor",
            "frontend",
            "postencoder",
            "postdecoder",
        ):
            if model_conf.get(key) is None and config.get(key) is not None:
                model_conf[key] = config.get(key)

        # Replace None sub-confs with empty dicts to avoid **None
        for key, value in list(model_conf.items()):
            if key.endswith("_conf") and value is None:
                model_conf[key] = {}

        if model_conf.get("encoder") is None:
            logger.warning("SenseVoice config missing encoder; model init may fail.")
        if "encoder_conf" not in model_conf:
            logger.warning("SenseVoice config missing encoder_conf; model init may fail.")

        vocab_size = model_conf.get("vocab_size")
        if not vocab_size or int(vocab_size) <= 0:
            inferred = SenseVoiceBackend._infer_vocab_size(model_path, tokenizer_conf, config)
            if inferred:
                model_conf["vocab_size"] = inferred
                logger.info("SenseVoice vocab_size inferred: %s", inferred)
            else:
                logger.warning("SenseVoice vocab_size missing or invalid.")

        return model_conf

    @staticmethod
    def _infer_vocab_size(model_path: str, tokenizer_conf: Dict[str, Any], config: Dict[str, Any]) -> int:
        candidates = [
            tokenizer_conf.get("token_list"),
            tokenizer_conf.get("token_list_file"),
            tokenizer_conf.get("tokenizer_path"),
            config.get("token_list"),
            os.path.join(model_path, "tokens.json"),
            os.path.join(model_path, "tokens.txt"),
        ]

        for path in candidates:
            if not path:
                continue
            path = str(path)
            if not os.path.isabs(path):
                path = os.path.join(model_path, path)
            if not os.path.exists(path):
                continue
            try:
                if path.endswith(".json"):
                    import json

                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        return len(data)
                    if isinstance(data, dict):
                        return len(data)
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        lines = [line.strip() for line in f if line.strip()]
                    if lines:
                        return len(lines)
            except Exception as exc:
                logger.warning("Failed to load token list from %s: %s", path, exc)
                continue
        return 0

    @staticmethod
    def _build_model(
        model_cls,
        config: Dict[str, Any],
        model_conf: Dict[str, Any],
        code_path: str,
        model_path: str,
        device: str,
    ):
        # Try repo-provided API first if available.
        api_model, api_tokenizer = SenseVoiceBackend._try_api_build(code_path, model_path, device, config)
        if api_model is not None:
            return api_model, api_tokenizer

        # Prefer class helper if available.
        if hasattr(model_cls, "from_config"):
            try:
                model = model_cls.from_config(config)  # type: ignore[attr-defined]
                return model, None
            except Exception as exc:
                logger.warning("SenseVoice from_config failed: %s", exc)

        return model_cls(**model_conf), None

    @staticmethod
    def _try_api_build(code_path: str, model_path: str, device: str, config: Dict[str, Any]):
        try:
            import api as sense_api  # type: ignore
        except Exception:
            return None, None

        candidates = ("load_model", "load_model_from_dir", "load_model_from_path", "get_model")
        device_id = 0
        if ":" in device:
            try:
                device_id = int(device.split(":")[-1])
            except ValueError:
                device_id = 0

        for name in candidates:
            fn = getattr(sense_api, name, None)
            if not callable(fn):
                continue
            res = SenseVoiceBackend._call_with_supported_args(
                fn,
                model_path=model_path,
                model_dir=model_path,
                model=model_path,
                device=device,
                device_id=device_id,
                config=config,
                code_path=code_path,
            )
            if res is None:
                continue
            model, tokenizer = SenseVoiceBackend._parse_build_result(res)
            if model is not None:
                return model, tokenizer
        return None, None

    @staticmethod
    def _call_with_supported_args(fn, **kwargs):
        try:
            sig = inspect.signature(fn)
            params = sig.parameters
            supported = {k: v for k, v in kwargs.items() if k in params}
            return fn(**supported)
        except Exception:
            return None

    @staticmethod
    def _parse_build_result(res):
        if isinstance(res, tuple):
            model = res[0] if res else None
            tokenizer = res[1] if len(res) > 1 else None
            return model, tokenizer
        if isinstance(res, dict):
            model = res.get("model") or res.get("net")
            tokenizer = res.get("tokenizer")
            return model, tokenizer
        return res, None

    @staticmethod
    def _load_tokenizer_class(code_path: str):
        # Try package import first
        try:
            from utils.tokenizer import SentencepiecesTokenizer  # type: ignore

            return SentencepiecesTokenizer
        except Exception:
            pass

        # Some repos don't ship utils as a package. Try adding utils/ to sys.path.
        utils_path = os.path.join(code_path, "utils")
        if os.path.isdir(utils_path) and utils_path not in sys.path:
            sys.path.insert(0, utils_path)

        try:
            from tokenizer import SentencepiecesTokenizer  # type: ignore

            return SentencepiecesTokenizer
        except Exception:
            pass

        # Fallback: load utils/tokenizer.py directly if it exists.
        tokenizer_py = os.path.join(utils_path, "tokenizer.py")
        if os.path.exists(tokenizer_py):
            spec = importlib.util.spec_from_file_location("sensevoice_tokenizer", tokenizer_py)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                return getattr(mod, "SentencepiecesTokenizer", None)

        return None
    @staticmethod
    def _clean_text(text: str) -> str:
        # Keep control tokens by default. Set SENSEVOICE_STRIP_TOKENS=1 to strip.
        strip_tokens = os.getenv("SENSEVOICE_STRIP_TOKENS", "0").strip().lower() in {"1", "true", "yes"}
        if not strip_tokens:
            return text.strip()
        import re

        cleaned = re.sub(r"<\\|[^>]+\\|>", "", text)
        return cleaned.strip()
