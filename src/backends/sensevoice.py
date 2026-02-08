from __future__ import annotations

import os
import sys
from typing import Any, Dict

import numpy as np
import torch


class SenseVoiceBackend:
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        code_path = os.getenv("SENSEVOICE_CODE_PATH", "/app/data/SenseVoice")
        if code_path not in sys.path:
            sys.path.insert(0, code_path)

        try:
            from model import SenseVoiceSmall  # type: ignore
            from utils.tokenizer import SentencepiecesTokenizer  # type: ignore
        except Exception as exc:
            raise ImportError(f"Failed to import SenseVoice code from {code_path}") from exc

        config_path = os.path.join(model_path, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing SenseVoice config: {config_path}")

        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        model_conf = config.get("model_conf") or {}
        tokenizer_conf = config.get("tokenizer_conf") or {}

        self.device = device
        self.model = SenseVoiceSmall(**model_conf)

        ckpt = os.path.join(model_path, "model.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Model weights not found: {ckpt}")

        state_dict = torch.load(ckpt, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = SentencepiecesTokenizer(tokenizer_conf)

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
    def _clean_text(text: str) -> str:
        # Keep control tokens by default. Set SENSEVOICE_STRIP_TOKENS=1 to strip.
        strip_tokens = os.getenv("SENSEVOICE_STRIP_TOKENS", "0").strip().lower() in {"1", "true", "yes"}
        if not strip_tokens:
            return text.strip()
        import re

        cleaned = re.sub(r"<\\|[^>]+\\|>", "", text)
        return cleaned.strip()
