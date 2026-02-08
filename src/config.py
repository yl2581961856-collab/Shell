from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


@dataclass(frozen=True)
class ModelPaths:
    asr: str
    vad: str
    punc: str
    spk: str


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    device_id: int


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int


@dataclass(frozen=True)
class AppConfig:
    model_paths: ModelPaths
    runtime: RuntimeConfig
    server: ServerConfig


def _get(d: Dict[str, Any], key: str, default: Any) -> Any:
    value = d.get(key)
    return default if value is None else value


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    mp = data.get("model_paths", {})
    rt = data.get("runtime", {})
    sv = data.get("server", {})

    model_paths = ModelPaths(
        asr=_get(mp, "asr", "/app/data/models/asr"),
        vad=_get(mp, "vad", "/app/data/models/vad"),
        punc=_get(mp, "punc", "/app/data/models/punc"),
        spk=_get(mp, "spk", "/app/data/models/speaker"),
    )
    runtime = RuntimeConfig(
        device=_get(rt, "device", "ascend"),
        device_id=int(_get(rt, "device_id", 0)),
    )
    server = ServerConfig(
        host=_get(sv, "host", "0.0.0.0"),
        port=int(_get(sv, "port", 6008)),
    )

    return AppConfig(model_paths=model_paths, runtime=runtime, server=server)
