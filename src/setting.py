"""Configuration helpers for the voice assistant stack."""
from __future__ import annotations

import logging.config
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("`pyyaml` is required for configuration loading. Install with `pip install pyyaml`.") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_LOGGING_PATH = PROJECT_ROOT / "config" / "logging.conf"


def _expand_env(value: Any) -> Any:
    """Recursively expand environment variable placeholders inside config values."""

    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        return expanded
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return _expand_env(data)


def configure_logging(config_path: Optional[str | Path] = None) -> None:
    log_path = Path(config_path) if config_path else DEFAULT_LOGGING_PATH
    if log_path.exists():
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.config.fileConfig(
            log_path,
            disable_existing_loggers=False,
            defaults={"log_dir": log_dir.as_posix()},
        )
    else:
        logging.basicConfig(level=logging.INFO)
