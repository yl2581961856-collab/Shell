"""Entry-point for running the voice assistant server or CLI pipeline."""
from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, Optional

import uvicorn

from .mcp_server import build_agent, build_conversation_manager, create_app
from .setting import configure_logging, load_config

logger = logging.getLogger(__name__)


def _load_config(config_path: Optional[str]) -> Dict[str, Any]:
    return load_config(config_path)


def run_server(config_path: Optional[str], host: str, port: int) -> None:
    app = create_app(config_path)
    uvicorn.run(app, host=host, port=port)


def run_cli(config_path: Optional[str], audio: Optional[str], text: Optional[str]) -> None:
    config = _load_config(config_path)
    manager = build_conversation_manager(config)
    if audio:
        turn = manager.handle_audio(audio)
    elif text:
        turn = manager.handle_text(text)
    else:
        raise ValueError("Either audio or text input must be provided for CLI mode")

    logger.info("Assistant response: %s", turn.assistant_text)
    if turn.audio:
        logger.info("Synthesized speech saved to %s", turn.audio)


def run_agent_cli(
    config_path: Optional[str],
    audio: Optional[str],
    text: Optional[str],
    auto_tts: Optional[bool],
) -> None:
    config = _load_config(config_path)
    manager = build_conversation_manager(config)
    agent = build_agent(manager, config)
    if agent is None:
        raise RuntimeError("Agent mode is disabled in the configuration file")

    result = agent.run(
        text=text if text and text.strip() else None,
        audio_path=audio,
        auto_tts=auto_tts,
    )
    logger.info("Agent response: %s", result.text)
    if result.audio_path:
        logger.info("Synthesized speech saved to %s", result.audio_path)
    if result.tool_messages:
        logger.info("Tools used: %s", ", ".join(result.tool_messages))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voice assistant orchestrator")
    parser.add_argument(
        "mode",
        choices=["server", "cli", "agent"],
        help="Run API server, deterministic pipeline CLI, or LangChain agent CLI",
    )
    parser.add_argument("--config", dest="config", help="Path to config.yaml override")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=9090, help="Server port")
    parser.add_argument("--audio", help="Audio file for CLI or agent mode")
    parser.add_argument("--text", help="Text input for CLI or agent mode")
    parser.add_argument(
        "--agent-auto-tts",
        choices=["true", "false"],
        help="Override agent auto TTS behaviour (defaults to config setting)",
    )
    return parser.parse_args()


def _parse_agent_auto_tts(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    return value.lower() == "true"


def main() -> None:
    args = parse_args()
    configure_logging()
    if args.mode == "server":
        run_server(args.config, args.host, args.port)
    elif args.mode == "cli":
        run_cli(args.config, args.audio, args.text)
    else:
        auto_tts = _parse_agent_auto_tts(args.agent_auto_tts)
        run_agent_cli(args.config, args.audio, args.text, auto_tts)


if __name__ == "__main__":
    main()
