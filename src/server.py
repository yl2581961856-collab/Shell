from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect
import uvicorn

from .asr_service import AsrService, AsrSession
from .config import load_config

app = FastAPI(title="ASR Service")
config = load_config()
asr = AsrService(config)

MAX_FRAME_BYTES = 1 * 1024 * 1024
IDLE_TIMEOUT_SEC = 30
SUPPORTED_SAMPLE_RATES = {16000}
SUPPORTED_CHANNELS = {1}
SUPPORTED_FMTS = {"pcm_s16le", "pcm16le", "s16le"}
SUPPORTED_ENCODINGS = {"pcm", "raw"}

# NOTE: WebSocket handlers are async and run on the event loop thread. ASR inference is
# synchronous and can be CPU/NPU bound, so we must offload it to a small worker pool.
ASR_MAX_WORKERS = int(os.getenv("ASR_MAX_WORKERS", "1"))
ASR_EXECUTOR = ThreadPoolExecutor(max_workers=ASR_MAX_WORKERS)


@dataclass
class SessionState:
    started: bool = False
    session_id: str = ""
    sr: int = 0
    ch: int = 0
    fmt: str = ""
    encoding: str = ""
    bytes_total: int = 0
    seq: int = 0
    started_at: float = 0.0


def _normalize_token(value: str) -> str:
    return value.lower().replace("-", "").replace("_", "").replace(" ", "")


def _fmt_ok(fmt: str) -> bool:
    return _normalize_token(fmt) in {_normalize_token(x) for x in SUPPORTED_FMTS}


def _encoding_ok(encoding: str) -> bool:
    return _normalize_token(encoding) in {_normalize_token(x) for x in SUPPORTED_ENCODINGS}


async def _send_error(ws: WebSocket, code: str, message: str, session_id: str = "") -> None:
    payload = {"type": "error", "code": code, "message": message}
    if session_id:
        payload["session_id"] = session_id
    await ws.send_json(payload)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _normalize_asr_result(result: Any) -> tuple[str, Optional[Any]]:
    if hasattr(result, "text"):
        text = getattr(result, "text", "")
        timestamps = getattr(result, "timestamps", None)
        return text, timestamps
    if isinstance(result, dict):
        text = result.get("text", "")
        timestamps = result.get("timestamps") or result.get("timestamp")
        return text, timestamps
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], str):
        return result[0], result[1]
    if isinstance(result, str):
        return result, None
    return "", None


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/asr/file")
async def asr_file(file: UploadFile = File(...)) -> JSONResponse:
    audio_bytes = await file.read()
    loop = asyncio.get_running_loop()
    session = asr.create_session(session_id=f"http-{uuid.uuid4().hex[:12]}")
    try:
        result = await loop.run_in_executor(ASR_EXECUTOR, session.transcribe_bytes, audio_bytes, False)
    finally:
        session.close()
    text, timestamps = _normalize_asr_result(result)
    payload: dict[str, Any] = {"text": text, "stub": asr.is_stub()}
    if timestamps is not None:
        payload["timestamps"] = timestamps
    return JSONResponse(payload)


@app.websocket("/asr/stream")
async def asr_stream(ws: WebSocket) -> None:
    await ws.accept()
    state = SessionState()
    loop = asyncio.get_running_loop()
    session: Optional[AsrSession] = None
    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=IDLE_TIMEOUT_SEC)
            except asyncio.TimeoutError:
                await _send_error(ws, "IDLE_TIMEOUT", "no data received", state.session_id)
                await ws.close(code=1000)
                return

            if msg.get("type") == "websocket.disconnect":
                return

            if msg.get("text") is not None:
                try:
                    payload = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await _send_error(ws, "INVALID_JSON", "invalid JSON frame", state.session_id)
                    await ws.close(code=1003)
                    return

                frame_type = payload.get("type")
                if frame_type in {"start", "config"}:
                    if state.started:
                        await _send_error(ws, "ALREADY_STARTED", "start frame already received", state.session_id)
                        await ws.close(code=1008)
                        return

                    sr = int(payload.get("sr", 0))
                    ch = int(payload.get("ch", 0))
                    fmt = str(payload.get("fmt", "")).strip()
                    encoding = str(payload.get("encoding", "")).strip()
                    session_id = str(payload.get("session_id", "")).strip() or uuid.uuid4().hex

                    if sr not in SUPPORTED_SAMPLE_RATES:
                        await _send_error(ws, "UNSUPPORTED_SR", f"unsupported sample rate: {sr}", session_id)
                        await ws.close(code=1003)
                        return
                    if ch not in SUPPORTED_CHANNELS:
                        await _send_error(ws, "UNSUPPORTED_CH", f"unsupported channels: {ch}", session_id)
                        await ws.close(code=1003)
                        return
                    if not fmt or not _fmt_ok(fmt):
                        await _send_error(ws, "UNSUPPORTED_FMT", f"unsupported format: {fmt}", session_id)
                        await ws.close(code=1003)
                        return
                    if encoding and not _encoding_ok(encoding):
                        await _send_error(
                            ws, "UNSUPPORTED_ENCODING", f"unsupported encoding: {encoding}", session_id
                        )
                        await ws.close(code=1003)
                        return

                    state.started = True
                    state.session_id = session_id
                    state.sr = sr
                    state.ch = ch
                    state.fmt = fmt
                    state.encoding = encoding or "pcm"
                    state.started_at = time.time()
                    session = asr.create_session(session_id=state.session_id)

                    await ws.send_json(
                        {
                            "type": "ack",
                            "session_id": state.session_id,
                            "sr": state.sr,
                            "ch": state.ch,
                            "fmt": state.fmt,
                            "encoding": state.encoding,
                            "ts": _now_ms(),
                        }
                    )
                    continue

                if frame_type == "end":
                    if not state.started:
                        await _send_error(ws, "MISSING_START", "start frame required before end", state.session_id)
                        await ws.close(code=1008)
                        return
                    if session is None:
                        session = asr.create_session(session_id=state.session_id)
                    result = await loop.run_in_executor(ASR_EXECUTOR, session.transcribe_bytes, b"", False)
                    final_text, final_timestamps = _normalize_asr_result(result)
                    payload: dict[str, Any] = {
                        "type": "final",
                        "text": final_text,
                        "session_id": state.session_id,
                        "bytes": state.bytes_total,
                        "seq": state.seq,
                        "ts": _now_ms(),
                        "stub": asr.is_stub(),
                    }
                    if final_timestamps is not None:
                        payload["timestamps"] = final_timestamps
                    await ws.send_json(payload)
                    session.close()
                    await ws.close(code=1000)
                    return

                await _send_error(ws, "INVALID_FRAME", "unknown text frame type", state.session_id)
                await ws.close(code=1003)
                return

            if msg.get("bytes") is not None:
                if not state.started:
                    await _send_error(ws, "MISSING_START", "start frame required before audio", state.session_id)
                    await ws.close(code=1008)
                    return

                chunk = msg["bytes"]
                if len(chunk) > MAX_FRAME_BYTES:
                    await _send_error(ws, "FRAME_TOO_LARGE", "audio frame too large", state.session_id)
                    await ws.close(code=1009)
                    return

                state.bytes_total += len(chunk)
                state.seq += 1
                if session is None:
                    await _send_error(ws, "MISSING_START", "start frame required before audio", state.session_id)
                    await ws.close(code=1008)
                    return
                result = await loop.run_in_executor(ASR_EXECUTOR, session.transcribe_bytes, chunk, True)
                partial, partial_timestamps = _normalize_asr_result(result)
                payload = {
                    "type": "partial",
                    "text": partial,
                    "session_id": state.session_id,
                    "bytes": state.bytes_total,
                    "seq": state.seq,
                    "ts": _now_ms(),
                    "stub": asr.is_stub(),
                }
                if partial_timestamps is not None:
                    payload["timestamps"] = partial_timestamps
                await ws.send_json(payload)
                continue

            await _send_error(ws, "INVALID_FRAME", "unsupported frame type", state.session_id)
            await ws.close(code=1003)
            return
    except WebSocketDisconnect:
        return
    finally:
        if session is not None:
            session.close()


def main() -> None:
    uvicorn.run(
        "src.server:app",
        host=config.server.host,
        port=config.server.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
