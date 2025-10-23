"""Simple WebSocket client to demo streaming ASR with faster-whisper.

Usage examples:
    # Stream from a WAV file
    python tools/ws_asr_client.py --file CosyVoice/asset/zero_shot_prompt.wav --language zh

    # Stream from microphone (requires sounddevice)
    python tools/ws_asr_client.py --mic --language zh

When --mic is enabled, audio is captured from the default microphone in real
-time and chunked to the /ws/asr endpoint so you can observe near real-time output.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
from pathlib import Path
from typing import Optional

import torch
import websockets

try:
    import torchaudio
except ImportError:  # pragma: no cover - optional for mic mode
    torchaudio = None  # type: ignore

try:  # optional - microphone mode only
    import sounddevice as sd
except ImportError:  # pragma: no cover
    sd = None  # type: ignore


async def _receiver(ws: websockets.WebSocketClientProtocol) -> None:
    try:
        async for message in ws:
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                print(f"[server] {message}")
                continue
            msg_type = payload.get("type")
            if msg_type == "segment":
                print(
                    f"[segment] {payload.get('start', 0.0):.2f}s → "
                    f"{payload.get('end', 0.0):.2f}s : {payload.get('text', '')}"
                )
            elif msg_type == "metadata":
                print(
                    f"[metadata] language={payload.get('language')} "
                    f"duration={payload.get('duration')}s"
                )
            elif msg_type == "flush_complete":
                print("[info] flush complete\n")
            elif msg_type == "ready":
                print("[server] ready")
            elif msg_type == "reset_ack":
                print("[server] reset acknowledged")
            else:
                print(f"[server] {payload}")
    except websockets.ConnectionClosed:
        return


def _load_audio(file_path: Path, target_sr: int = 16000) -> tuple[torch.Tensor, int]:
    if torchaudio is None:
        raise RuntimeError("torchaudio is required for file streaming mode (pip install torchaudio).")
    waveform, sr = torchaudio.load(str(file_path))
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    return waveform.squeeze(0), sr


def _pcm16_bytes(samples: torch.Tensor) -> bytes:
    return (
        torch.clamp(samples, -1.0, 1.0)
        .mul(32767)
        .to(torch.int16)
        .numpy()
        .tobytes()
    )


async def stream_file(
    uri: str,
    file_path: Path,
    *,
    language: Optional[str],
    chunk_ms: float,
    flush_interval: int,
) -> None:
    waveform, sr = _load_audio(file_path)
    chunk_samples = max(int(sr * (chunk_ms / 1000.0)), 1)
    pcm16 = torch.clamp(waveform, -1.0, 1.0).mul(32767).to(torch.int16).numpy()

    async with websockets.connect(uri) as ws:
        greeting = await ws.recv()
        print(f"[server] {greeting}")
        recv_task = asyncio.create_task(_receiver(ws))

        pending_chunks = 0
        for start in range(0, len(pcm16), chunk_samples):
            chunk = pcm16[start : start + chunk_samples]
            if len(chunk) == 0:
                continue
            payload = {
                "type": "chunk",
                "data": base64.b64encode(chunk.tobytes()).decode(),
                "sample_rate": sr,
            }
            if language:
                payload["language"] = language
            await ws.send(json.dumps(payload, ensure_ascii=False))
            pending_chunks += 1
            if pending_chunks >= flush_interval:
                await ws.send(json.dumps({"type": "flush"}))
                pending_chunks = 0
                await asyncio.sleep(chunk_ms / 1000.0)

        if pending_chunks:
            await ws.send(json.dumps({"type": "flush"}))

        await asyncio.sleep(1.0)
        await ws.send(json.dumps({"type": "reset"}))
        await asyncio.sleep(0.5)
        recv_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await recv_task


async def stream_mic(
    uri: str,
    *,
    language: Optional[str],
    chunk_ms: float,
    flush_interval: int,
    sample_rate: int,
) -> None:
    if sd is None:
        raise RuntimeError("sounddevice is required for microphone streaming (pip install sounddevice).")

    chunk_samples = max(int(sample_rate * (chunk_ms / 1000.0)), 1)
    buffer = bytearray()
    pending_chunks = 0

    async with websockets.connect(uri) as ws:
        greeting = await ws.recv()
        print(f"[server] {greeting}")
        recv_task = asyncio.create_task(_receiver(ws))
        loop = asyncio.get_running_loop()

        def audio_callback(indata, frames, time, status):  # type: ignore[override]
            nonlocal pending_chunks
            if status:
                print(f"[mic warning] {status}")
            pcm16 = torch.from_numpy(indata[:, 0]).clamp(-1.0, 1.0).mul(32767).to(torch.int16).numpy()
            buffer.extend(pcm16.tobytes())
            while len(buffer) >= chunk_samples * 2:
                chunk = buffer[: chunk_samples * 2]
                del buffer[: chunk_samples * 2]
                payload = {
                    "type": "chunk",
                    "data": base64.b64encode(chunk).decode(),
                    "sample_rate": sample_rate,
                }
                if language:
                    payload["language"] = language
                asyncio.run_coroutine_threadsafe(ws.send(json.dumps(payload, ensure_ascii=False)), loop)
                pending_chunks += 1
                if pending_chunks >= flush_interval:
                    asyncio.run_coroutine_threadsafe(ws.send(json.dumps({"type": "flush"})), loop)
                    pending_chunks = 0

        with sd.InputStream(
            channels=1,
            samplerate=sample_rate,
            dtype="float32",
            blocksize=chunk_samples,
            callback=audio_callback,
        ):
            print("[mic] streaming... Press Ctrl+C to stop.")
            try:
                while True:
                    await asyncio.sleep(0.5)
            except KeyboardInterrupt:
                print("\n[mic] stopping...")

        if buffer:
            await ws.send(
                json.dumps(
                    {
                        "type": "chunk",
                        "data": base64.b64encode(buffer).decode(),
                        "language": language,
                        "sample_rate": sample_rate,
                    },
                    ensure_ascii=False,
                )
            )
        await ws.send(json.dumps({"type": "flush"}))
        await asyncio.sleep(1.0)
        await ws.send(json.dumps({"type": "reset"}))
        await asyncio.sleep(0.5)
        recv_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await recv_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming ASR WebSocket client")
    parser.add_argument(
        "--uri",
        default="ws://127.0.0.1:9090/ws/asr",
        help="ASR WebSocket endpoint",
    )
    parser.add_argument("--file", help="Path to audio file to stream")
    parser.add_argument("--mic", action="store_true", help="Stream from default microphone in real time")
    parser.add_argument("--language", default=None, help="Language hint (e.g., zh, en)")
    parser.add_argument("--chunk-ms", type=float, default=500.0, help="Chunk size in milliseconds")
    parser.add_argument(
        "--flush-interval",
        type=int,
        default=2,
        help="Send a flush after this many chunks (default: 2)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate when streaming from microphone",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mic:
        asyncio.run(
            stream_mic(
                args.uri,
                language=args.language,
                chunk_ms=args.chunk_ms,
                flush_interval=args.flush_interval,
                sample_rate=args.sample_rate,
            )
        )
    else:
        if not args.file:
            raise ValueError("You must provide --file when --mic is not set.")
        file_path = Path(args.file)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        asyncio.run(
            stream_file(
                args.uri,
                file_path,
                language=args.language,
                chunk_ms=args.chunk_ms,
                flush_interval=args.flush_interval,
            )
        )


if __name__ == "__main__":
    main()
