from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from typing import Optional

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - optional runtime dependency
    sd = None

try:
    import websockets
except ImportError:  # pragma: no cover - optional runtime dependency
    websockets = None


def _require_websockets() -> None:
    if websockets is None:
        print("Missing dependency: websockets", file=sys.stderr)
        print("Install with: pip install websockets", file=sys.stderr)
        sys.exit(1)


def _require_sounddevice() -> None:
    if sd is None:
        print("Missing dependency: sounddevice", file=sys.stderr)
        print("Install with: pip install sounddevice", file=sys.stderr)
        sys.exit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Microphone streaming client for ASR WebSocket.")
    parser.add_argument("--uri", default="ws://127.0.0.1:8000/asr/stream")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate (Hz).")
    parser.add_argument("--ch", type=int, default=1, help="Channels.")
    parser.add_argument("--fmt", default="pcm_s16le", help="Audio format.")
    parser.add_argument("--encoding", default="pcm", help="Encoding.")
    parser.add_argument("--session-id", default="", help="Session id (optional).")
    parser.add_argument("--chunk-ms", type=int, default=100, help="Chunk size in milliseconds.")
    parser.add_argument("--queue", type=int, default=50, help="Max buffered chunks.")
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds (0 = until Ctrl+C).")
    parser.add_argument("--device", default=None, help="Input device index/name.")
    parser.add_argument("--list-devices", action="store_true", help="List input devices and exit.")
    return parser.parse_args()


async def _recv_loop(ws, stop_event: asyncio.Event) -> None:
    try:
        async for msg in ws:
            print(msg)
    except Exception as exc:  # pragma: no cover - best effort logging
        print(f"recv error: {exc}", file=sys.stderr)
    finally:
        stop_event.set()


async def _send_loop(ws, queue: asyncio.Queue, stop_event: asyncio.Event) -> None:
    while True:
        if stop_event.is_set() and queue.empty():
            break
        try:
            payload = await asyncio.wait_for(queue.get(), timeout=0.2)
        except asyncio.TimeoutError:
            continue
        await ws.send(payload)


async def _run(args: argparse.Namespace) -> None:
    _require_websockets()
    _require_sounddevice()

    if args.list_devices:
        print(sd.query_devices())
        return

    chunk_frames = int(args.sr * args.chunk_ms / 1000)
    if chunk_frames <= 0:
        raise ValueError("chunk-ms too small for given sample rate")

    session_id = args.session_id.strip() or f"py-{uuid.uuid4().hex[:12]}"
    start_frame = {
        "type": "start",
        "sr": args.sr,
        "ch": args.ch,
        "fmt": args.fmt,
        "encoding": args.encoding,
        "session_id": session_id,
    }
    end_frame = {"type": "end", "session_id": session_id}

    queue: asyncio.Queue = asyncio.Queue(maxsize=args.queue)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    drop_state = {"count": 0, "last_log": 0.0}

    def audio_callback(indata, frames, time_info, status) -> None:
        if status:
            print(f"audio status: {status}", file=sys.stderr)
        try:
            payload = indata.tobytes()
        except AttributeError:
            payload = bytes(indata)

        if queue.full():
            drop_state["count"] += 1
            now = time.time()
            if now - drop_state["last_log"] > 1.0:
                print(f"dropping audio frames: {drop_state['count']}", file=sys.stderr)
                drop_state["last_log"] = now
            return
        loop.call_soon_threadsafe(queue.put_nowait, payload)

    async with websockets.connect(args.uri, ping_interval=20, ping_timeout=20, max_size=None) as ws:
        await ws.send(json.dumps(start_frame))

        recv_task = asyncio.create_task(_recv_loop(ws, stop_event))
        send_task = asyncio.create_task(_send_loop(ws, queue, stop_event))

        try:
            with sd.RawInputStream(
                samplerate=args.sr,
                channels=args.ch,
                dtype="int16",
                blocksize=chunk_frames,
                callback=audio_callback,
                device=args.device,
            ):
                wait_tasks = [asyncio.create_task(stop_event.wait())]
                if args.duration and args.duration > 0:
                    wait_tasks.append(asyncio.create_task(asyncio.sleep(args.duration)))
                await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stop_event.set()
            try:
                await asyncio.wait_for(send_task, timeout=2.0)
            except asyncio.TimeoutError:
                pass
            await ws.send(json.dumps(end_frame))
            await ws.close()
            recv_task.cancel()


def main() -> None:
    args = _parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)


if __name__ == "__main__":
    main()
