from __future__ import annotations

import argparse
import json
import os
import time
import uuid
import wave
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


def _now_ms() -> int:
    return int(time.time() * 1000)


def _load_wav(path: str) -> Tuple[np.ndarray, int, int]:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        if sampwidth != 2:
            raise ValueError(f"unsupported sample width: {sampwidth} bytes")
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16)
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1).astype(np.int16)
    return audio.astype(np.float32) / 32768.0, sr, 1


def _load_pcm(path: str, sr: int, ch: int) -> Tuple[np.ndarray, int, int]:
    raw = open(path, "rb").read()
    if len(raw) % 2 == 1:
        raw = raw[:-1]
    audio = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1).astype(np.int16)
        ch = 1
    return audio.astype(np.float32) / 32768.0, sr, ch


def _extract_asr_result(raw: Any) -> Tuple[str, Optional[Any]]:
    if isinstance(raw, str):
        return raw, None
    if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], str):
        return raw[0], raw[1]
    item = None
    if isinstance(raw, dict):
        item = raw
    elif isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], dict):
        item = raw[0]
    if not item:
        return "", None
    text = item.get("text") or item.get("sentence") or ""
    timestamps = item.get("timestamps") or item.get("timestamp") or item.get("time_stamp")
    return text, timestamps


def _emit(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def _resolve_player() -> Callable[[np.ndarray, int], None]:
    try:
        import sounddevice as sd  # type: ignore

        def _play(chunk: np.ndarray, sr: int) -> None:
            if chunk.size == 0:
                return
            try:
                sd.play(chunk, sr, blocking=True)
            except Exception as exc:  # pragma: no cover - runtime dependency
                raise SystemExit(f"Audio playback failed (sounddevice): {exc}") from exc

        return _play
    except Exception:
        pass

    try:
        import simpleaudio as sa  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise SystemExit(
            "Playback requested but no audio backend is available. "
            "Install 'sounddevice' or 'simpleaudio'."
        ) from exc

    def _play(chunk: np.ndarray, sr: int) -> None:
        if chunk.size == 0:
            return
        pcm = np.clip(chunk, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype(np.int16)
        try:
            play_obj = sa.play_buffer(pcm16.tobytes(), 1, 2, sr)
            play_obj.wait_done()
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise SystemExit(f"Audio playback failed (simpleaudio): {exc}") from exc

    return _play


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fun-ASR-Nano pseudo-streaming evaluator.")
    parser.add_argument("--input", required=True, help="Input audio path (.wav or .pcm).")
    parser.add_argument("--input-format", default="", choices=["", "wav", "pcm"], help="Input format override.")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate for PCM input.")
    parser.add_argument("--ch", type=int, default=1, help="Channels for PCM input.")
    parser.add_argument("--chunk-ms", type=int, default=1200, help="Chunk size in milliseconds.")
    parser.add_argument("--update-every", type=int, default=1, help="Emit partial every N chunks.")
    parser.add_argument("--session-id", default="", help="Session id.")
    parser.add_argument("--model", default=os.getenv("FUNASR_NANO_MODEL", ""), help="Fun-ASR-Nano model path or id.")
    parser.add_argument("--device", default=os.getenv("FUNASR_NANO_DEVICE", "npu:0"), help="Device string.")
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play audio in real time while evaluating (requires sounddevice or simpleaudio).",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep per chunk to simulate real-time streaming.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.model:
        raise SystemExit("Missing --model or FUNASR_NANO_MODEL")

    try:
        from funasr import AutoModel
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise SystemExit(f"FunASR not available: {exc}") from exc

    model = AutoModel(model=args.model, device=args.device, disable_update=True)

    fmt = args.input_format.lower()
    if not fmt:
        fmt = "wav" if args.input.lower().endswith(".wav") else "pcm"

    if fmt == "wav":
        audio, sr, ch = _load_wav(args.input)
    else:
        audio, sr, ch = _load_pcm(args.input, args.sr, args.ch)

    if sr != 16000 or ch != 1:
        raise SystemExit(f"Expected 16k/mono audio, got sr={sr} ch={ch}")

    chunk_samples = int(sr * args.chunk_ms / 1000)
    if chunk_samples <= 0:
        raise SystemExit("chunk-ms too small")

    player = _resolve_player() if args.play else None

    session_id = args.session_id.strip() or f"nano-{uuid.uuid4().hex[:12]}"
    _emit(
        {
            "type": "ack",
            "session_id": session_id,
            "sr": sr,
            "ch": ch,
            "fmt": "pcm_s16le",
            "encoding": "pcm",
            "ts": _now_ms(),
        }
    )

    chunks = []
    bytes_total = 0
    seq = 0
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i : i + chunk_samples]
        if len(chunk) == 0:
            break
        chunks.append(chunk)
        bytes_total += len(chunk) * 2
        seq += 1

        if player is not None:
            player(chunk, sr)
        elif args.realtime:
            time.sleep(len(chunk) / sr)

        if seq % max(1, args.update_every) != 0:
            continue

        buf = np.concatenate(chunks, axis=0)
        t0 = time.perf_counter()
        res = model.generate(input=buf)
        t1 = time.perf_counter()
        text, timestamps = _extract_asr_result(res)
        audio_sec = len(buf) / sr
        rtf = (t1 - t0) / audio_sec if audio_sec > 0 else 0.0
        payload = {
            "type": "partial",
            "text": text,
            "session_id": session_id,
            "bytes": bytes_total,
            "seq": seq,
            "ts": _now_ms(),
            "rtf": round(rtf, 3),
        }
        if timestamps is not None:
            payload["timestamps"] = timestamps
        _emit(payload)

    buf = np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), dtype=np.float32)
    t0 = time.perf_counter()
    res = model.generate(input=buf)
    t1 = time.perf_counter()
    text, timestamps = _extract_asr_result(res)
    audio_sec = len(buf) / sr
    rtf = (t1 - t0) / audio_sec if audio_sec > 0 else 0.0
    payload = {
        "type": "final",
        "text": text,
        "session_id": session_id,
        "bytes": bytes_total,
        "seq": seq,
        "ts": _now_ms(),
        "rtf": round(rtf, 3),
    }
    if timestamps is not None:
        payload["timestamps"] = timestamps
    _emit(payload)


if __name__ == "__main__":
    main()
