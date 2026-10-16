#!/usr/bin/env python3
"""Demo client: stream a WAV file to EchoRelay and print transcripts.

    python clients/stream_wav.py path/to/audio.wav
    python clients/stream_wav.py --synthetic 3       # 3 s test tone, no file needed

Audio is sent as 16 kHz mono PCM16 frames, paced at real time by default
(``--speed 0`` sends as fast as possible). Partial hypotheses are shown as
provisional and overwritten in place; only the final line is the transcript.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import wave

import numpy as np
import websockets

SAMPLE_RATE = 16_000


def load_wav(path: str) -> bytes:
    with wave.open(path, "rb") as wf:
        rate, channels, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width != 2:
        sys.exit(f"{path}: expected 16-bit PCM, got {8 * width}-bit. "
                 "Convert with: ffmpeg -i in.wav -ar 16000 -ac 1 -sample_fmt s16 out.wav")
    pcm = np.frombuffer(raw, dtype="<i2").reshape(-1, channels).astype(np.float32).mean(axis=1)
    if rate != SAMPLE_RATE:
        # Linear resampling: fine for a demo; use ffmpeg/sox for evaluation audio.
        print(f"note: resampling {rate} Hz -> {SAMPLE_RATE} Hz (linear)", file=sys.stderr)
        n_out = int(len(pcm) * SAMPLE_RATE / rate)
        pcm = np.interp(np.linspace(0, len(pcm) - 1, n_out), np.arange(len(pcm)), pcm)
    return np.clip(pcm, -32768, 32767).astype("<i2").tobytes()


def synthetic(seconds: float) -> bytes:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2").tobytes()


async def stream(url: str, pcm: bytes, chunk_ms: int, speed: float, quiet: bool) -> int:
    chunk_bytes = SAMPLE_RATE * chunk_ms // 1000 * 2
    async with websockets.connect(url, max_size=None) as ws:
        ready = json.loads(await ws.recv())
        if ready.get("type") != "ready":
            print(f"server refused stream: {ready}", file=sys.stderr)
            return 1
        print(f"connected session={ready['session_id']}", file=sys.stderr)

        async def sender() -> None:
            t0 = time.monotonic()
            for i, off in enumerate(range(0, len(pcm), chunk_bytes)):
                await ws.send(pcm[off : off + chunk_bytes])
                if speed > 0:  # pace against the wall clock, not per-chunk sleeps
                    target = t0 + (i + 1) * chunk_ms / 1000 / speed
                    await asyncio.sleep(max(0.0, target - time.monotonic()))
            await ws.send(json.dumps({"type": "end"}))

        send_task = asyncio.create_task(sender())
        status = 0
        try:
            async for raw in ws:
                event = json.loads(raw)
                if event["type"] == "partial":
                    if not quiet:
                        print(f"\r\033[2K[provisional] {event['text']}", end="", file=sys.stderr, flush=True)
                elif event["type"] == "final":
                    print("\r\033[2K", end="", file=sys.stderr)
                    print(event["text"])
                    print(f"(audio {event['audio_duration_s']:.2f}s, final decode "
                          f"{event['inference_ms']:.0f} ms)", file=sys.stderr)
                elif event["type"] == "error":
                    print(f"\nerror: {event['code']}: {event['message']}", file=sys.stderr)
                    status = 1
        finally:
            send_task.cancel()
            await asyncio.gather(send_task, return_exceptions=True)
        return status


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("wav", nargs="?", help="16-bit PCM WAV file")
    src.add_argument("--synthetic", type=float, metavar="SECONDS", help="send a test tone instead")
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws/transcribe")
    p.add_argument("--chunk-ms", type=int, default=100)
    p.add_argument("--speed", type=float, default=1.0, help="1.0 = real time, 0 = unpaced")
    p.add_argument("--quiet", action="store_true", help="hide provisional partials")
    a = p.parse_args()
    pcm = synthetic(a.synthetic) if a.synthetic else load_wav(a.wav)
    sys.exit(asyncio.run(stream(a.url, pcm, a.chunk_ms, a.speed, a.quiet)))


if __name__ == "__main__":
    main()
