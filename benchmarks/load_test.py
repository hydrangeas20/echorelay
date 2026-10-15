#!/usr/bin/env python3
"""End-to-end streaming benchmark against a running EchoRelay server.

    python -m benchmarks.load_test --audio-dir eval_audio --streams 40 --concurrency 4 \\
        --output results/gpu_fp16_c4.json --label "T4 fp16 c=4"

Measurement boundaries (all client timestamps from ``time.monotonic()`` in a
single process, so they are mutually comparable):

* time to first partial (TTFP): first audio frame handed to the socket ->
  first *nonempty* ``partial`` event received. Includes network, buffering
  until ``partial_interval_s`` of audio has arrived, queueing, and decode.
* final latency: ``end`` message sent -> ``final`` event received.
* final inference RTF (server-reported): ``final.inference_ms / final.audio_duration_s``.
  Measured inside the worker thread around the final model call only. It
  EXCLUDES the partial decodes that ran during the stream, time spent queued
  for an inference slot, buffering, and network time, so it is not an
  end-to-end cost per audio second. Partial-decode RTF is reported separately;
  end-to-end behaviour is captured by final latency and throughput.
* failures: every stream runs under ``--stream-timeout-s`` (connect ->
  final). Streams that time out, get a server ``error`` event, or fail to
  connect are counted in ``streams_failed`` with their reason preserved.
* throughput: completed streams (and audio seconds) per wall-clock second,
  from the first connect to the last stream finishing. Warm-up streams are
  run beforehand and excluded.

Streams cycle through the evaluation files in sorted order, so a run is
reproducible given the same ``--audio-dir``, ``--streams``, and pacing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import websockets

from .common import (
    SAMPLE_RATE,
    Utterance,
    audio_manifest,
    corpus_wer,
    environment,
    load_audio_dir,
    percentiles,
    write_json,
)


METRIC_DEFINITIONS = {
    "time_to_first_partial_s": "client monotonic: first audio frame sent -> first nonempty partial received",
    "final_latency_s": "client monotonic: end message sent -> final received",
    "final_inference_rtf": (
        "server-reported final decode time / audio duration. Model call in the worker thread only; "
        "EXCLUDES partial decodes, queueing for an inference slot, buffering, and network time"
    ),
    "partial_inference_rtf": "server-reported partial decode time / audio covered, per partial; same exclusions",
    "throughput": "completed streams (and audio seconds) / wall clock from first connect to last stream end",
    "failures": "any stream without a final: server error event, connection error, or stream_timeout",
}


@dataclass
class StreamRecord:
    index: int
    utterance: str
    audio_s: float
    ok: bool = False
    error: str | None = None  # machine-readable failure category
    error_detail: str | None = None  # human-readable reason (server message, exception)
    t_connect: float | None = None
    t_first_audio: float | None = None
    t_first_partial: float | None = None
    t_end_sent: float | None = None
    t_final: float | None = None
    num_partials: int = 0
    final_text: str | None = None
    final_inference_ms: float | None = None
    final_audio_s: float | None = None
    partial_inference_ms: list[float] = field(default_factory=list)
    partial_audio_s: list[float] = field(default_factory=list)
    ready: dict | None = None


async def run_stream(url: str, utt: Utterance, index: int, chunk_ms: int, speed: float,
                     timeout_s: float) -> StreamRecord:
    """Run one stream under an overall deadline.

    The deadline covers connect -> final (or close). A stalled socket, a
    server that never sends ``final``, or a hung connect is recorded as a
    ``stream_timeout`` failure instead of hanging the benchmark.
    """
    rec = StreamRecord(index=index, utterance=utt.name, audio_s=utt.duration_s)
    try:
        async with asyncio.timeout(timeout_s):
            await _stream(url, utt, chunk_ms, speed, rec)
    except TimeoutError:
        rec.ok = False
        rec.error = "stream_timeout"
        stage = ("waiting for ready" if rec.ready is None else
                 "sending audio" if rec.t_end_sent is None else "waiting for final")
        rec.error_detail = f"no completion within {timeout_s:.1f}s (stalled while {stage})"
    except (OSError, websockets.WebSocketException) as exc:
        # The server closes with a non-1000 code right after an ``error``
        # event, which surfaces here as ConnectionClosedError. Keep the
        # server-reported reason; only classify as a connection failure if
        # nothing more specific was recorded.
        if rec.error is None:
            rec.error = f"connection:{type(exc).__name__}"
            rec.error_detail = str(exc) or type(exc).__name__
    except Exception as exc:  # never let one stream abort the whole run
        if rec.error is None:
            rec.error = f"client:{type(exc).__name__}"
            rec.error_detail = str(exc) or type(exc).__name__
    return rec


async def _stream(url: str, utt: Utterance, chunk_ms: int, speed: float, rec: StreamRecord) -> None:
    chunk_bytes = SAMPLE_RATE * chunk_ms // 1000 * 2
    rec.t_connect = time.monotonic()
    async with websockets.connect(url, max_size=None, open_timeout=30) as ws:
        first = json.loads(await ws.recv())
        if first.get("type") != "ready":
            rec.error = first.get("code", "unexpected_first_message")
            rec.error_detail = first.get("message")
            return
        rec.ready = first

        async def sender() -> None:
            t0 = None
            for i, off in enumerate(range(0, len(utt.pcm16), chunk_bytes)):
                await ws.send(utt.pcm16[off : off + chunk_bytes])
                if t0 is None:
                    t0 = rec.t_first_audio = time.monotonic()
                if speed > 0:
                    target = t0 + (i + 1) * chunk_ms / 1000 / speed
                    await asyncio.sleep(max(0.0, target - time.monotonic()))
            await ws.send(json.dumps({"type": "end"}))
            rec.t_end_sent = time.monotonic()

        send_task = asyncio.create_task(sender())
        try:
            async for raw in ws:
                now = time.monotonic()
                ev = json.loads(raw)
                if ev["type"] == "partial":
                    rec.num_partials += 1
                    rec.partial_inference_ms.append(ev["inference_ms"])
                    rec.partial_audio_s.append(ev["audio_duration_s"])
                    if ev["text"].strip() and rec.t_first_partial is None:
                        rec.t_first_partial = now
                elif ev["type"] == "final":
                    rec.t_final = now
                    rec.final_text = ev["text"]
                    rec.final_inference_ms = ev["inference_ms"]
                    rec.final_audio_s = ev["audio_duration_s"]
                    rec.ok = True
                elif ev["type"] == "error":
                    rec.error = ev["code"]
                    rec.error_detail = ev.get("message")
        finally:
            send_task.cancel()
            await asyncio.gather(send_task, return_exceptions=True)
        if not rec.ok and rec.error is None:
            rec.error = "closed_without_final"
            rec.error_detail = f"close code {ws.close_code}"


async def run_load(url: str, utts: list[Utterance], n: int, concurrency: int,
                   chunk_ms: int, speed: float, timeout_s: float,
                   offset: int = 0) -> tuple[list[StreamRecord], float]:
    sem = asyncio.Semaphore(concurrency)

    async def one(i: int) -> StreamRecord:
        async with sem:
            return await run_stream(url, utts[(offset + i) % len(utts)], offset + i, chunk_ms, speed,
                                    timeout_s)

    t0 = time.monotonic()
    records = await asyncio.gather(*(one(i) for i in range(n)))
    return list(records), time.monotonic() - t0


def summarize(records: list[StreamRecord], wall_s: float, utts: list[Utterance]) -> dict:
    ok = [r for r in records if r.ok]
    ttfp = [r.t_first_partial - r.t_first_audio for r in ok if r.t_first_partial and r.t_first_audio]
    final_lat = [r.t_final - r.t_end_sent for r in ok if r.t_final and r.t_end_sent]
    final_rtf = [r.final_inference_ms / 1000 / r.final_audio_s for r in ok if r.final_audio_s]
    partial_rtf = [ms / 1000 / a for r in ok for ms, a in zip(r.partial_inference_ms, r.partial_audio_s) if a]
    by_name = {u.name: u for u in utts}
    refs = [by_name[r.utterance].reference for r in ok if by_name[r.utterance].reference]
    hyps = [r.final_text or "" for r in ok if by_name[r.utterance].reference]
    return {
        "streams_total": len(records),
        "streams_ok": len(ok),
        "streams_failed": len(records) - len(ok),
        "errors": dict(Counter(r.error or "unknown" for r in records if not r.ok)),
        "failures": [
            {"index": r.index, "utterance": r.utterance, "error": r.error or "unknown",
             "detail": r.error_detail}
            for r in records if not r.ok
        ],
        "streams_without_nonempty_partial": sum(1 for r in ok if r.t_first_partial is None),
        "time_to_first_partial_s": percentiles(ttfp),
        "final_latency_s": percentiles(final_lat),
        "final_inference_rtf": percentiles(final_rtf),
        "partial_inference_rtf": percentiles(partial_rtf),
        "wall_clock_s": wall_s,
        "throughput_streams_per_s": len(ok) / wall_s if wall_s else None,
        "throughput_audio_s_per_s": sum(r.audio_s for r in ok) / wall_s if wall_s else None,
        "wer": corpus_wer(refs, hyps),
        "wer_num_utterances": len(refs),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws/transcribe")
    p.add_argument("--health-url", default=None, help="default: derived from --url")
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--streams", type=int, default=20, help="measured streams")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--warmup", type=int, default=2, help="unmeasured streams run first")
    p.add_argument("--chunk-ms", type=int, default=100)
    p.add_argument("--speed", type=float, default=1.0, help="1.0 = real-time pacing, 0 = unpaced")
    p.add_argument("--stream-timeout-s", type=float, default=120.0,
                   help="overall deadline per stream (connect -> final); exceeded streams are "
                        "recorded as stream_timeout failures")
    p.add_argument("--label", default="")
    p.add_argument("--output", required=True)
    a = p.parse_args()

    health_url = a.health_url or a.url.replace("ws://", "http://").replace("wss://", "https://").rsplit("/ws/", 1)[0] + "/health"
    health = json.loads(urllib.request.urlopen(health_url, timeout=10).read())
    utts = load_audio_dir(a.audio_dir)

    if a.warmup:
        warm, _ = asyncio.run(run_load(a.url, utts, a.warmup, 1, a.chunk_ms, 0.0, a.stream_timeout_s))
        print(f"warm-up: {sum(r.ok for r in warm)}/{len(warm)} ok")
    records, wall = asyncio.run(run_load(a.url, utts, a.streams, a.concurrency, a.chunk_ms, a.speed,
                                         a.stream_timeout_s))
    summary = summarize(records, wall, utts)
    ready = next((r.ready for r in records if r.ready), None)

    result = {
        "label": a.label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "clock": "time.monotonic (client); server inference timing also monotonic",
        "server": {"health": health, "ready_limits": ready and ready.get("limits")},
        "run_config": {
            "url": a.url, "streams": a.streams, "concurrency": a.concurrency, "warmup": a.warmup,
            "chunk_ms": a.chunk_ms, "speed": a.speed, "stream_timeout_s": a.stream_timeout_s,
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "environment_client": environment(),
        "audio": audio_manifest(utts),
        "summary": summary,
        "streams": [asdict(r) for r in records],
    }
    if health["model"]["backend"] == "fake":
        result["warning"] = "fake backend: harness check only, not a model measurement"
    write_json(a.output, result)

    s = summary
    print(f"backend={health['model']['backend']} model={health['model']['name']} "
          f"device={health['model']['device']} compute_type={health['model']['compute_type']}")
    print(f"ok {s['streams_ok']}/{s['streams_total']} failed={s['streams_failed']} errors={s['errors']}")
    for f in s["failures"][:5]:
        print(f"  failure #{f['index']} {f['utterance']}: {f['error']}: {f['detail']}")
    print("note: final_inference_rtf is model time for the final decode only "
          "(excludes partial decodes, queueing, and network)")
    for key in ("time_to_first_partial_s", "final_latency_s", "final_inference_rtf"):
        v = s[key]
        if v["n"]:
            print(f"{key}: p50={v['p50']:.3f} p95={v['p95']:.3f} (n={v['n']})")
    print(f"throughput: {s['throughput_streams_per_s']:.3f} streams/s, "
          f"{s['throughput_audio_s_per_s']:.2f} audio-s/s; WER={s['wer']}")
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
