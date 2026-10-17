#!/usr/bin/env python3
"""Stress matrix with explicit expected outcomes, against a running server.

    python -m benchmarks.stress --audio-dir eval_audio --streams 20 \\
        --concurrency 1 4 8 --burst-extra 4 --output-dir results/stress_fp16

Scenarios, in order (each writes its own result file):

1. ``c=N`` for each ``--concurrency`` value (real-time pacing).
2. ``burst``: ``limit + burst_extra`` connections opened at once and held,
   where ``limit`` is the server's max_concurrent_streams (from /metrics).
3. ``recovery``: the first concurrency level again, after everything above.

Expected outcomes (checked automatically; PASS/FAIL per check):

* every non-burst scenario: zero failed streams;
* after every scenario: server returns to idle (no active streams, buffered
  audio, in-flight or waiting inference) within 10 s;
* event-loop lag p99 sampled during each scenario stays below 50 ms;
* c=1: time-to-first-partial p95 <= pacing floor + 250 ms;
* burst: exactly ``burst_extra`` connections rejected with server_busy (1013),
  every admitted stream completes, peak active streams == limit;
* recovery: TTFP p95 within max(10 %, 50 ms) of the first run at that level.

Informational (reported, not pass/fail, because they are what Phase 2 is
measuring): queue waits, final-latency decomposition, partial deferrals and
staleness, GPU utilization and clocks, per-stream inference load.
Requires a server built from this branch (needs /metrics). Run it with
ECHORELAY_EVENT_TIMING=true to get the server-side decomposition.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import websockets

from . import sampler
from .analyze_results import pacing_floor_s
from .common import audio_manifest, concatenate, environment, load_audio_dir, write_json
from .load_test import SAMPLE_RATE, METRIC_DEFINITIONS, run_load, summarize


class Check:
    def __init__(self, scenario: str, name: str, ok: bool, detail: str) -> None:
        self.scenario, self.name, self.ok, self.detail = scenario, name, ok, detail

    def as_dict(self) -> dict:
        return {"scenario": self.scenario, "check": self.name, "pass": self.ok, "detail": self.detail}


async def wait_idle(metrics_url: str, timeout_s: float = 10.0) -> tuple[bool, dict | None]:
    deadline = time.monotonic() + timeout_s
    m = None
    while time.monotonic() < deadline:
        m = await asyncio.to_thread(sampler.fetch_json, metrics_url)
        if m and m["streams"]["active"] == 0 and m["streams"]["buffered_audio_bytes"] == 0 \
                and m["inference"]["in_flight"] == 0 and m["inference"]["waiting"] == 0:
            return True, m
        await asyncio.sleep(0.2)
    return False, m


async def run_paced(url, metrics_url, utts, streams, concurrency, a) -> dict:
    stop = asyncio.Event()
    sampling = asyncio.create_task(sampler.sample(metrics_url, a.sample_interval_s, stop))
    records, wall = await run_load(url, utts, streams, concurrency, a.chunk_ms, a.speed, a.stream_timeout_s)
    stop.set()
    samples = await sampling
    summary = summarize(records, wall, utts)
    summary["resources"] = sampler.summarize_samples(samples)
    return {"summary": summary, "streams": [asdict(r) for r in records], "resource_samples": samples}


async def run_burst(url, metrics_url, utt, limit: int, extra: int, a) -> dict:
    """Open limit+extra connections at once; admitted ones hold their slot
    until every connection has its first message, then stream one utterance."""
    gate = asyncio.Event()
    answered = 0
    total = limit + extra
    chunk = SAMPLE_RATE * a.chunk_ms // 1000 * 2

    async def one(i: int) -> dict:
        nonlocal answered
        out: dict = {"index": i}
        try:
            async with asyncio.timeout(a.stream_timeout_s):
                async with websockets.connect(url, max_size=None) as ws:
                    first = json.loads(await ws.recv())
                    answered += 1
                    if answered == total:
                        gate.set()
                    if first["type"] != "ready":
                        out["outcome"] = first.get("code")
                        try:
                            await ws.recv()
                        except websockets.ConnectionClosed as exc:
                            out["close_code"] = exc.rcvd.code if exc.rcvd else None
                        return out
                    await gate.wait()
                    for off in range(0, len(utt.pcm16), chunk):
                        await ws.send(utt.pcm16[off: off + chunk])
                        if a.speed > 0:
                            await asyncio.sleep(a.chunk_ms / 1000 / a.speed)
                    await ws.send(json.dumps({"type": "end"}))
                    events = []
                    try:
                        async for m in ws:
                            events.append(json.loads(m))
                    except websockets.ConnectionClosed:
                        pass
                    out["outcome"] = "completed" if events and events[-1]["type"] == "final" else (
                        events[-1].get("code") if events else "no_events")
                    out["close_code"] = ws.close_code
        except TimeoutError:
            out["outcome"] = "stream_timeout"
        except (OSError, websockets.WebSocketException) as exc:
            out["outcome"] = f"connection:{type(exc).__name__}"
        return out

    stop = asyncio.Event()
    sampling = asyncio.create_task(sampler.sample(metrics_url, a.sample_interval_s, stop))
    results = await asyncio.gather(*(one(i) for i in range(total)))
    stop.set()
    samples = await sampling
    return {"connections": results, "resource_samples": samples,
            "resources": sampler.summarize_samples(samples)}


def _p95(summary: dict, key: str) -> float | None:
    v = summary.get(key) or {}
    return v.get("p95")


async def main_async(a) -> int:
    url = a.url
    base = url.replace("ws://", "http://").replace("wss://", "https://").rsplit("/ws/", 1)[0]
    metrics_url = base + "/metrics"
    m0 = await asyncio.to_thread(sampler.fetch_json, metrics_url)
    if not m0 or "config" not in m0:
        raise SystemExit("server has no /metrics endpoint; run a server built from this branch")
    limit = m0["config"]["max_concurrent_streams"]
    utts = load_audio_dir(a.audio_dir)
    if a.concat_seconds:
        utts = concatenate(utts, a.concat_seconds)
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checks: list[Check] = []
    scenario_results: dict[str, dict] = {}

    if a.warmup:
        await run_load(url, utts, a.warmup, 1, a.chunk_ms, 0.0, a.stream_timeout_s)

    plan = [(f"c={c}", c) for c in a.concurrency]
    if a.burst_extra:
        plan.append(("burst", None))
    if a.concurrency:
        plan.append(("recovery", a.concurrency[0]))

    for name, c in plan:
        print(f"--- scenario {name}", flush=True)
        if name == "burst":
            r = await run_burst(url, metrics_url, utts[0], limit, a.burst_extra, a)
            outcomes = [x["outcome"] for x in r["connections"]]
            rejected = [x for x in r["connections"] if x["outcome"] == "server_busy"]
            completed = outcomes.count("completed")
            checks.append(Check(name, "rejected == burst_extra", len(rejected) == a.burst_extra,
                                f"rejected={len(rejected)} expected={a.burst_extra}"))
            checks.append(Check(name, "rejections close 1013",
                                all(x.get("close_code") == 1013 for x in rejected),
                                str(sorted({x.get('close_code') for x in rejected}, key=str))))
            checks.append(Check(name, "all admitted complete", completed == limit,
                                f"completed={completed} limit={limit} outcomes={sorted(set(outcomes))}"))
            res = r["resources"]
        else:
            r = await run_paced(url, metrics_url, utts, a.streams, c, a)
            s = r["summary"]
            checks.append(Check(name, "zero failed streams", s["streams_failed"] == 0,
                                f"failed={s['streams_failed']} errors={s['errors']}"))
            res = s["resources"]
            ttfp = _p95(s, "time_to_first_partial_s")
            if c == 1 and name != "recovery":
                floor = pacing_floor_s({"run_config": {"chunk_ms": a.chunk_ms, "speed": a.speed},
                                        "server": {"ready_limits": {"partial_interval_s":
                                                                    m0["config"]["partial_interval_s"]}}})
                if floor is not None and ttfp is not None:
                    checks.append(Check(name, "TTFP p95 <= floor + 250 ms", ttfp <= floor + 0.25,
                                        f"ttfp_p95={ttfp:.3f}s floor={floor:.3f}s"))
            if name == "recovery":
                first = scenario_results.get(f"c={c}", {}).get("summary", {})
                ref = _p95(first, "time_to_first_partial_s")
                if ref is not None and ttfp is not None:
                    tol = max(0.10 * ref, 0.05)
                    checks.append(Check(name, "TTFP p95 within max(10%, 50 ms) of first run",
                                        abs(ttfp - ref) <= tol,
                                        f"first={ref:.3f}s recovery={ttfp:.3f}s tol={tol:.3f}s"))
        lag = ((res or {}).get("loop_lag_max_s") or {}).get("max")
        idle, m_after = await wait_idle(metrics_url)
        checks.append(Check(name, "server idle within 10 s", idle,
                            "" if idle else json.dumps((m_after or {}).get("streams"))))
        p99s = [x["server"]["loop_lag_p99_s"] for x in r["resource_samples"]
                if x.get("server", {}).get("loop_lag_p99_s") is not None]
        if p99s:
            checks.append(Check(name, "event-loop lag p99 < 50 ms", max(p99s) < 0.05,
                                f"max sampled p99={max(p99s) * 1000:.1f} ms"
                                + (f", lifetime max {lag * 1000:.1f} ms" if lag is not None else "")))
        if name == "burst" and m_after:
            peak = m_after["streams"]["peak_active"]
            checks.append(Check(name, "peak active == limit", peak == limit, f"peak={peak} limit={limit}"))
        r["server_metrics_after"] = m_after
        scenario_results[name] = r
        write_json(out_dir / f"{name.replace('=', '')}.json", {
            "scenario": name, "label": a.label, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_config": {"url": url, "streams": a.streams, "concurrency": c, "chunk_ms": a.chunk_ms,
                           "speed": a.speed, "stream_timeout_s": a.stream_timeout_s,
                           "concat_seconds": a.concat_seconds, "burst_extra": a.burst_extra},
            "server": {"metrics_at_start": m0, "health": {"model": m0["model"],
                       "inference": {"max_concurrent": m0["config"]["max_concurrent_inference"]}},
                       "ready_limits": {"partial_interval_s": m0["config"]["partial_interval_s"]}},
            "metric_definitions": METRIC_DEFINITIONS, **r,
        })

    report = {
        "label": a.label, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": m0["model"], "config": m0["config"], "environment_client": environment(),
        "audio": audio_manifest(utts), "checks": [c.as_dict() for c in checks],
        "all_passed": all(c.ok for c in checks),
    }
    if m0["model"]["backend"] == "fake":
        report["warning"] = "fake backend: harness check only, not a model measurement"
    write_json(out_dir / "stress_report.json", report)
    width = max(len(c.scenario) for c in checks)
    for c in checks:
        print(f"[{'PASS' if c.ok else 'FAIL'}] {c.scenario:<{width}}  {c.name}: {c.detail}")
    print(f"wrote {out_dir}/stress_report.json")
    return 0 if report["all_passed"] else 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws/transcribe")
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--streams", type=int, default=20, help="streams per paced scenario")
    p.add_argument("--concurrency", type=int, nargs="*", default=[1, 4, 8])
    p.add_argument("--burst-extra", type=int, default=4, help="connections above the admission limit")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--chunk-ms", type=int, default=100)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--stream-timeout-s", type=float, default=300.0)
    p.add_argument("--concat-seconds", type=float, default=0.0)
    p.add_argument("--sample-interval-s", type=float, default=1.0)
    p.add_argument("--label", default="")
    p.add_argument("--output-dir", required=True)
    raise SystemExit(asyncio.run(main_async(p.parse_args())))


if __name__ == "__main__":
    main()
