"""Observability: metric correctness, timing decomposition, bounded memory,
event-loop health, and log hygiene."""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import numpy as np
import pytest
from conftest import chunks, collect, pcm16

from echorelay.inference import FakeTranscriber, InferenceRunner
from echorelay.metrics import Distribution, Metrics, monitor_event_loop_lag


def _stream(client, seconds: float) -> list[dict]:
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.receive_json()
        for frame in chunks(pcm16(seconds)):
            ws.send_bytes(frame)
        ws.send_json({"type": "end"})
        events, code = collect(ws)
    assert code == 1000, events
    return events


# ---------------------------------------------------------------- unit level
def test_distribution_window_is_bounded_and_lifetime_stats_are_kept():
    d = Distribution(window=100)
    for v in range(1000):
        d.observe(float(v))
    s = d.summary()
    assert s["count"] == 1000 and s["window"] == 100
    assert s["max"] == 999.0 and s["sum"] == sum(range(1000))
    assert s["p50"] == pytest.approx(949.5)  # percentiles cover the last 100 only


async def test_runner_separates_queue_wait_from_decode_time():
    runner = InferenceRunner(FakeTranscriber(delay_s=0.2), max_concurrency=1, timeout_s=5)
    audio = np.zeros(16000, dtype=np.float32)
    first, second = await asyncio.gather(
        runner.transcribe(audio, final=False), runner.transcribe(audio, final=True)
    )
    assert first.queue_s < 0.05 and first.inference_s >= 0.19
    assert second.queue_s >= 0.18  # waited for the first job's slot
    assert second.inference_s >= 0.19
    m = runner.metrics
    assert m.counter("inference.partial.completed") == 1
    assert m.counter("inference.final.completed") == 1
    assert m.summary("inference.final.queue_wait_s")["max"] >= 0.18
    assert m.summary("inference.partial.rtf")["count"] == 1
    runner.shutdown()


async def test_runner_counts_timeouts_while_queued():
    runner = InferenceRunner(FakeTranscriber(delay_s=0.5), max_concurrency=1, timeout_s=0.2)
    audio = np.zeros(1600, dtype=np.float32)
    results = await asyncio.gather(
        runner.transcribe(audio, final=False), runner.transcribe(audio, final=False),
        return_exceptions=True,
    )
    assert all(type(r).__name__ == "InferenceTimeout" for r in results)
    assert runner.metrics.counter("inference.partial.timeouts") == 2
    assert runner.metrics.counter("inference.partial.timeouts_while_queued") == 1
    await asyncio.sleep(0.4)
    assert runner.in_flight == 0 and runner.waiting == 0
    runner.shutdown()


async def test_event_loop_lag_monitor_detects_a_blocked_loop():
    """Negative control: proves the lag metric would catch inference that
    accidentally ran on the event loop."""
    metrics = Metrics()
    task = asyncio.create_task(monitor_event_loop_lag(metrics, interval_s=0.02))
    await asyncio.sleep(0.1)
    time.sleep(0.3)  # deliberately block the loop
    await asyncio.sleep(0.1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert metrics.summary("event_loop.lag_s")["max"] >= 0.25


# ------------------------------------------------------------ server level
def test_default_wire_format_has_no_timing(make_client):
    events = _stream(make_client(), 2.0)
    assert all("timing" not in e for e in events)


def test_event_timing_decomposes_latency(make_client):
    client = make_client(FakeTranscriber(delay_s=0.05), event_timing=True)
    events = _stream(client, 2.5)
    partials = [e for e in events if e["type"] == "partial"]
    final = events[-1]
    assert partials
    for p in partials:
        assert set(p["timing"]) == {"queue_ms", "decode_ms", "staleness_ms"}
        assert p["timing"]["decode_ms"] == p["inference_ms"]
        assert p["timing"]["staleness_ms"] >= 0
    assert set(final["timing"]) == {"wait_partial_ms", "queue_ms", "decode_ms"}
    assert final["timing"]["decode_ms"] == final["inference_ms"] >= 45


def test_metrics_endpoint_after_streams(make_client):
    client = make_client()
    events = _stream(client, 3.0)
    m = client.get("/metrics").json()
    assert m["streams"]["active"] == 0 and m["streams"]["buffered_audio_bytes"] == 0
    assert m["streams"]["completed"] == 1 and m["streams"]["peak_active"] == 1
    assert m["inference"] == {"in_flight": 0, "waiting": 0, "max_concurrent": 1}
    assert m["config"]["partial_interval_s"] == 1.0 and m["config"]["event_timing"] is False
    c, d = m["counters"], m["distributions"]
    n_partials = sum(1 for e in events if e["type"] == "partial")
    assert c["inference.final.completed"] == 1
    assert c["inference.partial.completed"] == n_partials  # fake text always changes
    assert c["streams.outcome.completed"] == 1
    # amplification = audio decoded by partials / stream audio
    partial_audio = sum(e["audio_duration_s"] for e in events if e["type"] == "partial")
    assert d["stream.partial_amplification"]["max"] == pytest.approx(partial_audio / 3.0, abs=1e-3)
    deadline = time.monotonic() + 2  # monitor samples every 100 ms
    while "event_loop.lag_s" not in client.get("/metrics").json()["distributions"]:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert m["process"]["threads"] >= 1 and m["process"]["cpu_user_s"] >= 0


def test_buffered_bytes_gauge_tracks_open_streams(make_client):
    client = make_client(partial_interval_s=10.0)
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.receive_json()
        ws.send_bytes(pcm16(1.0))
        deadline = time.monotonic() + 2
        while client.get("/metrics").json()["streams"]["buffered_audio_bytes"] != 32000:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        ws.send_json({"type": "end"})
        collect(ws)
    assert client.get("/metrics").json()["streams"]["buffered_audio_bytes"] == 0


def test_rejections_and_peak_are_counted(make_client):
    client = make_client(max_concurrent_streams=1)
    with client.websocket_connect("/ws/transcribe") as first:
        first.receive_json()
        with client.websocket_connect("/ws/transcribe") as second:
            collect(second)
        first.send_json({"type": "end"})
        collect(first)
    m = client.get("/metrics").json()
    assert m["streams"]["rejected"] == 1 and m["streams"]["peak_active"] == 1
    assert m["counters"]["streams.outcome.rejected"] == 1


def test_slow_inference_defers_partials_and_reports_staleness(make_client):
    client = make_client(FakeTranscriber(delay_s=1.0), event_timing=True)
    events = _stream(client, 3.0)  # frames arrive much faster than decodes
    m = client.get("/metrics").json()
    assert m["counters"]["stream.partials_deferred"] >= 1
    stale = [e["timing"]["staleness_ms"] for e in events if e["type"] == "partial"]
    assert stale and max(stale) > 0  # partial covered less audio than had arrived
    assert m["distributions"]["stream.partial_staleness_s"]["count"] == len(stale)


DECODE_S = 0.3
# If a decode ran on the event loop, every decode would show up as at least
# DECODE_S of lag. The threshold sits well below that and well above
# scheduler / GIL noise from the in-process test client threads (a fixed
# 50 ms bound was flaky on a 2-core runner: one 63 ms wake-up in ~600 runs).
LAG_LIMIT_S = DECODE_S / 2


def test_inference_does_not_lag_event_loop_under_load(make_client):
    """Several streams with slow (thread-blocking) decodes: no lag sample
    comes anywhere near one decode's duration."""
    fake = FakeTranscriber(delay_s=DECODE_S)
    # Cancellation would skip end-of-stream partials and starve the
    # precondition below of decodes; this test is about lag, not policy.
    client = make_client(fake, max_concurrent_inference=2, max_concurrent_streams=6,
                         cancel_queued_partial_on_end=False)
    threads = [threading.Thread(target=_stream, args=(client, 2.0)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    lag = client.get("/metrics").json()["distributions"]["event_loop.lag_s"]
    assert len(fake.calls) >= 8  # decodes really happened during the window
    assert lag["count"] >= 10
    assert lag["max"] < LAG_LIMIT_S, lag


def test_lag_threshold_would_catch_inference_on_the_loop():
    """Control for the test above: the same decode duration executed on the
    loop produces lag beyond LAG_LIMIT_S."""

    async def block_loop():
        time.sleep(DECODE_S)  # what a decode on the loop would do

    async def run():
        metrics = Metrics()
        task = asyncio.create_task(monitor_event_loop_lag(metrics))
        await asyncio.sleep(0.15)
        await block_loop()
        await asyncio.sleep(0.15)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return metrics.summary("event_loop.lag_s")["max"]

    assert asyncio.run(run()) >= LAG_LIMIT_S


def test_stream_summary_log_has_no_transcript_text(make_client, caplog):
    caplog.set_level(logging.INFO, logger="echorelay.transport")
    _stream(make_client(), 2.0)
    lines = [r.getMessage() for r in caplog.records if "stream_summary" in r.getMessage()]
    assert len(lines) == 1
    assert "outcome=completed" in lines[0] and "partial_amplification=" in lines[0]
    assert "w1" not in lines[0]  # fake transcript words never logged
