"""Level distributions (time at each active-stream count, queue depth and
in-flight count) and the queue depth recorded when a stream ends."""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import pytest
from conftest import chunks, collect, pcm16
from gated import GatedTranscriber, until
from test_priority import Stream, background

from echorelay.inference import InferenceRunner
from echorelay.metrics import LevelTracker, level_window


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def test_level_tracker_credits_time_to_the_level_held():
    c = Clock()
    lt = LevelTracker(clock=c)
    c.t += 2.0          # level 0 for 2 s
    lt.set(3)
    c.t += 1.0          # level 3 for 1 s
    lt.set(1)
    c.t += 0.5          # level 1 for 0.5 s (still open)
    s = lt.snapshot()
    assert s["time_at_level_s"] == {"0": 2.0, "1": 0.5, "3": 1.0}
    assert s["total_s"] == pytest.approx(3.5) and s["current"] == 1 and s["max"] == 3


def test_level_window_differences_two_snapshots():
    c = Clock()
    lt = LevelTracker(clock=c)
    c.t += 10.0         # idle before the window
    before = lt.snapshot()
    lt.set(2)
    c.t += 3.0
    lt.set(8)
    c.t += 1.0
    lt.set(0)
    after = lt.snapshot()
    w = level_window(before, after)
    assert w["total_s"] == pytest.approx(4.0)
    assert w["mean"] == pytest.approx((2 * 3 + 8 * 1) / 4)
    assert w["max_level_seen"] == 8
    assert w["share_at_level"] == {"2": pytest.approx(0.75), "8": pytest.approx(0.25)}
    assert w["share_at_or_above"]["8"] == pytest.approx(0.25)
    assert w["share_at_or_above"]["2"] == pytest.approx(1.0)
    assert level_window(after, after) == {"total_s": 0.0}


async def test_queue_depth_levels_are_exact():
    g = GatedTranscriber()
    runner = InferenceRunner(g, 1, timeout_s=30)
    hog = background(runner, 3200)
    await until(lambda: g.started == ["P3200"])
    queued = [background(runner, 1600 * (i + 2)) for i in range(2)]
    await until(lambda: runner.waiting == 2)
    lv = runner.scheduler.waiting_level.snapshot()
    assert lv["current"] == 2 and lv["max"] == 2
    assert runner.in_flight_level.snapshot()["current"] == 1

    def drain():
        g.release_all()
        return hog.done() and all(t.done() for t in queued)

    await until(drain)
    assert runner.scheduler.waiting_level.snapshot()["current"] == 0
    assert runner.in_flight_level.snapshot()["current"] == 0
    assert set(runner.scheduler.waiting_level.snapshot()["time_at_level_s"]) == {"0", "1", "2"}
    runner.shutdown()


async def test_queue_depth_recorded_at_end_before_own_final_is_submitted():
    for final_priority in (False, True):
        g = GatedTranscriber()
        runner = InferenceRunner(g, 1, timeout_s=30, final_priority=final_priority)
        hog = background(runner, 3200)
        await until(lambda: g.started == ["P3200"])
        other = background(runner, 4800)
        await until(lambda: runner.waiting == 1)
        s = Stream(runner)
        await s.feed_audio(1.0)  # own partial queued -> depth 2
        await until(lambda: runner.waiting == 2)
        await s.end()
        await until(lambda: s.session.stats.inference_waiting_at_end is not None)
        # observed before the early final (priority) adds a waiter
        assert s.session.stats.inference_waiting_at_end == 2
        assert s.session.stats.inference_in_flight_at_end == 1

        def drain():
            g.release_all()
            return s.task.done() and other.done() and hog.done()

        await until(drain)
        await s.task
        runner.shutdown()


def test_metrics_levels_and_summary_fields(make_client, caplog):
    caplog.set_level(logging.INFO, logger="echorelay.transport")
    client = make_client()
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.receive_json()
        for frame in chunks(pcm16(2.0)):
            ws.send_bytes(frame)
        ws.send_json({"type": "end"})
        _, code = collect(ws)
    assert code == 1000
    m = client.get("/metrics").json()
    lv = m["levels"]
    assert set(lv) == {"streams.active", "inference.in_flight", "inference.waiting"}
    assert lv["streams.active"]["max"] == 1 and lv["streams.active"]["current"] == 0
    for v in lv.values():
        assert v["total_s"] == pytest.approx(sum(v["time_at_level_s"].values()))
    assert m["distributions"]["stream.inference_waiting_at_end"]["count"] == 1
    line = next(r.getMessage() for r in caplog.records if "stream_summary" in r.getMessage())
    # values depend on timing (the stream's own partial may still be running)
    import re
    kv = dict(re.findall(r"(\w+)=(\S+)", line))
    assert int(kv["inference_waiting_at_end"]) >= 0 and int(kv["inference_in_flight_at_end"]) in (0, 1)
    assert np.isfinite(lv["inference.waiting"]["total_s"])
