"""Cancelling a stream's obsolete (not yet started) partial at end of stream:
execution order, wire output, slot accounting, and every partial state."""

from __future__ import annotations

import asyncio
import itertools

import numpy as np
import pytest
from conftest import make_settings, pcm16
from gated import GatedTranscriber, until
from test_priority import Stream, _idle, background

from echorelay.inference import InferenceRunner
from echorelay.session import StreamSession

POLICIES = {  # name -> (final_priority, cancel_queued_partial_on_end)
    "base": (False, False),
    "prio": (True, False),
    "cancel": (False, True),
    "both": (True, True),
}


async def _own_partial_queued_behind_other(policy: str):
    prio, cancel = POLICIES[policy]
    g = GatedTranscriber()
    runner = InferenceRunner(g, 1, timeout_s=30, final_priority=prio)
    hog = background(runner, 3200)
    await until(lambda: g.started == ["P3200"])
    other = background(runner, 4800)
    await until(lambda: runner.waiting == 1)
    s = Stream(runner, cancel_queued_partial_on_end=cancel)
    await s.feed_audio(1.0)
    await until(lambda: runner.waiting == 2, what="own partial queued")
    await s.end()
    await until(lambda: s.session.stats.partial_state_at_end != "none")

    def step():
        g.release_all()
        return s.task.done() and other.done() and hog.done()

    await until(step, timeout=10, what="all decodes finished")
    await s.task
    runner.shutdown()
    return g.started, s, runner


@pytest.mark.parametrize("policy, expected_order", [
    ("base", ["P3200", "P4800", "P16000", "F16000"]),
    ("prio", ["P3200", "P16000", "F16000", "P4800"]),
    ("cancel", ["P3200", "P4800", "F16000"]),
    ("both", ["P3200", "F16000", "P4800"]),
])
async def test_execution_order_by_policy(policy, expected_order):
    order, s, runner = await _own_partial_queued_behind_other(policy)
    assert order == expected_order
    st = s.session.stats
    assert st.partial_state_at_end == "queued"
    assert st.partial_cancelled_at_end is (policy in ("cancel", "both"))
    assert st.partial_promoted_at_end is (policy == "prio")
    # The final is identical under every policy.
    assert s.sent[-1] == {**s.sent[-1], "type": "final", "text": "F16000", "audio_duration_s": 1.0}
    # A cancelled partial is never sent; seq stays contiguous.
    partials = [e for e in s.sent if e["type"] == "partial"]
    assert len(partials) == (0 if st.partial_cancelled_at_end else 1)
    assert [p["seq"] for p in partials] == list(range(1, len(partials) + 1))
    cancelled = runner.metrics.counter("inference.partial.cancelled_while_queued")
    assert cancelled == (1 if st.partial_cancelled_at_end else 0)
    assert _idle(runner)


@pytest.mark.parametrize("policy", POLICIES)
async def test_running_partial_is_never_cancelled(policy):
    prio, cancel = POLICIES[policy]
    g = GatedTranscriber()
    runner = InferenceRunner(g, 1, timeout_s=30, final_priority=prio)
    s = Stream(runner, cancel_queued_partial_on_end=cancel)
    await s.feed_audio(1.0)
    await until(lambda: g.started == ["P16000"], what="own partial running")
    await s.end()
    await until(lambda: s.session.stats.partial_state_at_end == "running")
    assert s.session.stats.partial_cancelled_at_end is False

    def step():
        g.release_all()
        return s.task.done()

    await until(step, timeout=10)
    await s.task
    runner.shutdown()
    assert g.started == ["P16000", "F16000"]
    assert [e["type"] for e in s.sent] == ["partial", "final"]  # partial still precedes final


@pytest.mark.parametrize("policy", POLICIES)
async def test_partial_scheduled_but_not_yet_started(policy):
    """'created' state: the partial task exists but has not run, so it has
    not reached the queue. End is processed in the same loop iteration."""
    prio, cancel = POLICIES[policy]
    g = GatedTranscriber()
    runner = InferenceRunner(g, 1, timeout_s=30, final_priority=prio)
    sent = []

    async def send(p):
        sent.append(p)

    async def never():
        await asyncio.Event().wait()

    session = StreamSession("c", make_settings(cancel_queued_partial_on_end=cancel), runner, never, send)
    session.buffer.append(pcm16(1.0))
    session._maybe_schedule_partial()  # task created, not yet run
    job = session._partial_job
    assert job.state == "created"
    session._apply_end_of_stream_policy()  # no await in between
    st = session.stats
    assert st.partial_state_at_end == "created"
    assert st.partial_cancelled_at_end is cancel
    assert st.partial_promoted_at_end is (prio and not cancel)
    if prio and not cancel:
        assert job.priority == 0  # will queue at final priority
    if cancel:
        await asyncio.gather(session._partial_task, return_exceptions=True)
        assert session._partial_task.cancelled() and g.started == []
    else:
        await until(lambda: g.started == ["P16000"])
        g.release_all()
        await session._partial_task
    await session.cleanup()
    await until(lambda: _idle(runner))
    runner.shutdown()


@pytest.mark.parametrize("prio, cancel, workers", list(itertools.product([False, True], [False, True], [1, 2])))
async def test_final_text_and_wire_invariants_under_every_policy(prio, cancel, workers):
    """Several streams of different lengths, ending at once, under every
    policy and with 1 or 2 workers: each stream gets exactly one final whose
    text covers its full audio; partials precede it with contiguous seq."""
    g = GatedTranscriber()
    runner = InferenceRunner(g, workers, timeout_s=30, final_priority=prio)
    durations = [1.0, 1.5, 2.2, 3.1]
    streams = [Stream(runner, f"s{i}", cancel_queued_partial_on_end=cancel) for i in range(len(durations))]
    for s, d in zip(streams, durations):
        await s.feed_audio(d)
    for s in streams:
        await s.end()

    def step():
        g.release_all()
        return all(s.task.done() for s in streams)

    await until(step, timeout=20)
    for s in streams:
        await s.task
    assert _idle(runner)
    runner.shutdown()
    for s, d in zip(streams, durations):
        types = [e["type"] for e in s.sent]
        assert types.count("final") == 1 and types[-1] == "final"
        assert s.sent[-1]["text"] == f"F{int(round(d * 16000))}"
        seqs = [e["seq"] for e in s.sent if e["type"] == "partial"]
        assert seqs == list(range(1, len(seqs) + 1))
    # each final decoded exactly once
    assert sorted(x for x in g.started if x.startswith("F")) == sorted(
        f"F{int(round(d * 16000))}" for d in durations)


async def test_disconnect_after_cancellation_leaves_runner_idle():
    g = GatedTranscriber()
    runner = InferenceRunner(g, 1, timeout_s=30, final_priority=True)
    hog = background(runner, 3200)
    await until(lambda: g.started == ["P3200"])
    s = Stream(runner, cancel_queued_partial_on_end=True)
    await s.feed_audio(1.0)
    await until(lambda: runner.waiting == 1)
    await s.end()
    await until(lambda: s.session.stats.partial_cancelled_at_end)
    await until(lambda: runner.waiting == 1, what="only the early final queued")
    s.task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await s.task
    g.release_all()
    await hog
    await until(lambda: _idle(runner))
    assert g.started == ["P3200"]
    runner.shutdown()


def test_metrics_config_reports_both_flags(make_client, monkeypatch):
    client = make_client(final_priority=True, cancel_queued_partial_on_end=True)
    cfg = client.get("/metrics").json()["config"]
    assert cfg["final_priority"] is True and cfg["cancel_queued_partial_on_end"] is True
    monkeypatch.delenv("ECHORELAY_FINAL_PRIORITY", raising=False)
    monkeypatch.delenv("ECHORELAY_CANCEL_QUEUED_PARTIAL_ON_END", raising=False)
    cfg = make_client().get("/metrics").json()["config"]
    assert cfg["final_priority"] is False and cfg["cancel_queued_partial_on_end"] is False  # defaults


def test_default_wire_format_unchanged_with_both_flags(make_client):
    """Flags change scheduling only; event shapes are identical."""
    from conftest import chunks, collect

    for flags in ({}, {"final_priority": True, "cancel_queued_partial_on_end": True}):
        client = make_client(**flags)
        with client.websocket_connect("/ws/transcribe") as ws:
            ready = ws.receive_json()
            for frame in chunks(pcm16(2.0)):
                ws.send_bytes(frame)
            ws.send_json({"type": "end"})
            events, code = collect(ws)
        assert code == 1000
        assert set(ready) == {"type", "protocol_version", "session_id", "audio", "limits"}
        for e in events:
            keys = {"partial": {"type", "provisional", "seq", "text", "audio_duration_s", "inference_ms"},
                    "final": {"type", "provisional", "text", "audio_duration_s", "inference_ms"}}[e["type"]]
            assert set(e) == keys
        assert np.isclose(events[-1]["audio_duration_s"], 2.0)
