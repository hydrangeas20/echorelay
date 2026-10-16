"""Final-priority scheduling, end to end through StreamSession and the
runner, with a gated model: exact execution order, no timing assumptions.

Decodes are named by kind and sample count (see gated.py): another stream's
running partial "P3200" (the hog), another stream's queued partial "P4800",
and this stream's 1.0 s partial "P16000" and final "F16000"."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
from conftest import chunks, make_settings, pcm16
from gated import GatedTranscriber, until

from echorelay.inference import InferenceRunner
from echorelay.session import StreamSession

END = ("text", json.dumps({"type": "end"}))


class Stream:
    """A StreamSession fed from a queue, recording what it sends."""

    def __init__(self, runner: InferenceRunner, name: str = "s", **settings) -> None:
        # These scenarios assert exact orders, so pin the cancellation policy
        # (final priority is set on the runner) instead of inheriting env.
        settings.setdefault("cancel_queued_partial_on_end", False)
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.sent: list[dict] = []

        async def send(payload):
            self.sent.append(payload)

        self.session = StreamSession(
            name, make_settings(idle_timeout_s=30, **settings), runner, self.inbox.get, send
        )
        self.task = asyncio.create_task(self.session.run())

    async def feed_audio(self, seconds: float) -> None:
        for frame in chunks(pcm16(seconds)):
            await self.inbox.put(("bytes", frame))

    async def end(self) -> None:
        await self.inbox.put(END)


def background(runner: InferenceRunner, samples: int) -> asyncio.Task:
    """Another stream's partial decode."""
    return asyncio.create_task(runner.transcribe(np.zeros(samples, np.float32), final=False))


async def _scenario_own_partial_queued_behind_other(final_priority: bool):
    g = GatedTranscriber()
    runner = InferenceRunner(g, 1, timeout_s=30, final_priority=final_priority)
    hog = background(runner, 3200)
    await until(lambda: g.started == ["P3200"], what="hog running")
    other = background(runner, 4800)  # queued first
    await until(lambda: runner.waiting == 1, what="other partial queued")
    s = Stream(runner)
    await s.feed_audio(1.0)  # schedules P16000, queued behind P4800
    await until(lambda: runner.waiting == 2, what="own partial queued")
    await s.end()
    await until(lambda: s.session.stats.partial_state_at_end != "none", what="end processed")

    for _ in range(4):  # finish whatever is running, one decode at a time
        n = len(g.started)
        g.release_next()
        if n < 4:
            await until(lambda: len(g.started) == n + 1, what=f"decode #{n + 1} started")
    await asyncio.wait_for(s.task, 10)
    await asyncio.gather(hog, other)
    runner.shutdown()
    return g.started, s


async def test_baseline_fifo_final_waits_behind_other_streams_partial():
    order, s = await _scenario_own_partial_queued_behind_other(final_priority=False)
    assert order == ["P3200", "P4800", "P16000", "F16000"]
    assert s.session.stats.partial_state_at_end == "queued"
    assert s.session.stats.partial_promoted_at_end is False


async def test_final_priority_with_inheritance_runs_own_partial_then_final_first():
    order, s = await _scenario_own_partial_queued_behind_other(final_priority=True)
    # The queued own partial inherits final priority; the final then outranks P4800.
    assert order == ["P3200", "P16000", "F16000", "P4800"]
    assert s.session.stats.partial_promoted_at_end is True
    # Wire output is identical in content and order to the baseline.
    assert [e["type"] for e in s.sent] == ["partial", "final"]
    assert s.sent[-1]["text"] == "F16000"


async def test_final_priority_when_own_partial_already_running():
    for final_priority, expected in [(False, ["P16000", "P4800", "F16000"]),
                                     (True, ["P16000", "F16000", "P4800"])]:
        g = GatedTranscriber()
        runner = InferenceRunner(g, 1, timeout_s=30, final_priority=final_priority)
        s = Stream(runner)
        await s.feed_audio(1.0)
        await until(lambda: g.started == ["P16000"], what="own partial running")
        other = background(runner, 4800)
        await until(lambda: runner.waiting == 1, what="other queued")
        await s.end()
        await until(lambda: s.session.stats.partial_state_at_end == "running", what="end processed")
        assert s.session.stats.partial_promoted_at_end is False  # nothing to promote
        for _ in range(3):
            n = len(g.started)
            g.release_next()
            if n < 3:
                await until(lambda: len(g.started) == n + 1)
        await asyncio.wait_for(s.task, 10)
        await other
        runner.shutdown()
        assert g.started == expected, (final_priority, g.started)


async def test_partials_are_not_starved_once_finals_drain():
    """Fairness bound: with priority, a waiting partial runs as soon as no
    final is waiting. Finals are at most one per stream, so admission
    (max_concurrent_streams) bounds how many can overtake a partial."""
    g = GatedTranscriber()
    runner = InferenceRunner(g, 1, timeout_s=30, final_priority=True)
    hog = background(runner, 3200)
    await until(lambda: g.started == ["P3200"])
    partial = background(runner, 4800)
    finals = [asyncio.create_task(runner.transcribe(np.zeros(1600 * (i + 1), np.float32), final=True))
              for i in range(3)]
    await until(lambda: runner.waiting == 4)
    for _ in range(4):
        n = len(g.started)
        g.release_next()
        await until(lambda: len(g.started) == n + 1)
    g.release_next()
    await asyncio.gather(hog, partial, *finals)
    runner.shutdown()
    assert g.started == ["P3200", "F1600", "F3200", "F4800", "P4800"]


@pytest.mark.parametrize("final_priority", [False, True])
async def test_final_text_is_independent_of_scheduling_policy(final_priority):
    """Scheduling changes when a decode runs, never its input: every
    stream's final covers exactly its full audio."""
    g = GatedTranscriber()
    runner = InferenceRunner(g, 1, timeout_s=30, final_priority=final_priority)
    durations = [1.0, 1.5, 2.2]
    streams = [Stream(runner, f"s{i}") for i in range(3)]
    for s, d in zip(streams, durations):
        await s.feed_audio(d)
        await s.end()

    async def pump():
        while not all(s.task.done() for s in streams):
            g.release_all()
            await asyncio.sleep(0.005)

    await asyncio.wait_for(asyncio.gather(pump(), *(s.task for s in streams)), 20)
    runner.shutdown()
    for s, d in zip(streams, durations):
        assert s.sent[-1]["type"] == "final"
        assert s.sent[-1]["text"] == f"F{int(round(d * 16000))}"
        assert [e["type"] for e in s.sent].count("final") == 1
        seqs = [e["seq"] for e in s.sent if e["type"] == "partial"]
        assert seqs == list(range(1, len(seqs) + 1))


def _idle(runner: InferenceRunner) -> bool:
    s = runner.scheduler
    return runner.in_flight == 0 and s.waiting == 0 and s.free == s.slots


async def test_own_partial_fails_while_early_final_is_queued():
    """The partial's error ends the stream; the early final is cancelled even
    if it was already granted the slot (race S4); nothing leaks."""
    from echorelay.protocol import StreamError

    g = GatedTranscriber()
    g.fail.add("P16000")
    runner = InferenceRunner(g, 1, timeout_s=30, final_priority=True)
    hog = background(runner, 3200)
    await until(lambda: g.started == ["P3200"])
    s = Stream(runner)
    await s.feed_audio(1.0)
    await until(lambda: runner.waiting == 1, what="own partial queued")
    await s.end()
    await until(lambda: runner.waiting == 2, what="early final queued")
    g.release_next()  # hog done -> own (promoted) partial runs
    await until(lambda: g.started == ["P3200", "P16000"])
    g.release_next()  # own partial fails
    with pytest.raises(StreamError) as err:
        await asyncio.wait_for(s.task, 10)
    assert err.value.code == "inference_error"
    assert not any(e["type"] == "final" for e in s.sent)
    # The partial's slot can be handed to the queued final in the same
    # callback, so the final may start before the session sees the failure.
    # A running decode cannot be cancelled: it runs once, its result unused.
    await hog

    def released_and_idle():
        g.release_all()
        return _idle(runner)

    await until(released_and_idle, what="runner idle")
    assert g.started.count("F16000") <= 1
    runner.shutdown()


async def test_disconnect_during_finish_with_early_final_queued():
    g = GatedTranscriber()
    runner = InferenceRunner(g, 1, timeout_s=30, final_priority=True)
    hog = background(runner, 3200)
    await until(lambda: g.started == ["P3200"])
    s = Stream(runner)
    await s.feed_audio(1.0)
    await until(lambda: runner.waiting == 1)
    await s.end()
    await until(lambda: runner.waiting == 2, what="partial and early final queued")
    s.task.cancel()  # handler cancelled: client gone
    with pytest.raises(asyncio.CancelledError):
        await s.task
    assert runner.waiting == 0  # both queued requests withdrawn
    g.release_next()
    await hog
    await until(lambda: _idle(runner), what="runner idle")
    assert g.started == ["P3200"]  # neither the partial nor the final ever ran
    runner.shutdown()
