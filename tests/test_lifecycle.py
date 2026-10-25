"""Regression tests for task lifecycle under cancellation."""

from __future__ import annotations

import asyncio

import numpy as np

from echorelay.inference import FakeTranscriber, InferenceRunner
from echorelay.session import StreamSession

from conftest import make_settings
from gated import until


async def test_cancelling_session_cancels_and_awaits_pending_receive():
    """Cancelling the session while it waits for a frame must not leak the
    receive task: it has to be cancelled *and* finished before the
    cancellation propagates."""
    receive_started = asyncio.Event()
    receive_state = {"cancelled": False, "finished": False}

    async def blocking_receive():
        receive_started.set()
        try:
            await asyncio.Event().wait()  # never delivers a frame
        except asyncio.CancelledError:
            receive_state["cancelled"] = True
            await asyncio.sleep(0.05)  # cleanup that takes a moment
            raise
        finally:
            receive_state["finished"] = True

    async def send(_payload):  # pragma: no cover - nothing is sent here
        pass

    runner = InferenceRunner(FakeTranscriber(), max_concurrency=1, timeout_s=5)
    session = StreamSession(
        "lifecycle", make_settings(idle_timeout_s=60), runner, blocking_receive, send
    )
    before = asyncio.all_tasks()
    task = asyncio.create_task(session.run())
    await receive_started.wait()

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    else:  # pragma: no cover
        raise AssertionError("session.run() swallowed cancellation")

    # By the time run() has finished cancelling, the receive task must be done.
    assert receive_state == {"cancelled": True, "finished": True}
    leaked = asyncio.all_tasks() - before
    assert not leaked, f"leaked tasks: {leaked}"
    runner.shutdown()


async def test_cancelling_session_mid_partial_cancels_receive_and_partial():
    """Same guarantee while a partial decode is in flight."""
    frames = asyncio.Queue()
    pcm = (np.zeros(16000, dtype="<i2")).tobytes()  # 1.0 s triggers a partial
    await frames.put(("bytes", pcm))
    receive_state = {"cancelled": 0}

    async def receive():
        try:
            return await frames.get()  # second call blocks forever
        except asyncio.CancelledError:
            receive_state["cancelled"] += 1
            raise

    async def send(_payload):
        pass

    fake = FakeTranscriber(delay_s=0.3)
    runner = InferenceRunner(fake, max_concurrency=1, timeout_s=5)
    session = StreamSession("mid", make_settings(idle_timeout_s=60), runner, receive, send)
    before = asyncio.all_tasks()
    task = asyncio.create_task(session.run())
    try:
        await until(lambda: runner.in_flight > 0, what="partial decode to start")

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert receive_state["cancelled"] == 1
        assert not (asyncio.all_tasks() - before)
        # The worker thread finishes the decode, then its slot is released on
        # the loop; when depends on the machine, so wait for state, bounded.
        await until(lambda: runner.in_flight == 0, what="decode slot to be released")
        assert runner.in_flight == 0
    finally:
        if not task.done():  # a failed wait above: don't leave the session running
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        runner.shutdown()
