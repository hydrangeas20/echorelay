"""Unit tests for buffering, protocol parsing, and the inference runner."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from gated import until

from echorelay.audio import AudioBuffer, AudioLimitExceeded, InvalidAudioFrame
from echorelay.inference import FakeTranscriber, InferenceError, InferenceRunner, InferenceTimeout
from echorelay.protocol import StreamError, parse_control


def test_audio_buffer_converts_pcm16_to_float32():
    buf = AudioBuffer(max_bytes=100)
    buf.append(np.array([0, 16384, -32768, 32767], dtype="<i2").tobytes())
    out = buf.snapshot()
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0.0, 0.5, -1.0, 32767 / 32768], rtol=1e-6)
    assert buf.num_samples == 4 and buf.num_bytes == 8


def test_audio_buffer_limits():
    buf = AudioBuffer(max_bytes=4)
    with pytest.raises(InvalidAudioFrame):
        buf.append(b"\x00")
    buf.append(b"\x00" * 4)
    with pytest.raises(AudioLimitExceeded):
        buf.append(b"\x00\x00")
    assert buf.num_bytes == 4  # rejected frame not partially appended


def test_snapshot_is_independent_copy():
    buf = AudioBuffer(max_bytes=100)
    buf.append(b"\x01\x00")
    snap = buf.snapshot()
    buf.append(b"\x02\x00")  # must not raise BufferError or mutate snapshot
    assert snap.size == 1 and buf.num_samples == 2


def test_parse_control():
    assert parse_control('{"type": "end"}') == {"type": "end"}
    with pytest.raises(StreamError):
        parse_control("{")


async def test_runner_releases_slot_only_when_thread_finishes():
    fake = FakeTranscriber(delay_s=0.3)
    runner = InferenceRunner(fake, max_concurrency=1, timeout_s=0.1)
    audio = np.zeros(16000, dtype=np.float32)
    try:
        with pytest.raises(InferenceTimeout):
            await runner.transcribe(audio, final=True)
        assert runner.in_flight == 1  # thread is still running; slot still held
        # The release runs on the loop after the worker thread returns; when
        # depends on the machine, so wait for it (bounded) rather than sleeping.
        await until(lambda: runner.in_flight == 0, what="timed-out decode to release its slot")
        runner.timeout_s = 2.0
        fake.delay_s = 0.0
        result = await runner.transcribe(audio, final=True)
        assert result.text == "w1 w2" and result.inference_s >= 0
        assert fake.max_active == 1
    finally:
        runner.shutdown()


async def test_runner_wraps_unexpected_exceptions():
    class Boom(FakeTranscriber):
        def transcribe(self, audio, *, final):
            raise ValueError("bad")

    runner = InferenceRunner(Boom(), max_concurrency=1, timeout_s=1)
    audio = np.zeros(10, dtype=np.float32)
    with pytest.raises(InferenceError, match="ValueError: bad"):
        await runner.transcribe(audio, final=False)
    # The slot is released by the executor future's done callback. Whether
    # that callback runs before or after the caller resumes is not specified
    # (it runs first on 3.11, one loop iteration later on 3.13), so wait for
    # the release, with a bound, instead of assuming it already happened.
    async with asyncio.timeout(1.0):
        while runner.in_flight:
            assert runner.in_flight <= runner.max_concurrency
            await asyncio.sleep(0)
    assert runner.in_flight == 0 and runner.waiting == 0
    # The semaphore was released too: a second call gets the slot at once
    # (it would raise InferenceTimeout, not InferenceError, if the slot leaked).
    with pytest.raises(InferenceError, match="ValueError: bad"):
        await runner.transcribe(audio, final=False)
    runner.shutdown()
