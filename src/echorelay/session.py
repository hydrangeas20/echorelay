"""Per-connection streaming state machine.

Transport-agnostic: the session receives already-decoded messages through
an async ``receive`` callable and emits events through an async ``send``
callable. That keeps buffering/decoding policy testable without sockets.

Lifecycle::

    ready -> (binary audio frames)* -> {"type": "end"} -> final -> close

While audio is arriving, a provisional partial decode of the whole buffer is
scheduled every ``partial_interval_s`` of new audio. At most one partial
decode is in flight per session; frames that arrive meanwhile are simply
buffered (the next partial covers them). Partial decodes run in a background
task so the receive loop never waits on inference.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .audio import AudioBuffer, AudioLimitExceeded, InvalidAudioFrame
from .config import SAMPLE_RATE_HZ, Settings
from .inference import InferenceError, InferenceRunner, InferenceTimeout
from .protocol import (
    CloseCode,
    StreamError,
    final_event,
    malformed,
    parse_control,
    partial_event,
)

log = logging.getLogger(__name__)

# Messages delivered by the transport: ("bytes", b"..."), ("text", "..."), or
# ("disconnect", close_code).
Incoming = tuple[str, Any]


class ClientDisconnected(Exception):
    pass


@dataclass
class StreamStats:
    """Per-stream counters and timings (monotonic clock, seconds).

    Definitions: docs/measurements.md ("Per-stream statistics").
    """

    partials_scheduled: int = 0
    partials_completed: int = 0  # decodes that returned a result
    partials_sent: int = 0
    partials_skipped_empty: int = 0
    partials_skipped_duplicate: int = 0
    # Times the partial interval elapsed while a partial was still in flight
    # (counted once per in-flight partial): partials coalesced for lack of capacity.
    partials_deferred: int = 0
    partial_decode_s: float = 0.0
    partial_queue_s: float = 0.0
    partial_audio_s: float = 0.0  # sum of audio covered by completed partial decodes
    partial_staleness_s: list[float] = field(default_factory=list)  # per partial sent
    final_wait_partial_s: float | None = None
    final_queue_s: float | None = None
    final_decode_s: float | None = None

    def summary(self, audio_duration_s: float) -> dict[str, Any]:
        return {
            "audio_s": round(audio_duration_s, 3),
            "partials_scheduled": self.partials_scheduled,
            "partials_completed": self.partials_completed,
            "partials_sent": self.partials_sent,
            "partials_skipped_empty": self.partials_skipped_empty,
            "partials_skipped_duplicate": self.partials_skipped_duplicate,
            "partials_deferred": self.partials_deferred,
            "partial_decode_s": round(self.partial_decode_s, 4),
            "partial_queue_s": round(self.partial_queue_s, 4),
            "partial_amplification": (
                round(self.partial_audio_s / audio_duration_s, 3) if audio_duration_s else None
            ),
            "partial_staleness_max_s": (
                round(max(self.partial_staleness_s), 4) if self.partial_staleness_s else None
            ),
            "final_wait_partial_s": _r(self.final_wait_partial_s),
            "final_queue_s": _r(self.final_queue_s),
            "final_decode_s": _r(self.final_decode_s),
        }


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 4)


async def _cancel_and_wait(task: asyncio.Future) -> None:
    """Cancel ``task`` and wait until it has actually finished.

    ``asyncio.wait`` never raises the task's own exception, so this works
    whether or not the caller is itself being cancelled; if the caller is
    cancelled again while waiting, that cancellation propagates normally.
    """
    task.cancel()
    await asyncio.wait({task})
    if not task.cancelled():
        task.exception()  # mark retrieved; avoids "exception never retrieved"


def inference_failure(exc: Exception) -> StreamError:
    if isinstance(exc, InferenceTimeout):
        return StreamError("inference_timeout", str(exc), CloseCode.INTERNAL_ERROR)
    return StreamError("inference_error", str(exc), CloseCode.INTERNAL_ERROR)


class StreamSession:
    def __init__(
        self,
        session_id: str,
        settings: Settings,
        runner: InferenceRunner,
        receive: Callable[[], Awaitable[Incoming]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        self.session_id = session_id
        self.settings = settings
        self.runner = runner
        self._receive = receive
        self._send = send
        self.buffer = AudioBuffer(settings.max_audio_bytes)
        self._partial_task: asyncio.Task | None = None
        self._last_partial_samples = 0
        self._last_partial_text = ""
        self._partial_seq = 0
        self.audio_duration_s = 0.0  # total audio received; kept after cleanup
        self.stats = StreamStats()
        self._deferral_counted = False
        # Set by a failed partial decode so the receive loop can abort promptly.
        self._fatal: asyncio.Future[StreamError] = asyncio.get_running_loop().create_future()
        self._interval_samples = int(settings.partial_interval_s * SAMPLE_RATE_HZ)
        self._min_partial_samples = int(settings.min_partial_audio_s * SAMPLE_RATE_HZ)

    # ------------------------------------------------------------------ run
    async def run(self) -> None:
        """Drive the stream until the final transcript is sent.

        Raises :class:`StreamError` for protocol/resource/inference failures
        and :class:`ClientDisconnected` if the peer goes away.
        """
        try:
            while True:
                kind, payload = await self._next_message()
                if kind == "disconnect":
                    raise ClientDisconnected(payload)
                if kind == "bytes":
                    self._on_audio(payload)
                elif kind == "text":
                    parse_control(payload)  # only "end" is valid
                    await self._finish()
                    return
                else:  # pragma: no cover - transport guarantees kinds
                    raise malformed(f"unsupported frame kind {kind}")
        finally:
            await self.cleanup()

    async def _next_message(self) -> Incoming:
        recv = asyncio.ensure_future(self._receive())
        try:
            done, _ = await asyncio.wait(
                {recv, self._fatal},
                timeout=self.settings.idle_timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            # Runs on every exit path, including cancellation of this
            # coroutine: never leave the receive task pending.
            if not recv.done():
                await _cancel_and_wait(recv)
        if recv in done:
            return recv.result()
        if self._fatal in done:
            raise self._fatal.result()
        raise StreamError(
            "idle_timeout",
            f"no message received for {self.settings.idle_timeout_s:.1f}s",
            CloseCode.POLICY_VIOLATION,
        )

    async def _send_event(self, payload: dict[str, Any]) -> None:
        try:
            async with asyncio.timeout(self.settings.send_timeout_s):
                await self._send(payload)
        except TimeoutError as exc:
            raise StreamError(
                "send_timeout",
                f"client did not accept data within {self.settings.send_timeout_s:.1f}s",
                CloseCode.POLICY_VIOLATION,
            ) from exc

    # ---------------------------------------------------------------- audio
    def _on_audio(self, frame: bytes) -> None:
        if len(frame) > self.settings.max_frame_bytes:
            raise StreamError(
                "frame_too_large",
                f"frame of {len(frame)} bytes exceeds limit of {self.settings.max_frame_bytes}",
                CloseCode.MESSAGE_TOO_BIG,
            )
        if not frame:
            return
        try:
            self.buffer.append(frame)
        except InvalidAudioFrame as exc:
            raise StreamError("invalid_audio_frame", str(exc), CloseCode.UNSUPPORTED_DATA) from exc
        except AudioLimitExceeded as exc:
            raise StreamError("audio_too_large", str(exc), CloseCode.MESSAGE_TOO_BIG) from exc
        self._maybe_schedule_partial()

    def _maybe_schedule_partial(self) -> None:
        n = self.buffer.num_samples
        due = n >= self._min_partial_samples and n - self._last_partial_samples >= self._interval_samples
        if self._partial_task is not None and not self._partial_task.done():
            if due and not self._deferral_counted:
                self.stats.partials_deferred += 1
                self._deferral_counted = True
            return
        if not due:
            return
        self._last_partial_samples = n
        self._deferral_counted = False
        self.stats.partials_scheduled += 1
        self._partial_task = asyncio.create_task(
            self._partial(), name=f"partial-{self.session_id}"
        )

    async def _partial(self) -> None:
        audio = self.buffer.snapshot()
        try:
            result = await self.runner.transcribe(audio, final=False)
        except (InferenceError, InferenceTimeout) as exc:
            log.warning("session %s partial decode failed: %s", self.session_id, exc)
            if not self._fatal.done():
                self._fatal.set_result(inference_failure(exc))
            return
        audio_s = audio.size / SAMPLE_RATE_HZ
        st = self.stats
        st.partials_completed += 1
        st.partial_decode_s += result.inference_s
        st.partial_queue_s += result.queue_s
        st.partial_audio_s += audio_s
        text = result.text
        # Skip empty hypotheses and exact repeats: nothing new to show.
        if not text:
            st.partials_skipped_empty += 1
            return
        if text == self._last_partial_text:
            st.partials_skipped_duplicate += 1
            return
        self._last_partial_text = text
        self._partial_seq += 1
        # Staleness: audio received since this hypothesis' snapshot was taken.
        staleness_s = self.buffer.duration_s - audio_s
        st.partial_staleness_s.append(staleness_s)
        timing = None
        if self.settings.event_timing:
            timing = {"queue": result.queue_s, "decode": result.inference_s, "staleness": staleness_s}
        st.partials_sent += 1
        try:
            await self._send_event(
                partial_event(self._partial_seq, text, audio_s, result.inference_s, timing)
            )
        except StreamError as err:  # surface to the receive loop, like a failed decode
            if not self._fatal.done():
                self._fatal.set_result(err)

    # ---------------------------------------------------------------- final
    async def _finish(self) -> None:
        # Let an in-flight partial complete so events stay ordered
        # (partials always precede the final).
        t0 = time.monotonic()
        if self._partial_task is not None:
            await asyncio.gather(self._partial_task, return_exceptions=True)
        self.stats.final_wait_partial_s = time.monotonic() - t0
        if self._fatal.done():
            raise self._fatal.result()
        audio = self.buffer.snapshot()
        wait_s = self.stats.final_wait_partial_s
        if audio.size == 0:
            timing = {"wait_partial": wait_s, "queue": 0.0, "decode": 0.0} if self.settings.event_timing else None
            await self._send_event(final_event("", 0.0, 0.0, timing))
            return
        try:
            result = await self.runner.transcribe(audio, final=True)
        except (InferenceError, InferenceTimeout) as exc:
            raise inference_failure(exc) from exc
        self.stats.final_queue_s = result.queue_s
        self.stats.final_decode_s = result.inference_s
        timing = None
        if self.settings.event_timing:
            timing = {"wait_partial": wait_s, "queue": result.queue_s, "decode": result.inference_s}
        await self._send_event(
            final_event(result.text, audio.size / SAMPLE_RATE_HZ, result.inference_s, timing)
        )

    # -------------------------------------------------------------- cleanup
    async def cleanup(self) -> None:
        task, self._partial_task = self._partial_task, None
        if task is not None:
            task.cancel()  # no-op if already finished
            # Also retrieves any exception so it is not logged as unhandled.
            await asyncio.gather(task, return_exceptions=True)
        self.audio_duration_s = self.buffer.duration_s
        self.buffer.clear()
