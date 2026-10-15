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
from collections.abc import Awaitable, Callable
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
        done, _ = await asyncio.wait(
            {recv, self._fatal},
            timeout=self.settings.idle_timeout_s,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if recv in done:
            return recv.result()
        recv.cancel()
        await asyncio.gather(recv, return_exceptions=True)
        if self._fatal in done:
            raise self._fatal.result()
        raise StreamError(
            "idle_timeout",
            f"no message received for {self.settings.idle_timeout_s:.1f}s",
            CloseCode.POLICY_VIOLATION,
        )

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
        if self._partial_task is not None and not self._partial_task.done():
            return
        n = self.buffer.num_samples
        if n < self._min_partial_samples:
            return
        if n - self._last_partial_samples < self._interval_samples:
            return
        self._last_partial_samples = n
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
        text = result.text
        # Skip empty hypotheses and exact repeats: nothing new to show.
        if not text or text == self._last_partial_text:
            return
        self._last_partial_text = text
        self._partial_seq += 1
        await self._send(
            partial_event(self._partial_seq, text, audio.size / SAMPLE_RATE_HZ, result.inference_s)
        )

    # ---------------------------------------------------------------- final
    async def _finish(self) -> None:
        # Let an in-flight partial complete so events stay ordered
        # (partials always precede the final).
        if self._partial_task is not None:
            await asyncio.gather(self._partial_task, return_exceptions=True)
        if self._fatal.done():
            raise self._fatal.result()
        audio = self.buffer.snapshot()
        if audio.size == 0:
            await self._send(final_event("", 0.0, 0.0))
            return
        try:
            result = await self.runner.transcribe(audio, final=True)
        except (InferenceError, InferenceTimeout) as exc:
            raise inference_failure(exc) from exc
        await self._send(final_event(result.text, audio.size / SAMPLE_RATE_HZ, result.inference_s))

    # -------------------------------------------------------------- cleanup
    async def cleanup(self) -> None:
        task, self._partial_task = self._partial_task, None
        if task is not None:
            task.cancel()  # no-op if already finished
            # Also retrieves any exception so it is not logged as unhandled.
            await asyncio.gather(task, return_exceptions=True)
        self.audio_duration_s = self.buffer.duration_s
        self.buffer.clear()
