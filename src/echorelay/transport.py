"""WebSocket transport: admission control, framing, error reporting, cleanup."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from .metrics import LevelTracker, Metrics
from .protocol import CloseCode, StreamError, error_event, ready_event
from .session import ClientDisconnected, Incoming, StreamSession

log = logging.getLogger(__name__)
router = APIRouter()


class StreamRegistry:
    """Tracks active streams for admission control, /health and /metrics."""

    def __init__(self, max_streams: int) -> None:
        self.max_streams = max_streams
        self.active: set[str] = set()
        self.sessions: dict[str, StreamSession] = {}
        self.completed = 0
        self.failed = 0
        self.disconnected = 0
        self.rejected = 0
        self.peak_active = 0
        self.active_level = LevelTracker()  # time at each active-stream count

    def try_admit(self, session_id: str) -> bool:
        # Single event loop: check-and-add cannot interleave.
        if len(self.active) >= self.max_streams:
            self.rejected += 1
            return False
        self.active.add(session_id)
        self.peak_active = max(self.peak_active, len(self.active))
        self.active_level.set(len(self.active))
        return True

    def release(self, session_id: str) -> None:
        self.active.discard(session_id)
        self.active_level.set(len(self.active))
        self.sessions.pop(session_id, None)

    def buffered_audio_bytes(self) -> int:
        return sum(s.buffer.num_bytes for s in self.sessions.values())

    def snapshot(self) -> dict:
        return {
            "active": len(self.active),
            "peak_active": self.peak_active,
            "max": self.max_streams,
            "completed": self.completed,
            "failed": self.failed,
            "disconnected": self.disconnected,
            "rejected": self.rejected,
            "buffered_audio_bytes": self.buffered_audio_bytes(),
        }


def record_stream_end(metrics: Metrics, session: StreamSession, outcome: str) -> None:
    """Fold one finished stream's statistics into the metrics and log one
    summary line. The log never contains transcript text."""
    summary = session.stats.summary(session.audio_duration_s)
    metrics.inc(f"streams.outcome.{outcome}")
    metrics.observe("stream.audio_s", session.audio_duration_s)
    for key in ("partials_scheduled", "partials_sent", "partials_deferred",
                "partials_skipped_empty", "partials_skipped_duplicate"):
        metrics.inc(f"stream.{key}", summary[key])
    metrics.observe("stream.partial_decode_s", summary["partial_decode_s"])
    if summary["partial_amplification"] is not None:
        metrics.observe("stream.partial_amplification", summary["partial_amplification"])
    for v in session.stats.partial_staleness_s:
        metrics.observe("stream.partial_staleness_s", v)
    metrics.inc(f"stream.partial_at_end.{summary['partial_state_at_end']}")
    if summary["inference_waiting_at_end"] is not None:
        metrics.observe("stream.inference_waiting_at_end", summary["inference_waiting_at_end"])
    for key in ("final_wait_partial_s", "final_queue_s", "final_decode_s"):
        if summary[key] is not None:
            metrics.observe(f"stream.{key}", summary[key])
    log.info(
        "stream_summary session=%s outcome=%s %s",
        session.session_id, outcome, " ".join(f"{k}={v}" for k, v in summary.items()),
    )


async def _receive(ws: WebSocket) -> Incoming:
    msg = await ws.receive()
    if msg["type"] == "websocket.disconnect":
        return ("disconnect", msg.get("code"))
    if msg.get("bytes") is not None:
        return ("bytes", msg["bytes"])
    if msg.get("text") is not None:
        return ("text", msg["text"])
    return ("bytes", b"")  # pragma: no cover - empty frame


async def _send_json(ws: WebSocket, payload: dict[str, Any]) -> None:
    await ws.send_json(payload)


async def _close(ws: WebSocket, code: int, reason: str = "", timeout_s: float = 10.0) -> None:
    if ws.application_state == WebSocketState.CONNECTED and ws.client_state == WebSocketState.CONNECTED:
        try:
            async with asyncio.timeout(timeout_s):
                await ws.close(code=code, reason=reason[:120])
        except (RuntimeError, WebSocketDisconnect, TimeoutError, OSError):  # peer gone or stuck
            pass


@router.websocket("/ws/transcribe")
async def transcribe(ws: WebSocket) -> None:
    state = ws.app.state
    settings, runner, registry = state.settings, state.runner, state.registry
    metrics: Metrics = state.metrics
    session_id = uuid.uuid4().hex[:12]
    session: StreamSession | None = None
    outcome = "failed"
    await ws.accept()

    if not registry.try_admit(session_id):
        metrics.inc("streams.outcome.rejected")
        log.warning("session %s rejected: at capacity (%d)", session_id, registry.max_streams)
        await _safe_send(ws, error_event("server_busy", "maximum concurrent streams reached"),
                         settings.send_timeout_s)
        await _close(ws, CloseCode.TRY_AGAIN_LATER, "server_busy", settings.send_timeout_s)
        return

    log.info("session %s opened", session_id)
    try:
        await _send_bounded(ws, settings.send_timeout_s, lambda: ws.send_json(
            ready_event(
                session_id,
                {
                    "max_audio_seconds": settings.max_audio_seconds,
                    "max_frame_bytes": settings.max_frame_bytes,
                    "idle_timeout_s": settings.idle_timeout_s,
                    "partial_interval_s": settings.partial_interval_s,
                },
            )
        ))
        session = StreamSession(
            session_id,
            settings,
            runner,
            receive=lambda: _receive(ws),
            send=lambda payload: _send_json(ws, payload),
        )
        registry.sessions[session_id] = session
        await session.run()
        outcome = "completed"
        registry.completed += 1
        log.info("session %s completed (%.2fs audio)", session_id, session.audio_duration_s)
        await _close(ws, CloseCode.NORMAL, timeout_s=settings.send_timeout_s)
    except StreamError as err:
        outcome = f"error.{err.code}"
        registry.failed += 1
        log.warning("session %s failed: %s: %s", session_id, err.code, err.message)
        await _safe_send(ws, error_event(err.code, err.message), settings.send_timeout_s)
        await _close(ws, err.close_code, err.code, settings.send_timeout_s)
    except (ClientDisconnected, WebSocketDisconnect):
        outcome = "disconnected"
        registry.disconnected += 1
        log.info("session %s: client disconnected", session_id)
    except asyncio.CancelledError:
        # Handler task cancelled (server shutdown, or an ASGI server that
        # cancels on disconnect). Session cleanup already ran in run()'s finally.
        outcome = "cancelled"
        registry.disconnected += 1
        log.info("session %s: cancelled", session_id)
        raise
    except Exception:
        registry.failed += 1
        log.exception("session %s: unexpected server error", session_id)
        await _safe_send(ws, error_event("internal_error", "unexpected server error"),
                         settings.send_timeout_s)
        await _close(ws, CloseCode.INTERNAL_ERROR, "internal_error", settings.send_timeout_s)
    finally:
        registry.release(session_id)
        if session is not None:
            try:
                record_stream_end(metrics, session, outcome)
            except Exception:  # observability must never break a stream
                log.exception("session %s: failed to record metrics", session_id)


async def _send_bounded(ws: WebSocket, timeout_s: float, send) -> None:
    try:
        async with asyncio.timeout(timeout_s):
            await send()
    except TimeoutError as exc:
        raise StreamError(
            "send_timeout", f"client did not accept data within {timeout_s:.1f}s",
            CloseCode.POLICY_VIOLATION,
        ) from exc


async def _safe_send(ws: WebSocket, payload: dict[str, Any], timeout_s: float = 10.0) -> None:
    """Best-effort send of a terminal message; never raises, never hangs."""
    try:
        if ws.client_state == WebSocketState.CONNECTED:
            async with asyncio.timeout(timeout_s):
                await ws.send_json(payload)
    except (RuntimeError, WebSocketDisconnect, TimeoutError, OSError):
        pass
