"""WebSocket transport: admission control, framing, error reporting, cleanup."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from .protocol import CloseCode, StreamError, error_event, ready_event
from .session import ClientDisconnected, Incoming, StreamSession

log = logging.getLogger(__name__)
router = APIRouter()


class StreamRegistry:
    """Tracks active streams for admission control and /health."""

    def __init__(self, max_streams: int) -> None:
        self.max_streams = max_streams
        self.active: set[str] = set()
        self.completed = 0
        self.failed = 0
        self.disconnected = 0

    def try_admit(self, session_id: str) -> bool:
        # Single event loop: check-and-add cannot interleave.
        if len(self.active) >= self.max_streams:
            return False
        self.active.add(session_id)
        return True

    def release(self, session_id: str) -> None:
        self.active.discard(session_id)


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


async def _close(ws: WebSocket, code: int, reason: str = "") -> None:
    if ws.application_state == WebSocketState.CONNECTED and ws.client_state == WebSocketState.CONNECTED:
        try:
            await ws.close(code=code, reason=reason[:120])
        except (RuntimeError, WebSocketDisconnect):  # peer already gone
            pass


@router.websocket("/ws/transcribe")
async def transcribe(ws: WebSocket) -> None:
    state = ws.app.state
    settings, runner, registry = state.settings, state.runner, state.registry
    session_id = uuid.uuid4().hex[:12]
    await ws.accept()

    if not registry.try_admit(session_id):
        log.warning("session %s rejected: at capacity (%d)", session_id, registry.max_streams)
        await _safe_send(ws, error_event("server_busy", "maximum concurrent streams reached"))
        await _close(ws, CloseCode.TRY_AGAIN_LATER, "server_busy")
        return

    log.info("session %s opened", session_id)
    try:
        await ws.send_json(
            ready_event(
                session_id,
                {
                    "max_audio_seconds": settings.max_audio_seconds,
                    "max_frame_bytes": settings.max_frame_bytes,
                    "idle_timeout_s": settings.idle_timeout_s,
                    "partial_interval_s": settings.partial_interval_s,
                },
            )
        )
        session = StreamSession(
            session_id,
            settings,
            runner,
            receive=lambda: _receive(ws),
            send=lambda payload: _send_json(ws, payload),
        )
        await session.run()
        registry.completed += 1
        log.info("session %s completed (%.2fs audio)", session_id, session.audio_duration_s)
        await _close(ws, CloseCode.NORMAL)
    except StreamError as err:
        registry.failed += 1
        log.warning("session %s failed: %s: %s", session_id, err.code, err.message)
        await _safe_send(ws, error_event(err.code, err.message))
        await _close(ws, err.close_code, err.code)
    except (ClientDisconnected, WebSocketDisconnect):
        registry.disconnected += 1
        log.info("session %s: client disconnected", session_id)
    except asyncio.CancelledError:
        # Handler task cancelled (server shutdown, or an ASGI server that
        # cancels on disconnect). Session cleanup already ran in run()'s finally.
        registry.disconnected += 1
        log.info("session %s: cancelled", session_id)
        raise
    except Exception:
        registry.failed += 1
        log.exception("session %s: unexpected server error", session_id)
        await _safe_send(ws, error_event("internal_error", "unexpected server error"))
        await _close(ws, CloseCode.INTERNAL_ERROR, "internal_error")
    finally:
        registry.release(session_id)


async def _safe_send(ws: WebSocket, payload: dict[str, Any]) -> None:
    try:
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_json(payload)
    except (RuntimeError, WebSocketDisconnect):
        pass
