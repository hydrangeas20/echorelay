"""WebSocket protocol: message shapes, error codes, close codes.

See docs/protocol.md for the human-readable specification.
"""

from __future__ import annotations

import json
from enum import IntEnum
from typing import Any

PROTOCOL_VERSION = "1"


class CloseCode(IntEnum):
    NORMAL = 1000
    UNSUPPORTED_DATA = 1003
    POLICY_VIOLATION = 1008
    MESSAGE_TOO_BIG = 1009
    INTERNAL_ERROR = 1011
    TRY_AGAIN_LATER = 1013


class StreamError(Exception):
    """A protocol- or resource-level failure that ends the stream.

    The transport reports it to the client as an ``error`` event and closes
    the socket with ``close_code``.
    """

    def __init__(self, code: str, message: str, close_code: CloseCode) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.close_code = close_code


def malformed(message: str) -> StreamError:
    return StreamError("malformed_message", message, CloseCode.UNSUPPORTED_DATA)


def parse_control(text: str) -> dict[str, Any]:
    """Parse a client text frame. Only ``{"type": "end"}`` is defined."""
    try:
        msg = json.loads(text)
    except json.JSONDecodeError as exc:
        raise malformed(f"text frame is not valid JSON: {exc.msg}") from exc
    if not isinstance(msg, dict) or not isinstance(msg.get("type"), str):
        raise malformed('text frame must be a JSON object with a string "type" field')
    if msg["type"] != "end":
        raise malformed(f'unknown message type {msg["type"]!r}; expected "end"')
    return msg


def ready_event(session_id: str, limits: dict[str, Any]) -> dict[str, Any]:
    from .config import CHANNELS, ENCODING, SAMPLE_RATE_HZ

    return {
        "type": "ready",
        "protocol_version": PROTOCOL_VERSION,
        "session_id": session_id,
        "audio": {"encoding": ENCODING, "sample_rate": SAMPLE_RATE_HZ, "channels": CHANNELS},
        "limits": limits,
    }


def partial_event(seq: int, text: str, audio_s: float, inference_s: float) -> dict[str, Any]:
    return {
        "type": "partial",
        "provisional": True,  # may change or disappear in later partials / the final
        "seq": seq,
        "text": text,
        "audio_duration_s": round(audio_s, 3),
        "inference_ms": round(inference_s * 1000, 1),
    }


def final_event(text: str, audio_s: float, inference_s: float) -> dict[str, Any]:
    return {
        "type": "final",
        "provisional": False,
        "text": text,
        "audio_duration_s": round(audio_s, 3),
        "inference_ms": round(inference_s * 1000, 1),
    }


def error_event(code: str, message: str) -> dict[str, Any]:
    return {"type": "error", "code": code, "message": message}
