"""Malformed input, limits, timeouts, inference failures, disconnects."""

from __future__ import annotations

import time

import pytest
from conftest import chunks, collect, pcm16

from echorelay.inference import FakeTranscriber


def _error(events: list[dict]) -> dict:
    assert events and events[-1]["type"] == "error", events
    return events[-1]


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("not json", "not valid JSON"),
        ('["end"]', "JSON object"),
        ('{"type": "start"}', "unknown message type"),
        ('{"kind": "end"}', '"type"'),
    ],
)
def test_malformed_text_frames(make_client, text, fragment):
    client = make_client()
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.receive_json()
        ws.send_text(text)
        events, code = collect(ws)
    err = _error(events)
    assert err["code"] == "malformed_message" and fragment in err["message"]
    assert code == 1003
    assert client.get("/health").json()["streams"]["failed"] == 1


def test_odd_length_audio_frame(make_client):
    client = make_client()
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.receive_json()
        ws.send_bytes(b"\x00\x01\x02")
        events, code = collect(ws)
    assert _error(events)["code"] == "invalid_audio_frame"
    assert code == 1003


def test_oversized_stream_audio(make_client):
    client = make_client(max_audio_seconds=1.0)
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.receive_json()
        for frame in chunks(pcm16(1.5)):
            ws.send_bytes(frame)
        events, code = collect(ws)
    assert _error(events)["code"] == "audio_too_large"
    assert code == 1009


def test_oversized_frame(make_client):
    client = make_client(max_frame_bytes=1024)
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.receive_json()
        ws.send_bytes(b"\x00" * 2048)
        events, code = collect(ws)
    assert _error(events)["code"] == "frame_too_large"
    assert code == 1009


def test_final_inference_error(make_client):
    client = make_client(FakeTranscriber(fail_on_final=True))
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.receive_json()
        for frame in chunks(pcm16(1.0)):
            ws.send_bytes(frame)
        ws.send_json({"type": "end"})
        events, code = collect(ws)
    err = _error(events)
    assert err["code"] == "inference_error" and "injected failure" in err["message"]
    assert code == 1011
    assert not [e for e in events if e["type"] == "final"]


def test_partial_inference_error_aborts_stream_promptly(make_client):
    client = make_client(FakeTranscriber(fail_on_call=1))
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.receive_json()
        for frame in chunks(pcm16(1.0)):  # triggers the first (failing) partial
            ws.send_bytes(frame)
        # No "end" sent: the server must abort on its own.
        events, code = collect(ws)
    assert _error(events)["code"] == "inference_error"
    assert code == 1011


def test_inference_timeout(make_client):
    client = make_client(FakeTranscriber(delay_s=0.6), inference_timeout_s=0.2,
                         partial_interval_s=10.0)
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.receive_json()
        ws.send_bytes(pcm16(0.5))
        ws.send_json({"type": "end"})
        events, code = collect(ws)
    assert _error(events)["code"] == "inference_timeout"
    assert code == 1011


def test_idle_timeout(make_client):
    client = make_client(idle_timeout_s=0.3)
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.receive_json()
        t0 = time.monotonic()
        events, code = collect(ws)
    assert _error(events)["code"] == "idle_timeout"
    assert code == 1008
    assert time.monotonic() - t0 < 3


def test_capacity_limit_rejects_extra_streams(make_client):
    client = make_client(max_concurrent_streams=1)
    with client.websocket_connect("/ws/transcribe") as first:
        assert first.receive_json()["type"] == "ready"
        with client.websocket_connect("/ws/transcribe") as second:
            events, code = collect(second)
        assert _error(events)["code"] == "server_busy"
        assert code == 1013
        first.send_json({"type": "end"})
        _, code = collect(first)
        assert code == 1000
    # Capacity is released once the first stream ends.
    with client.websocket_connect("/ws/transcribe") as third:
        assert third.receive_json()["type"] == "ready"
        third.send_json({"type": "end"})
        collect(third)


# Client disconnect cleanup is tested against a real uvicorn server in
# test_live_server.py: Starlette's TestClient cancels the handler task on exit
# instead of delivering a network disconnect, so it cannot test that path.
