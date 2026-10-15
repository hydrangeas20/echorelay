"""End-to-end WebSocket behaviour against the fake adapter."""

from __future__ import annotations

import threading
import time

from conftest import chunks, collect, pcm16

from echorelay.inference import FakeTranscriber


def test_health(make_client):
    client = make_client()
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"]["backend"] == "fake"
    assert body["streams"] == {
        "active": 0, "max": 8, "completed": 0, "failed": 0, "disconnected": 0,
    }


def test_normal_stream_partials_then_final(make_client):
    client = make_client()
    with client.websocket_connect("/ws/transcribe") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["audio"] == {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1}
        for frame in chunks(pcm16(3.0)):
            ws.send_bytes(frame)
        ws.send_json({"type": "end"})
        events, code = collect(ws)

    assert code == 1000
    partials = [e for e in events if e["type"] == "partial"]
    finals = [e for e in events if e["type"] == "final"]
    assert len(finals) == 1 and events[-1]["type"] == "final"
    assert partials, "expected at least one provisional partial"
    assert all(p["provisional"] is True and p["text"] for p in partials)
    assert [p["seq"] for p in partials] == list(range(1, len(partials) + 1))
    final = finals[0]
    assert final["provisional"] is False
    assert final["text"] == "w1 w2 w3 w4 w5 w6"  # fake: 1 word per 0.5 s
    assert final["audio_duration_s"] == 3.0
    # Partial decodes use partial settings; the last call is the final decode.
    assert client.fake.calls[-1] == (48000, True)
    assert all(final_flag is False for _, final_flag in client.fake.calls[:-1])
    assert client.get("/health").json()["streams"]["completed"] == 1


def test_empty_stream_returns_empty_final(make_client):
    client = make_client()
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.receive_json()
        ws.send_json({"type": "end"})
        events, code = collect(ws)
    assert code == 1000
    assert events == [
        {"type": "final", "provisional": False, "text": "", "audio_duration_s": 0.0, "inference_ms": 0.0}
    ]
    assert client.fake.calls == []  # no inference for empty audio


def test_inference_does_not_block_event_loop(make_client):
    client = make_client(FakeTranscriber(delay_s=1.0))
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.receive_json()
        for frame in chunks(pcm16(1.0)):
            ws.send_bytes(frame)  # 1.0 s buffered -> partial decode starts (takes 1 s)
        deadline = time.monotonic() + 2
        while client.get("/health").json()["inference"]["in_flight"] == 0:
            assert time.monotonic() < deadline, "partial decode never started"
            time.sleep(0.01)
        t0 = time.monotonic()
        health = client.get("/health").json()
        elapsed = time.monotonic() - t0
        assert health["inference"]["in_flight"] == 1
        assert elapsed < 0.5, f"/health took {elapsed:.2f}s while inference was running"
        ws.send_json({"type": "end"})
        events, code = collect(ws)
    assert code == 1000 and events[-1]["type"] == "final"


def test_inference_concurrency_is_bounded(make_client):
    fake = FakeTranscriber(delay_s=0.2)
    client = make_client(fake, max_concurrent_inference=2, max_concurrent_streams=6)
    results = []

    def stream():
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.receive_json()
            for frame in chunks(pcm16(1.5)):
                ws.send_bytes(frame)
            ws.send_json({"type": "end"})
            results.append(collect(ws))

    threads = [threading.Thread(target=stream) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert len(results) == 5
    assert all(code == 1000 and ev[-1]["text"] == "w1 w2 w3" for ev, code in results)
    assert fake.max_active <= 2
