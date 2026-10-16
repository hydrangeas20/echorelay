"""Regression tests: a stalled WebSocket must not hang the benchmark."""

from __future__ import annotations

import asyncio
import json
import time

import websockets

from benchmarks.common import Utterance
from benchmarks.load_test import run_load, run_stream, summarize
from conftest import pcm16

UTT = Utterance(name="tone", pcm16=pcm16(0.5), reference=None, sha256="x")


async def _silent_server(ws):
    """Accepts the connection and never sends anything (returns once the client goes)."""
    await ws.wait_closed()


async def _no_final_server(ws):
    """Sends ready, swallows all audio and the end message, never answers."""
    await ws.send(json.dumps({"type": "ready", "session_id": "stall", "limits": {}}))
    async for _ in ws:  # ends when the client gives up and disconnects
        pass


async def _error_server(ws):
    await ws.send(json.dumps({"type": "ready", "session_id": "err", "limits": {}}))
    await ws.recv()
    await ws.send(json.dumps({"type": "error", "code": "inference_error", "message": "boom"}))
    await ws.close(1011)


async def _serve(handler):
    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, f"ws://127.0.0.1:{port}/ws/transcribe"


async def test_stream_times_out_waiting_for_ready():
    server, url = await _serve(_silent_server)
    try:
        t0 = time.monotonic()
        rec = await run_stream(url, UTT, 0, chunk_ms=100, speed=0.0, timeout_s=0.5)
        assert time.monotonic() - t0 < 2.0
    finally:
        server.close()
        await server.wait_closed()
    assert not rec.ok
    assert rec.error == "stream_timeout"
    assert "waiting for ready" in rec.error_detail


async def test_stream_times_out_waiting_for_final():
    server, url = await _serve(_no_final_server)
    try:
        t0 = time.monotonic()
        rec = await run_stream(url, UTT, 0, chunk_ms=100, speed=0.0, timeout_s=0.5)
        assert time.monotonic() - t0 < 2.0
    finally:
        server.close()
        await server.wait_closed()
    assert rec.error == "stream_timeout"
    assert "waiting for final" in rec.error_detail
    assert rec.t_end_sent is not None  # it got as far as sending end


async def test_failures_are_counted_with_reasons():
    stall, stall_url = await _serve(_no_final_server)
    err, err_url = await _serve(_error_server)
    try:
        records, wall = await run_load(stall_url, [UTT], 2, concurrency=2, chunk_ms=100,
                                       speed=0.0, timeout_s=0.5)
        err_records, _ = await run_load(err_url, [UTT], 1, concurrency=1, chunk_ms=100,
                                        speed=0.0, timeout_s=5.0)
    finally:
        for s in (stall, err):
            s.close()
            await s.wait_closed()
    assert wall < 2.0  # the two stalled streams ran concurrently and were cut off

    summary = summarize(records + err_records, wall, [UTT])
    assert summary["streams_total"] == 3
    assert summary["streams_ok"] == 0
    assert summary["streams_failed"] == 3
    assert summary["errors"] == {"stream_timeout": 2, "inference_error": 1}
    details = {f["error"]: f["detail"] for f in summary["failures"]}
    assert details["inference_error"] == "boom"
    assert "0.5s" in details["stream_timeout"]
