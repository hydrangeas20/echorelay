"""Tests against a real uvicorn server and a real WebSocket client.

These exercise the actual network disconnect path (uvicorn delivers
``websocket.disconnect``), which the in-process TestClient only approximates.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn
import websockets
from conftest import chunks, make_settings, pcm16

from echorelay.app import create_app
from echorelay.inference import FakeTranscriber


@pytest.fixture
def live_server():
    servers = []

    def _start(fake: FakeTranscriber, **overrides) -> str:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        app = create_app(make_settings(**overrides), transcriber=fake)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            assert time.monotonic() < deadline, "server did not start"
            time.sleep(0.02)
        servers.append((server, thread))
        return f"127.0.0.1:{port}"

    yield _start
    for server, thread in servers:
        server.should_exit = True
        thread.join(timeout=10)


def _health(addr: str) -> dict:
    return httpx.get(f"http://{addr}/health").json()


def test_live_stream_and_abrupt_disconnect(live_server):
    fake = FakeTranscriber(delay_s=0.5)
    addr = live_server(fake)

    async def scenario():
        # 1) a normal stream over a real socket
        async with websockets.connect(f"ws://{addr}/ws/transcribe") as ws:
            assert json.loads(await ws.recv())["type"] == "ready"
            for frame in chunks(pcm16(1.0)):
                await ws.send(frame)
            await ws.send(json.dumps({"type": "end"}))
            events = [json.loads(m) async for m in ws]
            assert ws.close_code == 1000
        assert events[-1] == {**events[-1], "type": "final", "text": "w1 w2"}

        # 2) disconnect while a partial decode is running in a worker thread
        ws = await websockets.connect(f"ws://{addr}/ws/transcribe")
        await ws.recv()
        for frame in chunks(pcm16(1.2)):
            await ws.send(frame)
        deadline = time.monotonic() + 3
        while _health(addr)["inference"]["in_flight"] == 0:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.01)
        ws.transport.abort()  # no close handshake: simulates a dropped client

    async def another_stream():
        async with websockets.connect(f"ws://{addr}/ws/transcribe") as ws:
            await ws.recv()
            await ws.send(pcm16(0.5))
            await ws.send(json.dumps({"type": "end"}))
            return [json.loads(m) async for m in ws]

    asyncio.run(scenario())

    deadline = time.monotonic() + 5
    while True:
        h = _health(addr)
        if h["streams"]["active"] == 0 and h["inference"]["in_flight"] == 0:
            break
        assert time.monotonic() < deadline, h
        time.sleep(0.05)
    assert h["streams"]["completed"] == 1
    assert h["streams"]["disconnected"] == 1
    assert h["streams"]["failed"] == 0

    # The server keeps serving after the dropped client.
    fake.delay_s = 0.0
    assert asyncio.run(another_stream())[-1]["text"] == "w1"
