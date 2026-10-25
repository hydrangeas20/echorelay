"""Reliability: disconnects at every stage, shutdown, slow clients, malformed
input mid-stream, and recovery. Runs a real uvicorn server with real sockets."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time

import httpx
import numpy as np
import pytest
import uvicorn
import websockets
from conftest import chunks, make_settings, pcm16

from echorelay.app import create_app
from echorelay.inference import FakeTranscriber, InferenceRunner
from echorelay.protocol import StreamError
from echorelay.session import StreamSession


class LiveServer:
    def __init__(self, fake: FakeTranscriber, **overrides) -> None:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.fake = fake
        app = create_app(make_settings(**overrides), transcriber=fake)
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            assert time.monotonic() < deadline, "server did not start"
            time.sleep(0.02)

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/ws/transcribe"

    def metrics(self) -> dict:
        return httpx.get(f"http://127.0.0.1:{self.port}/metrics").json()

    def wait_until(self, pred, timeout: float = 5.0) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            m = self.metrics()
            if pred(m):
                return m
            assert time.monotonic() < deadline, m
            time.sleep(0.02)

    def wait_idle(self, timeout: float = 5.0) -> dict:
        """Recovery invariant: nothing active, buffered, running, or queued."""
        return self.wait_until(
            lambda m: m["streams"]["active"] == 0
            and m["streams"]["buffered_audio_bytes"] == 0
            and m["inference"]["in_flight"] == 0
            and m["inference"]["waiting"] == 0,
            timeout,
        )

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=15)


@pytest.fixture
def live():
    servers: list[LiveServer] = []

    def _start(fake: FakeTranscriber | None = None, **overrides) -> LiveServer:
        srv = LiveServer(fake or FakeTranscriber(), **overrides)
        servers.append(srv)
        return srv

    yield _start
    for srv in servers:
        srv.stop()


async def _normal_stream(url: str, seconds: float = 1.0) -> list[dict]:
    async with websockets.connect(url) as ws:
        await ws.recv()
        for frame in chunks(pcm16(seconds)):
            await ws.send(frame)
        await ws.send(json.dumps({"type": "end"}))
        events = [json.loads(m) async for m in ws]
    assert ws.close_code == 1000, events
    return events


async def _drain(ws) -> list[dict]:
    """Read events until close. ``async for`` raises on non-1000 closes."""
    events = []
    try:
        async for m in ws:
            events.append(json.loads(m))
    except websockets.ConnectionClosed:
        pass
    return events


async def _await_metrics(srv: LiveServer, pred, what: str, timeout: float = 5.0) -> dict:
    """Wait (bounded) for server state from inside a client scenario, so a
    regression fails with a message instead of hanging the run."""
    deadline = time.monotonic() + timeout
    while True:
        m = srv.metrics()
        if pred(m):
            return m
        assert time.monotonic() < deadline, f"timed out waiting for {what}: {m}"
        await asyncio.sleep(0.01)


def _assert_recovered(srv: LiveServer) -> None:
    srv.wait_idle()
    srv.fake.delay_s = 0.0
    srv.fake.fail_on_final = False
    srv.fake.fail_on_call = None
    events = asyncio.run(_normal_stream(srv.url, 1.0))
    assert events[-1] == {**events[-1], "type": "final", "text": "w1 w2"}
    srv.wait_idle()


# ------------------------------------------------------------- disconnects
def test_disconnect_during_final_decode(live):
    srv = live(FakeTranscriber(delay_s=0.5), partial_interval_s=60)  # no partials

    async def scenario():
        ws = await websockets.connect(srv.url)
        await ws.recv()
        await ws.send(pcm16(1.0))
        await ws.send(json.dumps({"type": "end"}))
        await _await_metrics(srv, lambda m: m["inference"]["in_flight"] > 0, "final decode to start")
        ws.transport.abort()  # gone while the final decode runs

    asyncio.run(scenario())
    m = srv.wait_idle()
    # A peer that vanished mid-decode is a disconnect, not a server failure.
    assert m["streams"]["disconnected"] == 1, m["streams"]
    assert m["streams"]["failed"] == 0, m["streams"]
    _assert_recovered(srv)


def test_disconnect_while_queued_for_inference(live):
    srv = live(FakeTranscriber(delay_s=0.6), partial_interval_s=60, max_concurrent_inference=1)

    async def scenario():
        hog = await websockets.connect(srv.url)  # occupies the only inference slot
        await hog.recv()
        await hog.send(pcm16(1.0))
        await hog.send(json.dumps({"type": "end"}))
        victim = await websockets.connect(srv.url)
        await victim.recv()
        await victim.send(pcm16(1.0))
        await victim.send(json.dumps({"type": "end"}))
        await _await_metrics(srv, lambda m: m["inference"]["waiting"] > 0, "victim to queue")
        victim.transport.abort()  # leaves while waiting for a slot
        events = [json.loads(m) async for m in hog]
        assert events[-1]["type"] == "final"

    asyncio.run(scenario())
    m = srv.wait_idle()
    assert m["streams"]["completed"] == 1
    _assert_recovered(srv)


# -------------------------------------------------------------- slow clients
def test_slow_sender_within_idle_timeout_completes(live):
    srv = live(idle_timeout_s=0.5)

    async def scenario():
        async with websockets.connect(srv.url) as ws:
            await ws.recv()
            # 0.1 s of audio every 0.15 s: slower than real time (1.5x), with
            # gaps far enough under the 0.5 s idle limit (0.35 s of margin)
            # that a late wake-up on a loaded runner cannot fire it. The
            # stream still lasts well past 0.5 s in total, so an idle timer
            # that stopped resetting on each frame would still fail this.
            for frame in chunks(pcm16(1.0), seconds=0.1):
                await ws.send(frame)
                await asyncio.sleep(0.15)
            await ws.send(json.dumps({"type": "end"}))
            return [json.loads(m) async for m in ws], ws.close_code

    events, code = asyncio.run(scenario())
    assert code == 1000 and events[-1]["text"] == "w1 w2"


def test_stalled_sender_hits_idle_timeout_and_frees_slot(live):
    srv = live(idle_timeout_s=0.3, max_concurrent_streams=1)

    async def scenario():
        async with websockets.connect(srv.url) as ws:
            await ws.recv()
            await ws.send(pcm16(0.5))
            await asyncio.sleep(1.0)  # stalls without closing
            return await _drain(ws), ws.close_code

    events, code = asyncio.run(scenario())
    assert events[-1]["code"] == "idle_timeout" and code == 1008
    _assert_recovered(srv)


async def test_client_that_stops_reading_cannot_pin_a_stream():
    """H-d: if sends to the client block (client not reading, TCP window
    full), the stream must still end within the send timeout instead of
    holding its slot forever. Simulated at the session boundary because a
    full TCP window is not reproducible deterministically on loopback."""
    frames: asyncio.Queue = asyncio.Queue()
    for frame in chunks(pcm16(1.0)):
        await frames.put(("bytes", frame))
    await frames.put(("text", json.dumps({"type": "end"})))

    async def receive():
        return await frames.get()

    async def blocked_send(_payload):
        await asyncio.Event().wait()  # the peer never drains its socket

    runner = InferenceRunner(FakeTranscriber(), max_concurrency=1, timeout_s=5)
    session = StreamSession(
        "stuck", make_settings(send_timeout_s=0.3, idle_timeout_s=60), runner, receive, blocked_send
    )
    t0 = time.monotonic()
    with pytest.raises(StreamError) as err:
        await asyncio.wait_for(session.run(), timeout=5)
    assert err.value.code == "send_timeout"
    assert time.monotonic() - t0 < 2
    runner.shutdown()


# ------------------------------------------------- malformed mid-stream
def test_malformed_message_while_partial_in_flight(live):
    srv = live(FakeTranscriber(delay_s=0.5))

    async def scenario():
        async with websockets.connect(srv.url) as ws:
            await ws.recv()
            for frame in chunks(pcm16(1.0)):
                await ws.send(frame)
            await _await_metrics(srv, lambda m: m["inference"]["in_flight"] > 0, "partial decode to start")
            await ws.send("{not json")
            return await _drain(ws), ws.close_code

    events, code = asyncio.run(scenario())
    assert events[-1]["code"] == "malformed_message" and code == 1003
    _assert_recovered(srv)


# ----------------------------------------------------------------- shutdown
def test_shutdown_with_active_streams_terminates(live):
    srv = live(FakeTranscriber(delay_s=0.3))
    closed: list[int | None] = []

    async def client():
        try:
            async with websockets.connect(srv.url) as ws:
                await ws.recv()
                for frame in chunks(pcm16(30.0)):  # would take 30 s at real time
                    await ws.send(frame)
                    await asyncio.sleep(0.1)
        except websockets.ConnectionClosed as exc:
            closed.append(exc.rcvd.code if exc.rcvd else None)

    async def scenario():
        tasks = [asyncio.create_task(client()) for _ in range(3)]
        await _await_metrics(srv, lambda m: m["streams"]["active"] >= 3, "3 active streams", timeout=10)
        t0 = time.monotonic()
        srv.server.should_exit = True
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=15)
        return time.monotonic() - t0

    elapsed = asyncio.run(scenario())
    srv.thread.join(timeout=15)
    assert not srv.thread.is_alive(), "server did not shut down"
    assert len(closed) == 3, closed
    assert elapsed < 10, f"shutdown took {elapsed:.1f}s"


# ---------------------------------------------------------- admission burst
def test_burst_above_admission_limit(live):
    """c = limit + k at once: exactly k rejected with 1013, no other errors,
    admitted streams complete, active never exceeds the limit."""
    limit, extra = 3, 2
    srv = live(FakeTranscriber(delay_s=0.05), max_concurrent_streams=limit)
    gate = None

    async def one():
        async with websockets.connect(srv.url) as ws:
            first = json.loads(await ws.recv())
            if first["type"] == "error":
                return first["code"], await _drain(ws), ws.close_code
            await gate.wait()  # hold the slot until every connection has been answered
            for frame in chunks(pcm16(1.0)):
                await ws.send(frame)
            await ws.send(json.dumps({"type": "end"}))
            return "admitted", [json.loads(m) async for m in ws], ws.close_code

    async def scenario():
        nonlocal gate
        gate = asyncio.Event()
        tasks = [asyncio.create_task(one()) for _ in range(limit + extra)]
        # Poll from a thread: blocking here would stall these connections.
        srv_m = await asyncio.to_thread(
            srv.wait_until, lambda m: m["streams"]["rejected"] == extra, 5
        )
        gate.set()
        return await asyncio.gather(*tasks), srv_m

    results, during = asyncio.run(scenario())
    kinds = [r[0] for r in results]
    assert kinds.count("server_busy") == extra and kinds.count("admitted") == limit
    for kind, events, code in results:
        if kind == "server_busy":
            assert code == 1013 and events == []
        else:
            assert code == 1000 and events[-1]["type"] == "final"
    m = srv.wait_idle()
    assert m["streams"]["peak_active"] == limit
    assert m["streams"]["failed"] == 0 and m["streams"]["completed"] == limit
    _assert_recovered(srv)
