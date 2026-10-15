from __future__ import annotations

import contextlib
from collections.abc import Iterator

import numpy as np
import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from echorelay.app import create_app
from echorelay.config import SAMPLE_RATE_HZ, Settings
from echorelay.inference import FakeTranscriber

CHUNK_S = 0.1


def pcm16(seconds: float, freq: float = 440.0) -> bytes:
    n = int(seconds * SAMPLE_RATE_HZ)
    t = np.arange(n) / SAMPLE_RATE_HZ
    return (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2").tobytes()


def chunks(data: bytes, seconds: float = CHUNK_S) -> Iterator[bytes]:
    step = int(seconds * SAMPLE_RATE_HZ) * 2
    for i in range(0, len(data), step):
        yield data[i : i + step]


def collect(ws) -> tuple[list[dict], int | None]:
    """Read events until the server closes. Returns (events, close_code)."""
    events: list[dict] = []
    while True:
        try:
            events.append(ws.receive_json())
        except WebSocketDisconnect as exc:
            return events, exc.code


def make_settings(**overrides) -> Settings:
    base = dict(
        backend="fake",
        partial_interval_s=1.0,
        min_partial_audio_s=0.5,
        idle_timeout_s=5.0,
        inference_timeout_s=5.0,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def make_client():
    stack = contextlib.ExitStack()

    def _make(transcriber: FakeTranscriber | None = None, **overrides) -> TestClient:
        fake = transcriber or FakeTranscriber()
        app = create_app(make_settings(**overrides), transcriber=fake)
        client = stack.enter_context(TestClient(app))
        client.fake = fake  # type: ignore[attr-defined]
        return client

    yield _make
    stack.close()
