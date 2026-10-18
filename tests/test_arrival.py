"""Arrival schedules in the load generator: sync (default, unchanged) and
stagger (first-wave slot k starts k * stagger_s later)."""

from __future__ import annotations

import argparse
import json

import pytest
import websockets
from conftest import pcm16

from benchmarks.common import Utterance
from benchmarks.load_test import first_wave_offsets, run_load
from benchmarks.stress import scenario_plan

UTTS = [Utterance(f"u{i}", pcm16(0.3 + 0.1 * i), None, f"h{i}") for i in range(5)]


async def _echo_server(ws):
    """Protocol-compatible stand-in: ready, swallow audio, final on end."""
    await ws.send(json.dumps({"type": "ready", "session_id": "x", "limits": {"partial_interval_s": 1.0}}))
    async for msg in ws:
        if isinstance(msg, str):
            await ws.send(json.dumps({"type": "final", "provisional": False, "text": "t",
                                      "audio_duration_s": 0.0, "inference_ms": 0.0}))
            await ws.close()
            return


async def _serve():
    server = await websockets.serve(_echo_server, "127.0.0.1", 0)
    return server, f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/ws/transcribe"


def test_first_wave_offsets():
    assert first_wave_offsets(4) == [0.0, 0.0, 0.0, 0.0]
    assert first_wave_offsets(4, "sync", 9.0) == [0.0] * 4  # stagger_s ignored in sync
    assert first_wave_offsets(8, "stagger", 1.125) == [k * 1.125 for k in range(8)]
    with pytest.raises(ValueError, match="positive"):
        first_wave_offsets(4, "stagger", 0.0)
    with pytest.raises(ValueError, match="arrival"):
        first_wave_offsets(4, "poisson", 1.0)


def _max_overlap(records) -> int:
    events = sorted([(r.t_connect, 1) for r in records] + [(r.t_done, -1) for r in records],
                    key=lambda e: (e[0], e[1]))
    level = peak = 0
    for _, d in events:
        level += d
        peak = max(peak, level)
    return peak


@pytest.mark.parametrize("arrival, stagger", [("sync", 0.0), ("stagger", 0.25)])
async def test_run_load_applies_first_wave_offsets_only(arrival, stagger):
    server, url = await _serve()
    try:
        records, wall = await run_load(url, UTTS, 9, 3, chunk_ms=100, speed=0.0, timeout_s=10,
                                       arrival=arrival, stagger_s=stagger)
    finally:
        server.close()
        await server.wait_closed()
    assert all(r.ok for r in records)
    t0 = min(r.t_connect for r in records)
    first = [r.t_connect - t0 for r in records[:3]]
    expected = first_wave_offsets(3, arrival, stagger)
    for got, want in zip(first, expected):
        assert got == pytest.approx(want, abs=0.08), (first, expected)
    assert [r.start_delay_s for r in records] == expected + [0.0] * 6
    # later streams start only when a slot frees: never more than 3 at once
    assert _max_overlap(records) <= 3
    for r in records[3:]:
        assert r.t_connect >= min(x.t_done for x in records[:3]) - 1e-3
    # identical audio assignment in both modes
    assert [r.utterance for r in records] == [UTTS[i % 5].name for i in range(9)]
    assert all(r.t_done is not None and r.t_done >= r.t_connect for r in records)


async def test_stagger_does_not_shift_measurement_boundaries():
    """TTFP and final latency are measured from the stream's own first frame
    and end message, so the intentional delay is excluded."""
    server, url = await _serve()
    try:
        records, _ = await run_load(url, UTTS[:1], 2, 2, chunk_ms=100, speed=0.0, timeout_s=10,
                                    arrival="stagger", stagger_s=0.3)
    finally:
        server.close()
        await server.wait_closed()
    late = records[1]
    assert late.start_delay_s == 0.3
    assert late.t_first_audio - late.t_connect < 0.1  # clock starts after the delay
    assert late.t_final - late.t_end_sent < 0.1


def test_scenario_plan_names_repeated_blocks():
    assert scenario_plan([1, 4, 8]) == [("c=1", 1), ("c=4", 4), ("c=8", 8)]  # unchanged
    assert scenario_plan([8, 8, 8]) == [("c=8", 8), ("c=8_2", 8), ("c=8_3", 8)]
    assert scenario_plan([4, 8, 4]) == [("c=4", 4), ("c=8", 8), ("c=4_2", 4)]


def test_cli_defaults_are_sync(monkeypatch):
    import benchmarks.load_test as lt
    import benchmarks.stress as st

    captured = []

    def fake_parse(self, *args, **kwargs):
        ns = argparse.ArgumentParser.parse_known_args(self, ["--audio-dir", "x", "--output", "y",
                                                             "--output-dir", "z"])[0]
        captured.append(ns)
        raise SystemExit(0)  # stop before main() does any work

    for mod in (lt, st):
        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", fake_parse)
        monkeypatch.setattr("sys.argv", ["prog"])
        with pytest.raises(SystemExit):
            mod.main()
    assert len(captured) == 2
    for ns in captured:
        assert ns.arrival == "sync" and ns.stagger_s == 0.0
