"""Notebook helpers: real server process start/verify/restart/stop, commit and
audio verification. Uses the fake backend on a free port."""

from __future__ import annotations

import importlib
import json
import socket
import subprocess
import wave

import numpy as np
import pytest


@pytest.fixture
def helpers(monkeypatch, tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setenv("ECHORELAY_BENCH_PORT", str(port))
    import benchmarks.colab_helpers as h

    h = importlib.reload(h)
    monkeypatch.setattr(h, "PIDFILE", tmp_path / "server.pid")
    yield h
    h.stop_server()


def test_start_verify_restart_stop(helpers, tmp_path):
    h = helpers
    first = h.start_server(tmp_path / "a.log", backend="fake", event_timing="true")
    h.verify_server({"backend": "fake", "event_timing": True, "max_concurrent_inference": 1,
                     "partial_interval_s": 1.0, "final_beam_size": 5, "active_streams": 0})
    with pytest.raises(AssertionError, match="device"):
        h.verify_server({"device": "cuda"})

    second = h.start_server(tmp_path / "b.log", backend="fake", max_concurrent_inference=2)
    assert first.wait(timeout=10) is not None  # restart stopped the old server
    assert second.pid != first.pid
    h.verify_server({"max_concurrent_inference": 2, "event_timing": False})

    h.save_provenance(tmp_path / "prov", "selftest")
    prov = json.loads((tmp_path / "prov" / "provenance.json").read_text())
    assert prov["server_metrics"]["config"]["max_concurrent_inference"] == 2
    assert len(prov["git_commit"]) == 40

    h.stop_server()
    assert second.wait(timeout=10) is not None
    h.stop_server()  # idempotent


def test_start_server_reports_a_crashing_server(helpers, tmp_path):
    with pytest.raises(RuntimeError, match="exited"):
        helpers.start_server(tmp_path / "bad.log", backend="fake", max_concurrent_streams=0)


def test_verify_commit(helpers):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=helpers.REPO, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=helpers.REPO, text=True).strip()
    if not dirty:
        helpers.verify_commit(head)
    with pytest.raises(AssertionError, match="expected commit"):
        helpers.verify_commit("0" * 40)


def _wav(path, samples):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def test_verify_eval_audio_hashes_pcm_not_file(helpers, tmp_path):
    from benchmarks.common import audio_manifest, load_audio_dir

    audio = tmp_path / "audio"
    audio.mkdir()
    _wav(audio / "a.wav", np.arange(1600))
    _wav(audio / "b.wav", np.zeros(800))
    manifest = tmp_path / "result.json"
    manifest.write_text(json.dumps({"audio": audio_manifest(load_audio_dir(audio))}))
    helpers.verify_eval_audio(audio, manifest)

    _wav(audio / "b.wav", np.ones(800))  # same length, different samples
    with pytest.raises(AssertionError, match="differs"):
        helpers.verify_eval_audio(audio, manifest)


def test_stop_server_started_by_another_kernel(helpers, tmp_path):
    """After a notebook kernel restart only the pid file remains."""
    import time

    proc = helpers.start_server(tmp_path / "c.log", backend="fake")
    helpers._PROC = None  # simulate a fresh kernel
    t0 = time.monotonic()
    helpers.stop_server()
    assert time.monotonic() - t0 < 15
    assert proc.wait(timeout=5) is not None
