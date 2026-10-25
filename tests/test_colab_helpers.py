"""Notebook helpers: real server process start/verify/restart/stop, commit and
audio verification, and shutdown reliability (another kernel, stale pid
files, pid reuse, escalation, macOS-style liveness, port checks). Server tests
use the fake backend on a free port."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import wave

import numpy as np
import pytest

HAS_PS = shutil.which("ps") is not None
HAS_PROC = os.path.isdir("/proc/self")
HAS_PIDFD = hasattr(os, "pidfd_open")


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
    assert not h.PIDFILE.exists()


@pytest.fixture
def spawned():
    """Start helper processes; every one is killed and reaped at teardown,
    even if the test fails."""
    procs: list[subprocess.Popen] = []

    def spawn(code: str, *args: str) -> subprocess.Popen:
        p = subprocess.Popen([sys.executable, "-c", code, *args], stdout=subprocess.PIPE,
                             text=True, start_new_session=True)
        procs.append(p)
        return p

    yield spawn
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        p.wait(timeout=10)
        if p.stdout:
            p.stdout.close()


def _until(predicate, timeout: float = 10.0, what: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.01)


# Waits for SIGTERM (or ignores it), printing "ready" once its handler is set.
SLEEPER = ("import signal, sys, time\n"
           "if 'ignore-term' in sys.argv: signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
           "print('ready', flush=True)\n"
           "time.sleep(120)\n")


def _ready(p: subprocess.Popen) -> subprocess.Popen:
    assert p.stdout.readline().strip() == "ready"
    return p


def _record_signals(monkeypatch, h) -> list[tuple[int, int]]:
    """Record every real signal (not kill(pid, 0) probes) the helpers send."""
    sent: list[tuple[int, int]] = []
    real_kill, real_killpg = os.kill, os.killpg

    def kill(pid, sig):
        if sig:
            sent.append((pid, sig))
        return real_kill(pid, sig)

    def killpg(pgid, sig):
        sent.append((pgid, sig))
        return real_killpg(pgid, sig)

    monkeypatch.setattr(h.os, "kill", kill)
    monkeypatch.setattr(h.os, "killpg", killpg)
    return sent


# ----------------------------------------------------------- lifecycle
def test_start_verify_restart_stop(helpers, tmp_path):
    h = helpers
    first = h.start_server(tmp_path / "a.log", backend="fake", event_timing="true")
    rec = json.loads(h.PIDFILE.read_text())
    assert rec["pid"] == first.pid and rec["port"] == h.PORT and rec["start"]
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

    log = h._LOG
    h.stop_server()
    assert second.wait(timeout=10) is not None
    assert not h.PIDFILE.exists() and h._PROC is None and h._LOG is None and log.closed
    assert not h._port_listening()
    h.stop_server()  # idempotent


def test_start_server_reports_a_crashing_server(helpers, tmp_path):
    with pytest.raises(RuntimeError, match="exited"):
        helpers.start_server(tmp_path / "bad.log", backend="fake", max_concurrent_streams=0)
    # nothing left behind for the next start
    assert not helpers.PIDFILE.exists() and helpers._PROC is None and helpers._LOG is None


def test_start_server_cleans_up_when_never_healthy(helpers, tmp_path, monkeypatch):
    monkeypatch.setattr(helpers, "START_TIMEOUT_S", 0.0)
    with pytest.raises(RuntimeError, match="not healthy"):
        helpers.start_server(tmp_path / "slow.log", backend="fake")
    assert not helpers.PIDFILE.exists() and helpers._PROC is None and helpers._LOG is None
    assert not helpers._port_listening()


def test_repeated_stop_without_a_server(helpers):
    for _ in range(3):
        helpers.stop_server()
    assert not helpers.PIDFILE.exists()


@pytest.mark.parametrize("liveness", [
    pytest.param("proc", marks=pytest.mark.skipif(not HAS_PROC, reason="needs /proc")),
    pytest.param("ps", marks=pytest.mark.skipif(not HAS_PS, reason="needs ps")),
])
def test_stop_server_started_by_another_kernel(helpers, tmp_path, monkeypatch, liveness):
    """After a notebook kernel restart only the pid file remains. With
    liveness="ps" the helpers run as on macOS, which has no /proc."""
    monkeypatch.setattr(helpers, "_HAVE_PROC", liveness == "proc")
    proc = helpers.start_server(tmp_path / "c.log", backend="fake")
    helpers._PROC = None  # simulate a fresh kernel
    sent = _record_signals(monkeypatch, helpers)
    t0 = time.monotonic()
    helpers.stop_server()
    assert time.monotonic() - t0 < 15
    assert sent == [(proc.pid, signal.SIGTERM)]  # graceful: no SIGKILL needed
    assert proc.wait(timeout=5) is not None
    assert not helpers._port_listening()


def test_graceful_shutdown_sends_only_sigterm(helpers, tmp_path, monkeypatch):
    proc = helpers.start_server(tmp_path / "g.log", backend="fake")
    sent = _record_signals(monkeypatch, helpers)
    helpers.stop_server()
    assert sent == [(proc.pid, signal.SIGTERM)]


def test_already_exited_server_is_not_signalled(helpers, tmp_path, monkeypatch):
    proc = helpers.start_server(tmp_path / "x.log", backend="fake")
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=10)  # exited and reaped: its pid may be reused
    sent = _record_signals(monkeypatch, helpers)
    helpers.stop_server()  # same kernel
    assert sent == []
    helpers.PIDFILE.write_text(json.dumps({"pid": proc.pid, "port": helpers.PORT, "start": "x"}))
    helpers.stop_server()  # another kernel, stale pid file
    assert sent == [] and not helpers.PIDFILE.exists()


# ------------------------------------------------- stale pid files, reuse
@pytest.mark.parametrize("content", ["legacy", "wrong-start", "garbage"])
def test_stale_pid_file_never_signals_an_unrelated_process(helpers, spawned, monkeypatch, content):
    """The pid in the file now belongs to someone else's session leader, the
    dangerous case for killpg."""
    other = _ready(spawned(SLEEPER))
    text = {"legacy": str(other.pid),
            "wrong-start": json.dumps({"pid": other.pid, "port": helpers.PORT, "start": "proc:1"}),
            "garbage": "not a pid"}[content]
    helpers.PIDFILE.write_text(text)
    sent = _record_signals(monkeypatch, helpers)
    helpers.stop_server()
    assert sent == [] and other.poll() is None
    assert not helpers.PIDFILE.exists()


def test_pid_reuse_detected_by_start_time(helpers, spawned, monkeypatch):
    """Same pid, a command line that looks like the server, but a different
    start time: a reused pid must not be signalled."""
    lookalike = _ready(spawned(SLEEPER, "-m", "echorelay", "--port", str(helpers.PORT)))
    assert helpers._ownership({"pid": lookalike.pid, "start": helpers._identity(lookalike.pid)["start"]}) == "owned"
    helpers.PIDFILE.write_text(json.dumps({"pid": lookalike.pid, "port": helpers.PORT, "start": "proc:0"}))
    sent = _record_signals(monkeypatch, helpers)
    helpers.stop_server()
    assert sent == [] and lookalike.poll() is None


def test_unverifiable_ownership_fails_safe(helpers, spawned, monkeypatch):
    other = _ready(spawned(SLEEPER, "-m", "echorelay", "--port", str(helpers.PORT)))
    helpers.PIDFILE.write_text(json.dumps({"pid": other.pid, "port": helpers.PORT, "start": None}))
    monkeypatch.setattr(helpers, "_identity", lambda pid: None)  # no /proc and no ps
    sent = _record_signals(monkeypatch, helpers)
    with pytest.raises(RuntimeError, match="cannot verify"):
        helpers.stop_server()
    assert sent == [] and other.poll() is None
    assert helpers.PIDFILE.exists()  # kept so the user can act on it
    helpers.PIDFILE.unlink()


# ------------------------------------------------------------ escalation
@pytest.mark.parametrize("pidfd", [
    pytest.param(True, marks=pytest.mark.skipif(not HAS_PIDFD, reason="Linux pidfd")),
    False,
])
def test_sigterm_ignored_escalates_to_sigkill(helpers, spawned, monkeypatch, pidfd):
    if not pidfd:
        monkeypatch.setattr(helpers, "_pidfd", lambda pid: None)  # as on macOS
    stubborn = _ready(spawned(SLEEPER, "ignore-term", "-m", "echorelay", "--port", str(helpers.PORT)))
    ident = helpers._identity(stubborn.pid)
    helpers.PIDFILE.write_text(json.dumps({"pid": stubborn.pid, "port": helpers.PORT,
                                           "start": ident["start"]}))
    monkeypatch.setattr(helpers, "STOP_GRACE_S", 0.3)
    sent = _record_signals(monkeypatch, helpers)
    t0 = time.monotonic()
    helpers.stop_server()
    assert time.monotonic() - t0 < 5
    assert sent == [(stubborn.pid, signal.SIGTERM), (stubborn.pid, signal.SIGKILL)]
    assert stubborn.wait(timeout=5) == -signal.SIGKILL


def test_unkillable_process_raises_and_keeps_pid_file(helpers, spawned, monkeypatch):
    stubborn = _ready(spawned(SLEEPER, "ignore-term", "-m", "echorelay", "--port", str(helpers.PORT)))
    helpers.PIDFILE.write_text(json.dumps({"pid": stubborn.pid, "port": helpers.PORT,
                                           "start": helpers._identity(stubborn.pid)["start"]}))
    monkeypatch.setattr(helpers, "STOP_GRACE_S", 0.1)
    monkeypatch.setattr(helpers, "STOP_KILL_S", 0.2)
    monkeypatch.setattr(helpers, "_signal_group", lambda *a: True)  # signals never arrive
    with pytest.raises(RuntimeError, match="did not exit"):
        helpers.stop_server()
    assert helpers.PIDFILE.exists() and stubborn.poll() is None
    helpers.PIDFILE.unlink()


# A pid file without a recorded start time names a process whose command line
# may match but whose identity cannot be confirmed: never signal it.
@pytest.mark.parametrize("liveness", [
    pytest.param("proc", marks=pytest.mark.skipif(not HAS_PROC, reason="needs /proc")),
    pytest.param("ps", marks=pytest.mark.skipif(not HAS_PS, reason="needs ps")),
])
@pytest.mark.parametrize("content", ["legacy-int", "start-null", "start-missing"])
def test_pid_file_without_start_time_fails_safe(helpers, spawned, monkeypatch, content, liveness):
    monkeypatch.setattr(helpers, "_HAVE_PROC", liveness == "proc")
    lookalike = _ready(spawned(SLEEPER, "-m", "echorelay", "--port", str(helpers.PORT)))
    assert all(m in helpers._identity(lookalike.pid)["command"] for m in helpers._server_marker())
    text = {"legacy-int": str(lookalike.pid),
            "start-null": json.dumps({"pid": lookalike.pid, "port": helpers.PORT, "start": None}),
            "start-missing": json.dumps({"pid": lookalike.pid, "port": helpers.PORT})}[content]
    helpers.PIDFILE.write_text(text)
    sent = _record_signals(monkeypatch, helpers)
    for _ in range(2):  # stays safe when retried
        with pytest.raises(RuntimeError, match="no start time"):
            helpers.stop_server()
    assert sent == [] and lookalike.poll() is None
    assert helpers.PIDFILE.read_text() == text  # kept, unchanged, for the user to act on
    assert helpers._LOG is None
    helpers.PIDFILE.unlink()


def test_pid_file_without_start_time_for_an_exited_process_is_removed(helpers, spawned, monkeypatch):
    gone = spawned("pass")
    gone.wait(timeout=10)
    helpers.PIDFILE.write_text(str(gone.pid))
    sent = _record_signals(monkeypatch, helpers)
    helpers.stop_server()
    assert sent == [] and not helpers.PIDFILE.exists()


# ------------------------------------------------- pidfd (Linux) pinning
@pytest.mark.skipif(not HAS_PIDFD, reason="Linux pidfd")
def test_pinned_process_that_exited_is_never_signalled(helpers, spawned, monkeypatch):
    """Exit between verification and signalling: the pid may already be
    someone else's, so nothing is sent."""
    p = _ready(spawned(SLEEPER))
    fd = os.pidfd_open(p.pid)
    try:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait(timeout=10)  # exited and reaped: the pid number is free for reuse
        sent = _record_signals(monkeypatch, helpers)
        assert helpers._signal_group(p.pid, signal.SIGTERM, fd) is False
        assert helpers._wait_exit(p.pid, 5, pidfd=fd) is True
        assert sent == []
    finally:
        os.close(fd)


@pytest.mark.skipif(not HAS_PIDFD, reason="Linux pidfd")
def test_identity_read_after_pinned_exit_is_not_trusted(helpers, spawned, monkeypatch):
    """The identity is read by pid number; if the pinned process exited first,
    that read may describe a successor with the same pid, so it is 'gone'."""
    p = _ready(spawned(SLEEPER, "-m", "echorelay", "--port", str(helpers.PORT)))
    rec = {"pid": p.pid, "start": helpers._identity(p.pid)["start"]}
    fd = os.pidfd_open(p.pid)
    try:
        assert helpers._ownership(rec, fd) == "owned"
        ident = helpers._identity(p.pid)
        os.killpg(p.pid, signal.SIGKILL)
        p.wait(timeout=10)
        # pretend a successor with an identical identity took the pid
        monkeypatch.setattr(helpers, "_alive", lambda pid: True)
        monkeypatch.setattr(helpers, "_identity", lambda pid: ident)
        assert helpers._ownership(rec) == "owned"  # without the pin it would be fooled
        assert helpers._ownership(rec, fd) == "gone"
    finally:
        os.close(fd)


@pytest.mark.skipif(not HAS_PIDFD, reason="Linux pidfd")
def test_pidfd_wait_is_bounded_and_wakes_on_exit(helpers, spawned):
    p = _ready(spawned(SLEEPER))
    fd = os.pidfd_open(p.pid)
    try:
        t0 = time.monotonic()
        assert helpers._wait_exit(p.pid, 0.2, pidfd=fd) is False
        assert 0.15 < time.monotonic() - t0 < 2
        os.killpg(p.pid, signal.SIGTERM)
        t0 = time.monotonic()
        assert helpers._wait_exit(p.pid, 10, pidfd=fd) is True
        assert time.monotonic() - t0 < 2
    finally:
        os.close(fd)


def test_pidfd_unavailable_falls_back(helpers, monkeypatch):
    """Old kernels and seccomp sandboxes refuse pidfd_open: fall back to the
    unpinned path instead of failing."""
    if HAS_PIDFD:
        def refuse(pid):
            raise PermissionError(1, "Operation not permitted")
        monkeypatch.setattr(helpers.os, "pidfd_open", refuse)
    assert helpers._pidfd(os.getpid()) is None


# -------------------------------------------------------------- liveness
def _zombie(spawned):
    p = spawned("pass")
    return p  # exits at once; stays a zombie until p.wait()


@pytest.mark.skipif(not HAS_PS, reason="needs ps")
def test_liveness_without_proc_uses_ps(helpers, spawned, monkeypatch):
    """macOS has no /proc: live, zombie and reaped processes are told apart
    with ps."""
    monkeypatch.setattr(helpers, "_HAVE_PROC", False)
    live = _ready(spawned(SLEEPER))
    assert helpers._alive(live.pid)
    z = _zombie(spawned)
    _until(lambda: helpers._proc_state(z.pid) == "Z", what="zombie state")
    assert not helpers._alive(z.pid)  # kill(pid, 0) alone would call it alive
    z.wait(timeout=5)
    assert helpers._proc_state(z.pid) is None and not helpers._alive(z.pid)
    ident = helpers._identity(live.pid)
    assert ident["start"].startswith("ps:") and "time.sleep" in ident["command"]


def test_unknown_state_is_treated_as_alive(helpers, spawned, monkeypatch):
    """With neither /proc nor ps, a process that exists is assumed alive, so
    stop_server waits for it instead of declaring it dead."""
    monkeypatch.setattr(helpers, "_HAVE_PROC", False)
    monkeypatch.setattr(helpers, "_ps", lambda pid, field: None)
    live = _ready(spawned(SLEEPER))
    assert helpers._proc_state(live.pid) == "?" and helpers._alive(live.pid)
    live.kill()
    live.wait(timeout=5)
    assert helpers._proc_state(live.pid) is None


@pytest.mark.skipif(not HAS_PROC, reason="Linux /proc")
def test_linux_zombie_detection(helpers, spawned):
    z = _zombie(spawned)
    _until(lambda: helpers._proc_state(z.pid) == "Z", what="zombie state")
    assert not helpers._alive(z.pid)
    assert helpers._identity(z.pid)["start"].startswith("proc:")


# ------------------------------------------------------------------ port
def test_port_held_by_a_non_http_listener_is_not_reported_free(helpers, tmp_path, monkeypatch):
    """/health failing does not mean the port is free."""
    monkeypatch.setattr(helpers, "PORT_RELEASE_S", 0.2)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", helpers.PORT))
        listener.listen()
        with pytest.raises(RuntimeError, match="still in use by a process that is not"):
            helpers.stop_server()
        with pytest.raises(RuntimeError, match="still in use"):
            helpers.start_server(tmp_path / "busy.log", backend="fake")
        assert not helpers.PIDFILE.exists() and helpers._PROC is None
    helpers.stop_server()


# --------------------------------------------------- commit, audio checks
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
