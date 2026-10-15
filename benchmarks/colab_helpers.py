"""Notebook helpers for running Phase 2 experiments on Colab (or any VM).

    from benchmarks.colab_helpers import *

* ``verify_commit(sha)``: the checked-out code is exactly ``sha`` with no
  local edits to tracked files.
* ``verify_eval_audio(dir, manifest)``: the evaluation audio is the same
  PCM as a previous run (hashes PCM samples, the way result manifests do;
  ``sha256sum`` on the .wav files would include the header and never match).
* ``start_server(log, **settings)`` / ``stop_server()``: run EchoRelay in
  its own process group with ``ECHORELAY_<KEY>`` settings, and stop it
  reliably. Avoids ``pkill -f``, which can match the shell running it.
* ``verify_server(expected)``: assert model and config via ``/metrics``.
* ``save_provenance(out_dir, name)``: commit, packages, GPU, server metrics.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PORT = int(os.environ.get("ECHORELAY_BENCH_PORT", "8000"))
PIDFILE = REPO.parent / f"echorelay_server_{PORT}.pid"

_PROC: subprocess.Popen | None = None  # server started by this kernel

__all__ = ["REPO", "save_provenance", "start_server", "stop_server", "verify_commit",
           "verify_eval_audio", "verify_server"]


def _get(path: str, timeout: float = 3.0) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as r:
        return json.loads(r.read())


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def verify_commit(expected_sha: str) -> None:
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    print(f"HEAD {head}  local edits to tracked files: {'YES' if dirty else 'none'}")
    if head != expected_sha:
        raise AssertionError(f"expected commit {expected_sha}, checked out {head}")
    if dirty:
        raise AssertionError(f"tracked files modified:\n{dirty}")
    print("commit verified")


def verify_eval_audio(audio_dir: str | Path, manifest_result: str | Path) -> None:
    """Compare PCM hashes of audio_dir/*.wav with a result file's audio manifest."""
    sys.path.insert(0, str(REPO))
    from benchmarks.common import load_audio_dir

    want = {f["name"]: f["sha256"] for f in json.loads(Path(manifest_result).read_text())["audio"]["files"]}
    have = {u.name: u.sha256 for u in load_audio_dir(audio_dir)}
    for name in sorted(want.keys() | have.keys()):
        status = "OK" if want.get(name) == have.get(name) else (
            "MISSING" if name not in have else "EXTRA" if name not in want else "DIFFERENT")
        print(f"  {status:9} {name}")
    if have != want:
        raise AssertionError("evaluation audio differs from the reference manifest")
    print(f"evaluation audio verified: {len(have)} files identical to {Path(manifest_result).name}")


def _alive(pid: int) -> bool:
    """True if pid exists and is not a zombie (kill(pid, 0) succeeds on zombies)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, IndexError):
        return False


def stop_server() -> None:
    """Stop the server started by start_server (its whole process group)."""
    global _PROC
    if PIDFILE.exists():
        pid = int(PIDFILE.read_text())
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if _PROC is not None and _PROC.pid == pid:
            try:
                _PROC.wait(timeout=20)  # also reaps it
            except subprocess.TimeoutExpired:
                os.killpg(pid, signal.SIGKILL)
                _PROC.wait(timeout=10)
        else:  # started by an earlier kernel: cannot reap, poll instead
            deadline = time.monotonic() + 20
            while _alive(pid) and time.monotonic() < deadline:
                time.sleep(0.2)
            if _alive(pid):
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        _PROC = None
        PIDFILE.unlink()
    try:
        _get("/health", timeout=1)
    except (urllib.error.URLError, OSError):
        print(f"server stopped; port {PORT} free")
        return
    raise RuntimeError(f"something is still serving on :{PORT}; restart the runtime")


def start_server(log_path: str | Path, **settings) -> subprocess.Popen:
    """Start EchoRelay with ECHORELAY_<KEY>=value settings and wait until healthy."""
    global _PROC
    stop_server()
    env = dict(os.environ)
    env.update({f"ECHORELAY_{k.upper()}": str(v) for k, v in settings.items()})
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "echorelay", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    PIDFILE.write_text(str(proc.pid))
    _PROC = proc
    for _ in range(300):  # the first start downloads the model weights
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}; see {log_path}")
        try:
            _get("/health", timeout=1)
            print(f"server up (pid {proc.pid}, port {PORT}); log: {log_path}")
            return proc
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    raise RuntimeError(f"server not healthy after 300 s; see {log_path}")


def verify_server(expected: dict) -> dict:
    """Assert the running server's model and configuration; print all of it."""
    m = _get("/metrics")
    actual = {
        "backend": m["model"]["backend"], "model": m["model"]["name"],
        "device": m["model"]["device"], "compute_type": m["model"]["compute_type"],
        **m["config"], "active_streams": m["streams"]["active"], "version": m["version"],
    }
    bad = {k: (actual.get(k), v) for k, v in expected.items() if actual.get(k) != v}
    for k in sorted(actual):
        print(f"  {k:28} {actual[k]}{'  <-- MISMATCH' if k in bad else ''}")
    if bad:
        raise AssertionError(f"server configuration mismatch (actual, expected): {bad}")
    print("server configuration verified")
    return m


def save_provenance(out_dir: str | Path, experiment: str) -> None:
    """Record what produced the files in out_dir: code, packages, GPU, server."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def run(cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            return f"unavailable: {exc}"

    info = {
        "experiment": experiment,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(run(["git", "status", "--porcelain", "--untracked-files=no"])),
        "python": sys.version,
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm",
                           "--format=csv,noheader"]),
        "packages": run([sys.executable, "-m", "pip", "freeze"]).splitlines(),
    }
    try:
        info["server_metrics"] = _get("/metrics")
    except (urllib.error.URLError, OSError):
        info["server_metrics"] = None  # no server running (offline experiments)
    (out / "provenance.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"wrote {out / 'provenance.json'}")
