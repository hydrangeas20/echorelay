"""Notebook helpers for running Phase 2 experiments on Colab (or any VM).

    from benchmarks.colab_helpers import *

* ``verify_commit(sha)``: the checked-out code is exactly ``sha`` with no
  local edits to tracked files.
* ``verify_eval_audio(dir, manifest)``: the evaluation audio is the same
  PCM as a previous run (hashes PCM samples, the way result manifests do;
  ``sha256sum`` on the .wav files would include the header and never match).
* ``start_server(log, **settings)`` / ``stop_server()``: run EchoRelay in
  its own process group with ``ECHORELAY_<KEY>`` settings, and stop it
  reliably, including one started by an earlier notebook kernel. Works on
  Linux and macOS. Avoids ``pkill -f``, which can match the shell running it.
* ``verify_server(expected)``: assert model and config via ``/metrics``.
* ``save_provenance(out_dir, name)``: commit, packages, GPU, server metrics.
"""

from __future__ import annotations

import json
import os
import select
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import IO

REPO = Path(__file__).resolve().parents[1]
PORT = int(os.environ.get("ECHORELAY_BENCH_PORT", "8000"))
PIDFILE = REPO.parent / f"echorelay_server_{PORT}.pid"

# Bounded waits (seconds). Module constants rather than parameters because
# start_server(**settings) forwards every keyword to the server.
START_TIMEOUT_S = 300.0  # the first start downloads the model weights
STOP_GRACE_S = 20.0  # SIGTERM -> exit (uvicorn waits for open streams)
STOP_KILL_S = 10.0  # SIGKILL -> exit
PORT_RELEASE_S = 5.0  # process gone -> nothing listening on PORT

_HAVE_PROC = os.path.isdir("/proc/self")  # Linux; absent on macOS

_PROC: subprocess.Popen | None = None  # server started by this kernel
_LOG: IO[str] | None = None  # its log file, held open by this kernel

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


# ------------------------------------------------------------ processes
# A server started by an earlier notebook kernel is known only by the pid
# file, and its pid may since have been reused by an unrelated process. The
# pid file therefore records the process start time, and the server is
# signalled only if its start time and command line still match.


def _ps(pid: int, field: str) -> str | None:
    """One ``ps`` field for pid (Linux procps and macOS), or None."""
    try:
        r = subprocess.run(["ps", "-ww", "-o", f"{field}=", "-p", str(pid)], check=False,
                           capture_output=True, text=True, timeout=5,
                           env={**os.environ, "LC_ALL": "C"})  # returncode checked below
    except FileNotFoundError:  # no ps installed
        return None
    out = r.stdout.strip()
    return out if r.returncode == 0 and out else None


def _proc_state(pid: int) -> str | None:
    """Scheduler state of pid ('R', 'S', 'Z', ...), '?' if the process exists
    but its state cannot be read, or None if there is no such process.

    ``kill(pid, 0)`` alone is not enough: it succeeds on a zombie, which has
    exited but not been reaped. Linux reads the state from /proc; macOS has
    no /proc, so ``ps`` reads it there.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass  # exists, owned by another user
    if _HAVE_PROC:
        try:
            with open(f"/proc/{pid}/stat") as f:
                return f.read().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            return None  # exited between the two checks
    state = _ps(pid, "stat")
    if state:
        return state[0]
    try:
        os.kill(pid, 0)  # ps found nothing: gone, or ps unavailable
    except ProcessLookupError:
        return None
    except PermissionError:
        pass
    return "?"  # exists, state unknown: callers treat it as alive


def _alive(pid: int) -> bool:
    """True if pid exists and has not exited (a zombie counts as exited)."""
    return _proc_state(pid) not in (None, "Z", "X")


def _identity(pid: int) -> dict | None:
    """Start time and command line of pid, or None if they cannot be read."""
    if _HAVE_PROC:
        try:
            with open(f"/proc/{pid}/stat") as f:
                start = f.read().rsplit(")", 1)[1].split()[19]  # field 22: starttime
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                command = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
        except FileNotFoundError:
            return None
        return {"start": f"proc:{start}", "command": command}
    start, command = _ps(pid, "lstart"), _ps(pid, "command")
    if start is None or command is None:
        return None
    return {"start": f"ps:{start}", "command": command}


def _server_marker() -> tuple[str, str]:
    return ("-m echorelay", f"--port {PORT}")


def _pidfd(pid: int) -> int | None:
    """A Linux pidfd pinning this exact process, or None where unsupported
    (macOS, old kernels, sandboxes). Raises ProcessLookupError if pid is gone.

    A pidfd keeps referring to the process it was opened for even if that
    process exits and its pid is reused, which closes the window between
    verifying a process and signalling or waiting for it.
    """
    if not hasattr(os, "pidfd_open"):
        return None
    try:
        return os.pidfd_open(pid)
    except ProcessLookupError:
        raise
    except OSError:  # ENOSYS, EPERM under seccomp, ...
        return None


def _pinned_exited(pidfd: int, timeout: float = 0.0) -> bool:
    """True once the pinned process has exited (its pidfd becomes readable),
    waiting at most timeout seconds. poll, unlike select, takes any fd number."""
    poller = select.poll()
    poller.register(pidfd, select.POLLIN)
    return bool(poller.poll(max(timeout, 0.0) * 1000))


def _ownership(rec: dict, pidfd: int | None = None) -> str:
    """'owned', 'foreign' (pid reused or not our server), 'gone', 'unknown'
    (identity unreadable), or 'unconfirmed' (no recorded start time).

    A process is 'owned' only if its start time matches the one recorded at
    start-up AND its command line is the server's. A matching command line
    alone is never enough: another EchoRelay server, or anything started with
    the same arguments, would pass. With a pidfd, the identity read is
    confirmed to belong to the pinned process.
    """
    pid = rec["pid"]
    if not _alive(pid):
        return "gone"  # includes a zombie, whose command line is no longer readable
    ident = _identity(pid)
    if ident is None:
        return "gone" if _proc_state(pid) is None else "unknown"
    if not all(m in ident["command"] for m in _server_marker()):
        return "foreign"
    if rec.get("start") is None:
        return "unconfirmed"  # legacy pid file, or start time unreadable at start-up
    if rec["start"] != ident["start"]:
        return "foreign"  # same pid, different process: the pid was reused
    if pidfd is not None and _pinned_exited(pidfd):
        return "gone"  # pinned process gone: the identity read may be a successor's
    return "owned"


def _signal_group(pid: int, sig: int, pidfd: int | None = None) -> bool:
    """Send sig to pid's process group (the server leads its own session).
    Returns False if the process is already gone. Only called for this
    kernel's child or a process whose ownership was verified, both started
    with start_new_session=True, so their group id equals their pid. With a
    pidfd, nothing is sent once the pinned process has exited."""
    if pidfd is not None and _pinned_exited(pidfd):
        return False
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return False
    except PermissionError:  # some platforms restrict cross-session queries
        pgid = pid
    try:
        if pgid == pid:
            os.killpg(pgid, sig)
        else:  # not a group leader: never signal someone else's group
            os.kill(pid, sig)
    except ProcessLookupError:
        return False
    except PermissionError as exc:
        raise RuntimeError(f"not permitted to signal pid {pid}: {exc}") from exc
    return True


def _wait_exit(pid: int, timeout: float, proc: subprocess.Popen | None = None,
               pidfd: int | None = None) -> bool:
    """Wait at most timeout seconds for pid to exit. A child of this kernel is
    also reaped; a pinned process is waited for on its pidfd, which a reused
    pid cannot fool; anything else is polled with a growing interval."""
    if proc is not None:
        try:
            proc.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False
    if pidfd is not None:
        return _pinned_exited(pidfd, timeout)
    deadline = time.monotonic() + timeout
    delay = 0.02
    while _alive(pid):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(delay, remaining))
        delay = min(delay * 2, 0.25)
    return True


def _stop_pid(pid: int, proc: subprocess.Popen | None = None, still_ours=None,
              pidfd: int | None = None) -> None:
    """SIGTERM pid's group; SIGKILL it if it outlives STOP_GRACE_S."""
    if proc is not None:
        if proc.poll() is not None:
            return  # exited and reaped: its pid may already belong to someone else
    elif not _alive(pid):
        return
    if not _signal_group(pid, signal.SIGTERM, pidfd):
        return
    if _wait_exit(pid, STOP_GRACE_S, proc, pidfd):
        return
    if still_ours is not None and not still_ours():
        return  # it exited and the pid was reused while we waited
    print(f"pid {pid} still running {STOP_GRACE_S:g} s after SIGTERM; sending SIGKILL")
    _signal_group(pid, signal.SIGKILL, pidfd)
    if not _wait_exit(pid, STOP_KILL_S, proc, pidfd):
        raise RuntimeError(f"pid {pid} did not exit {STOP_KILL_S:g} s after SIGKILL")


# --------------------------------------------------------------- pid file
def _write_pidfile(proc: subprocess.Popen) -> None:
    ident = _identity(proc.pid)  # start time is fixed at fork, so it is valid before exec
    rec = {"pid": proc.pid, "port": PORT, "start": ident["start"] if ident else None}
    tmp = PIDFILE.with_name(PIDFILE.name + ".tmp")
    tmp.write_text(json.dumps(rec) + "\n")
    os.replace(tmp, PIDFILE)


def _read_pidfile() -> dict | None:
    """The pid file's record ({"pid", "port", "start"}), or None if absent or
    unreadable. A legacy file holds only the pid."""
    try:
        text = PIDFILE.read_text()
    except FileNotFoundError:
        return None
    try:
        rec = json.loads(text)
    except json.JSONDecodeError:
        rec = None
    if isinstance(rec, int) and not isinstance(rec, bool):
        rec = {"pid": rec}
    if not (isinstance(rec, dict) and isinstance(rec.get("pid"), int) and rec["pid"] > 0):
        print(f"ignoring unreadable pid file {PIDFILE}")
        return None
    return rec


# ------------------------------------------------------------------ port
def _port_listening() -> bool:
    """True if something accepts TCP connections on 127.0.0.1:PORT."""
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def _wait_port_free(timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while _port_listening():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def _close_log() -> None:
    global _LOG
    if _LOG is not None:
        _LOG.close()
        _LOG = None


def _stop_recorded(rec: dict) -> None:
    """Stop the server named by the pid file, only if it is provably that
    server. Raises, keeping the pid file, when ownership cannot be confirmed."""
    pid = rec["pid"]
    try:
        pidfd = _pidfd(pid)  # pin first, so the identity checked is the one signalled
    except ProcessLookupError:
        print(f"stale pid file: pid {pid} has already exited")
        return
    try:
        status = _ownership(rec, pidfd)
        if status == "owned":
            _stop_pid(pid, still_ours=lambda: _ownership(rec, pidfd) == "owned", pidfd=pidfd)
        elif status == "foreign":
            print(f"stale pid file: pid {pid} is not the EchoRelay server on :{PORT}; "
                  "not signalling it")
        elif status == "gone":
            print(f"stale pid file: pid {pid} has already exited")
        elif status == "unconfirmed":
            raise RuntimeError(
                f"{PIDFILE} records no start time (written by an older version, or the start "
                f"time was unreadable), so pid {pid} cannot be confirmed as the server even "
                f"though its command line matches; not signalling it. If it is the server, "
                f"stop it with `kill -TERM -- -{pid}`, then delete the pid file")
        else:  # "unknown"
            raise RuntimeError(
                f"cannot verify that pid {pid} from {PIDFILE} is the EchoRelay server "
                f"(no /proc and no ps); stop it yourself, then delete the pid file")
    finally:
        if pidfd is not None:
            os.close(pidfd)


# ---------------------------------------------------------------- public
def stop_server() -> None:
    """Stop the server started by start_server (its whole process group).

    Also stops a server started by an earlier kernel, found through the pid
    file, but only after verifying it is that server. Safe to call repeatedly.
    Raises if a process cannot be stopped, if ownership cannot be verified, or
    if anything still listens on PORT afterwards.
    """
    global _PROC
    try:
        own = _PROC
        if own is not None:
            _stop_pid(own.pid, proc=own)
            _PROC = None
        rec = _read_pidfile()
        if rec is not None and (own is None or rec["pid"] != own.pid):
            _stop_recorded(rec)
        PIDFILE.unlink(missing_ok=True)
    finally:
        _close_log()
    if not _wait_port_free(PORT_RELEASE_S):
        try:
            _get("/health", timeout=1)
            what = "an HTTP server answering /health (not one these helpers can stop)"
        except (urllib.error.URLError, OSError):
            what = "a process that is not an EchoRelay server"
        raise RuntimeError(f"port {PORT} is still in use by {what}; stop it or restart the runtime")
    print(f"server stopped; port {PORT} free")


def start_server(log_path: str | Path, **settings) -> subprocess.Popen:
    """Start EchoRelay with ECHORELAY_<KEY>=value settings and wait until healthy.

    Stops any previous server first. If the new server exits or never becomes
    healthy, it is stopped and the pid file removed before the error is raised.
    """
    global _PROC, _LOG
    stop_server()
    env = dict(os.environ)
    env.update({f"ECHORELAY_{k.upper()}": str(v) for k, v in settings.items()})
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "echorelay", "--host", "127.0.0.1", "--port", str(PORT)],
            cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    except BaseException:
        log.close()
        raise
    _PROC, _LOG = proc, log
    _write_pidfile(proc)
    deadline = time.monotonic() + START_TIMEOUT_S
    while True:
        if proc.poll() is not None:
            failure = f"server exited with code {proc.returncode}; see {log_path}"
            break
        try:
            _get("/health", timeout=1)
            print(f"server up (pid {proc.pid}, port {PORT}); log: {log_path}")
            return proc
        except (urllib.error.URLError, OSError):
            pass
        if time.monotonic() >= deadline:
            failure = f"server not healthy after {START_TIMEOUT_S:g} s; see {log_path}"
            break
        time.sleep(0.25)
    stop_server()  # clean up the failed server, its pid file and log handle
    raise RuntimeError(failure)


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
