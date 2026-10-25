#!/usr/bin/env python3
"""Run the real-model test suite (``pytest -m whisper``) and record what ran.

    python -m benchmarks.prepare_audio --out eval_audio5 --num 5
    python -m benchmarks.whisper_verification --audio-dir eval_audio5 --device cuda

Writes to ``--out`` (default ``results/whisper_verification``):

* ``record.json``  UTC time, git commit, GPU/driver (``nvidia-smi``), Python and
                   package versions, model, device, compute type, the exact
                   command, and pass/fail/skip/error counts from JUnit XML.
* ``pytest.log``   the complete pytest output.
* ``junit.xml``    per-test results, including recorded WER properties.

``complete`` in the record is true only if every collected real-model test
passed with nothing skipped. Exit status is pytest's (non-zero on any failure),
or 3 if tests passed but some were skipped.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from .common import environment, write_json

REPO = Path(__file__).resolve().parents[1]


def _nvidia_smi() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    q = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30, check=False)
    return q.stdout if q.returncode == 0 else f"nvidia-smi exited {q.returncode}: {q.stderr}"


def _junit_counts(path: Path) -> dict:
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    tests = []
    for case in suite.iter("testcase"):
        outcome = "passed"
        for tag in ("failure", "error", "skipped"):
            el = case.find(tag)
            if el is not None:
                outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[tag]
                tests.append({"name": case.get("name"), "outcome": outcome,
                              "message": (el.get("message") or "")[:500],
                              "time_s": float(case.get("time", 0))})
                break
        else:
            props = {p.get("name"): p.get("value") for p in case.iter("property")}
            tests.append({"name": case.get("name"), "outcome": outcome,
                          "time_s": float(case.get("time", 0)), **({"properties": props} if props else {})})
    counts = {k: sum(t["outcome"] == k for t in tests) for k in ("passed", "failed", "error", "skipped")}
    return {"collected": len(tests), **counts, "tests": tests}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio-dir", required=True, help="output of prepare_audio --num 5")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--compute-type", default=None, help="default: float16 on cuda, int8 on cpu")
    p.add_argument("--out", default="results/whisper_verification")
    a = p.parse_args()

    compute_type = a.compute_type or ("float16" if a.device == "cuda" else "int8")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    junit = out / "junit.xml"
    cmd = [sys.executable, "-m", "pytest", "-m", "whisper", "-rA", "-s", "-p", "no:cacheprovider",
           f"--junitxml={junit}", "tests/test_whisper.py"]
    env = {**os.environ, "ECHORELAY_TEST_DEVICE": a.device,
           "ECHORELAY_TEST_COMPUTE_TYPE": compute_type,
           "ECHORELAY_TEST_AUDIO_DIR": str(Path(a.audio_dir).resolve())}

    started = datetime.now(timezone.utc)
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, check=False)
    finished = datetime.now(timezone.utc)
    (out / "pytest.log").write_text(proc.stdout + proc.stderr)
    print(proc.stdout + proc.stderr)

    results = _junit_counts(junit) if junit.exists() else {"collected": 0, "junit_missing": True}
    complete = (proc.returncode == 0 and results.get("collected", 0) > 0
                and results.get("skipped", 1) == 0)
    write_json(out / "record.json", {
        "kind": "real_model_verification",
        "started_utc": started.isoformat(), "finished_utc": finished.isoformat(),
        "command": " ".join(["python", *cmd[1:]]),
        "env": {k: env[k] for k in ("ECHORELAY_TEST_DEVICE", "ECHORELAY_TEST_COMPUTE_TYPE",
                                    "ECHORELAY_TEST_AUDIO_DIR")},
        # Anything that could inject plugins or substitute modules is recorded,
        # so a stubbed run cannot pass for a real one.
        "pytest_overrides": {k: os.environ.get(k) for k in ("PYTEST_ADDOPTS", "PYTHONPATH",
                                                           "ECHORELAY_MODEL_DIR")},
        "model": "base.en", "device": a.device, "compute_type": compute_type,
        "environment": environment(),
        "nvidia_smi": _nvidia_smi(),
        "pytest_exit_code": proc.returncode,
        "results": results,
        "complete": complete,
    })
    status = "COMPLETE" if complete else "NOT COMPLETE"
    print(f"{status}: {results.get('passed', 0)} passed, {results.get('failed', 0)} failed, "
          f"{results.get('error', 0)} errors, {results.get('skipped', 0)} skipped; wrote {out}/")
    sys.exit(proc.returncode if proc.returncode else (0 if complete else 3))


if __name__ == "__main__":
    main()
