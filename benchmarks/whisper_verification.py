#!/usr/bin/env python3
"""Run the real-model test suite (``pytest -m whisper``) and record what ran.

    python -m benchmarks.prepare_audio --out eval_audio5 --num 5
    python -m benchmarks.whisper_verification --audio-dir eval_audio5 --device cuda

Writes to ``--out`` (default ``results/whisper_verification``):

* ``record.json``  UTC time, git commit, GPU/driver (``nvidia-smi``), Python and
                   package versions, model, device, compute type, the exact
                   command, per-test outcomes, the runtime identity the tests
                   observed, and every reason the run is not complete.
* ``pytest.log``   the complete pytest output.
* ``junit.xml``    per-test results, including recorded properties.

``complete`` is true only if all of these hold (``problems`` lists each one
that does not):

* exactly the expected real-model tests ran (``EXPECTED_TESTS``), all passed,
  and none failed, errored or was skipped;
* the identity the loaded runtime reported during the run matches the
  request: faster-whisper's ``WhisperModel`` over CTranslate2's ``Whisper``,
  the requested device and compute type, the ``base.en`` weights snapshot,
  and (for CUDA) a visible CUDA device and the expected GPU in ``nvidia-smi``.
  The requested settings in the record are not evidence of any of this;
* no ``PYTEST_ADDOPTS`` or ``PYTHONPATH`` override (they could inject plugins
  or substitute modules), and the checkout had no local edits.

Exit status: pytest's if non-zero, otherwise 0 if complete, else 3.
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
MODEL = "base.en"  # what tests/test_whisper.py loads
EXPECTED_TESTS = frozenset({
    "test_adapter_loads_with_requested_device_and_compute_type",
    "test_adapter_decodes_silence_and_empty_audio",
    "test_adapter_transcribes_real_speech",
    "test_adapter_final_decode_is_deterministic",
    "test_adapter_wraps_decode_errors",
    "test_server_loads_real_model_from_settings",
    "test_websocket_streams_real_speech",
    "test_server_shutdown_releases_inference_workers",
})
IDENTITY_TEST = "test_adapter_loads_with_requested_device_and_compute_type"
WHISPER_CLASS = "faster_whisper.transcribe.WhisperModel"
CT2_CLASS = "ctranslate2._ext.Whisper"
OVERRIDE_VARS = ("PYTEST_ADDOPTS", "PYTHONPATH")


def _nvidia_smi() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    q = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30, check=False)
    return q.stdout if q.returncode == 0 else f"nvidia-smi exited {q.returncode}: {q.stderr}"


def _junit_results(path: Path) -> dict:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    tests = []
    for case in (c for s in suites for c in s.iter("testcase")):
        outcome, message = "passed", None
        for tag, name in (("failure", "failed"), ("error", "error"), ("skipped", "skipped")):
            el = case.find(tag)
            if el is not None:
                outcome, message = name, (el.get("message") or "")[:500]
                break
        entry = {"name": case.get("name"), "classname": case.get("classname"),
                 "outcome": outcome, "time_s": float(case.get("time", 0))}
        if message is not None:
            entry["message"] = message
        props = {p.get("name"): p.get("value") for p in case.iter("property")}
        if props:
            entry["properties"] = props
        tests.append(entry)
    counts = {k: sum(t["outcome"] == k for t in tests) for k in ("passed", "failed", "error", "skipped")}
    return {"collected": len(tests), **counts, "tests": tests}


def validate(*, results: dict, device: str, compute_type: str, expect_gpu: str | None,
             gpus: list[str], overrides: dict, git: dict | None, exit_code: int) -> list[str]:
    """Every reason the run does not verify the requested configuration."""
    problems: list[str] = []
    if exit_code != 0:
        problems.append(f"pytest exited {exit_code}")
    if results.get("junit_missing"):
        return [*problems, "no JUnit results (pytest did not run the suite)"]

    tests = results["tests"]
    names = [t["name"] for t in tests]
    if len(names) != len(set(names)):
        problems.append(f"duplicate test names: {sorted(names)}")
    if set(names) != EXPECTED_TESTS:
        missing, extra = EXPECTED_TESTS - set(names), set(names) - EXPECTED_TESTS
        problems.append(f"ran the wrong test set (missing {sorted(missing)}, unexpected {sorted(extra)})")
    for t in tests:
        if t["outcome"] != "passed":
            problems.append(f"{t['name']}: {t['outcome']}" + (f" ({t['message']})" if t.get("message") else ""))
    if results["passed"] != len(EXPECTED_TESTS):
        problems.append(f"{results['passed']} passed, expected exactly {len(EXPECTED_TESTS)}")

    seen = next((t.get("properties", {}) for t in tests if t["name"] == IDENTITY_TEST), {})

    def check(key: str, ok, want: str) -> None:
        value = seen.get(key)
        if value is None:
            problems.append(f"runtime did not report {key}")
        elif not ok(value):
            problems.append(f"{key} = {value!r}, expected {want}")

    check("observed_whisper_class", lambda v: v == WHISPER_CLASS, WHISPER_CLASS)
    check("observed_ct2_class", lambda v: v == CT2_CLASS, CT2_CLASS)
    check("observed_ct2_device", lambda v: v == device, repr(device))
    if device == "cuda":
        check("observed_ct2_compute_type", lambda v: v == compute_type, repr(compute_type))
        check("observed_cuda_devices", lambda v: v.isdigit() and int(v) >= 1, ">= 1")
    else:
        check("observed_ct2_compute_type", lambda v: v.startswith(compute_type), f"{compute_type}*")
    snapshot = f"models--Systran--faster-whisper-{MODEL}"
    check("observed_model_path", lambda v: snapshot in Path(v).parts, f"a path inside {snapshot}/")
    check("observed_ct2_is_multilingual", lambda v: v == str(not MODEL.endswith(".en")),
          str(not MODEL.endswith(".en")))
    check("observed_model_bin_sha256", lambda v: len(v) == 64, "a SHA-256 of model.bin")

    if device == "cuda" and expect_gpu and not any(expect_gpu in g for g in gpus):
        problems.append(f"nvidia-smi lists {gpus or 'no GPU'}, expected one containing {expect_gpu!r}")
    for k, v in overrides.items():
        if v:
            problems.append(f"{k} is set ({v!r}); it can inject plugins or substitute modules")
    if not git or git.get("dirty") is not False:
        problems.append(f"checkout is not a clean commit: {git}")
    return problems


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio-dir", required=True, help="output of prepare_audio --num 5")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--compute-type", default=None, help="default: float16 on cuda, int8 on cpu")
    p.add_argument("--expect-gpu", default="T4",
                   help="substring nvidia-smi must report for --device cuda ('' to skip the check)")
    p.add_argument("--out", default="results/whisper_verification")
    a = p.parse_args()

    compute_type = a.compute_type or ("float16" if a.device == "cuda" else "int8")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    junit = out / "junit.xml"
    junit.unlink(missing_ok=True)  # never read a previous run's results
    # junit_family=xunit1: the format that carries record_property values.
    cmd = [sys.executable, "-m", "pytest", "-m", "whisper", "-rA", "-s", "-p", "no:cacheprovider",
           "-o", "junit_family=xunit1", f"--junitxml={junit}", "tests/test_whisper.py"]
    env = {**os.environ, "ECHORELAY_TEST_DEVICE": a.device,
           "ECHORELAY_TEST_COMPUTE_TYPE": compute_type,
           "ECHORELAY_TEST_AUDIO_DIR": str(Path(a.audio_dir).resolve())}
    overrides = {k: os.environ.get(k) for k in OVERRIDE_VARS}

    started = datetime.now(timezone.utc)
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, check=False)
    finished = datetime.now(timezone.utc)
    (out / "pytest.log").write_text(proc.stdout + proc.stderr)
    print(proc.stdout + proc.stderr)

    results = _junit_results(junit) if junit.exists() else {"collected": 0, "junit_missing": True}
    env_info = environment()
    problems = validate(results=results, device=a.device, compute_type=compute_type,
                        expect_gpu=a.expect_gpu or None, gpus=env_info.get("gpus", []),
                        overrides=overrides, git=env_info.get("git"), exit_code=proc.returncode)
    complete = not problems
    write_json(out / "record.json", {
        "kind": "real_model_verification",
        "started_utc": started.isoformat(), "finished_utc": finished.isoformat(),
        "command": " ".join(["python", *cmd[1:]]),
        "requested": {"model": MODEL, "device": a.device, "compute_type": compute_type,
                      "expect_gpu": a.expect_gpu or None,
                      "note": "requested settings; the observed identity below is the evidence"},
        "env": {k: env[k] for k in ("ECHORELAY_TEST_DEVICE", "ECHORELAY_TEST_COMPUTE_TYPE",
                                    "ECHORELAY_TEST_AUDIO_DIR")},
        "pytest_overrides": {**overrides, "ECHORELAY_MODEL_DIR": os.environ.get("ECHORELAY_MODEL_DIR")},
        "observed": next((t.get("properties", {}) for t in results.get("tests", [])
                          if t["name"] == IDENTITY_TEST), {}),
        "environment": env_info,
        "nvidia_smi": _nvidia_smi(),
        "pytest_exit_code": proc.returncode,
        "results": results,
        "expected_tests": sorted(EXPECTED_TESTS),
        "complete": complete,
        "problems": problems,
    })
    if complete:
        print(f"COMPLETE: {results['passed']} passed, 0 failed, 0 errors, 0 skipped; "
              f"observed {a.device}/{compute_type}; wrote {out}/")
    else:
        print(f"NOT COMPLETE ({len(problems)} problem(s)); wrote {out}/")
        for prob in problems:
            print(f"  - {prob}")
    sys.exit(proc.returncode if proc.returncode else (0 if complete else 3))


if __name__ == "__main__":
    main()
