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
* no ``PYTEST_ADDOPTS``, ``PYTEST_PLUGINS`` or ``PYTHONPATH`` override (they
  could inject plugins or substitute modules), and the checkout had no local
  edits;
* the checkout audit, taken before pytest starts, finds no untracked file that
  could execute in the test process: no untracked, non-ignored file anywhere
  in the repository, and no ignored file that Python or pytest would load from
  the directories on the test process's import / conftest path (the repository
  root, ``tests/``, ``src/``): ``*.py``, ``*.pyc``, ``*.pth``, ``*.so``, or a
  pytest/Python configuration file. ``git status`` alone ignores untracked
  files, so a stray ``conftest.py`` would otherwise run unnoticed;
* every ``provenance`` section below was collected. Provenance *values* never
  change ``complete``; failing to collect a section does (fail closed).

The record also carries ``provenance``:

* ``environment_metadata``: allowlisted CUDA / dynamic-loader / pytest
  variables (``LD_LIBRARY_PATH`` and ``LD_PRELOAD`` split into entries,
  ``CUDA_*``, ``CT2_*``...) as passed to the test process, with credentials
  redacted where recognisable; where the CUDA 12 cuBLAS / cuDNN 9 libraries
  sit on that search path (directories that cannot be read are reported as
  errors, not skipped); installed ``nvidia-*`` wheels. This describes the
  environment. It is **not** proof of which library was loaded.
* ``pytest_plugins``: ``installed_entry_points`` (``pytest11`` entry points
  with distribution versions: what pytest can auto-load) and
  ``header_reported`` (the ``plugins:`` line pytest printed). The header lists
  entry-point plugins only: plugins loaded with ``-p``, ``PYTEST_PLUGINS``, a
  ``conftest.py`` or a ``pytest_plugins`` variable do not appear in it. Those
  routes are covered by the override and checkout rules above, not by this
  inventory, which is therefore not a list of every plugin that ran.
* ``loaded_libraries``: CUDA-related shared objects mapped into the pytest
  process after the real-speech decodes (``/proc/self/maps``, Linux only), as
  recorded by that test. This is what was actually loaded, in that process.

Nothing outside the allowlist is read from the environment. Redaction removes
URL credentials, ``KEY=value`` assignments with secret-like keys and common
token formats; it cannot recognise every secret, which is why only
allowlisted variables are read at all.

Exit status: pytest's if non-zero, otherwise 0 if complete, else 3.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from importlib import metadata
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
OVERRIDE_VARS = ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTHONPATH")
LOADED_LIBRARIES_TEST = "test_adapter_transcribes_real_speech"

# Environment provenance: only these variables are read, never the whole
# environment. Path lists are recorded entry by entry.
PROVENANCE_VARS = (
    "LD_LIBRARY_PATH", "LD_PRELOAD", "CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES",
    "CUDA_MODULE_LOADING", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
)
PROVENANCE_PREFIXES = ("CT2_",)  # CTranslate2 runtime switches
PATH_LIST_VARS = frozenset({"LD_LIBRARY_PATH", "LD_PRELOAD"})
# The CUDA libraries the CTranslate2 4.x wheels load at run time.
CUDA_RUNTIME_LIBRARIES = ("libcublas.so.12", "libcublasLt.so.12", "libcudnn.so.9")
# Shared objects that show which CUDA / CTranslate2 libraries were loaded.
CUDA_LIBRARY_PATTERN = re.compile(r"/(libcublas|libcudnn|libcudart|libcuda\.so|libctranslate2)[^/]*$")

_SECRET_WORDS = r"token|secret|passw|pwd|apikey|api_key|api-key|credential|auth|private_key|access_key"
_SECRET_LIKE = re.compile(rf"({_SECRET_WORDS})", re.IGNORECASE)
# userinfo after "//" (with or without a scheme), up to the LAST "@" before
# whitespace, so a password containing "@" or "/" is removed whole. May
# over-redact a path that itself contains "@"; never under-redacts that form.
_URL_USERINFO = re.compile(r"(?P<scheme>//)[^\s]*@")
# KEY=value with a secret-like key, anywhere in the value (query strings
# included). The value runs to "&", ";" or whitespace, so it is removed even
# if it contains ":" (over-redacting the rest of a path list is acceptable).
_SECRET_ASSIGNMENT = re.compile(rf"(?i)([\w.\-]*(?:{_SECRET_WORDS})[\w.\-]*)=[^&;\s]*")
# Recognisable token formats (GitHub, Hugging Face, AWS access keys, OpenAI-
# style, Slack, GitLab, JWTs). Not exhaustive: unknown formats are not caught.
_TOKEN_FORMATS = re.compile(
    r"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}"
    r"|(?:AKIA|ASIA)[0-9A-Z]{16}|sk-[A-Za-z0-9_\-]{20,}|xox[abprs]-[A-Za-z0-9\-]{10,}"
    r"|glpat-[A-Za-z0-9_\-]{20,}|eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,})")

# Checkout audit (F5): what may execute in the test process from outside git.
# pytest runs from the repository root with rootdir there; the root, tests/
# (rootdir-relative test import path) and src/ (editable install) are on the
# import path, and conftest.py files are loaded from the root down to tests/.
IMPORT_ROOTS = ("", "tests", "src")
EXECUTABLE_SUFFIXES = (".py", ".pyc", ".pyo", ".pth", ".so")
CONFIG_NAMES = frozenset({"conftest.py", "pytest.ini", ".pytest.ini", "tox.ini", "setup.cfg",
                          "pyproject.toml", "sitecustomize.py", "usercustomize.py"})


def _nvidia_smi() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    q = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30, check=False)
    return q.stdout if q.returncode == 0 else f"nvidia-smi exited {q.returncode}: {q.stderr}"


def redact(value: str) -> str:
    """Remove recognisable credentials: URL userinfo, secret-like KEY=value
    assignments and common token formats. Best effort, not a guarantee."""
    value = _URL_USERINFO.sub(lambda m: m["scheme"] + "[redacted]@", value)
    value = _SECRET_ASSIGNMENT.sub(lambda m: m.group(1) + "=[redacted]", value)
    return _TOKEN_FORMATS.sub("[redacted]", value)


def _split_redacted(raw: str, sep: str) -> list[str]:
    """Redact the whole value first, so splitting can never separate a
    credential from the context that marks it as one."""
    return [e[:4096] for e in re.split(sep, redact(raw)) if e]


def _probe(check, path: Path) -> bool | str:
    """A filesystem check that reports an unreadable path instead of raising
    (Path.exists / is_dir raise PermissionError under an unreadable parent)."""
    try:
        return check(path)
    except OSError as exc:
        return f"error: {type(exc).__name__}: {exc.strerror or exc}"


def environment_metadata(environ: dict[str, str]) -> dict:
    """Allowlisted CUDA / loader / pytest variables from the environment the
    tests ran with. Metadata only: it says where the loader *would* look, not
    what it loaded."""
    names = [n for n in PROVENANCE_VARS if n in environ]
    names += sorted(n for n in environ if n.startswith(PROVENANCE_PREFIXES) and n not in names)
    variables: dict[str, object] = {}
    for name in names:
        raw = environ[name]
        if _SECRET_LIKE.search(name):
            variables[name] = "[redacted]"
        elif name == "LD_PRELOAD":  # ld.so accepts spaces and colons, mixed
            variables[name] = _split_redacted(raw, r"[:\s]+")
        elif name in PATH_LIST_VARS:
            variables[name] = _split_redacted(raw, ":")
        else:
            variables[name] = redact(raw)[:4096]
    search_path = variables.get("LD_LIBRARY_PATH", [])
    found: dict[str, list[str]] = {lib: [] for lib in CUDA_RUNTIME_LIBRARIES}
    errors: list[dict] = []
    for d in search_path:
        for lib in CUDA_RUNTIME_LIBRARIES:
            hit = _probe(Path.exists, Path(d) / lib)
            if hit is True:
                found[lib].append(d)
            elif hit is not False:
                errors.append({"directory": d, "library": lib, "error": hit})
    return {
        "note": ("environment metadata, not proof of loading: see loaded_libraries "
                 "for what the test process actually mapped"),
        "variables": variables,
        "unset": [n for n in PROVENANCE_VARS if n not in environ],
        "ld_library_path_entries_exist": {d: _probe(Path.is_dir, Path(d)) for d in search_path},
        "cuda_libraries_on_search_path": found,
        # Unreadable directories: whether the library is there is unknown.
        "search_errors": errors,
        "nvidia_wheels": nvidia_wheels(),
    }


def nvidia_wheels() -> dict[str, str]:
    """Installed ``nvidia-*`` distributions (pip-provided CUDA libraries)."""
    out: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"] or ""
        if name.lower().startswith("nvidia-"):
            out[name] = dist.version
    return dict(sorted(out.items()))


def pytest_plugin_inventory() -> list[dict]:
    """Installed ``pytest11`` entry points: what pytest *can* auto-load in this
    interpreter (the one the tests run in). Not a list of every plugin that
    ran: ``-p``, ``PYTEST_PLUGINS``, ``conftest.py`` and ``pytest_plugins``
    routes are not entry points."""
    plugins = []
    for ep in metadata.entry_points(group="pytest11"):
        dist = getattr(ep, "dist", None)
        plugins.append({"name": ep.name, "module": ep.value,
                        "distribution": dist.name if dist else None,
                        "version": dist.version if dist else None})
    return sorted(plugins, key=lambda p: p["name"])


def plugins_reported_by_pytest(log: str) -> list[str] | None:
    """The ``plugins:`` header pytest printed for the run. pytest lists only
    plugins that came from installed distributions (entry points); a plugin
    loaded with ``-p``, ``PYTEST_PLUGINS``, a ``conftest.py`` or a
    ``pytest_plugins`` variable is not listed."""
    m = re.search(r"^plugins: (.*)$", log, re.MULTILINE)
    return [p.strip() for p in m.group(1).split(",")] if m else None


def parse_loaded_libraries(maps_lines) -> list[str]:
    """CUDA / CTranslate2 shared objects in ``/proc/<pid>/maps`` lines. The
    pathname is everything after the fifth field, so paths containing spaces
    (e.g. Google Drive's "My Drive") are kept whole; a " (deleted)" suffix
    (file replaced while mapped) is dropped from the path."""
    paths = set()
    for line in maps_lines:
        fields = line.rstrip("\n").split(maxsplit=5)
        if len(fields) < 6 or not fields[5].startswith("/"):
            continue  # anonymous mapping, [heap], [stack], [vdso]...
        path = fields[5].removesuffix(" (deleted)")
        if CUDA_LIBRARY_PATTERN.search(path):
            paths.add(path)
    return sorted(paths)


def loaded_libraries(results: dict) -> dict:
    """CUDA-related libraries the test process had mapped after decoding, as
    recorded by the real-speech test. Absent if that test did not record it."""
    props = next((t.get("properties", {}) for t in results.get("tests", [])
                  if t["name"] == LOADED_LIBRARIES_TEST), {})
    raw = props.get("observed_loaded_libraries")
    if raw is None:
        return {"available": False, "reason": "not recorded by the test run"}
    try:
        value = json.loads(raw)
    except ValueError:
        return {"available": False, "reason": f"unreadable value {raw[:200]!r}"}
    if not isinstance(value, list):
        return {"available": False, "reason": str(value)[:200]}
    return {"available": True, "source": "/proc/self/maps of the pytest process, after the real-speech test's decodes",
            "libraries": value}


def _plugins(environ: dict[str, str], log: str) -> dict:
    return {
        "installed_entry_points": pytest_plugin_inventory(),
        "header_reported": plugins_reported_by_pytest(log),
        "note": ("header_reported lists entry-point plugins only; plugins loaded via -p, "
                 "PYTEST_PLUGINS, conftest.py or pytest_plugins are not listed (see the "
                 "override and checkout rules)"),
        "autoload_disabled": bool(environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD")),
    }


def provenance(environ: dict[str, str], results: dict, log: str) -> tuple[dict, list[str]]:
    """The provenance sections, and a problem for each that could not be
    collected. A failure is recorded in place of the section rather than
    aborting the record, and fails the run closed."""
    sections = {
        "environment_metadata": lambda: environment_metadata(environ),
        "pytest_plugins": lambda: _plugins(environ, log),
        "loaded_libraries": lambda: loaded_libraries(results),
    }
    out: dict = {}
    problems: list[str] = []
    for name, collect in sections.items():
        try:
            out[name] = collect()
        except Exception as exc:  # noqa: BLE001  recorded, never raised: the record must be written
            out[name] = {"collection_error": f"{type(exc).__name__}: {exc}"}
            problems.append(f"provenance section {name} could not be recorded ({type(exc).__name__}: {exc})")
    return out, problems


def _git(*args: str, root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, timeout=30, check=True)


def checkout_audit(root: Path = REPO) -> dict:
    """Untracked files that could execute in the test process, found before
    pytest starts. ``git status --untracked-files=no`` (the dirty check) does
    not see them."""
    try:
        raw = _git("status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignored=matching",
                   root=root).stdout.decode("utf-8", "surrogateescape")
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": f"git status failed: {type(exc).__name__}: {exc}"}
    untracked, ignored = [], []
    for entry in filter(None, raw.split("\0")):
        code, path = entry[:2], entry[3:]
        if code == "??":
            untracked.append(path)
        elif code == "!!":
            ignored.append(path)
    return {"untracked": sorted(untracked), "ignored_executable": sorted(
        p for p in ignored if _executable_at_import_root(p))}


def _executable_at_import_root(path: str) -> bool:
    """An ignored *file* that Python or pytest would load: directly inside an
    import root, with an executable suffix or a configuration name."""
    if path.endswith("/"):
        return False  # an ignored directory (e.g. __pycache__/, .cache/, *.egg-info/)
    parent, _, name = path.rpartition("/")
    return parent in IMPORT_ROOTS and (name.endswith(EXECUTABLE_SUFFIXES) or name in CONFIG_NAMES)


def checkout_problems(audit: dict | None) -> list[str]:
    if not audit:
        return ["checkout audit missing"]
    if "error" in audit:
        return [f"checkout audit failed: {audit['error']}"]
    problems = []
    if audit.get("untracked"):
        shown = audit["untracked"][:10]
        problems.append(f"untracked files present ({len(audit['untracked'])}): {shown}; "
                        "they can execute in the test process (e.g. conftest.py)")
    if audit.get("ignored_executable"):
        problems.append(f"ignored files Python/pytest would load: {audit['ignored_executable'][:10]}")
    return problems


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
             gpus: list[str], overrides: dict, git: dict | None, exit_code: int,
             checkout: dict | None) -> list[str]:
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
    problems += checkout_problems(checkout)
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
    checkout = checkout_audit()  # before pytest: what could have run inside it

    started = datetime.now(timezone.utc)
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, check=False)
    finished = datetime.now(timezone.utc)
    (out / "pytest.log").write_text(proc.stdout + proc.stderr)
    print(proc.stdout + proc.stderr)

    results = _junit_results(junit) if junit.exists() else {"collected": 0, "junit_missing": True}
    env_info = environment()
    prov, prov_problems = provenance(env, results, proc.stdout + proc.stderr)
    problems = validate(results=results, device=a.device, compute_type=compute_type,
                        expect_gpu=a.expect_gpu or None, gpus=env_info.get("gpus", []),
                        overrides=overrides, git=env_info.get("git"), exit_code=proc.returncode,
                        checkout=checkout) + prov_problems
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
        "checkout_audit": checkout,
        "provenance": prov,
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
