"""The real-model verification record is complete only for the exact expected
test set, all passing, with a runtime identity that matches the request.
Offline: exercises the validator on synthetic JUnit results."""

from __future__ import annotations

import copy
import json

import pytest

from benchmarks import whisper_verification as wv
from benchmarks.whisper_verification import (
    CT2_CLASS,
    EXPECTED_TESTS,
    IDENTITY_TEST,
    WHISPER_CLASS,
    validate,
)

OBSERVED_CUDA = {
    "observed_whisper_class": WHISPER_CLASS,
    "observed_ct2_class": CT2_CLASS,
    "observed_ct2_device": "cuda",
    "observed_ct2_compute_type": "float16",
    "observed_ct2_is_multilingual": "False",
    "observed_cuda_devices": "1",
    "observed_model_path": "/root/.cache/huggingface/hub/models--Systran--faster-whisper-base.en/"
                           "snapshots/abc123",
    "observed_model_bin_sha256": "0" * 64,
}


def _results(observed=None, outcomes=None, names=EXPECTED_TESTS) -> dict:
    outcomes = outcomes or {}
    tests = []
    for name in sorted(names):
        t = {"name": name, "outcome": outcomes.get(name, "passed")}
        if name == IDENTITY_TEST:
            t["properties"] = dict(OBSERVED_CUDA if observed is None else observed)
        tests.append(t)
    counts = {k: sum(t["outcome"] == k for t in tests) for k in ("passed", "failed", "error", "skipped")}
    return {"collected": len(tests), **counts, "tests": tests}


def _validate(results=None, **kw) -> list[str]:
    args = dict(results=results or _results(), device="cuda", compute_type="float16",
                expect_gpu="T4", gpus=["Tesla T4, 580.82.07, 15360 MiB"],
                overrides={"PYTEST_ADDOPTS": None, "PYTHONPATH": None},
                git={"commit": "f" * 40, "dirty": False}, exit_code=0,
                checkout={"untracked": [], "ignored_executable": []})
    args.update(kw)
    return validate(**args)


def test_complete_run_has_no_problems():
    assert _validate() == []


def test_cpu_int8_may_report_int8_float32():
    observed = {**OBSERVED_CUDA, "observed_ct2_device": "cpu",
                "observed_ct2_compute_type": "int8_float32", "observed_cuda_devices": "0"}
    assert _validate(_results(observed), device="cpu", compute_type="int8", gpus=[]) == []


@pytest.mark.parametrize("outcome", ["failed", "error", "skipped"])
def test_any_non_passing_test_is_incomplete(outcome):
    problems = _validate(_results(outcomes={"test_adapter_transcribes_real_speech": outcome}))
    assert any(f"test_adapter_transcribes_real_speech: {outcome}" in p for p in problems)
    assert any("expected exactly 8" in p for p in problems)


def test_missing_or_extra_tests_are_incomplete():
    fewer = EXPECTED_TESTS - {"test_websocket_streams_real_speech"}
    assert any("wrong test set" in p for p in _validate(_results(names=fewer)))
    more = EXPECTED_TESTS | {"test_something_else"}
    assert any("wrong test set" in p for p in _validate(_results(names=more)))


@pytest.mark.parametrize(("key", "value", "match"), [
    ("observed_ct2_device", "cpu", "observed_ct2_device"),
    ("observed_ct2_compute_type", "float32", "observed_ct2_compute_type"),  # silent fallback
    ("observed_ct2_compute_type", "int8_float16", "observed_ct2_compute_type"),
    ("observed_cuda_devices", "0", "observed_cuda_devices"),
    ("observed_whisper_class", "stub_plugin.StubModel", "observed_whisper_class"),
    ("observed_ct2_class", "stub_plugin._CT2", "observed_ct2_class"),
    ("observed_model_path", "/cache/models--Systran--faster-whisper-base/snapshots/x",
     "observed_model_path"),
    ("observed_model_path", "/cache/models--Systran--faster-whisper-small.en/snapshots/x",
     "observed_model_path"),
    ("observed_ct2_is_multilingual", "True", "observed_ct2_is_multilingual"),
])
def test_observed_identity_must_match_the_request(key, value, match):
    observed = {**OBSERVED_CUDA, key: value}
    assert any(match in p for p in _validate(_results(observed)))


@pytest.mark.parametrize("key", sorted(OBSERVED_CUDA))
def test_unreported_identity_is_incomplete(key):
    observed = {k: v for k, v in OBSERVED_CUDA.items() if k != key}
    assert any(f"did not report {key}" in p for p in _validate(_results(observed)))


def test_requested_settings_alone_are_not_evidence():
    """A run whose identity test recorded nothing cannot be complete, whatever
    was requested."""
    problems = _validate(_results(observed={}))
    assert sum("did not report" in p for p in problems) == len(OBSERVED_CUDA)


def test_wrong_or_missing_gpu_is_incomplete():
    assert any("expected one containing 'T4'" in p
               for p in _validate(gpus=["NVIDIA A100-SXM4-40GB, 550.0, 40960 MiB"]))
    assert any("no GPU" in p for p in _validate(gpus=[]))
    assert _validate(gpus=[], expect_gpu=None) == []  # check explicitly disabled


@pytest.mark.parametrize("var", ["PYTEST_ADDOPTS", "PYTHONPATH"])
def test_plugin_or_path_overrides_are_incomplete(var):
    overrides = {"PYTEST_ADDOPTS": None, "PYTHONPATH": None, var: "-p stub_plugin"}
    assert any(f"{var} is set" in p for p in _validate(overrides=overrides))


@pytest.mark.parametrize("git", [{"commit": "f" * 40, "dirty": True}, None, {"commit": None}])
def test_dirty_or_unknown_checkout_is_incomplete(git):
    assert any("not a clean commit" in p for p in _validate(git=git))


def test_pytest_failure_or_missing_junit_is_incomplete():
    assert any("pytest exited 1" in p for p in _validate(exit_code=1))
    problems = _validate({"collected": 0, "junit_missing": True}, exit_code=4)
    assert problems == ["pytest exited 4", "no JUnit results (pytest did not run the suite)"]


def test_duplicate_test_names_are_incomplete():
    results = _results()
    results["tests"].append(copy.deepcopy(results["tests"][0]))
    assert any("duplicate" in p for p in _validate(results))


# ------------------------------------------------- environment provenance
# These cover the record's `provenance` section, which describes the run but
# never changes `complete`.


def test_pytest_plugins_env_is_an_override():
    """PYTEST_PLUGINS loads arbitrary plugin modules, like PYTEST_ADDOPTS."""
    assert "PYTEST_PLUGINS" in wv.OVERRIDE_VARS
    overrides = {"PYTEST_ADDOPTS": None, "PYTHONPATH": None, "PYTEST_PLUGINS": "evil_plugin"}
    assert any("PYTEST_PLUGINS is set" in p for p in _validate(overrides=overrides))


def test_environment_metadata_reads_only_allowlisted_variables(tmp_path):
    env = {"LD_LIBRARY_PATH": "/a:/b", "CT2_VERBOSE": "1", "HOME": "/root",
           "AWS_SECRET_ACCESS_KEY": "abc", "GH_TOKEN": "ghp_x", "HF_TOKEN": "hf_x"}
    meta = wv.environment_metadata(env)
    assert set(meta["variables"]) == {"LD_LIBRARY_PATH", "CT2_VERBOSE"}
    assert meta["variables"]["LD_LIBRARY_PATH"] == ["/a", "/b"]
    assert "LD_PRELOAD" in meta["unset"] and "LD_LIBRARY_PATH" not in meta["unset"]
    flat = repr(meta)
    for secret in ("abc", "ghp_x", "hf_x", "/root"):
        assert secret not in flat
    assert "not proof of loading" in meta["note"]


def test_environment_metadata_redacts_credentials():
    env = {"CUDA_PATH": "https://user:pw@mirror.example/cuda",
           "LD_LIBRARY_PATH": "/usr/lib64-nvidia:https://u:pw2@host/lib:/opt/x",
           "CT2_AUTH_TOKEN": "s3cr3t", "LD_PRELOAD": "/lib/libtcmalloc.so.4 TOKEN=xyz"}
    meta = wv.environment_metadata(env)
    v = meta["variables"]
    assert v["CUDA_PATH"] == "https://[redacted]@mirror.example/cuda"
    # a URL inside a ':'-separated list is redacted before the list is split
    assert v["LD_LIBRARY_PATH"] == ["/usr/lib64-nvidia", "https", "//[redacted]@host/lib", "/opt/x"]
    assert v["CT2_AUTH_TOKEN"] == "[redacted]"
    assert v["LD_PRELOAD"] == ["/lib/libtcmalloc.so.4", "TOKEN=[redacted]"]
    flat = repr(meta)
    for secret in ("pw", "pw2", "s3cr3t", "xyz", "user:", "u:"):
        assert secret not in flat, secret


def test_environment_metadata_locates_cuda_libraries_on_the_search_path(tmp_path):
    have, empty = tmp_path / "cublas", tmp_path / "empty"
    have.mkdir()
    empty.mkdir()
    (have / "libcublas.so.12").write_bytes(b"")
    env = {"LD_LIBRARY_PATH": f"{empty}:{have}:{tmp_path / 'missing'}"}
    meta = wv.environment_metadata(env)
    assert meta["cuda_libraries_on_search_path"]["libcublas.so.12"] == [str(have)]
    assert meta["cuda_libraries_on_search_path"]["libcudnn.so.9"] == []
    assert meta["ld_library_path_entries_exist"] == {
        str(empty): True, str(have): True, str(tmp_path / "missing"): False}


def test_nvidia_wheels_lists_only_nvidia_distributions(monkeypatch):
    class Dist:
        def __init__(self, name, version):
            self.metadata, self.version = {"Name": name}, version

    fake = [Dist("nvidia-cublas-cu12", "12.4.5.8"), Dist("numpy", "2.1.3"),
            Dist("nvidia-cudnn-cu12", "9.1.0.70")]
    monkeypatch.setattr(wv.metadata, "distributions", lambda: fake)
    assert wv.nvidia_wheels() == {"nvidia-cublas-cu12": "12.4.5.8", "nvidia-cudnn-cu12": "9.1.0.70"}


def test_pytest_plugin_inventory_includes_versions():
    inventory = {p["name"]: p for p in wv.pytest_plugin_inventory()}
    assert inventory["asyncio"]["distribution"] == "pytest-asyncio"  # a dev dependency
    assert inventory["asyncio"]["version"]


def test_plugins_reported_by_pytest_parses_the_header():
    log = "platform linux\nplugins: asyncio-1.4.0, typeguard-4.6.0, anyio-4.15.1\ncollected 8\n"
    assert wv.plugins_reported_by_pytest(log) == ["asyncio-1.4.0", "typeguard-4.6.0", "anyio-4.15.1"]
    assert wv.plugins_reported_by_pytest("no header here") is None


def _with_loaded(value: str | None) -> dict:
    results = _results()
    for t in results["tests"]:
        if t["name"] == wv.LOADED_LIBRARIES_TEST and value is not None:
            t["properties"] = {"observed_loaded_libraries": value}
    return results


def test_loaded_libraries_come_only_from_the_test_process():
    libs = ["/usr/local/lib/python3.13/dist-packages/nvidia/cublas/lib/libcublas.so.12"]
    got = wv.loaded_libraries(_with_loaded(json.dumps(libs)))
    assert got == {"available": True, "libraries": libs,
                   "source": "/proc/self/maps of the pytest process, after the real-speech test's decodes"}
    assert wv.loaded_libraries(_with_loaded(None)) == {
        "available": False, "reason": "not recorded by the test run"}
    assert wv.loaded_libraries(_with_loaded(json.dumps("unavailable: no maps")))["available"] is False
    assert wv.loaded_libraries(_with_loaded("{not json"))["available"] is False


def test_provenance_values_never_change_completeness():
    """A complete run stays complete whatever provenance says, and an
    incomplete one stays incomplete: provenance is not evidence of identity."""
    assert _validate() == []
    prov, problems = wv.provenance({"LD_LIBRARY_PATH": "/x", "LD_PRELOAD": "/odd.so"}, _results(),
                                   "plugins: asyncio-1.4.0\n")
    assert problems == []
    assert set(prov) == {"environment_metadata", "pytest_plugins", "loaded_libraries"}
    assert prov["pytest_plugins"]["header_reported"] == ["asyncio-1.4.0"]
    assert "not listed" in prov["pytest_plugins"]["note"]
    assert prov["loaded_libraries"]["available"] is False


# --------------------------------------------- F1: unreadable directories
def test_unreadable_search_path_entry_is_reported_not_raised(monkeypatch, tmp_path):
    """Path.exists / is_dir raise PermissionError under an unreadable parent
    (as a non-root user). Simulated, so the test does not depend on running
    as root or not."""
    blocked, ok = tmp_path / "locked" / "sub", tmp_path / "ok"
    ok.mkdir()
    (ok / "libcudnn.so.9").write_bytes(b"")
    real_exists, real_is_dir = wv.Path.exists, wv.Path.is_dir

    def guarded(real):
        def check(self, *a, **k):
            if str(self).startswith(str(blocked)):
                raise PermissionError(13, "Permission denied", str(self))
            return real(self, *a, **k)
        return check

    monkeypatch.setattr(wv.Path, "exists", guarded(real_exists))
    monkeypatch.setattr(wv.Path, "is_dir", guarded(real_is_dir))
    meta = wv.environment_metadata({"LD_LIBRARY_PATH": f"{blocked}:{ok}"})
    assert meta["ld_library_path_entries_exist"][str(blocked)].startswith("error: PermissionError")
    assert meta["ld_library_path_entries_exist"][str(ok)] is True
    assert meta["cuda_libraries_on_search_path"]["libcudnn.so.9"] == [str(ok)]
    assert {e["library"] for e in meta["search_errors"]} == set(wv.CUDA_RUNTIME_LIBRARIES)
    assert all(e["directory"] == str(blocked) and "Permission denied" in e["error"]
               for e in meta["search_errors"])


def test_provenance_collection_failure_is_recorded_and_fails_closed(monkeypatch):
    def boom(_environ):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(wv, "environment_metadata", boom)
    prov, problems = wv.provenance({}, _results(), "")
    assert prov["environment_metadata"] == {"collection_error": "RuntimeError: unexpected"}
    assert prov["loaded_libraries"]["available"] is False  # other sections still collected
    assert problems == [("provenance section environment_metadata could not be recorded "
                         "(RuntimeError: unexpected)")]


# ----------------------------------------------- F2: /proc/self/maps parsing
MAPS = [
    "7f00-7f10 r--p 00000000 fe:00 1024295                    /content/drive/My Drive/x/libcublas.so.12\n",
    ("7f10-7f20 r-xp 00000000 fe:00 1024296                    /usr/local/lib/python3.13/dist-packages/"
     "nvidia/cublas/lib/libcublasLt.so.12\n"),
    "7f20-7f30 r--p 00000000 fe:00 1024297                    /opt/ct2/libctranslate2-bdb19cbc.so.4.8.3 (deleted)\n",
    "7f30-7f40 r--p 00000000 fe:00 1024298                    /usr/lib/x86_64-linux-gnu/libm.so.6\n",
    "7f40-7f50 rw-p 00000000 00:00 0 \n",
    "7f50-7f60 rw-p 00000000 00:00 0                          [heap]\n",
    "7f60-7f70 r--p 00000000 fe:00 1024295                    /content/drive/My Drive/x/libcublas.so.12\n",
]


def test_maps_parsing_keeps_paths_with_spaces_whole():
    assert wv.parse_loaded_libraries(MAPS) == [
        "/content/drive/My Drive/x/libcublas.so.12",
        "/opt/ct2/libctranslate2-bdb19cbc.so.4.8.3",
        "/usr/local/lib/python3.13/dist-packages/nvidia/cublas/lib/libcublasLt.so.12",
    ]


@pytest.mark.skipif(not __import__("os").path.exists("/proc/self/maps"), reason="Linux /proc")
def test_maps_parsing_on_a_real_mapping_with_a_space(tmp_path):
    """End to end on this process: a library loaded from a directory whose
    name contains a space is reported with its full path."""
    import ctypes
    import shutil

    src = next(p for p in ("/lib/x86_64-linux-gnu/libm.so.6", "/usr/lib/x86_64-linux-gnu/libm.so.6",
                           "/lib64/libm.so.6", "/usr/lib/aarch64-linux-gnu/libm.so.6")
               if __import__("os").path.exists(p))
    lib = tmp_path / "My Drive" / "libcudart.so.12"
    lib.parent.mkdir()
    shutil.copy(src, lib)
    ctypes.CDLL(str(lib))
    with open("/proc/self/maps") as maps:
        assert str(lib) in wv.parse_loaded_libraries(maps)


# ------------------------------------------------------- F4: redaction
@pytest.mark.parametrize(("raw", "secret"), [
    ("https://user:p@ss@host/x", "p@ss"),           # "@" inside the password
    ("https://user:pa/ss@host/x", "pa/ss"),         # "/" inside the password
    ("https://user:pa/s@s@host/x", "pa/s@s"),
    ("//user:pw@host/x", "user:pw"),                # scheme-less
    ("https://host/x?a=1&token=abc123&b=2", "abc123"),
    ("https://host/x?api_key=k3y:with:colons", "k3y"),
    ("ghp_" + "A" * 36, "A" * 36),                  # bare GitHub token
    ("hf_" + "b" * 34, "b" * 34),                   # bare Hugging Face token
    ("AKIA" + "ABCDEFGHIJKLMNOP", "ABCDEFGHIJKLMNOP"),
    ("/opt/x/github_pat_" + "c" * 30, "c" * 30),
])
def test_redaction_removes_demonstrated_credential_forms(raw, secret):
    for name in ("CUDA_HOME", "LD_LIBRARY_PATH"):
        meta = wv.environment_metadata({name: raw})
        assert secret not in repr(meta["variables"]), (name, meta["variables"])


def test_redaction_keeps_ordinary_paths_intact():
    paths = "/usr/lib64-nvidia:/usr/local/lib/python3.13/dist-packages/nvidia/cublas/lib:/opt/sk-tools"
    meta = wv.environment_metadata({"LD_LIBRARY_PATH": paths, "CUDA_HOME": "/usr/local/cuda-12.4"})
    assert meta["variables"]["LD_LIBRARY_PATH"] == paths.split(":")
    assert meta["variables"]["CUDA_HOME"] == "/usr/local/cuda-12.4"


# ---------------------------------------------------- F6: LD_PRELOAD split
def test_ld_preload_accepts_mixed_separators():
    meta = wv.environment_metadata({"LD_PRELOAD": "/a/libx.so /b/liby.so:/c/libz.so"})
    assert meta["variables"]["LD_PRELOAD"] == ["/a/libx.so", "/b/liby.so", "/c/libz.so"]


# --------------------------------------------------- F5: checkout audit
def _repo(tmp_path):
    import subprocess

    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    (tmp_path / ".gitignore").write_text("*.pyc\nlocal_settings.py\n.cache/\n*.egg-info/\n__pycache__/\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "conftest.py").write_text("")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "pkg.py").write_text("")
    git("add", "-A")
    git("commit", "-qm", "init")
    return tmp_path


def test_checkout_audit_clean_repo(tmp_path):
    repo = _repo(tmp_path)
    (repo / ".cache").mkdir()
    (repo / ".cache" / "audio.tar.gz").write_bytes(b"")          # ignored data directory
    (repo / "src" / "pkg.egg-info").mkdir()
    (repo / "src" / "pkg.egg-info" / "PKG-INFO").write_text("")  # ignored build metadata
    (repo / "tests" / "__pycache__").mkdir()
    (repo / "tests" / "__pycache__" / "conftest.cpython-313.pyc").write_bytes(b"")
    audit = wv.checkout_audit(repo)
    assert audit == {"untracked": [], "ignored_executable": []}
    assert wv.checkout_problems(audit) == []


@pytest.mark.parametrize("path", ["conftest.py", "tests/conftest_extra.py", "faster_whisper.py",
                                  "pytest.ini", "tests/sub/conftest.py", "notes.txt"])
def test_untracked_files_fail_closed(tmp_path, path):
    repo = _repo(tmp_path)
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text("raise SystemExit('should never run')\n")
    problems = wv.checkout_problems(wv.checkout_audit(repo))
    assert len(problems) == 1 and path in problems[0] and "untracked files present" in problems[0]


@pytest.mark.parametrize("path", ["stale.pyc", "local_settings.py", "tests/helper.pyc", "src/mod.pyc"])
def test_ignored_executable_files_at_import_roots_fail_closed(tmp_path, path):
    repo = _repo(tmp_path)
    (repo / path).write_bytes(b"")
    problems = wv.checkout_problems(wv.checkout_audit(repo))
    assert problems == [f"ignored files Python/pytest would load: ['{path}']"]


def test_checkout_audit_failure_fails_closed(tmp_path):
    audit = wv.checkout_audit(tmp_path)  # not a git repository
    assert "error" in audit
    assert wv.checkout_problems(audit)[0].startswith("checkout audit failed: git status failed")
    assert wv.checkout_problems(None) == ["checkout audit missing"]


def test_untracked_file_makes_an_otherwise_complete_run_incomplete():
    problems = _validate(checkout={"untracked": ["conftest.py"], "ignored_executable": []})
    assert len(problems) == 1 and "conftest.py" in problems[0]
