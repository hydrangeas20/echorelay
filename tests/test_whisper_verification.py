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
                git={"commit": "f" * 40, "dirty": False}, exit_code=0)
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
    assert v["LD_PRELOAD"] == ["/lib/libtcmalloc.so.4", "[redacted]"]
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


def test_provenance_never_changes_completeness():
    """A complete run stays complete whatever provenance says, and an
    incomplete one stays incomplete: provenance is not evidence of identity."""
    assert _validate() == []
    prov = wv.provenance({"LD_LIBRARY_PATH": "/x"}, _results(), "plugins: asyncio-1.4.0\n")
    assert set(prov) == {"environment_metadata", "pytest_plugins", "loaded_libraries"}
    assert prov["pytest_plugins"]["reported_loaded"] == ["asyncio-1.4.0"]
    assert prov["loaded_libraries"]["available"] is False
