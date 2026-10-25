"""The real-model verification record is complete only for the exact expected
test set, all passing, with a runtime identity that matches the request.
Offline: exercises the validator on synthetic JUnit results."""

from __future__ import annotations

import copy

import pytest

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
