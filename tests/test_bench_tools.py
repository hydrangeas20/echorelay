"""Phase 2 benchmark tooling: analysis, sampling, long-stream audio, profiling
helpers. Pure functions on synthetic inputs; no model, no GPU."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.analyze_results import analyze, pacing_floor_s
from benchmarks.common import Utterance, concatenate
from benchmarks.profile_decode import build_signal, fit_linear
from benchmarks.sampler import parse_nvidia_smi, summarize_samples
from conftest import pcm16


def _utt(name: str, seconds: float, ref: str | None = "A B") -> Utterance:
    return Utterance(name, pcm16(seconds), ref, name)


def _synthetic_result(intercept=30.0, slope=6.0, concurrency=4, wall=40.0) -> dict:
    """Streams whose partial decode cost is exactly intercept + slope * audio."""
    streams = []
    for i, dur in enumerate([10.0, 4.0] * 4):
        n = int(dur)
        audio = [float(k) for k in range(1, n + 1)]
        streams.append({
            "index": i, "utterance": f"u{dur:g}", "audio_s": dur, "ok": True,
            "t_first_audio": 100.0, "t_first_partial": 100.0 + 0.9 + (intercept + slope) / 1000 + 0.004,
            "t_end_sent": 200.0, "t_final": 200.0 + 0.120,
            "num_partials": n, "partial_audio_s": audio,
            "partial_inference_ms": [intercept + slope * a for a in audio],
            "final_inference_ms": 100.0, "final_audio_s": dur,
            "partial_timing": [], "final_timing": None,
        })
    return {
        "label": "synthetic", "run_config": {"concurrency": concurrency, "chunk_ms": 100, "speed": 1.0},
        "server": {"health": {"model": {"backend": "fake"}, "inference": {"max_concurrent": 1}},
                   "ready_limits": {"partial_interval_s": 1.0}},
        "summary": {"wall_clock_s": wall, "streams_ok": len(streams)},
        "streams": streams,
    }


def test_analysis_recovers_partial_cost_line_and_utilization():
    a = analyze(_synthetic_result())
    fit = a["partial_cost"]["linear_fit"]
    assert fit["intercept_ms"] == pytest.approx(30.0) and fit["slope_ms_per_audio_s"] == pytest.approx(6.0)
    assert fit["r"] == pytest.approx(1.0)
    u = a["utilization"]
    partial_ms = 4 * sum(30 + 6 * k for k in range(1, 11)) + 4 * sum(30 + 6 * k for k in range(1, 5))
    assert u["partial_decode_s"] == pytest.approx(partial_ms / 1000)
    assert u["final_decode_s"] == pytest.approx(0.8)
    assert u["busy_fraction"] == pytest.approx((partial_ms / 1000 + 0.8) / 40.0)
    row = next(r for r in a["per_utterance"] if r["utterance"] == "u10")
    assert row["amplification"] == pytest.approx(5.5)  # (1+...+10)/10
    # TTFP remainder: 4 ms after removing the 0.9 s floor and the first decode
    assert a["time_to_first_partial"]["remainder_s"]["p50"] == pytest.approx(0.004, abs=1e-6)
    assert a["final_latency"]["remainder_s"]["p50"] == pytest.approx(0.020, abs=1e-6)
    assert a["projections"]["label"].startswith("PROJECTION")


def test_projection_marks_extrapolation_beyond_fitted_range():
    rows = analyze(_synthetic_result())["projections"]["by_stream_length"]
    assert [r["extrapolated"] for r in rows] == [False, False, True, True, True, True]
    sixty = next(r for r in rows if r["stream_audio_s"] == 60)
    assert sixty["partial_compute_s"] == pytest.approx(sum(30 + 6 * k for k in range(1, 61)) / 1000)


@pytest.mark.parametrize("chunk_ms, interval, speed, expected", [
    (100, 1.0, 1.0, 0.9),   # Phase 1: 10th frame completes 1.0 s, sent 0.9 s after the first
    (100, 1.0, 2.0, 0.45),
    (30, 1.0, 1.0, 0.99),   # 34 frames of 30 ms
    (100, 0.5, 1.0, 0.4),
])
def test_pacing_floor(chunk_ms, interval, speed, expected):
    r = {"run_config": {"chunk_ms": chunk_ms, "speed": speed},
         "server": {"ready_limits": {"partial_interval_s": interval}}}
    assert pacing_floor_s(r) == pytest.approx(expected)


def test_pacing_floor_unpaced_is_undefined():
    r = {"run_config": {"chunk_ms": 100, "speed": 0.0}, "server": {"ready_limits": {"partial_interval_s": 1.0}}}
    assert pacing_floor_s(r) is None


def test_concatenate_is_deterministic_and_keeps_references_exact():
    utts = [_utt("a", 2.0, "ONE"), _utt("b", 3.0, "TWO"), _utt("c", 1.0, "THREE")]
    long = concatenate(utts, 5.0)
    assert [u.name for u in long] == ["concat5s_00", "concat5s_01", "concat5s_02"]
    assert [u.duration_s for u in long] == [5.0, 6.0, 6.0]
    assert [u.reference for u in long] == ["ONE TWO", "TWO THREE ONE", "THREE ONE TWO"]
    assert concatenate(utts, 5.0)[1].sha256 == long[1].sha256


def test_concatenate_without_references_gives_none():
    long = concatenate([_utt("a", 2.0, None), _utt("b", 2.0)], 3.0)
    assert long[0].reference is None


def test_build_signal_has_exact_length():
    utts = [_utt("a", 1.5), _utt("b", 0.7)]
    for length in (0.5, 2.0, 7.3):
        assert build_signal(utts, length).size == round(length * 16000)


def test_fit_linear_respects_length_cap():
    lengths = [1, 2, 4, 8, 16, 30, 60]
    ms = [30 + 6 * x for x in lengths[:-1]] + [9999.0]  # >30 s point is off-line
    f = fit_linear(lengths, ms, max_len=30.0)
    assert f["slope_ms_per_audio_s"] == pytest.approx(6.0) and f["max_length_s"] == 30.0


def test_parse_nvidia_smi_handles_unavailable_fields():
    g = parse_nvidia_smi("37, 1234, 1590, 1590, [N/A], 61")
    assert g["gpu_util_pct"] == 37.0 and g["sm_clock_mhz"] == 1590.0
    assert g["power_w"] is None and g["temp_c"] == 61.0


def test_summarize_samples():
    samples = [
        {"t": 0.0, "server": {"active": 1, "in_flight": 0, "waiting": 0, "buffered_audio_bytes": 0,
                              "rss_bytes": 100, "cpu_s": 1.0, "threads": 5, "loop_lag_max_s": 0.001},
         "gpus": [{"gpu_util_pct": 10.0, "sm_clock_mhz": 585.0}]},
        {"t": 2.0, "server": {"active": 4, "in_flight": 1, "waiting": 2, "buffered_audio_bytes": 64000,
                              "rss_bytes": 120, "cpu_s": 2.0, "threads": 5, "loop_lag_max_s": 0.002},
         "gpus": [{"gpu_util_pct": 50.0, "sm_clock_mhz": 1590.0}]},
        {"t": 3.0, "server_error": "metrics unavailable"},
    ]
    s = summarize_samples(samples)
    assert s["num_samples"] == 3
    assert s["waiting"]["max"] == 2 and s["active"]["max"] == 4
    assert s["server_cpu_cores_mean"] == pytest.approx(0.5)
    assert s["sm_clock_mhz"] == {"min": 585.0, "p50": 1087.5, "mean": 1087.5, "max": 1590.0}
    assert np.isclose(s["gpu_util_pct"]["mean"], 30.0)
