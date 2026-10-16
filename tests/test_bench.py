"""Benchmark helper correctness (pure functions; no model, no server)."""

from __future__ import annotations

from benchmarks.common import corpus_wer, normalize_text, percentiles


def test_normalize_text():
    assert normalize_text("  Hello, WORLD -- it's   twenty-one! ") == "hello world it's twenty one"


def test_corpus_wer_uses_normalization():
    assert corpus_wer(["HELLO WORLD"], ["Hello, world."]) == 0.0
    # 1 substitution over 4 reference words
    assert corpus_wer(["a b c d", "  "], ["a b x d", "ignored"]) == 0.25
    assert corpus_wer([], []) is None


def test_percentiles():
    p = percentiles([float(i) for i in range(1, 101)])
    assert p["n"] == 100 and p["p50"] == 50.5 and abs(p["p95"] - 95.05) < 1e-9
    assert percentiles([])["p50"] is None


def test_aggregate_rtf_is_total_time_over_total_audio_mean_of_repeats():
    from benchmarks.compare_precision import rtf_stats

    # two files (10 s and 2 s), two repeats
    per_file = {"long": [2.0, 3.0], "short": [1.0, 1.0]}
    stats = rtf_stats(per_file, {"long": 10.0, "short": 2.0})
    # repeat 1: 3.0/12 = 0.25, repeat 2: 4.0/12 = 0.3333
    assert stats["aggregate_rtf_per_repeat"] == [0.25, 4.0 / 12]
    assert abs(stats["aggregate_rtf"] - (0.25 + 4.0 / 12) / 2) < 1e-12
    assert abs(stats["aggregate_rtf_spread"] - (4.0 / 12 - 0.25)) < 1e-12
    # secondary view uses per-file minima: long 0.2, short 0.5
    assert stats["rtf_per_file_best_of_repeats"]["min"] == 0.2
    assert stats["rtf_per_file_best_of_repeats"]["max"] == 0.5


def test_int8_float16_is_labelled_mixed_precision():
    from benchmarks.compare_precision import precision_label

    assert precision_label("float16") == "FP16"
    assert "mixed precision" in precision_label("int8_float16")
