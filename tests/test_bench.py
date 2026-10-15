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
