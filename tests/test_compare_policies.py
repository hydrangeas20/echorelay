"""Matched-comparison tooling: counterbalanced order, grouping, deltas,
text identity, and bootstrap intervals, on small synthetic run directories."""

from __future__ import annotations

import json

import pytest

from benchmarks.compare_policies import analyze, bootstrap_p95_diff, to_markdown
from benchmarks.matched_comparison import POLICIES, run_order


def test_run_order_is_counterbalanced():
    order = run_order(["base", "prio", "cancel", "both"], 2)
    assert order == [(1, "base"), (1, "prio"), (1, "cancel"), (1, "both"),
                     (2, "both"), (2, "cancel"), (2, "prio"), (2, "base")]
    # every policy's mean position is identical, so linear drift cancels
    pos = {p: [i for i, (_, q) in enumerate(order) if q == p] for p in POLICIES}
    assert len({sum(v) for v in pos.values()}) == 1


def test_bootstrap_detects_a_shift_and_not_noise():
    base = [0.1 + 0.001 * i for i in range(40)]
    same = bootstrap_p95_diff(base, list(base))
    assert same["diff_p95"] == 0 and same["ci95"][0] <= 0 <= same["ci95"][1]
    faster = bootstrap_p95_diff(base, [v - 0.05 for v in base])
    assert faster["diff_p95"] == pytest.approx(-0.05)
    assert faster["ci95"][1] < 0  # interval excludes zero
    assert bootstrap_p95_diff(base[:3], base[:3]) == {"n_base": 3, "n_cand": 3}


def _stream(i, utt, latency, queue, text, ok=True):
    return {"index": i, "utterance": utt, "ok": ok, "error": None if ok else "inference_error",
            "t_end_sent": 10.0, "t_final": 10.0 + latency, "t_first_audio": 1.0, "t_first_partial": 1.95,
            "final_inference_ms": 100.0, "final_text": text, "num_partials": 4,
            "final_timing": {"queue_ms": queue * 1000, "wait_partial_ms": 0.0, "decode_ms": 100.0},
            "partial_timing": [{"queue_ms": 5.0}]}


def _write_run(root, i, rnd, policy, latency, queue, text="HELLO", counters=None):
    d = root / f"{i:02d}_r{rnd}_{policy}"
    d.mkdir()
    for scen in ("c4", "c8", "recovery"):
        streams = [_stream(k, f"u{k % 2}", latency, queue, text) for k in range(10)]
        (d / f"{scen}.json").write_text(json.dumps({
            "streams": streams, "summary": {"throughput_streams_per_s": 0.5},
            "server_metrics_after": {"counters": counters or {}}}))
    return {"index": i, "round": rnd, "policy": policy, "dir": d.name, "stress_exit": 0, "failed_checks": []}


def test_analyze_groups_rounds_and_computes_deltas(tmp_path):
    entries = [
        _write_run(tmp_path, 1, 1, "base", 0.40, 0.25),
        _write_run(tmp_path, 2, 1, "prio", 0.20, 0.05, counters={"stream.partials_promoted_at_end": 2}),
        _write_run(tmp_path, 3, 2, "prio", 0.22, 0.06, counters={"stream.partials_promoted_at_end": 1}),
        _write_run(tmp_path, 4, 2, "base", 0.42, 0.26),
    ]
    (tmp_path / "manifest.json").write_text(json.dumps({
        "commit": "a" * 40, "common_settings": {}, "stress_args": {}, "order": entries}))
    a = analyze(tmp_path, {"u0": "hello", "u1": "hello"})
    g = a["groups"]
    assert set(g) == {"base c=4", "base c=8", "prio c=4", "prio c=8"}
    assert g["base c=4"]["ok"] == 40  # c4 + recovery, two rounds
    assert g["base c=8"]["ok"] == 20
    assert g["prio c=4"]["final_latency_s"]["p95"] == pytest.approx(0.22)
    assert g["prio c=4"]["final_text_identical_to_baseline"] == 1.0
    assert g["prio c=4"]["wer_v2"] == 0.0
    assert set(g["prio c=8"]["per_round_p95"]) == {"1", "2"}
    d = a["deltas"]["prio vs base c=8"]
    assert d["final_latency_s"]["diff_p95"] == pytest.approx(0.22 - 0.42)
    assert d["final_decode_s"]["diff_p95"] == 0  # control unchanged
    assert a["work_counters"]["prio"]["stream.partials_promoted_at_end"] == [2, 1]
    md = to_markdown(a)
    assert "| prio vs base c=8 |" in md and "Final text = base" in md


def test_text_identity_flags_a_changed_final(tmp_path):
    entries = [_write_run(tmp_path, 1, 1, "base", 0.4, 0.2, text="HELLO"),
               _write_run(tmp_path, 2, 1, "both", 0.3, 0.1, text="HULLO")]
    (tmp_path / "manifest.json").write_text(json.dumps({
        "commit": "b" * 40, "common_settings": {}, "stress_args": {}, "order": entries}))
    a = analyze(tmp_path, None)
    assert a["groups"]["both c=4"]["final_text_identical_to_baseline"] == 0.0
