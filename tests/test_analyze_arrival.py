"""Phase 4 plan, analysis and simulator: registered order balance, block
bootstrap, workload descriptors, decision rules, simulator consistency."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from benchmarks import analyze_arrival as aa
from benchmarks.arrival_sim import DUR, simulate_block, simulate_cell
from benchmarks.load_test import first_wave_offsets
from benchmarks.matched_comparison import PHASE4, PHASE4_ORDER


@pytest.fixture(autouse=True)
def fewer_resamples(monkeypatch):
    monkeypatch.setattr(aa, "BOOT_BLOCK_ITERS", 1000)  # registered value is 10,000


# ------------------------------------------------------------- the plan
def test_phase4_constants_are_the_registered_values():
    assert PHASE4["policies"] == ["base", "prio", "both"]
    assert PHASE4["arrivals"] == ["sync", "stagger"]
    assert PHASE4["concurrency"] == [8, 8, 8] and PHASE4["streams"] == 20
    assert PHASE4["stagger_s"] == 1.125 and PHASE4["speed"] == 1.0 and PHASE4["chunk_ms"] == 100


def test_phase4_order_is_balanced():
    assert len(PHASE4_ORDER) == 12
    pos = defaultdict(list)
    for i, (rnd, arr, pol) in enumerate(PHASE4_ORDER, 1):
        pos[(arr, pol)].append(i)
        assert rnd == (1 if i <= 6 else 2)
    assert len(pos) == 6
    assert all(len(v) == 2 and sum(v) == 13 for v in pos.values())  # mean position 6.5
    for arr in ("sync", "stagger"):  # within-mode balance
        assert {sum(pos[(arr, p)]) for p in PHASE4["policies"]} == {13}
    arrivals = [a for _, a, _ in PHASE4_ORDER]
    repeats = [i for i in range(1, 12) if arrivals[i] == arrivals[i - 1]]
    assert repeats == [6]  # only the mirror point repeats a mode


# ---------------------------------------------------------- statistics
def test_block_bootstrap_detects_shift_and_respects_clusters():
    rng = np.random.default_rng(1)
    base = [list(0.3 + 0.05 * rng.standard_normal(20)) for _ in range(8)]
    assert aa.BOOT_BLOCK_ITERS == 1000
    same = aa.block_bootstrap(base, [list(b) for b in base], aa._p95_stat, iters=2000)
    assert same["diff"] == 0 and same["ci95"][0] <= 0 <= same["ci95"][1]
    faster = aa.block_bootstrap(base, [[v - 0.1 for v in b] for b in base], aa._p95_stat, iters=2000)
    assert faster["diff"] == pytest.approx(-0.1) and faster["ci95"][1] < 0
    # one extreme block: cluster resampling widens the interval vs stream resampling
    outlier = [list(b) for b in base]
    outlier[0] = [v + 0.5 for v in outlier[0]]
    wide = aa.block_bootstrap(base, outlier, aa._p95_stat, iters=2000)
    assert wide["ci95"][1] - wide["ci95"][0] > 0.1
    assert aa.block_bootstrap(base[:1], base[:1], aa._p95_stat) == {"n_blocks_base": 1, "n_blocks_cand": 1}


# --------------------------------------------------------- descriptors
def _s(t0, dur, **kw):
    return {"ok": True, "t_connect": t0 - 0.003, "t_first_audio": t0, "t_end_sent": t0 + dur,
            "t_final": t0 + dur + 0.1, "t_done": t0 + dur + 0.101, "partial_audio_s": [], **kw}


def test_end_sync_counts_neighbours_within_250ms():
    streams = [_s(0, 5.0), _s(0, 5.2), _s(0, 9.0)]
    d = aa.end_sync(streams)
    assert d["share_with_neighbour"] == pytest.approx(2 / 3) and d["mean_neighbours"] == pytest.approx(2 / 3)


def test_phase_alignment_aligned_vs_spread():
    aligned = [_s(0.0, 3.0, partial_audio_s=[1.0, 2.0, 3.0]) for _ in range(4)]
    spread = [_s(0.25 * k, 3.0, partial_audio_s=[1.0, 2.0, 3.0]) for k in range(4)]
    a = aa.phase_alignment(aligned, 100, 1.0)
    s = aa.phase_alignment(spread, 100, 1.0)
    assert a["mean_neighbours"] == 3 and a["phase_R"] == pytest.approx(1.0)
    assert s["mean_neighbours"] == 0 and s["phase_R"] < 0.01
    # request time = first frame + (frames - 1) * chunk: 1.0 s of audio -> 0.9 s
    assert aa.partial_request_times(aligned[0], 100, 1.0)[0] == pytest.approx(0.9)
    assert aa.partial_request_times(aligned[0], 100, 4.0)[0] == pytest.approx(0.225)
    # at speed 4 partials recur every 0.25 s of wall time: phases use that period
    fast = [_s(0.0625 * k, 3.0, partial_audio_s=[1.0, 2.0, 3.0]) for k in range(4)]
    assert aa.phase_alignment(fast, 100, 4.0)["phase_R"] < 0.01


def test_observed_concurrency_time_weighted():
    streams = [_s(0, 1.0), _s(0, 3.0)]  # 2 active for ~1.1 s, then 1 active
    d = aa.observed_concurrency(streams, scheduled=2)
    assert d["max"] == 2 and 1.0 < d["mean"] < 2.0
    assert d["share_at_scheduled"] == pytest.approx(1.104 / 3.104, abs=0.01)
    assert d["mean_active_at_end"] == pytest.approx(1.5)


# -------------------------------------------------------- decision rules
def _block(arrival, policy, rnd, pos, lat, text="T", decode=0.1, ttfp=0.95, state="done", wait=0.0, fail=0):
    streams = []
    for i in range(20):
        s = {"ok": i >= fail, "error": None if i >= fail else "inference_error",
             "utterance": f"u{i % 5}", "final_text": text, "t_end_sent": 10.0, "t_final": 10.0 + lat[i],
             "t_first_audio": 1.0, "t_first_partial": 1.0 + ttfp, "final_inference_ms": decode * 1000,
             "final_timing": {"queue_ms": 0.0, "wait_partial_ms": wait * 1000, "decode_ms": decode * 1000},
             "partial_timing": [], "_end": {"state": state, "promoted": False, "cancelled": False,
                                            "waiting_at_end": 0}}
        streams.append(s)
    return {"streams": streams, "arrival": arrival, "policy": policy, "round": rnd, "block_pos": pos,
            "run_config": {"chunk_ms": 100, "speed": 1.0}}


def _cells(base_lat, cand_lat, **cand_kw):
    blocks = []
    for rnd in (1, 2):
        for pos in (1, 2, 3, 4):
            blocks.append(_block("stagger", "base", rnd, pos, base_lat(rnd, pos)))
            blocks.append(_block("stagger", "both", rnd, pos, cand_lat(rnd, pos), **cand_kw))
    return blocks


def _verdict(blocks):
    bs = aa.cell_summary(aa.cell_blocks(blocks, "stagger", "base"), None)
    cs = aa.cell_summary(aa.cell_blocks(blocks, "stagger", "both"), None)
    return aa.compare(blocks, "stagger", "base", "both", None, bs, cs)


def _lat(center, rnd, pos):
    rng = np.random.default_rng(rnd * 10 + pos)
    return list(center + 0.02 * rng.standard_normal(20))


def test_verdict_demonstrated_when_all_criteria_hold():
    v = _verdict(_cells(lambda r, p: _lat(0.5, r, p), lambda r, p: _lat(0.3, r, p)))["verdict"]
    assert v["outcome"] == "benefit demonstrated under this workload" and v["failed"] == []


def test_verdict_not_demonstrated_without_a_difference():
    v = _verdict(_cells(lambda r, p: _lat(0.3, r, p), lambda r, p: _lat(0.3, r + 5, p)))["verdict"]
    assert v["outcome"] == "benefit not demonstrated under this workload"
    assert any(k.startswith("C1") for k in v["failed"])


@pytest.mark.parametrize("kw, criterion", [
    ({"text": "DIFFERENT"}, "C4"), ({"fail": 1}, "C3"), ({"ttfp": 1.2}, "C5"), ({"decode": 0.2}, "C6")])
def test_each_guard_criterion_can_fail(kw, criterion):
    v = _verdict(_cells(lambda r, p: _lat(0.5, r, p), lambda r, p: _lat(0.3, r, p), **kw))["verdict"]
    assert v["outcome"] == "benefit not demonstrated under this workload"
    assert [k[:2] for k in v["failed"]] == [criterion]


def test_round_criterion_requires_every_round():
    v = _verdict(_cells(lambda r, p: _lat(0.5, r, p),
                        lambda r, p: _lat(0.3 if r == 1 else 0.6, r, p)))["verdict"]
    assert any(k.startswith("C2") for k in v["failed"])


def test_mechanism_verdict_frequency_and_duration():
    blocks = []
    for rnd in (1, 2):
        for pos in (1, 2, 3, 4):
            blocks.append(_block("sync", "base", rnd, pos, [0.3] * 20))
            b = _block("sync", "prio", rnd, pos, [0.3] * 20)
            for s in b["streams"][:5]:
                s["_end"]["state"] = "queued"
                s["final_timing"]["wait_partial_ms"] = 200.0
            blocks.append(b)
    bs = aa.cell_summary(aa.cell_blocks(blocks, "sync", "base"), None)
    cs = aa.cell_summary(aa.cell_blocks(blocks, "sync", "prio"), None)
    m = aa.compare(blocks, "sync", "base", "prio", None, bs, cs)
    assert m["mechanism_verdict"]["frequency"]["supported"] is True
    assert m["mechanism_verdict"]["duration"]["supported"] is True
    assert m["mechanism"]["share_unfinished"]["diff"] == pytest.approx(0.25)


# ------------------------------------------------------------ simulator
def test_simulator_schedules_partials_like_the_harness():
    """Without queueing (one stream), a stream gets floor(duration) partials,
    as every Phase 1-3 stream did, at 1.0, 2.0, ... s of audio."""
    blk = simulate_block("base", [0.0], np.random.default_rng(0), n=5)
    for s in blk["streams"]:
        assert len(s["partial_audio_s"]) == int(s["audio_s"])
        assert s["partial_audio_s"][:2] == pytest.approx([1.0, 2.0])
        assert s["final_timing"]["queue_ms"] == pytest.approx(0.0)


def test_simulator_is_deterministic_and_policy_ordered():
    a = simulate_cell("sync", "base", blocks=2, seed=3)
    b = simulate_cell("sync", "base", blocks=2, seed=3)
    assert [s["t_final"] for x in a for s in x["streams"]] == [s["t_final"] for x in b for s in x["streams"]]
    p95 = {p: aa.p95(aa.vals([s for x in simulate_cell("sync", p, 4, seed=3) for s in x["streams"]],
                             aa.final_latency)) for p in ("base", "both")}
    assert p95["both"] < p95["base"]


def test_simulator_stagger_spreads_partial_phases():
    sync = aa.describe_cell(simulate_cell("sync", "base", 2, seed=0))["block_mean"]
    stag = aa.describe_cell(simulate_cell("stagger", "base", 2, seed=0))["block_mean"]
    assert stag["partial_mean_neighbours_50ms"] < sync["partial_mean_neighbours_50ms"] / 3
    assert stag["end_share_with_neighbour_250ms"] < sync["end_share_with_neighbour_250ms"]
    last = first_wave_offsets(8, "stagger", 1.125)[7]
    assert last == pytest.approx(7.875) and last < max(DUR)
