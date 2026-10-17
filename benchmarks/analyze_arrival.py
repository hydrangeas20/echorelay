#!/usr/bin/env python3
"""Analysis for the registered staggered-arrival experiment (Phase 4).

    python -m benchmarks.analyze_arrival results/phase4 --audio-dir eval_audio5 \\
        --output results/phase4/analysis.json --markdown results/phase4/analysis.md

Registration (methodology, predictions, decision rules):
docs/experiments/phase4-staggered-arrival.md. This file implements those
rules exactly; changing them after data collection must be documented as
a deviation.

Units
* block: one 20-stream scenario (c8, c8_2, c8_3, recovery), starting
  from an idle server. Blocks are the unit of resampling.
* cell: arrival mode x policy. 2 runs x 4 blocks x 20 streams = 160.
* Comparisons are only ever made within one arrival mode.

Intervals (all EXPLORATORY)
* block bootstrap (decision interval): resample the 8 blocks of each arm
  with replacement, pool their streams, recompute the statistic;
  10,000 resamples, seed 0, percentile 95 % interval. Respects dependence
  within a block (shared queue, synchronized waves) but not between blocks
  of the same run or session, and 8 clusters is few: still optimistic.
* stream bootstrap (Phase 3 method, for continuity): streams resampled
  independently, 2,000 resamples, seed 0. Ignores dependence; narrower.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from echorelay.metrics import level_window

from .common import corpus_wer
from .compare_policies import bootstrap_p95_diff
from .end_state_breakdown import summaries

BLOCKS = ("c8", "c8_2", "c8_3", "recovery")  # files, in execution order: B1..B4
UNFINISHED = ("created", "queued", "running")
END_SYNC_WINDOW_S = 0.25   # "another end within 250 ms" (about two final decodes)
PHASE_WINDOW_S = 0.05      # "partial requests within 50 ms" (about one short partial decode)
BOOT_BLOCK_ITERS = 10_000
SEED = 0
TTFP_MAX_REGRESSION = 0.10
DECODE_CONTROL_TOLERANCE = 0.10


# ----------------------------------------------------------- per-stream data
def final_latency(s): return s["t_final"] - s["t_end_sent"]
def final_queue(s): return s["final_timing"]["queue_ms"] / 1000
def wait_own(s): return s["final_timing"]["wait_partial_ms"] / 1000
def final_decode(s): return s["final_inference_ms"] / 1000
def ttfp(s): return (s["t_first_partial"] - s["t_first_audio"]) if s.get("t_first_partial") is not None else None


METRICS = {"final_latency_s": final_latency, "final_queue_s": final_queue,
           "wait_own_partial_s": wait_own, "final_decode_s": final_decode, "ttfp_s": ttfp}


def vals(streams, f) -> list[float]:
    out = []
    for s in streams:
        if not s.get("ok"):
            continue
        try:
            v = f(s)
        except (KeyError, TypeError):
            continue
        if v is not None:
            out.append(v)
    return out


def p95(v):
    return float(np.percentile(v, 95)) if len(v) else None


def partial_request_times(s, chunk_ms: int, speed: float) -> list[float]:
    """When each sent partial was scheduled (client clock). A partial covering
    a seconds of audio was scheduled on the frame that made the buffer a s
    long, sent (round(a / chunk) - 1) chunks after the first frame."""
    if s.get("t_first_audio") is None or speed <= 0:
        return []
    chunk = chunk_ms / 1000
    return [s["t_first_audio"] + (round(a / chunk) - 1) * chunk / speed for a in s.get("partial_audio_s", [])]


# -------------------------------------------------------- block descriptors
def observed_concurrency(streams, scheduled: int) -> dict:
    """Client-side active streams, from [t_connect, t_done] intervals."""
    def end(s):
        return s["t_done"] if s.get("t_done") is not None else s.get("t_final")

    iv = [(s["t_connect"], end(s)) for s in streams if s.get("t_connect") is not None and end(s) is not None]
    if not iv:
        return {}
    ev = sorted([(a, 1) for a, _ in iv] + [(b, -1) for _, b in iv], key=lambda e: (e[0], e[1]))
    t_prev, level, area, at_sched, peak = ev[0][0], 0, 0.0, 0.0, 0
    for t, d in ev:
        area += level * (t - t_prev)
        at_sched += (t - t_prev) if level >= scheduled else 0.0
        t_prev = t
        level += d
        peak = max(peak, level)
    span = ev[-1][0] - ev[0][0]
    ends = [s["t_end_sent"] for s in streams if s.get("t_end_sent") is not None]
    at_end = [sum(a <= e < b for a, b in iv) for e in ends]
    return {"mean": area / span if span else None, "max": peak,
            "share_at_scheduled": at_sched / span if span else None,
            "mean_active_at_end": float(np.mean(at_end)) if at_end else None, "span_s": span}


def end_sync(streams) -> dict:
    ends = np.array([s["t_end_sent"] for s in streams if s.get("t_end_sent") is not None])
    if not ends.size:
        return {}
    nb = np.array([(np.abs(ends - e) <= END_SYNC_WINDOW_S).sum() - 1 for e in ends])
    return {"share_with_neighbour": float((nb > 0).mean()), "mean_neighbours": float(nb.mean())}


def phase_alignment(streams, chunk_ms: int, speed: float, interval_s: float = 1.0) -> dict:
    req = [(i, t) for i, s in enumerate(streams) for t in partial_request_times(s, chunk_ms, speed)]
    if not req:
        return {}
    owner = np.array([i for i, _ in req])
    t = np.array([x for _, x in req])
    nb = np.array([((np.abs(t - x) <= PHASE_WINDOW_S) & (owner != o)).sum() for o, x in req])
    period = interval_s / speed  # partials recur every interval_s of audio = interval_s / speed of wall time
    phi = 2 * math.pi * np.mod(t, period) / period
    return {"mean_neighbours": float(nb.mean()), "share_3plus_neighbours": float((nb >= 3).mean()),
            "phase_R": float(abs(np.exp(1j * phi).mean())), "requests": int(t.size)}


def server_levels(block: dict) -> dict:
    b, a = block.get("server_metrics_before"), block.get("server_metrics_after")
    if not (b and a and "levels" in b and "levels" in a):
        return {}
    return {k: level_window(b["levels"][k], a["levels"][k]) for k in a["levels"]}


def describe_block(block: dict, scheduled: int = 8) -> dict:
    s = block["streams"]
    cfg = block.get("run_config", {})
    chunk, speed = cfg.get("chunk_ms", 100), cfg.get("speed", 1.0)
    end_states = [x["_end"] for x in s if x.get("_end")]
    waiting = [e["waiting_at_end"] for e in end_states if e.get("waiting_at_end") is not None]
    return {
        "observed_concurrency": observed_concurrency(s, scheduled),
        "end_sync": end_sync(s),
        "phase_alignment": phase_alignment(s, chunk, speed),
        "server_levels": server_levels(block),
        "queue_depth_at_end": {"mean": float(np.mean(waiting)), "p95": p95(waiting),
                               "share_ge1": float(np.mean([w >= 1 for w in waiting]))} if waiting else {},
    }


# -------------------------------------------------------------- loading
class MissingEvidence(RuntimeError):
    """Required raw evidence is absent. The analysis refuses to run on an
    incomplete set rather than silently dropping the affected measurements
    (end states, workload blocks, WER)."""


def attach_end_states(streams, log: Path | None) -> None:
    # Fail-fast checks added after data collection (evidence repair). They
    # change no computed value when the evidence is complete.
    if log is None or not log.is_file():
        raise MissingEvidence(f"server log not found: {log}")
    summ = summaries(log)
    unmatched = []
    for s in streams:
        sid = (s.get("ready") or {}).get("session_id", "")
        k = summ.get(sid)
        if not k:
            if sid:  # a stream that got a session must have a stream_summary line
                unmatched.append(sid)
            continue
        w = k.get("inference_waiting_at_end")
        s["_end"] = {"state": k.get("partial_state_at_end"),
                     "promoted": k.get("partial_promoted_at_end") == "True",
                     "cancelled": k.get("partial_cancelled_at_end") == "True",
                     "waiting_at_end": int(w) if w not in (None, "None") else None}
    if unmatched:
        raise MissingEvidence(f"{log}: no stream_summary for {len(unmatched)} session(s), "
                              f"e.g. {unmatched[:3]}")


def load_references(audio_dir: Path, root: Path) -> dict[str, str]:
    """Reference transcripts, ``<audio_dir>/<utterance>.txt``. Every utterance
    played in the experiment must have one, or WER would silently cover only
    a subset (or nothing) and C4's WER comparison would be vacuous."""
    if not audio_dir.is_dir():
        raise MissingEvidence(f"--audio-dir not found: {audio_dir}")
    refs = {f.stem: f.read_text().strip() for f in audio_dir.glob("*.txt")}
    manifest = json.loads((root / "manifest.json").read_text())
    used = {s["utterance"] for e in manifest["order"] for name in BLOCKS
            if (root / e["dir"] / f"{name}.json").is_file()
            for s in json.loads((root / e["dir"] / f"{name}.json").read_text())["streams"]}
    missing = sorted(used - refs.keys())
    if missing:
        raise MissingEvidence(f"no reference transcript in {audio_dir} for {missing}")
    return refs


def load(root: Path) -> tuple[dict, list[dict]]:
    """Returns the manifest and a list of blocks with their cell metadata."""
    manifest = json.loads((root / "manifest.json").read_text())
    blocks = []
    for e in manifest["order"]:
        d = root / e["dir"]
        for pos, name in enumerate(BLOCKS, 1):
            f = d / f"{name}.json"
            if not f.is_file():
                raise MissingEvidence(f"block result not found: {f}")
            blk = json.loads(f.read_text())
            attach_end_states(blk["streams"], d / "server.log")
            blocks.append({**blk, "arrival": e.get("arrival", "sync"), "policy": e["policy"],
                           "round": e["round"], "run": e["dir"], "block_pos": pos})
    return manifest, blocks


# --------------------------------------------------------------- statistics
def block_bootstrap(base_blocks, cand_blocks, stat, iters=None, seed=SEED) -> dict:
    """Cluster bootstrap of stat(cand) - stat(base); each arm resampled by block."""
    iters = iters or BOOT_BLOCK_ITERS
    base_blocks = [np.asarray(b, dtype=float) for b in base_blocks if len(b)]
    cand_blocks = [np.asarray(b, dtype=float) for b in cand_blocks if len(b)]
    if len(base_blocks) < 2 or len(cand_blocks) < 2:
        return {"n_blocks_base": len(base_blocks), "n_blocks_cand": len(cand_blocks)}
    rng = np.random.default_rng(seed)
    point = stat(np.concatenate(cand_blocks)) - stat(np.concatenate(base_blocks))
    diffs = np.empty(iters)
    nb, nc = len(base_blocks), len(cand_blocks)
    for k in range(iters):
        c = np.concatenate([cand_blocks[i] for i in rng.integers(0, nc, nc)])
        b = np.concatenate([base_blocks[i] for i in rng.integers(0, nb, nb)])
        diffs[k] = stat(c) - stat(b)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(point), "ci95": [float(lo), float(hi)],
            "n_blocks_base": nb, "n_blocks_cand": nc}


def _p95_stat(v): return float(np.percentile(v, 95))
def _mean_stat(v): return float(np.mean(v)) if len(v) else 0.0


def cell_blocks(blocks, arrival, policy):
    return sorted([b for b in blocks if b["arrival"] == arrival and b["policy"] == policy],
                  key=lambda b: (b["round"], b["block_pos"]))


def per_stream(blocks, f):
    return [vals(b["streams"], f) for b in blocks]


def unfinished(s):
    e = s.get("_end")
    return None if not e else float(e["state"] in UNFINISHED)


def queued_at_end(s):
    e = s.get("_end")
    return None if not e else float(e["state"] == "queued")


def wait_own_if_unfinished(s):
    e = s.get("_end")
    return wait_own(s) if e and e["state"] in UNFINISHED else None


def cell_summary(blocks, refs) -> dict:
    streams = [s for b in blocks for s in b["streams"]]
    ok = [s for s in streams if s.get("ok")]
    out = {"blocks": len(blocks), "streams": len(streams), "ok": len(ok),
           "failures": len(streams) - len(ok),
           "errors": dict(Counter(s.get("error") or "unknown" for s in streams if not s.get("ok")))}
    for m, f in METRICS.items():
        v = vals(ok, f)
        out[m] = {"p50": float(np.median(v)) if v else None, "p95": p95(v),
                  "max": float(max(v)) if v else None, "mean": float(np.mean(v)) if v else None}
    out["partial_queue_s_p95"] = p95([t["queue_ms"] / 1000 for s in ok for t in s.get("partial_timing", [])])
    out["per_round_p95"] = {str(r): {m: p95(vals([s for b in blocks if b["round"] == r for s in b["streams"]], f))
                                     for m, f in METRICS.items()}
                            for r in sorted({b["round"] for b in blocks})}
    out["per_block_p95"] = [{"round": b["round"], "block": f"B{b['block_pos']}",
                             "final_latency_s": p95(vals(b["streams"], final_latency)),
                             "final_queue_s": p95(vals(b["streams"], final_queue)),
                             "ttfp_s": p95(vals(b["streams"], ttfp))} for b in blocks]
    ends = [s["_end"] for s in ok if s.get("_end")]
    out["end_state"] = {
        "n_joined": len(ends),
        "share_unfinished": float(np.mean([e["state"] in UNFINISHED for e in ends])) if ends else None,
        "share_queued": float(np.mean([e["state"] == "queued" for e in ends])) if ends else None,
        "counts": dict(Counter(e["state"] for e in ends)),
        "promoted": sum(e["promoted"] for e in ends), "cancelled": sum(e["cancelled"] for e in ends),
        "mean_wait_own_s": _mean_stat(vals(ok, wait_own)),
        "mean_wait_own_unfinished_s": _mean_stat(vals(ok, wait_own_if_unfinished)),
    }
    if refs:
        scored = [s for s in ok if s["utterance"] in refs]
        out["wer_v2"] = corpus_wer([refs[s["utterance"]] for s in scored],
                                   [s.get("final_text") or "" for s in scored], "v2")
    return out


def describe_cell(blocks) -> dict:
    d = [describe_block(b) for b in blocks]

    def mean_of(path):
        xs = []
        for x in d:
            for k in path:
                x = (x or {}).get(k)
            if isinstance(x, (int, float)):
                xs.append(x)
        return float(np.mean(xs)) if xs else None

    keys = {
        "observed_active_mean": ("observed_concurrency", "mean"),
        "observed_share_time_at_8": ("observed_concurrency", "share_at_scheduled"),
        "observed_active_at_end": ("observed_concurrency", "mean_active_at_end"),
        "end_share_with_neighbour_250ms": ("end_sync", "share_with_neighbour"),
        "end_mean_neighbours_250ms": ("end_sync", "mean_neighbours"),
        "partial_mean_neighbours_50ms": ("phase_alignment", "mean_neighbours"),
        "partial_share_3plus_neighbours": ("phase_alignment", "share_3plus_neighbours"),
        "partial_phase_R": ("phase_alignment", "phase_R"),
        "server_active_mean": ("server_levels", "streams.active", "mean"),
        "server_queue_depth_mean": ("server_levels", "inference.waiting", "mean"),
        "server_queue_share_ge1": ("server_levels", "inference.waiting", "share_at_or_above", "1"),
        "server_queue_max": ("server_levels", "inference.waiting", "max_level_seen"),
        "server_in_flight_mean": ("server_levels", "inference.in_flight", "mean"),
        "queue_depth_at_end_mean": ("queue_depth_at_end", "mean"),
    }
    return {"block_mean": {k: mean_of(p) for k, p in keys.items()}, "per_block": d}


# ------------------------------------------------------------- decisions
def compare(blocks, arrival, base, cand, refs, base_summary, cand_summary) -> dict:
    bb, cb = cell_blocks(blocks, arrival, base), cell_blocks(blocks, arrival, cand)
    out = {"deltas": {}, "stream_bootstrap": {}, "paired_blocks": {}}
    for m, f in METRICS.items():
        out["deltas"][m] = block_bootstrap(per_stream(bb, f), per_stream(cb, f), _p95_stat)
        out["stream_bootstrap"][m] = bootstrap_p95_diff(
            [v for x in per_stream(bb, f) for v in x], [v for x in per_stream(cb, f) for v in x])
    pairs = {(b["round"], b["block_pos"]): b for b in bb}
    better, total = 0, 0
    for c in cb:
        b = pairs.get((c["round"], c["block_pos"]))
        if b:
            total += 1
            better += p95(vals(c["streams"], final_latency)) < p95(vals(b["streams"], final_latency))
    out["paired_blocks"] = {"candidate_lower_p95": better, "pairs": total}
    out["mechanism"] = {
        "share_unfinished": block_bootstrap(per_stream(bb, unfinished), per_stream(cb, unfinished), _mean_stat),
        "share_queued": block_bootstrap(per_stream(bb, queued_at_end), per_stream(cb, queued_at_end), _mean_stat),
        "mean_wait_own_s": block_bootstrap(per_stream(bb, wait_own), per_stream(cb, wait_own), _mean_stat),
    }
    rounds = sorted({b["round"] for b in bb})
    rnd = lambda bl, r, f: [v for b in bl if b["round"] == r for v in vals(b["streams"], f)]  # noqa: E731
    out["per_round"] = {str(r): {
        "final_latency_p95": (p95(rnd(bb, r, final_latency)), p95(rnd(cb, r, final_latency))),
        "share_unfinished": (_mean_stat(rnd(bb, r, unfinished)), _mean_stat(rnd(cb, r, unfinished))),
        "mean_wait_own_s": (_mean_stat(rnd(bb, r, wait_own)), _mean_stat(rnd(cb, r, wait_own)))}
        for r in rounds}
    out["verdict"] = adoption_verdict(out, rounds, base_summary, cand_summary, bb, cb)
    out["mechanism_verdict"] = mechanism_verdict(out, rounds)
    return out


def text_identity(base_blocks, cand_blocks) -> float | None:
    modal = defaultdict(Counter)
    for b in base_blocks:
        for s in b["streams"]:
            if s.get("ok"):
                modal[s["utterance"]][s.get("final_text")] += 1
    ref = {u: c.most_common(1)[0][0] for u, c in modal.items()}
    same = [s.get("final_text") == ref.get(s["utterance"]) for b in cand_blocks for s in b["streams"]
            if s.get("ok") and s["utterance"] in ref]
    return float(np.mean(same)) if same else None


def adoption_verdict(cmp, rounds, bs, cs, bb, cb) -> dict:
    d = cmp["deltas"]["final_latency_s"]
    ident = text_identity(bb, cb)
    wer_equal = (bs.get("wer_v2") is None or cs.get("wer_v2") is None
                 or abs(bs["wer_v2"] - cs["wer_v2"]) < 1e-12)
    ttfp_ratio = (cs["ttfp_s"]["p95"] / bs["ttfp_s"]["p95"] - 1) if bs["ttfp_s"]["p95"] else None
    dec_ratio = (cs["final_decode_s"]["p95"] / bs["final_decode_s"]["p95"] - 1) if bs["final_decode_s"]["p95"] else None
    crit = {
        "C1 block-bootstrap interval for final latency p95 difference entirely below 0":
            bool(d.get("ci95") and d["ci95"][1] < 0),
        "C2 candidate final latency p95 below baseline in every round":
            all(cmp["per_round"][str(r)]["final_latency_p95"][1] < cmp["per_round"][str(r)]["final_latency_p95"][0]
                for r in rounds),
        "C3 zero failed candidate streams": cs["failures"] == 0,
        "C4 final text 100 % identical to baseline and WER v2 equal": ident == 1.0 and wer_equal,
        "C5 TTFP p95 regression <= 10 %": ttfp_ratio is not None and ttfp_ratio <= TTFP_MAX_REGRESSION,
        "C6 final decode p95 within +/-10 % (control)":
            dec_ratio is not None and abs(dec_ratio) <= DECODE_CONTROL_TOLERANCE,
    }
    ok = all(crit.values())
    return {"criteria": crit, "text_identity": ident, "ttfp_p95_change": ttfp_ratio,
            "decode_p95_change": dec_ratio,
            "outcome": "benefit demonstrated under this workload" if ok
            else "benefit not demonstrated under this workload",
            "failed": [k for k, v in crit.items() if not v]}


def mechanism_verdict(cmp, rounds) -> dict:
    out = {}
    for key, label in (("share_unfinished", "frequency"), ("mean_wait_own_s", "duration")):
        d = cmp["mechanism"][key]
        lo_pos = bool(d.get("ci95") and d["ci95"][0] > 0)
        all_rounds = all(cmp["per_round"][str(r)][key][1] > cmp["per_round"][str(r)][key][0] for r in rounds)
        out[label] = {"supported": lo_pos and all_rounds, "interval_above_0": lo_pos,
                      "higher_in_every_round": all_rounds}
    return out


def analyze(root: Path, refs: dict | None, baseline: str = "base") -> dict:
    manifest, blocks = load(root)
    result = {"manifest": {k: manifest.get(k) for k in ("commit", "plan", "plan_constants", "deviations",
                                                        "common_settings")},
              "cells": {}, "workload": {}, "comparisons": {}}
    arrivals = sorted({b["arrival"] for b in blocks})
    policies = sorted({b["policy"] for b in blocks})
    summ = {}
    for a in arrivals:
        for p in policies:
            bl = cell_blocks(blocks, a, p)
            summ[(a, p)] = cell_summary(bl, refs)
            result["cells"][f"{a} {p}"] = summ[(a, p)]
            result["workload"][f"{a} {p}"] = describe_cell(bl)
    for a in arrivals:
        for p in policies:
            if p != baseline:
                result["comparisons"][f"{a}: {p} vs {baseline}"] = compare(
                    blocks, a, baseline, p, refs, summ[(a, baseline)], summ[(a, p)])
    return result


# --------------------------------------------------------------- report
def _ms(v):
    return "–" if v is None else f"{v * 1000:.0f}"


def _f(v, fmt):
    return "–" if v is None else format(v, fmt)


def to_markdown(r: dict) -> str:
    m = r["manifest"]
    L = ["# Phase 4 staggered-arrival analysis", "",
         f"Commit `{(m.get('commit') or '?')[:12]}`; plan `{m.get('plan')}`; "
         f"deviations from the registered plan: {m.get('deviations') or 'none'}.", "",
         "All intervals are exploratory (see docs/experiments/phase4-staggered-arrival.md).", "",
         "## Workload descriptors (mean over blocks)", "",
         "| Cell | Active (client) mean / share at 8 | Active at end | Ends ±250 ms share / mean | "
         "Partial nbrs ±50 ms / share ≥3 / phase R | Server queue depth mean / share ≥1 / max | "
         "Queue depth at end |", "|---|---|---|---|---|---|---|"]
    for cell, w in r["workload"].items():
        x = w["block_mean"]
        L.append(f"| {cell} | {_f(x['observed_active_mean'], '.2f')} / {_f(x['observed_share_time_at_8'], '.0%')} | "
                 f"{_f(x['observed_active_at_end'], '.2f')} | {_f(x['end_share_with_neighbour_250ms'], '.0%')} / "
                 f"{_f(x['end_mean_neighbours_250ms'], '.2f')} | {_f(x['partial_mean_neighbours_50ms'], '.2f')} / "
                 f"{_f(x['partial_share_3plus_neighbours'], '.0%')} / {_f(x['partial_phase_R'], '.2f')} | "
                 f"{_f(x['server_queue_depth_mean'], '.2f')} / {_f(x['server_queue_share_ge1'], '.0%')} / "
                 f"{_f(x['server_queue_max'], '.0f')} | {_f(x['queue_depth_at_end_mean'], '.2f')} |")
    L += ["", "## Latency by cell (pooled over 8 blocks; ms)", "",
          "| Cell | n | Fail | Final lat p50 / p95 / max | Final queue p95 | Wait own p95 / max | "
          "Final decode p95 | TTFP p50 / p95 | Partial queue p95 | WER v2 | Unfinished own partial at end | "
          "Mean wait own (all / unfinished) |", "|---|---:|---:|---|---:|---|---:|---|---:|---:|---|---|"]
    for cell, c in r["cells"].items():
        e = c["end_state"]
        L.append(f"| {cell} | {c['ok']} | {c['failures']} | {_ms(c['final_latency_s']['p50'])} / "
                 f"{_ms(c['final_latency_s']['p95'])} / {_ms(c['final_latency_s']['max'])} | "
                 f"{_ms(c['final_queue_s']['p95'])} | {_ms(c['wait_own_partial_s']['p95'])} / "
                 f"{_ms(c['wait_own_partial_s']['max'])} | {_ms(c['final_decode_s']['p95'])} | "
                 f"{_ms(c['ttfp_s']['p50'])} / {_ms(c['ttfp_s']['p95'])} | {_ms(c['partial_queue_s_p95'])} | "
                 f"{_f(c.get('wer_v2'), '.2%')} | {_f(e['share_unfinished'], '.1%')} ({e['counts']}) | "
                 f"{_ms(e['mean_wait_own_s'])} / {_ms(e['mean_wait_own_unfinished_s'])} |")
    L += ["", "## Final latency p95 per block (ms)", "",
          "| Cell | " + " | ".join(f"r{r} B{b}" for r in (1, 2) for b in (1, 2, 3, 4)) + " | per round |",
          "|---|" + "---:|" * 9]
    for cell, c in r["cells"].items():
        by = {(x["round"], x["block"]): x["final_latency_s"] for x in c["per_block_p95"]}
        cells = [_ms(by.get((rr, f"B{b}"))) for rr in (1, 2) for b in (1, 2, 3, 4)]
        pr = " / ".join(_ms(v["final_latency_s"]) for v in c["per_round_p95"].values())
        L.append(f"| {cell} | " + " | ".join(cells) + f" | {pr} |")
    L += ["", "## Comparisons (within arrival mode)", ""]
    for name, cmp in r["comparisons"].items():
        v = cmp["verdict"]
        L += [f"### {name}", "", f"**Outcome: {v['outcome']}.**" + (f" Failed: {'; '.join(v['failed'])}."
                                                                    if v["failed"] else ""), "",
              "| Metric | Δ p95 (ms) | block bootstrap 95 % | stream bootstrap 95 % |", "|---|---:|---|---|"]
        for met in METRICS:
            d, s = cmp["deltas"][met], cmp["stream_bootstrap"][met]
            bi = f"[{d['ci95'][0]*1000:+.0f}, {d['ci95'][1]*1000:+.0f}]" if "ci95" in d else "–"
            si = f"[{s['ci95'][0]*1000:+.0f}, {s['ci95'][1]*1000:+.0f}]" if "ci95" in s else "–"
            L.append(f"| {met} | {d.get('diff', 0)*1000:+.0f} | {bi} | {si} |")
        pb = cmp["paired_blocks"]
        L += ["", f"Paired blocks (same round and position) with lower candidate p95: "
                  f"{pb['candidate_lower_p95']} of {pb['pairs']}.", ""]
        mv = cmp["mechanism_verdict"]
        mech = cmp["mechanism"]
        L.append("Own-partial mechanism (candidate − baseline): " + "; ".join(
            f"{k} {mech[key]['diff']*(100 if key.startswith('share') else 1000):+.1f}"
            f"{'pp' if key.startswith('share') else ' ms'} "
            f"[{mech[key]['ci95'][0]*(100 if key.startswith('share') else 1000):+.1f}, "
            f"{mech[key]['ci95'][1]*(100 if key.startswith('share') else 1000):+.1f}]"
            for k, key in (("unfinished share", "share_unfinished"), ("queued share", "share_queued"),
                           ("mean wait own", "mean_wait_own_s")) if "ci95" in mech[key])
            + (f". Registered mechanism test: frequency increase supported: {mv['frequency']['supported']}; "
               f"duration increase supported: {mv['duration']['supported']}." if name.split(": ")[1].startswith("prio")
               else ". (Descriptive; the registered mechanism test is prio vs base.)"))
        L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root")
    p.add_argument("--audio-dir")
    p.add_argument("--output")
    p.add_argument("--markdown")
    a = p.parse_args()
    try:
        refs = load_references(Path(a.audio_dir), Path(a.root)) if a.audio_dir else None
        if refs is None:
            print("warning: no --audio-dir; WER is not computed and the C4 WER-equality check "
                  "has nothing to compare", file=sys.stderr)
        r = analyze(Path(a.root), refs)
    except MissingEvidence as exc:
        sys.exit(f"error: incomplete evidence: {exc}")
    md = to_markdown(r)
    print(md)
    if a.output:
        Path(a.output).write_text(json.dumps(r, indent=2, default=str) + "\n")
    if a.markdown:
        Path(a.markdown).write_text(md)


if __name__ == "__main__":
    main()
