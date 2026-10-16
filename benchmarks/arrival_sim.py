#!/usr/bin/env python3
"""Simulator used to register Phase 4 predictions (no GPU, no server).

    python -m benchmarks.arrival_sim --validate-phase3     # model vs Phase 3 measurements
    python -m benchmarks.arrival_sim --predict-phase4      # registered predictions

Models one block exactly as the harness runs it: closed-loop client (8
slots, stream i plays file i % 5, a slot restarts when its final arrives),
100 ms frames at real time, partial scheduling as in StreamSession (due
every 1.0 s of audio, at most one per stream in flight), end of stream
after the last frame, and one inference worker serving requests under
base (FIFO), prio (finals first, own queued partial promoted, final
submitted at end) or both (prio + own not-started partial cancelled).

Decode cost is linear in audio length with multiplicative log-normal
noise, fitted on the Phase 3 raw data (policy-independent):
partial 28.6 + 6.12 ms/s (sd 0.138), final 35.9 + 7.19 ms/s (sd 0.101).
Client overheads from the same data: final -> next first frame 4.8 ms.

It omits GPU clock changes, Python/GIL scheduling, network jitter and the
real decode-time dependence on content. Its output is a model, not a
measurement. It emits stream records in the load_test format, so workload
descriptors and latency metrics are computed by the same code
(benchmarks.analyze_arrival) as for the real experiment.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from . import analyze_arrival as aa
from .load_test import first_wave_offsets

DUR = (10.885, 11.13, 11.475, 4.755, 4.225)  # Phase 1-4 evaluation files, sorted by id
NAMES = tuple(f"1272-135031-000{i}" for i in range(5))
CONNECT_S = 0.003   # connect + ready before the first frame (first wave)
RESTART_S = 0.0048  # client receives final -> next stream's first frame
NET_S = 0.0015      # server finishes decode -> client receives the event
ARRIVAL_SEED = {"sync": 0, "stagger": 1}  # fixed (str hash is randomized per process)


@dataclass(frozen=True)
class Costs:
    pa: float = 28.6
    pb: float = 6.12
    ps: float = 0.138
    fa: float = 35.9
    fb: float = 7.19
    fs: float = 0.101


class _Levels:
    def __init__(self) -> None:
        self.level, self.t, self.at = 0, 0.0, defaultdict(float)

    def set(self, t: float, level: int) -> None:
        self.at[self.level] += t - self.t
        self.t, self.level = t, level

    def snapshot(self) -> dict:
        return {"time_at_level_s": {str(k): v for k, v in self.at.items()}}


def simulate_block(policy: str, offsets: list[float], rng: np.random.Generator, n: int = 20,
                   costs: Costs = Costs(), chunk: float = 0.1, interval: float = 1.0,
                   min_partial: float = 0.5, block_id: str = "b") -> dict:
    prio, cancel = policy in ("prio", "both"), policy == "both"
    c = len(offsets)
    ev: list = []
    seq = itertools.count()
    queue: list[dict] = []
    worker: dict | None = None
    lv = {k: _Levels() for k in ("streams.active", "inference.in_flight", "inference.waiting")}
    streams: list[dict] = []
    next_idx = 0

    def push(t, kind, data):
        heapq.heappush(ev, (t, next(seq), kind, data))

    def cost(kind, audio):
        a, b, sd = (costs.pa, costs.pb, costs.ps) if kind == "partial" else (costs.fa, costs.fb, costs.fs)
        return (a + b * audio) / 1000 * math.exp(rng.normal(0, sd) - sd * sd / 2)

    def enqueue(t, job):
        job["t_req"], job["state"] = t, "queued"
        job["key"] = (0 if (prio and job["kind"] == "final") else 1, next(seq))
        queue.append(job)
        lv["inference.waiting"].set(t, len(queue))
        dispatch(t)

    def dispatch(t):
        nonlocal worker
        if worker is not None or not queue:
            return
        job = min(queue, key=lambda j: j["key"])
        queue.remove(job)
        lv["inference.waiting"].set(t, len(queue))
        job["state"], job["t_start"] = "running", t
        worker = job
        lv["inference.in_flight"].set(t, 1)
        push(t + job["cost"], "done", job)

    def start_stream(t_first):
        nonlocal next_idx
        if next_idx >= n:
            return
        i = next_idx
        next_idx += 1
        d = DUR[i % 5]
        s = {"i": i, "d": d, "N": math.ceil(d / chunk - 1e-9), "t_first": t_first, "last": 0.0,
             "partial": None, "partials": [], "ended": False, "final_pending": False, "rec": {
                 "index": i, "utterance": NAMES[i % 5], "audio_s": d, "ok": True,
                 "t_connect": t_first - CONNECT_S, "t_first_audio": t_first, "t_first_partial": None,
                 "partial_audio_s": [], "partial_timing": [], "final_text": NAMES[i % 5],
                 "ready": {"session_id": f"{block_id}-{i}"}}}
        streams.append(s)
        active = sum(1 for x in streams if x["rec"].get("t_done") is None)
        lv["streams.active"].set(t_first - CONNECT_S, active)
        push(t_first, "frame", (s, 0))

    def submit_final(t, s):
        s["final"] = {"kind": "final", "s": s, "audio": s["d"], "cost": cost("final", s["d"])}
        s["t_final_submit"] = t
        enqueue(t, s["final"])

    for k in range(c):
        start_stream(offsets[k] + CONNECT_S)

    while ev:
        t, _, kind, data = heapq.heappop(ev)
        if kind == "frame":
            s, j = data
            b = min((j + 1) * chunk, s["d"])
            p = s["partial"]
            free = p is None or p["state"] in ("done", "cancelled")
            if free and b >= min_partial and b - s["last"] >= interval - 1e-9:
                s["last"] = b
                s["partial"] = {"kind": "partial", "s": s, "audio": b, "cost": cost("partial", b)}
                enqueue(t, s["partial"])
            if j + 1 < s["N"]:
                push(s["t_first"] + (j + 1) * chunk, "frame", (s, j + 1))
            else:
                push(s["t_first"] + s["N"] * chunk, "end", s)
        elif kind == "end":
            s = data
            s["ended"], s["t_end"] = True, t
            r = s["rec"]
            r["t_end_sent"] = t
            p = s["partial"]
            state = "none" if p is None else {"queued": "queued", "running": "running"}.get(p["state"], "done")
            end = {"state": state, "promoted": False, "cancelled": False, "waiting_at_end": len(queue)}
            r["_end"] = end
            s["wait_own"] = 0.0
            if state == "queued" and cancel:
                queue.remove(p)
                p["state"] = "cancelled"
                lv["inference.waiting"].set(t, len(queue))
                end["cancelled"] = True
                state = "done"
            elif state == "queued" and prio:
                p["key"] = (0, next(seq))
                end["promoted"] = True
            if prio:
                submit_final(t, s)
            elif state in ("queued", "running"):
                s["final_pending"] = True
            else:
                submit_final(t, s)
        elif kind == "done":
            job = data
            s = job["s"]
            job["state"] = "done"
            worker = None
            lv["inference.in_flight"].set(t, 0)
            r = s["rec"]
            if job["kind"] == "partial":
                r["partial_audio_s"].append(job["audio"])
                r["partial_timing"].append({"queue_ms": (job["t_start"] - job["t_req"]) * 1000})
                if r["t_first_partial"] is None:
                    r["t_first_partial"] = t + NET_S
                if s["ended"]:
                    s["wait_own"] = t - s["t_end"]
                    if s["final_pending"]:
                        s["final_pending"] = False
                        submit_final(t, s)
            else:
                r["t_final"] = t + NET_S
                r["final_inference_ms"] = job["cost"] * 1000
                r["final_audio_s"] = s["d"]
                r["final_timing"] = {"queue_ms": (job["t_start"] - s["t_final_submit"]) * 1000,
                                     "wait_partial_ms": s["wait_own"] * 1000,
                                     "decode_ms": job["cost"] * 1000}
                r["t_done"] = t + NET_S + 0.001
                active = sum(1 for x in streams if x["rec"].get("t_done") is None)
                lv["streams.active"].set(r["t_done"], active)
                start_stream(t + NET_S + RESTART_S)
            dispatch(t)

    zero = {k: {"time_at_level_s": {}} for k in lv}
    return {"streams": [s["rec"] for s in sorted(streams, key=lambda s: s["i"])],
            "run_config": {"chunk_ms": int(chunk * 1000), "speed": 1.0},
            "server_metrics_before": {"levels": zero},
            "server_metrics_after": {"levels": {k: v.snapshot() for k, v in lv.items()}}}


def simulate_cell(arrival: str, policy: str, blocks: int, seed: int, stagger_s: float = 1.125) -> list[dict]:
    rng = np.random.default_rng(seed)
    offs = first_wave_offsets(8, arrival, stagger_s)
    out = []
    for b in range(blocks):
        blk = simulate_block(policy, offs, rng, block_id=f"{arrival}-{policy}-{b}")
        out.append({**blk, "arrival": arrival, "policy": policy, "round": 1 + b // (blocks // 2 or 1),
                    "block_pos": 1 + b % 4, "run": "sim"})
    return out


def _quant(xs, qs=(10, 50, 90)):
    xs = [x for x in xs if x is not None]
    return [float(np.percentile(xs, q)) for q in qs] if xs else [None] * len(qs)


def predict(arrivals, policies, blocks: int, seeds: int, stagger_s: float = 1.125) -> dict:
    """For each cell: distribution over seeds of the pooled metrics, using the
    same code as the real analysis. Seeds share nothing (independent noise)."""
    out = {}
    for a in arrivals:
        per = defaultdict(lambda: defaultdict(list))
        for sd in range(seeds):
            cells = {p: simulate_cell(a, p, blocks, seed=1000 * sd + ARRIVAL_SEED[a], stagger_s=stagger_s)
                     for p in policies}
            for p, bl in cells.items():
                cs = aa.cell_summary(bl, None)
                wl = aa.describe_cell(bl)["block_mean"]
                for m in ("final_latency_s", "final_queue_s", "ttfp_s", "wait_own_partial_s"):
                    per[p][f"{m}_p95"].append(cs[m]["p95"])
                per[p]["share_unfinished"].append(cs["end_state"]["share_unfinished"])
                per[p]["share_queued"].append(cs["end_state"]["share_queued"])
                per[p]["mean_wait_own_s"].append(cs["end_state"]["mean_wait_own_s"])
                for k, v in wl.items():
                    per[p][f"wl_{k}"].append(v)
            for p in policies:
                if p != "base":
                    per[p]["delta_final_latency_p95"].append(
                        per[p]["final_latency_s_p95"][-1] - per["base"]["final_latency_s_p95"][-1])
                    per[p]["delta_share_unfinished"].append(
                        per[p]["share_unfinished"][-1] - per["base"]["share_unfinished"][-1])
        out[a] = {p: {k: _quant(v) for k, v in d.items()} for p, d in per.items()}
    return out


PHASE3_MEASURED = {  # c=8 block, 2 runs x 20 streams per policy (docs/benchmarks/scheduling-t4)
    "base": {"final_latency_s_p95": 0.5624, "final_queue_s_p95": 0.3314, "unfinished": 3 / 40},
    "prio": {"final_latency_s_p95": 0.3898, "final_queue_s_p95": 0.2677, "unfinished": 8 / 40},
    "both": {"final_latency_s_p95": 0.2663, "final_queue_s_p95": 0.1481, "unfinished": 8 / 40},
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--validate-phase3", action="store_true")
    p.add_argument("--predict-phase4", action="store_true")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--output")
    a = p.parse_args()
    result = {}
    if a.validate_phase3:
        # Phase 3: one sync c=8 block per run, 2 runs per policy
        v = predict(["sync"], ["base", "prio", "both"], blocks=2, seeds=a.seeds)["sync"]
        result["validate_phase3"] = v
        print("Phase 3 c=8 (2 blocks per policy): model 10/50/90 % over seeds vs measured")
        for pol in ("base", "prio", "both"):
            m = PHASE3_MEASURED[pol]
            f = lambda q: "/".join(f"{x * 1000:.0f}" for x in q)  # noqa: E731
            print(f"  {pol:5s} final p95 {f(v[pol]['final_latency_s_p95'])} (meas {m['final_latency_s_p95']*1000:.0f})"
                  f" | queue p95 {f(v[pol]['final_queue_s_p95'])} (meas {m['final_queue_s_p95']*1000:.0f})"
                  f" | unfinished {'/'.join(f'{x:.0%}' for x in v[pol]['share_unfinished'])}"
                  f" (meas {m['unfinished']:.0%})")
    if a.predict_phase4:
        v = predict(["sync", "stagger"], ["base", "prio", "both"], blocks=8, seeds=a.seeds)
        result["predict_phase4"] = v
        for arr, cells in v.items():
            print(f"== {arr} (8 blocks x 20 streams per cell; 10/50/90 % over {a.seeds} seeds)")
            for pol, d in cells.items():
                f = lambda q: "/".join("–" if x is None else f"{x * 1000:.0f}" for x in q)  # noqa: E731
                line = (f"  {pol:5s} final p95 {f(d['final_latency_s_p95'])} | queue p95 {f(d['final_queue_s_p95'])}"
                        f" | ttfp p95 {f(d['ttfp_s_p95'])} | unfinished {'/'.join(f'{x:.0%}' for x in d['share_unfinished'])}")
                if "delta_final_latency_p95" in d:
                    line += f" | Δ final p95 vs base {f(d['delta_final_latency_p95'])}"
                print(line)
            w = cells["base"]
            print(f"  workload (base): ends±250ms {w['wl_end_share_with_neighbour_250ms'][1]:.0%}, partial nbrs "
                  f"{w['wl_partial_mean_neighbours_50ms'][1]:.2f}, phase R {w['wl_partial_phase_R'][1]:.2f}, "
                  f"active mean {w['wl_observed_active_mean'][1]:.2f}, queue depth mean "
                  f"{w['wl_server_queue_depth_mean'][1]:.2f}")
    if a.output:
        with open(a.output, "w") as fh:
            json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
