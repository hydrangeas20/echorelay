#!/usr/bin/env python3
"""Analyze saved load_test results: where inference time goes, and why.

    python -m benchmarks.analyze_results results/stream_fp16_c4.json \\
        results/stream_int8_c4.json --output results/analysis.json --markdown results/analysis.md

Works on Phase 1 files (no timing fields) and on Phase 2 files recorded
with ECHORELAY_EVENT_TIMING=true (adds direct queue / wait-for-partial data).
Everything is derived from per-stream records; nothing is re-measured.

Reported per file:

* partial decode cost vs audio covered: per-bucket stats and a least-squares
  line ``decode_ms = intercept + slope * audio_s``. Tests whether repeated
  full-buffer partial decoding gets more expensive as the buffer grows.
* inference busy fraction: sum of all decode times / run wall clock, and the
  share of that spent on partials.
* per-utterance re-decode amplification (audio decoded by partials / audio).
* time-to-first-partial decomposition: pacing floor + first partial decode +
  remainder (queueing, network, scheduling).
* final-latency decomposition: final decode + remainder (wait for in-flight
  partial, queueing, network); split directly when timing fields exist.
* capacity projections (clearly labelled as projections, not measurements).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .common import percentiles

BUCKETS_S = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 16), (16, 20),
             (20, 30), (30, 60), (60, 120)]


def _fit(x: np.ndarray, y: np.ndarray) -> dict:
    if x.size < 3 or np.ptp(x) == 0:
        return {"n": int(x.size)}
    slope, intercept = np.polyfit(x, y, 1)
    return {"n": int(x.size), "intercept_ms": float(intercept), "slope_ms_per_audio_s": float(slope),
            "r": float(np.corrcoef(x, y)[0, 1]) if np.ptp(y) > 0 else None, "x_min_s": float(x.min()), "x_max_s": float(x.max())}


def partial_cost_profile(result: dict) -> dict:
    ok = [s for s in result["streams"] if s["ok"]]
    x = np.array([a for s in ok for a in s["partial_audio_s"]], dtype=float)
    y = np.array([ms for s in ok for ms in s["partial_inference_ms"]], dtype=float)
    buckets = []
    for lo, hi in BUCKETS_S:
        m = (x >= lo) & (x < hi)
        if m.any():
            buckets.append({"audio_s": [lo, hi], "n": int(m.sum()), "p50_ms": float(np.median(y[m])),
                            "mean_ms": float(y[m].mean()), "max_ms": float(y[m].max())})
    return {"buckets": buckets, "linear_fit": _fit(x, y)}


def final_cost_fit(result: dict) -> dict:
    ok = [s for s in result["streams"] if s["ok"] and s["final_audio_s"]]
    return _fit(np.array([s["final_audio_s"] for s in ok], dtype=float),
                np.array([s["final_inference_ms"] for s in ok], dtype=float))


def utilization(result: dict) -> dict:
    ok = [s for s in result["streams"] if s["ok"]]
    partial_s = sum(sum(s["partial_inference_ms"]) for s in ok) / 1000
    final_s = sum(s["final_inference_ms"] for s in ok) / 1000
    wall = result["summary"]["wall_clock_s"]
    workers = result["server"]["health"]["inference"]["max_concurrent"]
    busy = (partial_s + final_s) / wall / workers
    conc = result["run_config"]["concurrency"]
    return {
        "partial_decode_s": partial_s,
        "final_decode_s": final_s,
        "partial_share": partial_s / (partial_s + final_s) if partial_s + final_s else None,
        "wall_clock_s": wall,
        "inference_workers": workers,
        "busy_fraction": busy,
        "busy_fraction_per_concurrent_stream": busy / conc,
        "note": "busy_fraction = total decode time / (wall clock x workers); concurrency ramps "
                "down at the end of a run, so per-stream load is slightly underestimated",
    }


def per_utterance(result: dict) -> list[dict]:
    rows: dict[str, dict] = {}
    for s in result["streams"]:
        if not s["ok"]:
            continue
        r = rows.setdefault(s["utterance"], {"utterance": s["utterance"], "audio_s": s["audio_s"],
                                             "runs": 0, "partials": [], "partial_ms": [], "final_ms": [],
                                             "amplification": []})
        r["runs"] += 1
        r["partials"].append(s["num_partials"])
        r["partial_ms"].append(sum(s["partial_inference_ms"]))
        r["final_ms"].append(s["final_inference_ms"])
        r["amplification"].append(sum(s["partial_audio_s"]) / s["audio_s"])
    out = []
    for r in rows.values():
        pm, fm = float(np.median(r["partial_ms"])), float(np.median(r["final_ms"]))
        out.append({"utterance": r["utterance"], "audio_s": r["audio_s"], "runs": r["runs"],
                    "partials": int(np.median(r["partials"])), "partial_decode_ms_median": pm,
                    "final_decode_ms_median": fm, "partial_to_final_ratio": pm / fm if fm else None,
                    "amplification": float(np.median(r["amplification"]))})
    return sorted(out, key=lambda r: r["utterance"])


def pacing_floor_s(result: dict) -> float | None:
    """Earliest possible first partial at the run's pacing, measured from the
    first frame: the frame that completes ``max(interval, min_partial)`` of
    audio is sent ``(k - 1) * chunk`` after the first one."""
    cfg, limits = result["run_config"], result["server"].get("ready_limits") or {}
    interval = limits.get("partial_interval_s")
    if interval is None or not cfg.get("speed"):
        return None
    chunk_s = cfg["chunk_ms"] / 1000
    frames = math.ceil(round(interval / chunk_s, 9))
    return (frames - 1) * chunk_s / cfg["speed"]


def ttfp_decomposition(result: dict) -> dict:
    floor = pacing_floor_s(result)
    ok = [s for s in result["streams"] if s["ok"] and s["t_first_partial"]]
    first_decode = [s["partial_inference_ms"][0] / 1000 for s in ok]
    ttfp = [s["t_first_partial"] - s["t_first_audio"] for s in ok]
    out = {"pacing_floor_s": floor, "ttfp_s": percentiles(ttfp), "first_partial_decode_s": percentiles(first_decode)}
    if floor is not None:
        out["remainder_s"] = percentiles([t - floor - d for t, d in zip(ttfp, first_decode)])
        out["remainder_definition"] = "ttfp - pacing floor - first partial decode = queue + network + scheduling"
    timed = [s for s in ok if s.get("partial_timing")]
    if timed:
        out["first_partial_queue_s"] = percentiles([s["partial_timing"][0]["queue_ms"] / 1000 for s in timed])
    return out


def final_latency_decomposition(result: dict) -> dict:
    ok = [s for s in result["streams"] if s["ok"] and s["t_final"] and s["t_end_sent"]]
    lat = [s["t_final"] - s["t_end_sent"] for s in ok]
    dec = [s["final_inference_ms"] / 1000 for s in ok]
    out = {"final_latency_s": percentiles(lat), "final_decode_s": percentiles(dec),
           "remainder_s": percentiles([a - b for a, b in zip(lat, dec)]),
           "remainder_definition": "final latency - final decode = wait for in-flight partial + queue + network"}
    timed = [s for s in ok if s.get("final_timing")]
    if timed:
        out["wait_partial_s"] = percentiles([s["final_timing"]["wait_partial_ms"] / 1000 for s in timed])
        out["queue_s"] = percentiles([s["final_timing"]["queue_ms"] / 1000 for s in timed])
    return out


def projections(result: dict, partial_fit: dict, final_fit: dict, util: dict) -> dict:
    """Capacity projections. These are models fitted to one run, not measurements."""
    out: dict = {"label": "PROJECTION (not measured)"}
    per_stream = util["busy_fraction_per_concurrent_stream"]
    if per_stream:
        out["saturation_concurrency_same_audio_mix"] = 1 / per_stream
        out["saturation_assumption"] = ("decode cost per stream unchanged as concurrency grows; "
                                        "queueing delay rises well before this point")
    interval = (result["server"].get("ready_limits") or {}).get("partial_interval_s")
    if interval and "slope_ms_per_audio_s" in partial_fit and "slope_ms_per_audio_s" in final_fit:
        a, b = partial_fit["intercept_ms"], partial_fit["slope_ms_per_audio_s"]
        fa, fb = final_fit["intercept_ms"], final_fit["slope_ms_per_audio_s"]
        rows = []
        for t in (5, 10, 20, 30, 60, 120):
            n = int(t // interval)
            partial_ms = sum(a + b * k * interval for k in range(1, n + 1))
            final_ms = fa + fb * t
            load = (partial_ms + final_ms) / 1000 / t  # worker-seconds per audio-second
            rows.append({"stream_audio_s": t, "partials": n, "partial_compute_s": partial_ms / 1000,
                         "final_compute_s": final_ms / 1000, "worker_load_per_stream": load,
                         "saturation_streams_per_worker": 1 / load if load else None,
                         "extrapolated": t > partial_fit["x_max_s"]})
        out["by_stream_length"] = rows
        out["by_stream_length_assumption"] = (
            f"linear partial cost fitted on {partial_fit['x_min_s']:.1f}-{partial_fit['x_max_s']:.1f} s; "
            "rows marked extrapolated go beyond the data, and above 30 s Whisper decodes in "
            "multiple windows, which this linear model does not capture")
    return out


def analyze(result: dict) -> dict:
    pfit = partial_cost_profile(result)
    ffit = final_cost_fit(result)
    util = utilization(result)
    return {
        "label": result.get("label"),
        "model": result["server"]["health"]["model"],
        "run_config": result["run_config"],
        "streams_ok": result["summary"]["streams_ok"],
        "partial_cost": pfit,
        "final_cost_fit": ffit,
        "utilization": util,
        "per_utterance": per_utterance(result),
        "time_to_first_partial": ttfp_decomposition(result),
        "final_latency": final_latency_decomposition(result),
        "projections": projections(result, pfit["linear_fit"], ffit, util),
    }


def _ms(v: float | None) -> str:
    return "–" if v is None else f"{v * 1000:.0f}"


def _x(v: float | None, fmt: str) -> str:
    return "–" if v is None else format(v, fmt)


def to_markdown(analyses: dict[str, dict]) -> str:
    lines = ["# EchoRelay results analysis", ""]
    lines += ["| Run | Partial cost fit (ms) | r | Busy | Partial share | TTFP p95 remainder (ms) "
              "| Final p95 remainder (ms) |", "|---|---|---:|---:|---:|---:|---:|"]
    for path, a in analyses.items():
        f = a["partial_cost"]["linear_fit"]
        fit = (f"{f['intercept_ms']:.1f} + {f['slope_ms_per_audio_s']:.2f}·s"
               if "slope_ms_per_audio_s" in f else "–")
        rem = a["time_to_first_partial"].get("remainder_s", {}).get("p95")
        lines.append(f"| {a['label'] or path} | {fit} | {f.get('r', float('nan')):.2f} | "
                     f"{_x(a['utilization']['busy_fraction'], '.1%')} | "
                     f"{_x(a['utilization']['partial_share'], '.0%')} | "
                     f"{_ms(rem)} | {_ms(a['final_latency']['remainder_s']['p95'])} |")
    for path, a in analyses.items():
        lines += ["", f"## {a['label'] or path}", "", "Partial decode cost by audio covered:", "",
                  "| Audio (s) | n | p50 ms | mean ms | max ms |", "|---|---:|---:|---:|---:|"]
        for b in a["partial_cost"]["buckets"]:
            lines.append(f"| {b['audio_s'][0]}–{b['audio_s'][1]} | {b['n']} | {b['p50_ms']:.1f} | "
                         f"{b['mean_ms']:.1f} | {b['max_ms']:.1f} |")
        lines += ["", "| Utterance | Audio s | Partials | Partial ms | Final ms | Partial/final | Amplification |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for r in a["per_utterance"]:
            lines.append(f"| {r['utterance']} | {r['audio_s']:.2f} | {r['partials']} | "
                         f"{r['partial_decode_ms_median']:.0f} | {r['final_decode_ms_median']:.0f} | "
                         f"{_x(r['partial_to_final_ratio'], '.1f')}× | {_x(r['amplification'], '.1f')}× |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+", help="load_test result JSON files")
    p.add_argument("--output", help="write full analysis JSON here")
    p.add_argument("--markdown", help="write a markdown summary here")
    a = p.parse_args()
    analyses = {path: analyze(json.loads(Path(path).read_text())) for path in a.results}
    md = to_markdown(analyses)
    print(md)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(json.dumps(analyses, indent=2) + "\n")
    if a.markdown:
        Path(a.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(a.markdown).write_text(md)


if __name__ == "__main__":
    main()
