#!/usr/bin/env python3
"""Compare scheduling policies from a matched_comparison run.

    python -m benchmarks.compare_policies results/sched --audio-dir eval_audio5 \\
        --output results/sched/comparison.json --markdown results/sched/comparison.md

Groups: per policy x concurrency. c=4 pools the ``c4`` and ``recovery``
scenarios (identical workloads); c=8 is ``c8``. Rounds are pooled, and
per-round p95 values are reported so drift and order effects stay visible.

Per group: final latency, final queue wait, wait for the stream's own
partial, final decode time (control: scheduling must not change it),
time to first partial, partial queue wait, throughput, failures, partials
sent per stream, WER (v2), and final-text identity with the baseline
(decoding is deterministic, so anything below 100 % is a red flag).

Differences from the baseline policy come with a percentile-bootstrap 95 %
interval: streams are resampled within each arm, 2,000 resamples, fixed
seed. Streams within a run are not independent (shared GPU, synchronized
starts), so these intervals are optimistic; read them with the per-round
values.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .common import corpus_wer

SCENARIOS = {"c4": 4, "recovery": 4, "c8": 8}
METRICS = {  # name -> per-stream extractor (seconds); None when unavailable
    "final_latency_s": lambda s: s["t_final"] - s["t_end_sent"],
    "final_queue_s": lambda s: s["final_timing"]["queue_ms"] / 1000,
    "final_wait_own_partial_s": lambda s: s["final_timing"]["wait_partial_ms"] / 1000,
    "final_decode_s": lambda s: s["final_inference_ms"] / 1000,
    "ttfp_s": lambda s: (s["t_first_partial"] - s["t_first_audio"]) if s["t_first_partial"] else None,
}
DELTA_METRICS = ("final_latency_s", "final_queue_s", "ttfp_s", "final_decode_s")


def values(streams: list[dict], metric: str) -> list[float]:
    """Per-stream values of a metric; streams lacking the field are skipped
    (e.g. no timing object, or no nonempty partial for TTFP)."""
    out = []
    for s in streams:
        try:
            v = METRICS[metric](s)
        except (KeyError, TypeError):
            continue
        if v is not None:
            out.append(v)
    return out


def _pct(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def summarize_values(v: list[float]) -> dict:
    if not v:
        return {"n": 0}
    a = np.asarray(v)
    return {"n": int(a.size), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "mean": float(a.mean()), "max": float(a.max())}


def bootstrap_p95_diff(base: list[float], cand: list[float], iters: int = 2000, seed: int = 0) -> dict:
    if len(base) < 5 or len(cand) < 5:
        return {"n_base": len(base), "n_cand": len(cand)}
    rng = np.random.default_rng(seed)
    b, c = np.asarray(base), np.asarray(cand)
    diffs = np.percentile(rng.choice(c, (iters, c.size)), 95, axis=1) - \
        np.percentile(rng.choice(b, (iters, b.size)), 95, axis=1)
    point = float(np.percentile(c, 95) - np.percentile(b, 95))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff_p95": point, "ci95": [float(lo), float(hi)],
            "relative": point / float(np.percentile(b, 95)) if np.percentile(b, 95) else None,
            "n_base": int(b.size), "n_cand": int(c.size)}


def load_runs(root: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((root / "manifest.json").read_text())
    runs = []
    for entry in manifest["order"]:
        d = root / entry["dir"]
        scenarios = {}
        for name in SCENARIOS:
            f = d / f"{name}.json"
            if f.exists():
                scenarios[name] = json.loads(f.read_text())
        runs.append({**entry, "scenarios": scenarios})
    return manifest, runs


def analyze(root: Path, refs: dict[str, str] | None, baseline: str = "base") -> dict:
    manifest, runs = load_runs(root)
    groups: dict[tuple[str, int], dict] = defaultdict(lambda: {"streams": [], "throughput": [],
                                                               "per_round": defaultdict(list)})
    counters: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        last_metrics = None
        for name, data in run["scenarios"].items():
            g = groups[(run["policy"], SCENARIOS[name])]
            g["streams"] += data["streams"]
            g["throughput"].append(data["summary"]["throughput_streams_per_s"])
            g["per_round"][run["round"]] += data["streams"]
            last_metrics = data.get("server_metrics_after") or last_metrics
        if last_metrics:
            c = last_metrics.get("counters", {})
            counters[run["policy"]].append({k: v for k, v in c.items() if k.startswith(
                ("stream.partials_", "stream.partial_at_end", "inference.partial.completed",
                 "inference.final.completed", "inference.partial.cancelled_while_queued"))})

    # Baseline final text per utterance (deterministic decoding -> one text).
    base_text: dict[str, Counter] = defaultdict(Counter)
    for (pol, _), g in groups.items():
        if pol == baseline:
            for s in g["streams"]:
                if s["ok"]:
                    base_text[s["utterance"]][s["final_text"]] += 1
    modal = {u: c.most_common(1)[0][0] for u, c in base_text.items()}

    out: dict = {"manifest": {k: manifest[k] for k in ("commit", "common_settings", "stress_args")},
                 "run_order": [(r["round"], r["policy"]) for r in runs], "groups": {}, "deltas": {},
                 "work_counters": {}}
    for (pol, conc), g in sorted(groups.items()):
        ok = [s for s in g["streams"] if s["ok"]]
        row: dict = {"streams": len(g["streams"]), "ok": len(ok),
                     "failures": len(g["streams"]) - len(ok),
                     "errors": dict(Counter(s["error"] or "unknown" for s in g["streams"] if not s["ok"])),
                     "throughput_streams_per_s": {"mean": float(np.mean(g["throughput"])),
                                                  "per_scenario_run": g["throughput"]},
                     "partials_sent_per_stream": float(np.mean([s["num_partials"] for s in ok])) if ok else None,
                     "streams_without_nonempty_partial": sum(1 for s in ok if not s["t_first_partial"])}
        for m in METRICS:
            row[m] = summarize_values(values(ok, m))
        pq = [t["queue_ms"] / 1000 for s in ok for t in s.get("partial_timing", [])]
        row["partial_queue_s"] = summarize_values(pq)
        row["per_round_p95"] = {
            str(r): {m: _pct(values([s for s in ss if s["ok"]], m), 95)
                     for m in ("final_latency_s", "final_queue_s", "ttfp_s")}
            for r, ss in sorted(g["per_round"].items())}
        if refs:
            scored = [s for s in ok if s["utterance"] in refs]
            row["wer_v2"] = corpus_wer([refs[s["utterance"]] for s in scored],
                                       [s["final_text"] or "" for s in scored], "v2")
        if modal:
            same = [s["final_text"] == modal.get(s["utterance"]) for s in ok if s["utterance"] in modal]
            row["final_text_identical_to_baseline"] = float(np.mean(same)) if same else None
        out["groups"][f"{pol} c={conc}"] = row

    for (pol, conc), g in sorted(groups.items()):
        if pol == baseline or (baseline, conc) not in groups:
            continue
        b_ok = [s for s in groups[(baseline, conc)]["streams"] if s["ok"]]
        c_ok = [s for s in g["streams"] if s["ok"]]
        out["deltas"][f"{pol} vs {baseline} c={conc}"] = {
            m: bootstrap_p95_diff(values(b_ok, m), values(c_ok, m))
            for m in DELTA_METRICS}
    for pol, rows in counters.items():
        keys = sorted({k for r in rows for k in r})
        out["work_counters"][pol] = {k: [r.get(k, 0) for r in rows] for k in keys}
    return out


def _ms(v) -> str:
    return "–" if v is None else f"{v * 1000:.0f}"


def to_markdown(a: dict) -> str:
    L = ["# Scheduling policy comparison", "",
         f"Commit `{a['manifest']['commit'][:12]}`; run order: "
         + ", ".join(f"r{r}:{p}" for r, p in a["run_order"]), "",
         "All times in ms. c=4 pools the c4 and recovery scenarios.", "",
         "| Group | n | Fail | Final lat p50 / p95 / max | Final queue p50 / p95 | Wait own partial p95 / max "
         "| Final decode p50 / p95 (control) | TTFP p50 / p95 | Partial queue p95 | Throughput (streams/s) "
         "| Partials / stream | WER v2 | Final text = base |",
         "|---|---:|---:|---|---|---|---|---|---:|---:|---:|---:|---:|"]
    for name, g in a["groups"].items():
        fl, fq, fw, fd, tt = (g[k] for k in ("final_latency_s", "final_queue_s", "final_wait_own_partial_s",
                                             "final_decode_s", "ttfp_s"))
        wer = "–" if g.get("wer_v2") is None else f"{g['wer_v2']:.2%}"
        ident = g.get("final_text_identical_to_baseline")
        L.append(f"| {name} | {g['ok']} | {g['failures']} | {_ms(fl.get('p50'))} / {_ms(fl.get('p95'))} / "
                 f"{_ms(fl.get('max'))} | {_ms(fq.get('p50'))} / {_ms(fq.get('p95'))} | {_ms(fw.get('p95'))} / "
                 f"{_ms(fw.get('max'))} | {_ms(fd.get('p50'))} / {_ms(fd.get('p95'))} | {_ms(tt.get('p50'))} / "
                 f"{_ms(tt.get('p95'))} | {_ms(g['partial_queue_s'].get('p95'))} | "
                 f"{g['throughput_streams_per_s']['mean']:.3f} | {g['partials_sent_per_stream']:.2f} | {wer} | "
                 f"{'–' if ident is None else f'{ident:.0%}'} |")
    L += ["", "## Differences in p95 vs baseline (bootstrap 95 % interval, ms)", "",
          "| Comparison | Final latency | Final queue | TTFP | Final decode (control) |", "|---|---|---|---|---|"]
    for name, d in a["deltas"].items():
        cells = []
        for m in DELTA_METRICS:
            x = d[m]
            cells.append("–" if "diff_p95" not in x else
                         f"{x['diff_p95'] * 1000:+.0f} [{x['ci95'][0] * 1000:+.0f}, {x['ci95'][1] * 1000:+.0f}]")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "## p95 per round (ms): consistency check", "",
          "| Group | Round | Final latency | Final queue | TTFP |", "|---|---|---:|---:|---:|"]
    for name, g in a["groups"].items():
        for r, v in g["per_round_p95"].items():
            L.append(f"| {name} | {r} | {_ms(v['final_latency_s'])} | {_ms(v['final_queue_s'])} | {_ms(v['ttfp_s'])} |")
    L += ["", "## Work done per run (server counters, whole run)", ""]
    for pol, c in a["work_counters"].items():
        L.append(f"* **{pol}**: " + "; ".join(f"{k} = {v}" for k, v in c.items()))
    return "\n".join(L) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="matched_comparison output directory")
    p.add_argument("--audio-dir", help="for WER: <utterance>.txt references")
    p.add_argument("--baseline", default="base")
    p.add_argument("--output")
    p.add_argument("--markdown")
    a = p.parse_args()
    refs = None
    if a.audio_dir:
        refs = {f.stem: f.read_text().strip() for f in Path(a.audio_dir).glob("*.txt")}
    result = analyze(Path(a.root), refs, a.baseline)
    md = to_markdown(result)
    print(md)
    if a.output:
        Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    if a.markdown:
        Path(a.markdown).write_text(md)


if __name__ == "__main__":
    main()
