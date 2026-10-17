#!/usr/bin/env python3
"""Post-hoc analysis: final latency by the state of each stream's own
partial at end of stream, per scheduling policy.

    python -m benchmarks.end_state_breakdown results/sched --scenario c8

Read-only; it changes nothing about how measurements are taken. Each
client stream record (``<run>/<scenario>.json``) is joined to the server's
``stream_summary`` log line for the same session (``ready.session_id``),
which records the partial's state when ``end`` was processed (done /
running / queued / created) and whether it was promoted or cancelled.

Added after the Phase 3 GPU run to explain its tail; see
docs/benchmarks/scheduling-t4/README.md.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

STATES = ("done", "running", "queued", "created", "none")


def summaries(log: Path) -> dict[str, dict]:
    out = {}
    for line in log.read_text().splitlines():
        if "stream_summary" in line:
            kv = dict(re.findall(r"(\w+)=(\S+)", line.split("stream_summary", 1)[1]))
            out[kv["session"]] = kv
    return out


def breakdown(root: Path, scenario: str) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    rows: dict[str, list[dict]] = defaultdict(list)
    for entry in manifest["order"]:
        run = root / entry["dir"]
        summ = summaries(run / "server.log")
        for s in json.loads((run / f"{scenario}.json").read_text())["streams"]:
            if not s["ok"]:
                continue
            k = summ[s["ready"]["session_id"]]
            t = s["final_timing"]
            rows[entry["policy"]].append({
                "state": k["partial_state_at_end"],
                "promoted": k.get("partial_promoted_at_end") == "True",
                "cancelled": k.get("partial_cancelled_at_end") == "True",
                "wait_own_ms": t["wait_partial_ms"], "queue_ms": t["queue_ms"],
                "latency_ms": (s["t_final"] - s["t_end_sent"]) * 1000,
            })
    out = {}
    for pol, rs in rows.items():
        lat = sorted(r["latency_ms"] for r in rs)
        p95 = float(np.percentile(lat, 95))
        out[pol] = {"n": len(rs), "latency_p95_ms": p95, "classes": {}}
        for st in STATES:
            c = [r for r in rs if r["state"] == st]
            if not c:
                continue
            out[pol]["classes"][st] = {
                "n": len(c), "share": len(c) / len(rs),
                "promoted": sum(r["promoted"] for r in c), "cancelled": sum(r["cancelled"] for r in c),
                "wait_own_ms": {"p50": float(np.median([r["wait_own_ms"] for r in c])),
                                "max": max(r["wait_own_ms"] for r in c)},
                "queue_ms": {"p50": float(np.median([r["queue_ms"] for r in c])),
                             "max": max(r["queue_ms"] for r in c)},
                "latency_ms": {"p50": float(np.median([r["latency_ms"] for r in c])),
                               "max": max(r["latency_ms"] for r in c)},
                "at_or_above_p95": sum(r["latency_ms"] >= p95 for r in c),
            }
    return out


def to_markdown(b: dict, scenario: str) -> str:
    L = [f"Scenario `{scenario}`; latencies in ms; class = state of the stream's own partial at `end`.", "",
         "| Policy | Class | n (share) | Promoted / cancelled | Wait own p50 / max | Final queue p50 / max "
         "| Final latency p50 / max | Streams ≥ policy p95 |", "|---|---|---|---|---|---|---|---:|"]
    for pol in ("base", "prio", "cancel", "both"):
        if pol not in b:
            continue
        for st, c in b[pol]["classes"].items():
            L.append(f"| {pol} (p95 {b[pol]['latency_p95_ms']:.0f}) | {st} | {c['n']} ({c['share']:.0%}) | "
                     f"{c['promoted']} / {c['cancelled']} | {c['wait_own_ms']['p50']:.0f} / {c['wait_own_ms']['max']:.0f} | "
                     f"{c['queue_ms']['p50']:.0f} / {c['queue_ms']['max']:.0f} | "
                     f"{c['latency_ms']['p50']:.0f} / {c['latency_ms']['max']:.0f} | {c['at_or_above_p95']} |")
    return "\n".join(L) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root")
    p.add_argument("--scenario", default="c8")
    p.add_argument("--output")
    p.add_argument("--markdown")
    a = p.parse_args()
    b = breakdown(Path(a.root), a.scenario)
    md = to_markdown(b, a.scenario)
    print(md)
    if a.output:
        Path(a.output).write_text(json.dumps(b, indent=2) + "\n")
    if a.markdown:
        Path(a.markdown).write_text(md)


if __name__ == "__main__":
    main()
