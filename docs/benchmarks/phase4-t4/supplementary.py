#!/usr/bin/env python3
"""POST-HOC, supplementary computations for the Phase 4 report.

NOT part of the registered analysis (benchmarks/analyze_arrival.py at the
registration commit) and not used for any registered decision. Read-only:
it never writes into the raw evidence directory.

    python docs/benchmarks/phase4-t4/supplementary.py docs/benchmarks/phase4-t4/raw/phase4_results

Computes:
1. Throughput (completed streams per second, from each block's existing
   stress summary) per cell. It was a registered secondary outcome that the
   registered analysis code did not compute. Compare within an arrival mode
   only: the stagger ramp is part of each block's wall clock.
2. Per-block counts of streams whose own partial was unfinished at end,
   and which files they played (explains zero-width mechanism intervals).
3. TTFP p95 per sync block, and the slowest first partials of run 10's
   recovery block (the one failed stress check).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from benchmarks import analyze_arrival as aa  # noqa: E402


def main(root: Path) -> None:
    manifest, blocks = aa.load(root)
    print("# Phase 4 supplementary (post-hoc) computations\n")

    print("## 1. Throughput by cell (streams/s; per-block mean, min, max; within arrival mode only)\n")
    print("| Arrival | Policy | Mean | Min | Max | Blocks |\n|---|---|---:|---:|---:|---:|")
    for arr in ("sync", "stagger"):
        for pol in ("base", "prio", "both"):
            v = [b["summary"]["throughput_streams_per_s"] for b in aa.cell_blocks(blocks, arr, pol)]
            print(f"| {arr} | {pol} | {np.mean(v):.4f} | {min(v):.4f} | {max(v):.4f} | {len(v)} |")

    print("\n## 2. Streams with an unfinished own partial at end, per block (B1-B4 round 1, then round 2)\n")
    print("| Arrival | Policy | Per block | Total | Files |\n|---|---|---|---:|---|")
    for arr in ("sync", "stagger"):
        for pol in ("base", "prio", "both"):
            per, files = [], Counter()
            for b in aa.cell_blocks(blocks, arr, pol):
                u = [s for s in b["streams"] if s.get("_end", {}).get("state") in aa.UNFINISHED]
                per.append(len(u))
                files.update(s["utterance"] for s in u)
            print(f"| {arr} | {pol} | {per} | {sum(per)} | {dict(sorted(files.items())) or '-'} |")

    print("\n## 3. TTFP p95 per sync block (ms) and run 10's recovery block\n")
    print("| Policy | " + " | ".join(f"r{r} B{p}" for r in (1, 2) for p in (1, 2, 3, 4)) + " |")
    print("|---|" + "---:|" * 8)
    for pol in ("base", "prio", "both"):
        v = [aa.p95(aa.vals(b["streams"], aa.ttfp)) for b in aa.cell_blocks(blocks, "sync", pol)]
        print(f"| {pol} | " + " | ".join(f"{x * 1000:.0f}" for x in v) + " |")
    run10 = next(b for b in blocks if b["run"].startswith("10_") and b["block_pos"] == 4)
    t = aa.vals(run10["streams"], aa.ttfp)
    print(f"\nRun 10 recovery block: TTFP p95 {aa.p95(t) * 1000:.0f} ms, median {np.median(t) * 1000:.0f} ms. "
          "Three slowest first partials:\n")
    print("| Stream | File | TTFP (ms) | First-partial queue wait (ms) | First-partial decode (ms) | First frame, s after block start |")
    print("|---:|---|---:|---:|---:|---:|")
    t0 = min(s["t_first_audio"] for s in run10["streams"])
    for s in sorted(run10["streams"], key=aa.ttfp)[-3:]:
        pt = s["partial_timing"][0]
        print(f"| {s['index']} | {s['utterance']} | {aa.ttfp(s) * 1000:.0f} | {pt['queue_ms']:.0f} | "
              f"{pt['decode_ms']:.0f} | {s['t_first_audio'] - t0:.2f} |")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
