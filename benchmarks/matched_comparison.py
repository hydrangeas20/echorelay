#!/usr/bin/env python3
"""Matched baseline-vs-candidate comparison of scheduling policies.

    python -m benchmarks.matched_comparison --audio-dir eval_audio5 \\
        --out results/sched --rounds 2

Every run uses the same commit, the same evaluation audio, the same server
settings except the two policy flags, and the unchanged measurement code
(``benchmarks.stress`` at --concurrency 4 8, which also repeats c=4 as its
recovery scenario). Only the policy flags differ:

    base    final_priority=false  cancel_queued_partial_on_end=false
    prio    final_priority=true   cancel_queued_partial_on_end=false
    cancel  final_priority=false  cancel_queued_partial_on_end=true
    both    final_priority=true   cancel_queued_partial_on_end=true

Runs are counterbalanced: round 1 in the given order, round 2 reversed
(ABCD DCBA), so a linear drift in GPU state affects every policy equally.
Each run restarts the server, verifies its configuration via /metrics,
and saves provenance. Analyze with ``benchmarks.compare_policies``.

``--plan phase4`` runs the registered staggered-arrival experiment
instead (docs/experiments/phase4-staggered-arrival.md): the fixed
PHASE4_ORDER of 12 runs, each executing four 20-stream c=8 blocks with
the run's arrival schedule. Analyze with ``benchmarks.analyze_arrival``.
Any override of a plan constant is recorded under ``deviations``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .colab_helpers import PORT, REPO, save_provenance, start_server, stop_server, verify_server

POLICIES = {
    "base": {"final_priority": False, "cancel_queued_partial_on_end": False},
    "prio": {"final_priority": True, "cancel_queued_partial_on_end": False},
    "cancel": {"final_priority": False, "cancel_queued_partial_on_end": True},
    "both": {"final_priority": True, "cancel_queued_partial_on_end": True},
}


# Registered Phase 4 plan. Do not edit after data collection has started.
PHASE4 = {
    "policies": ["base", "prio", "both"],
    "arrivals": ["sync", "stagger"],
    "concurrency": [8, 8, 8],  # stress adds a 4th c=8 block ("recovery"): blocks c8, c8_2, c8_3, recovery
    "streams": 20,             # per block, the Phase 2/3 scenario size
    "stagger_s": 1.125,        # partial interval x (1 + 1/8): first-wave partial phases evenly spread
    "speed": 1.0,
    "chunk_ms": 100,
    "warmup": 2,
    "burst_extra": 0,
}
_HALF = [("sync", "base"), ("stagger", "prio"), ("sync", "both"),
         ("stagger", "base"), ("sync", "prio"), ("stagger", "both")]
# Mirror order: run k and run 13-k are the same cell, so every cell's mean
# position is 6.5 (linear drift cancels) and the arrival mode alternates.
PHASE4_ORDER = [(1, a, p) for a, p in _HALF] + [(2, a, p) for a, p in reversed(_HALF)]


def run_order(policies: list[str], rounds: int) -> list[tuple[int, str]]:
    """Counterbalanced order: odd rounds forward, even rounds reversed."""
    order = []
    for r in range(1, rounds + 1):
        seq = policies if r % 2 else list(reversed(policies))
        order += [(r, p) for p in seq]
    return order


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--policies", nargs="+", default=list(POLICIES), choices=list(POLICIES))
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--concurrency", type=int, nargs="+", default=[4, 8])
    p.add_argument("--streams", type=int, default=20)
    p.add_argument("--speed", type=float, default=1.0, help="keep 1.0 for GPU runs (Phase 1/2 methodology)")
    p.add_argument("--backend", default="whisper", choices=["whisper", "fake"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-type", default="float16")
    p.add_argument("--fake-delay-s", type=float, default=0.0, help="fake backend only (harness tests)")
    p.add_argument("--plan", choices=["phase4"], default=None,
                   help="run a registered plan; its constants replace --policies/--rounds/--concurrency")
    p.add_argument("--stagger-s", type=float, default=None, help="phase4 only: override (recorded as a deviation)")
    a = p.parse_args()
    deviations = []
    if a.plan == "phase4":
        explicit = {x.split("=")[0] for x in sys.argv[1:] if x.startswith("--")}
        for key, flag in (("streams", "--streams"), ("speed", "--speed")):
            if flag in explicit and getattr(a, key) != PHASE4[key]:
                deviations.append(f"{key}={getattr(a, key)} (registered {PHASE4[key]})")
        if a.stagger_s is not None and a.stagger_s != PHASE4["stagger_s"]:
            deviations.append(f"stagger_s={a.stagger_s} (registered {PHASE4['stagger_s']})")
        if a.backend != "whisper":
            deviations.append(f"backend={a.backend} (registered whisper)")
        if "--streams" not in explicit:
            a.streams = PHASE4["streams"]
        if "--speed" not in explicit:
            a.speed = PHASE4["speed"]
        a.concurrency = PHASE4["concurrency"]
        stagger_s = a.stagger_s if a.stagger_s is not None else PHASE4["stagger_s"]
        runs = PHASE4_ORDER
        if deviations:
            print("WARNING: NOT the registered configuration: " + "; ".join(deviations), flush=True)
    else:
        stagger_s = None
        runs = [(r, "sync", n) for r, n in run_order(a.policies, a.rounds)]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    common = {"event_timing": True, "max_concurrent_inference": 1, "max_concurrent_streams": 8,
              "backend": a.backend}
    if a.backend == "whisper":
        common.update(device=a.device, compute_type=a.compute_type)
    else:
        common.update(fake_delay_s=a.fake_delay_s)
    expect_common = {"event_timing": True, "max_concurrent_inference": 1, "max_concurrent_streams": 8,
                     "partial_interval_s": 1.0, "min_partial_audio_s": 0.5, "partial_beam_size": 1,
                     "final_beam_size": 5, "active_streams": 0, "backend": a.backend}
    if a.backend == "whisper":
        expect_common.update(model="base.en", device=a.device, compute_type=a.compute_type)

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    manifest = {"commit": commit, "order": [], "common_settings": common, "policies": POLICIES,
                "stress_args": {"concurrency": a.concurrency, "streams": a.streams, "speed": a.speed,
                                "burst_extra": 0, "warmup": 2},
                "audio_dir": str(Path(a.audio_dir).resolve())}
    if a.plan:
        manifest.update(plan=a.plan, plan_constants=PHASE4, deviations=deviations,
                        planned_order=[list(x) for x in runs], stagger_s=stagger_s)
    print(f"commit {commit[:12]}; {len(runs)} runs: "
          + " ".join(f"r{r}:{arr + '/' if a.plan else ''}{n}" for r, arr, n in runs), flush=True)
    for i, (rnd, arrival, name) in enumerate(runs, 1):
        run_dir = out / (f"{i:02d}_r{rnd}_{arrival}_{name}" if a.plan else f"{i:02d}_r{rnd}_{name}")
        run_dir.mkdir(parents=True, exist_ok=True)
        flags = POLICIES[name]
        print(f"\n=== run {i}/{len(runs)}: round {rnd}, "
              + (f"arrival {arrival}, " if a.plan else "") + f"policy {name} {flags}", flush=True)
        start_server(run_dir / "server.log", **common, **{k: str(v).lower() for k, v in flags.items()})
        verify_server({**expect_common, **flags})
        t0 = time.monotonic()
        cmd = [sys.executable, "-m", "benchmarks.stress", "--audio-dir", a.audio_dir,
               "--url", f"ws://127.0.0.1:{PORT}/ws/transcribe",
               "--streams", str(a.streams), "--concurrency", *map(str, a.concurrency),
               "--burst-extra", "0", "--speed", str(a.speed),
               "--label", f"{name} r{rnd}", "--output-dir", str(run_dir)]
        if a.plan:  # the default (Phase 3) command line is unchanged
            cmd += ["--arrival", arrival, "--chunk-ms", str(PHASE4["chunk_ms"]),
                    "--warmup", str(PHASE4["warmup"])]
            if arrival == "stagger":
                cmd += ["--stagger-s", str(stagger_s)]
        with open(run_dir / "console.txt", "w") as log:
            rc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
        save_provenance(run_dir, f"matched comparison: {name}, round {rnd}"
                        + (f", arrival {arrival} (plan {a.plan})" if a.plan else ""))
        stop_server()
        report = run_dir / "stress_report.json"
        if report.exists():
            checks = json.loads(report.read_text())["checks"]
            failed = [c for c in checks if not c["pass"]]
        else:  # stress could not run; record it and continue with the other runs
            checks = []
            failed = [{"scenario": "-", "check": "stress produced no report",
                       "detail": f"exit {rc}; see {run_dir.name}/console.txt"}]
        print(f"stress exit {rc}; {len(checks) - len(failed)}/{len(checks)} checks "
              f"passed; {time.monotonic() - t0:.0f} s", flush=True)
        for c in failed:
            print(f"  FAIL {c['scenario']}: {c['check']}: {c['detail']}", flush=True)
        entry = {"index": i, "round": rnd, "policy": name, "dir": run_dir.name,
                 "stress_exit": rc, "failed_checks": failed}
        if a.plan:
            entry["arrival"] = arrival
        manifest["order"].append(entry)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
