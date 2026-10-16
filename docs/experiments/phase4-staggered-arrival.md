# Phase 4 registration: does the scheduling benefit survive staggered arrivals?

**Registered before any Phase 4 data was collected.** The registration is
the commit that adds this file. Its git history, together with
`benchmarks/analyze_arrival.py`, `benchmarks/matched_comparison.py`
(`PHASE4`, `PHASE4_ORDER`) and `docs/experiments/phase4-predictions.json`
at that commit, is the record. The results report must cite that commit and
list every deviation from it.

Scheduling flags stay **off by default** whatever the outcome. A positive
result supports proposing a default change for review; it does not make one.

## 1. Question

Phase 3 (`docs/benchmarks/scheduling-t4`) found that final priority plus
cancellation of a stream's queued partial cut c=8 final-latency p95 from
562 to 266 ms. That was measured on a workload where every 20-stream
scenario starts all eight streams at the same instant. **Is the benefit
still present when the first wave's arrivals are staggered?** In
particular:

* **Primary:** combined policy (`both`) vs `base`, within each arrival mode.
* **Secondary:** priority alone (`prio`) vs `base`.
* **Mechanism** (from the Phase 3 post-hoc analysis): does `prio` increase
  the frequency, or the duration, of waiting for the stream's own
  unfinished partial at end of stream?

## 2. Why this design (pre-data analysis)

These findings came from simulation and Phase 3 data before any Phase 4
run. They are why the design differs from the first proposal.

* **Closed-loop refill re-randomizes timing after the first wave.** In an
  80-stream scenario even `sync` would be mostly desynchronized (share of
  ends with another end within 250 ms: 50 % sync vs 45 % staggered,
  client-only simulation), so the arms would barely differ. Phase 3's
  20-stream scenarios are dominated by the synchronized first wave (70 %).
  **Design: four 20-stream blocks per run, each starting from an idle
  server**, which preserves Phase 3's scenario structure.
* **A 1.0 s stagger keeps partial bursts aligned.** Partials are requested
  every 1.0 s of audio, so offsets that are whole seconds leave every
  stream's partial requests on the same sub-second phase. **Design:
  stagger 1.125 s = 1.0 s × (1 + 1/8).** The eight first-wave streams' partial
  phases are then evenly spread over the second, and the first wave spans
  7.9 s, less than the longest file (11.5 s).

## 3. Design

### Factors

| Factor | Levels |
|---|---|
| Arrival | `sync`: all 8 first-wave streams start at t = 0 (existing behaviour). `stagger`: slot k's first stream starts at k × 1.125 s, k = 0..7 |
| Policy | `base` (both flags off), `prio` (`FINAL_PRIORITY`), `both` (`FINAL_PRIORITY` + `CANCEL_QUEUED_PARTIAL_ON_END`) |

In both arrival modes every later stream starts as soon as a slot frees
(closed loop, unchanged). The arrival schedule is the **only** difference
between the modes.

### Fixed in every run

* Commit: the registration commit (identical for all 12 runs).
* Model `base.en`, FP16, CUDA (T4); partial beam 1, final beam 5; partial
  interval 1.0 s; 1 inference worker; max 8 streams; `EVENT_TIMING=true`.
  Each run's configuration is verified via `/metrics` before it starts.
* Audio: the five Phase 1–3 utterances (PCM-hash verified against Phase 1).
* **Audio assignment:** in every block, stream i (0–19) plays file
  i mod 5, in sorted file order. This is identical across blocks, runs,
  policies and arrival modes.
* Client: 100 ms PCM16 frames at real-time pacing; 300 s per-stream deadline.

### Units and structure

| Unit | Definition | Count |
|---|---|---|
| Block | One 20-stream c=8 scenario: `c8`, `c8_2`, `c8_3`, `recovery` (B1–B4), always in that order | 4 per run |
| Run | Server start + config check + 2 warm-up streams + B1–B4 + provenance + server stop | 12 |
| Cell | Arrival × policy | 6 cells × 2 runs × 4 blocks = 8 blocks, 160 streams per cell |
| Total | | 48 blocks, 960 measured streams (+ 24 warm-up streams, excluded) |

`recovery` is the stress tool's existing fourth scenario. It has the same
workload as B1–B3 and adds one health check: TTFP p95 within max(10 %,
50 ms) of B1. Stress checks per run: 3 per block for B1–B3 (zero failed
streams, server idle, event-loop lag) plus 4 for B4, so 13 per run and
156 in total.

### Scheduler and server state between blocks

The server is **not** restarted between blocks of a run; it is restarted
between runs. After each block the stress tool waits until the server is
idle: no active streams, no buffered audio, nothing in flight or queued.
The next block starts only then, so the scheduler queue is empty and every
slot is free. No per-stream state carries over (sessions are released
and buffers cleared). Lifetime counters, distributions and level trackers
continue. Per-block values come from client records and from `/metrics`
snapshots taken immediately before and after each block. The GPU and model
stay warm across blocks. Block position is the same for every cell and is
reported per block.

### Run order (fixed)

| # | Round | Arrival | Policy |
|---:|---:|---|---|
| 1 | 1 | sync | base |
| 2 | 1 | stagger | prio |
| 3 | 1 | sync | both |
| 4 | 1 | stagger | base |
| 5 | 1 | sync | prio |
| 6 | 1 | stagger | both |
| 7 | 2 | stagger | both |
| 8 | 2 | sync | prio |
| 9 | 2 | stagger | base |
| 10 | 2 | sync | both |
| 11 | 2 | stagger | prio |
| 12 | 2 | sync | base |

Run k and run 13−k are the same cell, so every cell's mean position is 6.5
and linear drift over the session cancels, within each arrival mode as
well. The arrival mode alternates every run except at the mirror point
(runs 6–7). This balances linear drift, not curved drift: a U-shaped
session effect would favour cells at positions (1, 12) over (6, 7).
Estimated GPU time: about 30–35 minutes.

## 4. Measurements

Existing definitions are unchanged (`docs/measurements.md`). Added in this
registration, before any data, and defined in `docs/measurements.md` §10:

| Descriptor | Source | Definition |
|---|---|---|
| Observed active streams | client | time-weighted count of streams in [connect, done] over the block; share of time at 8; mean active at each stream's end |
| Server active / queue depth / in-flight | server | time-weighted distributions from `/metrics` level snapshots before and after the block |
| Queue depth at end | server | inference queue depth when each stream's `end` is processed, before its final is submitted |
| End-time synchronization | client | share of streams with another stream's end within ±250 ms, and mean such neighbours |
| Partial-request phase alignment | client | for each sent partial, other streams' partial requests within ±50 ms (mean, share ≥ 3); phase concentration R of request times modulo 1.0 s (1 = all aligned, ≈ 0 = evenly spread) |

## 5. Outcomes

* **Primary:** final latency p95, `both` − `base`, c=8, within each mode.
* **Secondary:** the same for `prio` − `base`; final queue wait p95; TTFP
  p50 and p95; partial queue wait p95; throughput; failures; WER v2 and
  final-text identity; final decode p95 (control).
* **Mechanism:** `prio` − `base`, within each mode:
  * *frequency*: share of streams whose own partial is unfinished
    (created, queued or running) at `end`, plus the queued-only share;
  * *duration*: mean wait for the own partial over all streams, plus the
    mean among unfinished streams and the p95.
* **Reporting:** every metric per block (8 per cell), per round, and pooled.
  Arrival modes are never pooled or compared to each other quantitatively.

## 6. Statistical analysis (exploratory)

* **Decision interval: block bootstrap.** Resample each arm's 8 blocks
  with replacement, pool their streams, and recompute the statistic;
  10,000 resamples, seed 0, percentile 95 % interval. It accounts for
  dependence within a block (shared queue, synchronized waves). It does not
  account for dependence between blocks of the same run or session, and 8
  clusters is few, so the interval is still optimistic. **All intervals are
  exploratory.**
* **Continuity interval:** the Phase 3 stream bootstrap (2,000 resamples,
  seed 0), which ignores dependence. Reported alongside, never used for
  decisions.
* **Consistency:** per-round pooled p95, and the number of paired blocks
  (same round and position) in which the candidate's p95 is lower (of 8).
* There is no correction for multiple comparisons. The primary outcome is
  one comparison per mode; everything else is descriptive.

## 7. Decision rules (registered; implemented in `analyze_arrival.adoption_verdict`)

For a comparison in one arrival mode, the outcome is **"benefit
demonstrated under this workload"** only if all of these hold:

* C1: block-bootstrap 95 % interval for the final-latency p95 difference
  lies entirely below 0;
* C2: candidate pooled final-latency p95 below baseline in **each** round;
* C3: zero failed candidate streams;
* C4: final text 100 % identical to the baseline's text for the same
  utterance, and WER v2 equal;
* C5: TTFP p95 regression ≤ 10 %;
* C6: final-decode p95 within ±10 % of baseline (control).

Otherwise the outcome is **"benefit not demonstrated under this
workload"**, reported with the effect size, interval and failed criteria.
This includes inconclusive results. It is never reported as the
optimization failing to generalize.

**Sync replication.** The `sync` arm repeats Phase 3's block structure.
If `both` vs `base` is not demonstrated under `sync`, the session did not
reproduce Phase 3. The `stagger` result is then reported but not
interpreted as evidence about generalization.

**Mechanism** (`prio` vs `base`, per mode; `analyze_arrival.mechanism_verdict`):
an increase in frequency (or duration) is **supported** if its
block-bootstrap interval lies entirely above 0 **and** the increase appears
in each round. Otherwise it is **not supported** in that mode.

**Health checks.** A failed stress check (e.g. the recovery TTFP check) is
reported as data. It does not trigger a rerun or an exclusion.

## 8. Predictions (registered)

Source: `benchmarks/arrival_sim.py`, a discrete-event model of the
closed-loop client and one inference worker. Decode costs were fitted on
Phase 3 data (partial 28.6 + 6.12 ms/s, final 35.9 + 7.19 ms/s,
log-normal noise). Values are 10th / 50th / 90th percentiles over 200
simulated replications of the full design. Full output:
`docs/experiments/phase4-predictions.json`.

**Model check against Phase 3.** The latency, queue and own-partial values
below were not fitted:

| Phase 3 c=8 | base | prio | both |
|---|---|---|---|
| final p95, model / measured (ms) | 515–559–604 / 562 | 351–385–429 / 390 | 250–267–290 / 266 |
| final queue p95, model / measured (ms) | 318–351–388 / 331 | 236–268–307 / 268 | 132–153–174 / 148 |
| unfinished own partial at end, model / measured | 8 % / 8 % | 20 % / 20 % | 20 % / 20 % |

**Phase 4 predictions:**

| Cell | Final p95 (ms) | Final queue p95 | TTFP p95 | Unfinished own partial | Mean wait own |
|---|---|---|---|---|---|
| sync base | 551 / 573 / 593 | 353 / 375 / 396 | 1152 / 1159 / 1168 | 8 / 8 / 9 % | 11 / 12 / 13 ms |
| sync prio | 387 / 412 / 435 | 265 / 292 / 316 | 1173 / 1182 / 1191 | 18 / 19 / 20 % | 36 / 39 / 41 ms |
| sync both | 261 / 274 / 286 | 145 / 156 / 169 | 1173 / 1182 / 1191 | 18 / 19 / 20 % | 3 / 4 / 5 ms |
| stagger base | 162 / 167 / 176 | 72 / 78 / 84 | 967 / 974 / 980 | 0 % | 0 ms |
| stagger prio | 159 / 165 / 172 | 64 / 70 / 77 | 969 / 976 / 982 | 0 % | 0 ms |
| stagger both | 159 / 165 / 172 | 64 / 70 / 77 | 969 / 976 / 982 | 0 % | 0 ms |

| Workload (base) | sync | stagger |
|---|---|---|
| Observed active streams, mean / share of time at 8 | 7.4 / 86 % | 5.8 / 47 % |
| Ends with another end within 250 ms | 80 % | 35 % |
| Partial-request neighbours within 50 ms / phase R | 2.78 / 0.54 | 0.19 / 0.19 |
| Server queue depth, time-weighted mean | 0.75 | 0.05 |

**Predicted outcomes:**

1. `sync`, `both` vs `base`: **demonstrated** (Δ p95 −318 / −299 / −280 ms).
   This replicates Phase 3.
2. `sync`, `prio` vs `base`: Δ p95 −194 / −160 / −131 ms; likely
   demonstrated.
3. `stagger`, `both` and `prio` vs `base`: **not demonstrated under this
   workload.** Staggering alone is predicted to remove most of the
   queueing (baseline p95 about 167 ms), leaving the policies nothing to
   act on (Δ p95 −7 / −2 / 0 ms).
4. Mechanism: frequency increase **supported** under `sync` (+9.4 /
   +10.6 / +11.9 percentage points, about 8 % → 19 %), duration increase
   supported under `sync`, and **neither supported** under `stagger`
   (no unfinished own partials predicted).
5. TTFP p95 under `sync` rises by about 2 % with priority (Phase 3:
   +2.5 %), within C5. Final decode, throughput, WER and final text are
   unchanged in all cells.

**What would contradict the model:** a `stagger` baseline p95 well above
about 200 ms, any unfinished own partials under `stagger`, or a
demonstrated benefit under `stagger`. Each would mean the real system
queues more than the model's single-worker, fixed-cost assumptions allow.
Examples are GPU clock changes, Python scheduling delays, or content-dependent
decode time. Contradictions are findings to report, not reasons to rerun.

## 9. Data handling

* No reruns, except a run that produces no data because of an
  infrastructure failure (e.g. a Colab disconnect). That run is repeated
  once, at the end of the session, and documented.
* No exclusions. Failed streams count as failures (C3).
* The complete result archive is committed byte-for-byte with SHA-256
  checksums, after scanning for credentials and transcript text in logs.
* Any change to analysis code or decision rules after data collection is
  a documented deviation, reported alongside the registered analysis.

## 10. Limitations

* One T4, one Colab session, one inference worker, FP16 only, five
  utterances of 4–11 s.
* Staggered first waves are one deterministic, minimally bursty schedule
  (partial phases evenly spread). They are not real traffic. Random
  arrivals would fall between the two arms, and open-loop arrivals at
  c=8 would cause admission rejections that confound the comparison.
* The predictions come from a model whose cost parameters were fitted on
  Phase 3. The design choices in section 2 used simulation, not the
  real system.
* Block bootstrap intervals with 8 clusters per arm are optimistic, and
  session-level dependence is not modelled.

## 11. Running it

Commands: `benchmarks/COLAB.md`, "Phase 4: staggered arrival
(registered)". Analysis: `python -m benchmarks.analyze_arrival <out>
--audio-dir eval_audio5`.
