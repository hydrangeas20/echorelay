# Phase 4 results: final-priority scheduling under synchronized and staggered arrivals

**Status:** complete. The experiment was registered before data collection
in [the Phase 4 registration](../../experiments/phase4-staggered-arrival.md) at commit
`b9e86349b6e5fa4fac788c27ed1c05c312737c99`. It was executed once on one Colab T4,
with no reruns, exclusions or deviations. Both scheduling flags remain
**off by default**.

Labels used throughout this report:

| Label | Meaning |
|---|---|
| **Registered** | Fixed before data collection: design, decision criteria C1–C6, the mechanism test, and the model predictions |
| **Measured** | Computed from the committed raw files by the registered analysis code (`benchmarks/analyze_arrival.py` at the registration commit) |
| **Exploratory interval** | Bootstrap 95 % intervals. None is a significance test. The block bootstrap is the registered decision interval; the stream bootstrap is shown for continuity with Phase 3 |
| **Model prediction** | 10th / 50th / 90th percentiles from `benchmarks/arrival_sim.py`, registered in [`phase4-predictions.json`](../../experiments/phase4-predictions.json) |
| **Post-hoc** | Computed or interpreted after the data were seen. It is not part of any registered decision |

## Summary

| Arrival mode | Comparison | Δ final-latency p95 | Block-bootstrap 95 % (decision) | Registered outcome |
|---|---|---:|---|---|
| sync | `both` vs `base` | −327.5 ms (−52.9 %) | [−343.9, −301.8] ms | **Benefit demonstrated under this workload** (C1–C6 pass) |
| sync | `prio` vs `base` | −199.5 ms (−32.2 %) | [−228.2, −174.4] ms | **Benefit demonstrated under this workload** (C1–C6 pass) |
| stagger | `both` vs `base` | −19.6 ms (−10.7 %) | [−28.4, **+0.2**] ms | **Benefit not demonstrated under this workload.** C1 fails |
| stagger | `prio` vs `base` | −5.4 ms (−3.0 %) | [−14.6, +15.0] ms | **Benefit not demonstrated under this workload.** C1 and C2 fail |

* **Demonstrated (sync):** with synchronized first waves (Phase 3's
  workload), both candidates cut final-latency p95, as in Phase 3. The
  reduction held in every one of the 8 paired blocks, and in both rounds.
  `both` is the larger effect. The cost is a TTFP p95 increase of about 2 %.
* **Not demonstrated (stagger):** with first waves staggered by 1.125 s,
  the baseline final p95 was already 183 ms (vs 619 ms in sync). `both`
  still had a lower point estimate in both rounds and in 6 of 8 paired
  blocks. However, its decision interval's upper bound is **+0.205 ms**, so
  it does not lie entirely below 0 and C1 fails. This is a close miss.
  Under the registered rules it is reported as *not demonstrated under
  this workload*, not as a small benefit. It is not evidence that the
  optimization fails to generalize.
* **Mechanism:** the registered mechanism (priority makes a stream's own
  partial more often unfinished at `end`, and makes it wait longer) is
  **supported under sync** and **not supported under stagger**. Under
  stagger, no stream in any cell had an unfinished own partial at `end`.
* **Health:** 960 of 960 streams completed. WER v2 was 5.31 % and final
  text was identical in every cell. **155 of 156 stress checks passed.
  One recovery check failed; see the next section.**
* **Predictions:** all five registered qualitative predictions held. Several
  quantitative values fell outside their model prediction bands (see
  [Predictions vs observations](#predictions-vs-observations)).

## Failed health check: run 10 recovery TTFP

> **One of 156 registered stress checks failed.** Run `10_r2_sync_both`,
> `recovery` block: *"TTFP p95 within max(10%, 50 ms) of first run"*.
> First block TTFP p95 was 1.159 s; recovery TTFP p95 was **1.285 s**.
> That is an increase of **0.126 s** against a tolerance of **0.116 s**.
> The stress driver exited with status 1 for this run.

Per the registration (§7, health checks), the failure is reported as data.
It did not trigger a rerun or exclusion, and the block is included in every
analysis above and below. The driver console contains no deviation warning
for this run.

**What the data show (post-hoc).** This block's TTFP p95 is the highest of
all 24 sync blocks. The next highest is 1,211 ms (sync prio, run 1
B2). Its median TTFP was 954 ms. The high p95 comes from the two slowest
first partials in the block. Both were second-wave streams that started about 9.3 s into the block:

| Stream | File | TTFP | First-partial queue wait | First-partial decode |
|---:|---|---:|---:|---:|
| 11 | 1272-135031-0001 | 1,283 ms | 348 ms | 32 ms |
| 10 | 1272-135031-0000 | 1,330 ms | 382 ms | 44 ms |

Their decode times are normal. The excess is queue wait for the first
partial. **The cause is not established beyond these observed queue
waits.** The data do not show why those two requests queued longer in this
block than in the equivalent blocks. Candidate explanations are untested
and are not offered as findings. Other blocks of the same cell had TTFP p95
of 1,159–1,188 ms (see
[supplementary §3](supplementary.md#3-ttfp-p95-per-sync-block-ms-and-run-10s-recovery-block)).

## Provenance and verification

| Item | Value |
|---|---|
| Registered commit | `b9e86349b6e5fa4fac788c27ed1c05c312737c99` (`git_dirty: false` in all 12 runs) |
| Plan / deviations | `phase4` / `[]`. Plan constants and executed run order equal the registered `PHASE4` and `PHASE4_ORDER` |
| Hardware | Tesla T4, driver 580.82.07, 15,360 MiB, 1,590 MHz max SM clock; one Colab session |
| Server | faster-whisper `base.en`, CUDA, `float16`; 1 inference worker; max 8 streams; event timing on; Python 3.13.15 |
| Flags per policy | `base`: both off. `prio`: `FINAL_PRIORITY` on. `both`: `FINAL_PRIORITY` and `CANCEL_QUEUED_PARTIAL_ON_END` on. All checked in each run's `provenance.json` |
| Evidence archive | `phase4_evidence.zip`, 843,442 bytes, SHA-256 `3e8a64065e9715698e43fc8e1b45893ef6b55fa439b6766b7b926340ba4056e2` |
| Committed evidence | [`raw/`](raw/): the archive's 105 files, byte for byte, with [`raw/SHA256SUMS`](raw/SHA256SUMS) (6.4 MB uncompressed) |

The following checks were completed before this report was written:

* **Archive integrity.** The archive SHA-256 matched the value supplied
  with it. Every committed file matches its `SHA256SUMS` entry.
* **Design execution.**
  * Each of the 48 blocks had the registered configuration.
  * Stream *i* played file *i* mod 5.
  * First-wave start offsets were within 5.5 ms of the registered schedule.
* **Counts.** There were 960 streams with 0 failures, 160 per cell.
* **Reproducibility.** Re-running the registered analysis at the
  registration commit reproduced `analysis.json` and `analysis.md` byte for
  byte.
* **Credential and data scan.**
  * No credentials, tokens, e-mail addresses or Drive paths were found.
  * The only filesystem path is the generic Colab `/content/echorelay`.
  * Server logs contain no transcript text.
  * `provenance.json` includes the environment's pip freeze.
* **Server logs.** All 12 logs (`server.log`, 1.6 MB in total) are committed
  unmodified so the checksums hold. About 70 % of their lines are library
  and HTTP-access noise. The registered analysis needs the logs: it joins
  each stream's `stream_summary` end state to the client records.

> **Evidence correction.** The first version of this PR committed only 93
> of the 105 listed files. The repository's `*.log` ignore rule silently
> dropped all 12 `server.log` files. The checks above had passed only
> because ignored copies were present in the author's working tree. On a
> fresh clone, `sha256sum -c` failed, and the analysis ran without the
> end-state join, exited 0 and produced a different `analysis.json`.
> The 12 logs are now committed unmodified, taken from the original archive
> (SHA-256 above), and match the hashes that `raw/SHA256SUMS` has listed
> since the first version. The verification was repeated on a fresh clone:
> all 105 files match, and the registered analysis reproduces
> `analysis.json` and `analysis.md` byte for byte. No result changed.
>
> **Change to analysis code after data collection.** To prevent a repeat,
> `benchmarks/analyze_arrival.py` now stops with an error (exit status 1)
> instead of silently continuing when any of these is missing:
>
> * a run's `server.log`;
> * a `stream_summary` line for a stream that received a session;
> * a block result file;
> * a reference transcript for a played utterance, when `--audio-dir` is
>   given.
>
> Without `--audio-dir` it prints a warning that WER is not computed.
> These checks add no computation and alter no statistic, criterion or
> output. With complete evidence, the outputs are byte-identical to those
> of the registration commit `b9e8634`.

**Display rounding.** `raw/phase4_results/analysis.md` rounds intervals to
whole milliseconds. It therefore shows the stagger `both` decision interval
as `[-28, +0]`. The unrounded values in `analysis.json` are
[−28.392, +0.205] ms. The registered verdict was computed from the unrounded
values, and this report quotes them.

## Design (registered; summary)

Full design: [registration](../../experiments/phase4-staggered-arrival.md).

* **Factors.** There are 3 policies (`base`, `prio`, `both`) and 2 arrival
  modes:
  * `sync`: all 8 first-wave streams start together.
  * `stagger`: slot *k* starts at *k* × 1.125 s.
  * In both modes, later streams start when a slot frees.
* **Runs and blocks.**
  * There are 12 runs, one per cell per round. Rounds 1 and 2 follow a
    fixed mirror order.
  * Each run has four 20-stream blocks at concurrency 8: `c8`, `c8_2`,
    `c8_3` and `recovery`.
  * Every block starts from an idle server.
  * The total is 8 blocks and 160 streams per cell.
* **Audio.** Five Mini LibriSpeech `dev-clean-2` utterances of 4–11 s,
  streamed at real-time pace in 100 ms frames ([attribution](DATA-ATTRIBUTION.md)).
* **Decision rule.** A comparison counts as *benefit demonstrated under
  this workload* only if all of C1–C6 hold:
  * C1: the block-bootstrap 95 % interval for Δ final p95 lies entirely
    below 0.
  * C2: the candidate p95 is lower in each round.
  * C3: there are zero failed candidate streams.
  * C4: final text is identical and WER is equal.
  * C5: the TTFP p95 regression is ≤ 10 %.
  * C6: final-decode p95 is within ±10 % (control).
* **Mode comparisons.** Arrival modes are never pooled or compared
  quantitatively with each other.

## Registered decisions

| Criterion | sync `both` | sync `prio` | stagger `both` | stagger `prio` |
|---|---|---|---|---|
| C1 block interval for Δ p95 < 0 | **pass** [−343.9, −301.8] | **pass** [−228.2, −174.4] | **FAIL** [−28.4, +0.2] | **FAIL** [−14.6, +15.0] |
| C2 lower p95 in each round (base → cand., ms) | **pass** 614.7 → 291.4; 619.3 → 285.6 | **pass** 614.7 → 420.2; 619.3 → 398.9 | **pass** 183.8 → 173.8; 174.3 → 163.0 | **FAIL** 183.8 → 182.0; 174.3 → **175.2** |
| C3 zero failed streams | pass (0/160) | pass (0/160) | pass (0/160) | pass (0/160) |
| C4 final text identical, WER equal | pass (100 %, 5.31 %) | pass | pass | pass |
| C5 TTFP p95 regression ≤ 10 % | pass (+2.09 %) | pass (+2.43 %) | pass (−1.46 %) | pass (−1.93 %) |
| C6 final decode p95 within ±10 % | pass (−0.58 %) | pass (−0.64 %) | pass (−0.55 %) | pass (−0.12 %) |
| **Outcome** | **demonstrated** | **demonstrated** | **not demonstrated** | **not demonstrated** |
| Stream-bootstrap 95 % (continuity, not used) | [−359.7, −273.2] | [−247.1, −66.7] | [−28.7, +6.4] | [−21.1, +24.1] |
| Paired blocks with lower candidate p95 | 8 of 8 | 8 of 8 | 6 of 8 | 3 of 8 |

**Sync replication.** The registration made the stagger result
interpretable only if sync `both` vs `base` was demonstrated (i.e. the
session reproduced Phase 3). It was demonstrated.

## Measured latency by cell

Pooled over 8 blocks (160 streams per cell). Values in ms. "Wait own" is the
time from `end` until the stream's own in-flight or queued partial finished.

| Cell | Final p50 / p95 / max | Final p95 per round | Final queue p95 | Wait own p95 | Final decode p95 | TTFP p50 / p95 | Partial queue p95 |
|---|---|---|---:|---:|---:|---|---:|
| sync base | 128 / 618.9 / 651 | 614.7 / 619.3 | 390 | 59.6 | 137.3 | 960 / 1,150.7 | 358 |
| sync prio | 132 / 419.4 / 637 | 420.2 / 398.9 | 282 | 282.6 | 136.4 | 1,007 / 1,178.7 | 430 |
| sync both | 122 / 291.4 / 353 | 291.4 / 285.6 | 165 | 60.6 | 136.5 | 961 / 1,174.7 | 368 |
| stagger base | 122 / 183.4 / 226 | 183.8 / 174.3 | 82 | 0.4 | 141.5 | 937 / 988.7 | 55 |
| stagger prio | 118 / 178.0 / 225 | 182.0 / 175.2 | 77 | 0.4 | 141.3 | 936 / 969.6 | 61 |
| stagger both | 118 / 163.8 / 233 | 173.8 / 163.0 | 77 | 0.4 | 140.7 | 938 / 974.3 | 62 |

Final-latency p95 for each block (ms). Blocks are ordered B1–B4 =
`c8`, `c8_2`, `c8_3`, `recovery`.

| Cell | r1 B1 | r1 B2 | r1 B3 | r1 B4 | r2 B1 | r2 B2 | r2 B3 | r2 B4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sync base | 615 | 590 | 624 | 594 | 567 | 636 | 619 | 608 |
| sync prio | 390 | 426 | 398 | 383 | 400 | 370 | 380 | 400 |
| sync both | 288 | 268 | 263 | 294 | 273 | 286 | 282 | 267 |
| stagger base | 193 | 168 | 176 | 149 | 161 | 164 | 174 | 177 |
| stagger prio | 176 | 178 | 193 | 173 | 171 | 176 | 165 | 164 |
| stagger both | 164 | 162 | 174 | 166 | 179 | 163 | 153 | 161 |

Under stagger, block-to-block spread within a cell (e.g. base 149–193 ms) is
larger than the differences between cells. This is why the stagger intervals
are wide relative to the effects.

Other within-mode Δ p95 values, block-bootstrap 95 % (exploratory):

| Δ p95 (ms) | sync both | sync prio | stagger both | stagger prio |
|---|---|---|---|---|
| Final queue wait | −225 [−230, −210] | −108 [−118, −94] | −5 [−25, +5] | −5 [−26, +3] |
| TTFP | +24 [+4, +56] | +28 [+6, +50] | −14 [−22, +11] | −19 [−26, +10] |

Under sync, the TTFP increase has an interval above zero. It is a small
measured cost of both policies, well within C5.

## Mechanism (registered test: `prio` vs `base`)

A stream's own partial is *unfinished* at `end` if it is created, queued or
running when `end` is processed. Under `prio`, a waiting final outranks
waiting partials. The registered hypothesis was that this leaves more
streams' own partials unfinished at `end`, and that those partials wait
longer.

| Cell | End states (done / queued / running) | Unfinished | Queued | Mean wait own, all / among unfinished |
|---|---|---:|---:|---|
| sync base | 144 / 8 / 8 | 10.0 % | 5.0 % | 13.1 / 128.4 ms |
| sync prio | 128 / 24 / 8 (24 promoted) | 20.0 % | 15.0 % | 47.4 / 236.3 ms |
| sync both | 128 / 18 / 14 (18 cancelled) | 20.0 % | 11.25 % | 6.3 / 30.7 ms |
| stagger (all policies) | 160 / 0 / 0 | 0 % | 0 % | 0.2–0.3 / – ms |

| `prio` − `base` | sync | stagger |
|---|---|---|
| Unfinished share | +10.0 pp [+10.0, +10.0]; round 1 10 % → 20 %, round 2 10 % → 20 % | +0 pp [+0, +0] |
| Queued share | +10.0 pp [+10.0, +10.0] | +0 pp |
| Mean wait own | +34.35 ms [+32.03, +37.18]; round 1 12.34 → 49.61, round 2 13.86 → 45.28 ms | ≈ 0 |
| Registered verdict | frequency **supported**, duration **supported** | **neither supported** |

**Zero-width intervals are a property of the workload, not statistical
certainty.** The intervals for the sync unfinished and queued shares have
zero width. Every block of a cell had the same number of streams with an
unfinished own partial:

| Cell | Per block (8 blocks) | Files |
|---|---|---|
| sync base | 2, 2, 2, 2, 2, 2, 2, 2 | always `…0001` and `…0002` |
| sync prio | 4, 4, 4, 4, 4, 4, 4, 4 | always `…0001` and `…0002` |
| sync both | 4, 4, 4, 4, 4, 4, 4, 4 | always `…0001` and `…0002` |

Each block replays the same 20-stream schedule of 5 fixed files from an
idle server. The same streams therefore reach `end` in the same scheduler
state each time. Resampling blocks that are identical in this count cannot
produce variation, so the interval collapses to a point. The result shows
the effect is **reproducible under this deterministic workload**. It does
not show the effect size is known with certainty, and it says nothing about
variability under other arrivals or audio.
Source: [supplementary §2](supplementary.md#2-streams-with-an-unfinished-own-partial-at-end-per-block-b1-b4-round-1-then-round-2).

**Descriptive only, not registered.** Sync `both` vs `base`:
* The unfinished share rose by the same +10 pp, and the queued share by
  +6.25 pp [+5.0, +8.1].
* The mean wait for the own partial *fell* by 6.8 ms [−8.1, −5.5].
* This is because `both` cancels the queued own partials at `end` (18 of
  160) instead of running them before the final.

## Workload descriptors and what the comparison does not isolate

Block means. Registered model medians are shown in parentheses for the
baselines.

| Cell | Observed active, mean / time at 8 | Server queue depth, mean | Queue depth at `end` | Ends with another end ±250 ms | Partial neighbours ±50 ms / phase R |
|---|---|---:|---:|---:|---|
| sync base | 7.40 (7.41) / 86 % (86 %) | 0.76 (0.75) | 1.19 | 80 % (80 %) | 2.76 (2.78) / 0.54 (0.54) |
| sync prio | 7.37 / 86 % | 0.81 | 1.20 | 70 % | 2.83 / 0.51 |
| sync both | 7.42 / 87 % | 0.75 | 1.15 | 70 % | 2.68 / 0.40 |
| stagger base | 5.80 (5.80) / 47 % (47 %) | 0.06 (0.05) | 0.08 | 35 % (35 %) | 0.21 (0.19) / 0.18 (0.19) |
| stagger prio | 5.81 / 48 % | 0.07 | 0.12 | 35 % | 0.23 / 0.19 |
| stagger both | 5.80 / 48 % | 0.07 | 0.12 | 35 % | 0.21 / 0.20 |

The two arrival modes differ in more than arrival phase:

* Staggered runs spend about half their time below 8 active streams. Sync
  runs spend 86 %. So staggered runs have **lower average concurrency**
  (5.8 vs 7.4).
* Staggered runs have about one-tenth of the server queue depth, and much
  less queue depth when streams end.

The experiment therefore **does not isolate arrival-phase alignment as the
sole causal factor**. "Staggered arrivals" here means the whole
combination: lower concurrency, less contention, and spread partial
phases. The result supports this narrow statement: *under this staggered
workload, the queueing that the policies act on was largely absent, and no
benefit was demonstrated.* Each mode's comparisons hold within that mode
only.

## Predictions vs observations

The registered **decision criteria** (C1–C6, mechanism test) decide
outcomes. The registered **model prediction bands** are 10th–90th
percentile ranges from a simulator fitted on Phase 3. They are tests of
the model, not of the policies. A value outside its band says the model was
quantitatively off. It does not change any outcome.

**Qualitative predictions (registered §8):**

| # | Prediction | Observed | Held? |
|---|---|---|---|
| 1 | sync `both` vs `base` demonstrated | demonstrated | yes |
| 2 | sync `prio` vs `base` likely demonstrated | demonstrated | yes |
| 3 | stagger `both`, `prio` not demonstrated | not demonstrated (both) | yes |
| 4 | mechanism supported under sync, not under stagger; no unfinished own partials under stagger | as predicted; 0 of 480 stagger streams unfinished | yes |
| 5 | sync TTFP p95 +~2 % with priority; final decode, WER, final text unchanged | +2.09 %, +2.43 %; decode within 0.7 %; WER and text identical | yes, with the throughput component evaluated post-hoc below |

None of the registered "would contradict the model" conditions occurred:

* The stagger baseline p95 was 183 ms, below about 200 ms.
* No stagger stream had an unfinished own partial.
* No stagger benefit was demonstrated.

**Quantitative band misses (model prediction 10 / 50 / 90 vs measured):**

| Quantity | Cell | Band | Measured | Position |
|---|---|---|---:|---|
| Final p95 (ms) | sync base | 551 / 573 / 593 | 619 | above |
| | sync both | 261 / 274 / 286 | 291 | above |
| | stagger base | 162 / 167 / 176 | 183 | above |
| | stagger prio | 159 / 165 / 172 | 178 | above |
| Δ final p95 (ms) | sync prio | −194 / −160 / −131 | −200 | below (larger benefit) |
| | sync both | −318 / −299 / −280 | −328 | below (larger benefit) |
| | stagger both | −7 / −2 / 0 | −20 | below (larger point estimate; still not demonstrated) |
| Final queue p95 (ms) | stagger both | 64.2 / 70.1 / 76.8 | 76.9 | marginally above |
| TTFP p95 (ms) | sync base | 1,152 / 1,159 / 1,168 | 1,151 | marginally below |
| | stagger base | 967 / 974 / 980 | 989 | above |
| Unfinished own partial | sync base | 7.5 / 8.1 / 9.4 %* | 10.0 % | above |
| Mean wait own (ms) | sync base | 10.8 / 11.9 / 13.0 | 13.1 | above |
| | sync prio | 36.5 / 38.8 / 41.0 | 47.4 | above |
| | sync both | 2.8 / 3.6 / 4.6 | 6.3 | above |

\* Values from `phase4-predictions.json`; the registration's rounded table shows 8 / 8 / 9 %.

All other predicted values were inside their bands: final p95 for sync prio
and stagger both, Δ p95 for stagger prio, the remaining queue and TTFP
values, and the remaining unfinished shares. Stagger mean wait own measured
0.2–0.3 ms against a predicted 0 ms, a trivial difference.

*Post-hoc reading:*
* The model ran consistently slightly optimistic on absolute final
  latency, by about 3–8 %.
* It underestimated how long own partials wait.
* It underestimated the size of the policy effects.

These are model-calibration observations. They are not new findings about
the system.

## Supplementary: throughput (post-hoc)

> **Post-hoc, outside the registered analysis.** Throughput was a
> registered secondary outcome, but the registered analysis code did not
> compute it. The values below are each block's existing stress-summary
> throughput (completed streams per second), averaged over the 8 blocks per
> cell by [`supplementary.py`](supplementary.py). Compare only **within an
> arrival mode**: each staggered block's wall clock includes the 7.9 s
> stagger ramp. Neither mode measures server capacity.

| Arrival | base | prio | both |
|---|---:|---:|---:|
| sync | 0.8414 | 0.8439 (+0.3 %) | 0.8525 (+1.3 %) |
| stagger | 0.6690 | 0.6699 (+0.1 %) | 0.6693 (+0.0 %) |

Block ranges (min–max): sync base 0.8402–0.8426, sync both 0.8506–0.8542.

* Registered prediction 5 said throughput would be unchanged. Under
  stagger it was. Under sync, `both` was 1.3 % higher, and the block ranges
  do not overlap.
* Clients here are closed-loop at real-time pace. Each slot's throughput
  is therefore about 1 / (stream duration + final latency)
  ([measurements §1](../../measurements.md#1-client-side-stream-metrics-benchmarksload_testpy)).
  A shorter final latency frees a slot sooner.
* The small increase is consistent with the measured final-latency
  reduction. It is not evidence of greater server capacity, and no capacity
  claim is made.

## Limitations

* **One session, one GPU, one configuration.** One Colab T4 was used, with
  one inference worker, FP16 `base.en`, and five read-speech utterances of
  4–11 s. The results do not describe other hardware, models, utterance
  lengths, concurrency limits or worker counts.
* **Two deterministic arrival schedules, not traffic.** Real, random or
  open-loop arrivals were not tested. Outcomes are stated "under this
  workload" only.
* **Confounded arrival comparison.** Arrival modes differ in concurrency and
  contention as well as phase (see [above](#workload-descriptors-and-what-the-comparison-does-not-isolate)).
* **Intervals are exploratory and optimistic.**
  * There are only 8 blocks per arm.
  * Dependence between blocks of one run or one session is not modelled.
  * There is no multiple-comparison correction.
  * Zero-width intervals reflect the repeated workload.
* **Stagger `both` is a near miss.** Its interval's upper bound is
  +0.205 ms. A different session could plausibly land on either side of
  zero. This experiment does not distinguish a small stagger benefit from
  none.
* **One failed health check.** The run 10 recovery TTFP check failed. Its
  cause is not established beyond the two observed first-partial queue
  waits.
* **Throughput was computed post-hoc.**

## Reproduction

From the repository root, on a fresh clone of this branch. The registered
computation is unchanged from `b9e8634`; only the fail-fast evidence checks
described above were added.

```bash
# Verify the committed evidence (run on a fresh clone: an ignored local copy
# of a file would also pass this check)
(cd docs/benchmarks/phase4-t4/raw && sha256sum -c SHA256SUMS)

# Registered analysis. Writes nothing into raw/ unless --output/--markdown is given.
python -m benchmarks.analyze_arrival docs/benchmarks/phase4-t4/raw/phase4_results \
  --audio-dir docs/benchmarks/phase4-t4/raw/eval_audio5 \
  --output /tmp/phase4_analysis.json --markdown /tmp/phase4_analysis.md
cmp /tmp/phase4_analysis.json docs/benchmarks/phase4-t4/raw/phase4_results/analysis.json
cmp /tmp/phase4_analysis.md   docs/benchmarks/phase4-t4/raw/phase4_results/analysis.md

# Post-hoc supplementary tables
python docs/benchmarks/phase4-t4/supplementary.py docs/benchmarks/phase4-t4/raw/phase4_results
```

Re-running the experiment itself (GPU) follows
[the registration §11](../../experiments/phase4-staggered-arrival.md#11-running-it) and
[benchmarks/COLAB.md](../../../benchmarks/COLAB.md). A new session is a new
experiment and would not replace these results.

## Files

| Path | Contents |
|---|---|
| [`raw/phase4_results/`](raw/phase4_results/) | `manifest.json`, driver `console.txt`, registered `analysis.json` / `analysis.md`, and 12 run directories. Each run directory holds 4 block result files, `stress_report.json`, `provenance.json`, `console.txt` and `server.log` |
| [`raw/eval_audio5/`](raw/eval_audio5/) | Reference transcripts for the 5 utterances ([attribution](DATA-ATTRIBUTION.md)); audio not committed |
| [`raw/SHA256SUMS`](raw/SHA256SUMS) | Checksums for all 105 evidence files |
| [`supplementary.py`](supplementary.py), [`supplementary.md`](supplementary.md) | Post-hoc script and its output |
| [`DATA-ATTRIBUTION.md`](DATA-ATTRIBUTION.md) | LibriSpeech source and license. Kept outside `raw/` so the evidence checksums are unchanged |
