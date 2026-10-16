# Phase 3: scheduling policies on an NVIDIA T4 (matched comparison)

**Status: one session, synchronized-start workload.** Both flags remain
**off by default** pending broader validation (see "Next validation").
Design, invariants and the predictions written before this run:
[`docs/scheduling-design.md`](../../scheduling-design.md).

All numbers come from [`raw/`](raw/), the Colab output byte-for-byte
(60 files, SHA-256 in [`raw/SHA256SUMS`](raw/SHA256SUMS)). Regenerating
`comparison.json` locally from these files with
`benchmarks/compare_policies.py` reproduces the Colab output exactly,
including the bootstrap intervals (fixed seed).

## Configuration

| | |
|---|---|
| Commit | `53161b588a716b2c3c1ce847ec46843ba1442d62`, clean tree in all 8 runs (`provenance.json`) |
| GPU | Tesla T4, 15,360 MiB, driver 580.82.07, Colab |
| Model | Whisper `base.en`, faster-whisper, `float16`; partial beam 1, final beam 5 |
| Server (every run) | 1 inference worker, max 8 streams, partial interval 1.0 s, `ECHORELAY_EVENT_TIMING=true`; only the two policy flags differ, each verified via `/metrics` before the run |
| Policies | `base` (both off), `prio` (`FINAL_PRIORITY`), `cancel` (`CANCEL_QUEUED_PARTIAL_ON_END`), `both` |
| Order | counterbalanced: base, prio, cancel, both, both, cancel, prio, base |
| Workload per run | `benchmarks.stress --concurrency 4 8 --burst-extra 0`: c=4 (20 streams), c=8 (20), c=4 again as recovery (20); 2 unpaced warm-up streams |
| Audio | the five Phase 1 utterances (PCM-hash verified), replayed; real-time pacing, 100 ms frames; all streams of a wave start together |
| Samples per policy | 80 streams at c=4, 40 at c=8 |

## Results

8/8 runs completed; 80/80 stress checks passed; 0 failed streams.

| Policy | c=8 final latency p95 | c=8 final queue p95 | c=8 TTFP p95 | c=4 final latency p95 | Final text = base | WER v2 |
|---|---:|---:|---:|---:|---:|---:|
| base | 562.4 ms | 331.4 ms | 1,143.7 ms | 153.0 ms | 100 % | 5.31 % |
| prio | 389.8 ms | 267.7 ms | 1,176.2 ms | 143.0 ms | 100 % | 5.31 % |
| cancel | 475.2 ms | 341.9 ms | 1,158.3 ms | 156.2 ms | 100 % | 5.31 % |
| **both** | **266.3 ms** | **148.1 ms** | 1,172.3 ms | 156.0 ms | 100 % | 5.31 % |

Difference in p95 from `base`, with bootstrap 95 % interval:

| | c=8 final latency | c=8 final queue | c=8 TTFP | c=8 final decode (control) |
|---|---|---|---|---|
| prio | −173 [−307, +72] | −64 [−241, +80] | +33 [−28, +89] | −8 [−22, +18] |
| cancel | −87 [−179, +73] | +11 [−97, +66] | +15 [−67, +75] | −8 [−22, +53] |
| **both** | **−296 [−358, −142]** | **−183 [−294, −119]** | +29 [−29, +86] | −11 [−25, +17] |

The p95 per round is consistent for every policy. c=8 final latency:
base 563 / 560, prio 376 / 391, cancel 474 / 475, both 253 / 268 ms.
At c=4 no policy differs from base beyond noise. Every interval includes
zero, and the largest p95 change is 10 ms. The final-decode control is
unchanged everywhere. Throughput is unchanged (0.455 / ~0.85 streams/s at
c=4 / c=8), as expected under real-time pacing. Full tables:
[`raw/comparison.md`](raw/comparison.md).

## Why `both` beats either flag alone (post-hoc analysis)

Each c=8 stream was joined to its server `stream_summary` log line by
session id. That line records the state of the stream's own partial when
`end` arrived. Tool: `benchmarks/end_state_breakdown.py`; output:
[`end_state_c8.md`](end_state_c8.md). The same breakdown at c=4 shows every
own partial already finished at `end` under every policy, so neither policy
had anything to act on there. That fits the unchanged c=4 numbers.

| Policy (c=8 p95) | Own partial at `end` | n of 40 | Wait own partial p50 / max | Final latency p50 / max |
|---|---|---:|---:|---:|
| base (562) | done | 37 | 0 / 1 | 114 / **560** |
| | queued | 2 | 206 / 224 | 569 / **574** |
| prio (390) | done | 32 | 0 / 0 | 115 / 252 |
| | **queued (promoted)** | **6** | **257 / 394** | 377 / **518** |
| cancel (475) | done | 38 | 0 / 0 | 126 / **508** |
| | queued (cancelled) | 2 | 0 / 1 | 473 / 474 |
| both (266) | done | 32 | 0 / 1 | 113 / 238 |
| | queued (cancelled) | 4 | 0 / 0 | 275 / **317** |

Streams whose own partial was *running* at `end` (1, 2, 0 and 4 per
policy) are omitted. Under `prio` and `both` they stay well below the p95
(max 180 and 202 ms). Base's single such stream (562 ms) sits at its p95,
part of the same queueing tail. See the full table.

1. **Two separate causes make up the base tail.** Finals queue behind other
   streams' partials (the `done` class, up to 560 ms). Streams whose own
   partial is still queued at `end` also wait for it, then queue again
   (569–574 ms).
2. **`prio` fixes the first cause and amplifies the second.** Because finals
   now overtake partials, partials finish later. Three times as many streams
   (2 → 6 of 40) still have their own partial queued at `end`. Promotion
   puts that partial at the back of the final class, behind finals from
   streams that ended at the same moment, so it still waits (median
   257 ms). These 6 streams are 15 % of the sample, so the p95 falls inside
   this class: 390 ms. **The design assumed inheritance alone would remove
   this wait. It did not.**
3. **`cancel` alone fixes the second cause only.** Cancelled streams no
   longer wait for their partial, but their finals still queue behind
   partials first in, first out (339 ms), and the `done` class still
   reaches 508 ms.
4. **`both` removes both causes.** Priority takes the `done` class to
   ≤ 238 ms, and cancellation removes the class priority had inflated.
   The tail is then finals queued behind other finals that ended at the
   same moment (≤ 317 ms).

**Why so few cancellations matter.** Over the c=8 scenarios of both runs,
`both` cancelled 4 partials. The p95 of 40 samples sits at roughly the
second-worst stream, so the tail is decided by a handful of streams, and
those are exactly the ones cancellation acts on. The effect is real but
concentrated in a small, structurally defined group. The size of the
improvement depends on how often own partials are pending at `end`, which
in turn depends on synchronized ends. That is the dependency the next
experiment tests.

## Predictions vs observed

| Prediction (written before the run) | Observed | Verdict |
|---|---|---|
| prio c=8 final queue p95 336 → 100–250 ms | 268 ms | slightly above range |
| prio c=8 final latency p95 → 250–400 ms | 390 ms; interval includes 0 | at the edge; not significant |
| prio c=4: small, possibly noise | −10 ms [−32, +11] | as predicted |
| cancel alone: p95 change not detectable | −87 ms in both rounds; interval includes 0 | underestimated tail leverage |
| both ≈ prio; own-partial wait ≈ 0 | both far better than prio; own-partial wait p95 34 ms (running partials) | **wrong**: interaction not anticipated |
| TTFP p95 unchanged or up to +130 ms | +29 ms [−29, +86] | as predicted |
| decode control, throughput unchanged | unchanged | as predicted |
| WER / final text identical | 5.31 %, 100 % | as predicted |

## Adoption criteria (from the design doc)

| Criterion (c=8) | both | prio alone |
|---|---|---|
| p95 reduction, interval excludes 0, both rounds | ✓ | ✗ (interval includes 0) |
| zero failures | ✓ | ✓ |
| 100 % identical text, WER unchanged | ✓ | ✓ |
| TTFP p95 regression ≤ 10 % (c=4 and c=8) | ✓ (+0.6 %, +2.5 %) | ✓ |
| decode control within ±10 % | ✓ | ✓ |

`both` meets every criterion on this workload; `prio` alone does not.
Per the review decision, **both flags stay off by default** until the
result is shown to generalize beyond synchronized starts.

## Limitations

* **Synchronized starts.** All streams of a wave start together on 5
  cycled files, so identical streams end at the same moment. That burst of
  ends is what creates the pending partials both policies act on. Real
  traffic is less synchronized; the effect could be smaller. This is the
  main open question.
* **Small samples.** 40 streams per policy at c=8. The p95 is decided by
  about two streams, and 4–6 streams drive the policy differences.
  Bootstrap intervals treat streams as independent, but they are
  correlated within a run, so the intervals are optimistic.
* One Colab session, one T4, one inference worker, FP16 only, five
  utterances of 4–11 s.
* TTFP p50 at c=8 rose under `prio` (993 / 1,000 vs base 959 / 945 ms per
  run) but not under `both` (953 / 953). That is consistent across rounds
  but unexplained; it is not part of the adoption criteria and needs a
  closer look.
* The end-state analysis was designed after seeing the results. It
  explains this run's tail; it was not a pre-registered test.

## Reproduce

* Run: [`benchmarks/COLAB.md`](../../../benchmarks/COLAB.md),
  "Scheduling experiment".
* Re-derive from the committed files (no GPU needed):

```bash
python -m benchmarks.compare_policies docs/benchmarks/scheduling-t4/raw --audio-dir eval_audio5
python -m benchmarks.end_state_breakdown docs/benchmarks/scheduling-t4/raw --scenario c8
```
