# Phase 2: profiling and streaming reliability

Evidence base: [Phase 1 T4 pilot report](benchmarks/phase1-t4/README.md).
Metric definitions: [measurements.md](measurements.md).

## What Phase 2 changed (this branch)

No architectural change. The default wire format, `/health` and Phase 1
behaviour are unchanged; all Sprint 1 tests pass unmodified.

| Change | Why |
|---|---|
| Queue wait measured separately from decode time; `/metrics` endpoint; per-stream stats; event-loop lag monitor; optional per-event `timing` | Phase 1 could not tell what the non-decode part of latency was |
| `send_timeout_s` (default 10 s) | Found by inspection and confirmed by a test: a client that stops reading could hold a stream slot indefinitely |
| WER normalizer v2 (v1 kept) | Phase 1 WER gap was a quote mark scored as an error |
| `profile_decode`, `analyze_results`, `stress`, resource sampling, long-stream mode | To test the hypotheses below |

## Hypotheses

| ID | Statement | Status | Evidence / test |
|---|---|---|---|
| H1 | Real-time throughput is bounded by the client | **Supported** (derived) | 0.455 measured vs ≈0.46 pacing bound |
| H2 | TTFP is configuration-bound | **Supported** (measured) | c=1: floor 0.9 s + decode + 4 ms |
| H3 | Partial cost is ~flat below 30 s (encoder padding) | **Refuted** (measured) | cost grows ~6.2 ms per audio-second, r = 0.92 |
| H4 | Final latency at load includes partial wait and queueing | **Partly supported** | remainder p95 43–86 ms at c=4, ~1 ms at c=1; split needs E2 |
| H-a | One worker saturates near c≈14 at Phase 1 lengths (FP16) | Projection | E2 |
| H-b | Long streams saturate much earlier (≈4 streams at 60 s) | Projection, extrapolated | E1, E4 |
| H-c | Slower single decodes at c=1 are a GPU clock / warm-up effect | Untested | E5 |
| H-d | A non-reading client can pin a stream slot | **Confirmed and fixed** | `test_client_that_stops_reading_cannot_pin_a_stream` |

## Experiments (GPU, Colab T4)

Run order and exact commands: [`benchmarks/COLAB.md`](../benchmarks/COLAB.md),
"Phase 2". Run every server with `ECHORELAY_EVENT_TIMING=true`.

Predictions come from the Phase 1 fits. A result outside a prediction is a
finding, not a failure.

| ID | Experiment | Prediction | Pass/fail checks |
|---|---|---|---|
| E1 | `profile_decode` 1–60 s, FP16 and INT8/FP16 | partial cost ≈ 31 + 6.2 ms/s up to 30 s; a step above 30 s | — (measurement) |
| E2 | `stress` c=1, 4, 8 + burst + recovery, FP16 | busy ≈ 7 % per stream → ~55 % at c=8; queue waits grow at c=8; 0 deferrals | zero failures; exact rejections; idle after each; loop lag p99 < 50 ms |
| E3 | E2 with INT8/FP16 | busy slightly higher than FP16 | same |
| E4 | `stress --concat-seconds 60 --concurrency 1 2 4` | per-stream load ≈ 0.23 → deferrals and rising staleness at c=4 (extrapolated) | zero failures |
| E5 | Sampled SM clock during c=1 vs c=4 | lower clocks at c=1 | — |
| E6 | `load_test --speed 0` at c=4 | real capacity, not pacing-bound | zero failures |
| E7 | `rescore` Phase 1 files with references | both precisions 6/113 = 5.31 % | — |
| E8 | E2 with `ECHORELAY_MAX_CONCURRENT_INFERENCE=2` (config only) | lower queue wait; unknown effect on GPU contention | same as E2 |

## Decisions that need approval

Each option is gated on the evidence listed. Nothing here is implemented.

| Option | Evidence so far | Gate | Trade-off |
|---|---|---|---|
| A. Finals ahead of partials in the inference queue | final remainder p95 43–86 ms at c=4 | E2 shows `final_queue_s` is a material share of final latency | small; partials slightly later under load |
| B. Drop a stream's queued (not running) partial when `end` arrives | same | E2 shows `final_wait_partial_s` > 0 at load | small; less wasted work |
| C. Partial interval grows with buffer length | partial share 83 %; cost grows with length | E1/E4 confirm growth beyond 11 s | simple, no accuracy change; long streams get fewer partials |
| D. Sliding-window partial decoding with committed prefix | same | long streams are in scope **and** C is insufficient | caps cost; accuracy risk at the window edge; moderate complexity |
| Batching across streams | worker ≤ 30 % busy at c=4 | E2/E6 show worker saturation | not justified by current data |
| VAD / endpointing | none | product requirement | out of scope |

## Risks

* Instrumentation overhead. The per-decode work is a few dict and deque
  updates under a lock, and the lag monitor wakes every 100 ms. E2 at c=4
  should match Phase 1 within noise; if it does not, the instrumentation is
  suspect.
* `timing` adds fields to events when enabled. It is off by default.
* Fake-backend tests verify control flow, bounds and recovery, not GPU
  behaviour.
* Colab variance between sessions. Compare configurations within one session.

## Acceptance criteria for this branch

1. All Sprint 1 tests pass unmodified; default events and `/health` are
   byte-compatible with Phase 1.
2. New tests cover: queue/decode separation, loop lag (including a negative
   control), bounded memory, disconnect during partial / final / queue,
   slow and stalled senders, non-reading client, malformed input mid-stream,
   shutdown, burst above the admission limit, and recovery after each fault.
3. Every reported metric has a definition in `measurements.md`.
4. No benchmark numbers that were not measured; projections are labelled.
