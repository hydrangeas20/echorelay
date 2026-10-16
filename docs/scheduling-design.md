# Final-priority scheduling and obsolete-partial cancellation

Status: **experiment.** GPU results (Phase 3, T4):
[`docs/benchmarks/scheduling-t4`](benchmarks/scheduling-t4/README.md).
The prediction that inheritance alone would remove the own-partial wait
under final priority was **wrong**; see that report. Both policies are behind flags that default to off.
The default code path, model settings, audio, benchmark methodology and
wire protocol are unchanged. Adoption depends on the matched GPU comparison
(section 9).

## 1. Evidence (E2, T4, FP16, commit `93c1bf2`)

| | c=4 | c=8 |
|---|---:|---:|
| Final latency p95 | 164 ms | 567 ms |
| Final inference-queue wait p95 | 77 ms | 336 ms |
| Final decode p95 | 125 ms | 130 ms |
| Wait for own partial p95 / max | 0 / 0 ms | 10 / 207 ms |
| Partial queue wait p95 | 120 ms | 341 ms |

At c=8, queue wait is the largest part of tail final latency. It is waiting
for one inference worker (`max_concurrent_inference=1`) that is busy with
other streams' partials, served first in, first out.

Per-stream details from the raw files:
* 9 of 20 finals at c=8 waited 29–433 ms. The long waits cluster in the
  first two waves. The load generator starts 8 streams together on 5 cycled
  files, so streams with the same file end at the same moment and their
  finals arrive together. This affects every policy equally.
* One stream (#6, file `…-0001`, which ends 0.13 s after its last partial
  is scheduled) waited 207 ms for its **own** partial, then 249 ms more in
  the queue. Only streams whose partial has not started at `end` can benefit
  from cancellation: 1 of 20 at c=8, 0 of 20 at c=4.

## 2. Policies

| Flag (`ECHORELAY_…`) | Effect |
|---|---|
| `FINAL_PRIORITY` | Waiting finals are served before waiting partials (FIFO within each class). Includes priority inheritance and early final submission (section 3). |
| `CANCEL_QUEUED_PARTIAL_ON_END` | At `end`, cancel the stream's partial if it has not started decoding. |

The flags are independent. The four combinations (`base`, `prio`, `cancel`,
`both`) are compared on the same commit.

## 3. Mechanism

**Slot scheduler** (`inference/scheduler.py`) replaces `asyncio.Semaphore`.
It hands slots over explicitly and keeps a priority queue of waiters. Each
request is a `Job` whose state (`created → queued → running → done`, or
`cancelled`) can be read without awaiting. With priorities off it grants in
exactly the order `asyncio.Semaphore` does (tested on random operation
sequences).

**Final priority** needed two additions. The deterministic tests found two
priority inversions without them:

1. *Inheritance.* `_finish` waits for the stream's own partial before
   sending the final. If that partial is still queued, the final would wait
   behind every other stream's partial. At `end`, a not-yet-running own
   partial is therefore promoted to final priority. This includes a partial
   whose task has not reached the queue yet.
2. *Early submission.* The own partial's slot is released in its
   completion callback, and at that instant the final had not been
   submitted yet, so the slot went to another stream's partial. Under final
   priority the final is submitted at `end`, concurrently with waiting for
   the own partial. The final is still **sent** only after the partial task
   finishes.

**Cancellation.** At `end`, if the stream's partial is `created` or `queued`,
its task is cancelled. A running partial is never interrupted: a decode in a
worker thread cannot be stopped, and its result is still a valid provisional
hypothesis that precedes the final.

## 4. Invariants

Scheduler (tested in `tests/test_scheduler.py`):

* **S1 Conservation.** `free + held = slots`; no double grant, no lost slot.
* **S2 No barging.** A freed slot goes straight to the next waiter, so a
  free slot and a live waiter never coexist.
* **S3 Order.** (priority, arrival); FIFO when priority is off.
* **S4 Cancellation.** Cancelling a queued job never consumes a slot. A slot
  granted to a job that is cancelled before it resumes is passed on.

Stream (tested in `tests/test_priority.py`, `tests/test_cancellation.py`):

* **I1 One final.** Exactly one `final` per successful stream; nothing after it.
* **I2 Order.** Every partial a stream sends precedes its final; `seq` is
  contiguous (a cancelled partial never sends, so it leaves no gap).
* **I3 Final input.** The final decodes the complete buffer. No audio is
  accepted after `end`, so the input is identical under every policy.
* **I4 At most one partial** per stream is created, queued or running.
* **I5 Bounded work.** A cancelled partial never reaches the model. A
  running decode always finishes and releases its slot when its thread exits.

## 5. Race conditions and how they are resolved

| Race | Resolution | Test |
|---|---|---|
| A slot is handed to a waiter that is cancelled in the same loop iteration | The waiter detects the grant and passes the slot on (S4) | `test_grant_then_cancel_race_passes_slot_on` |
| `end` arrives while the own partial moves from queued to running | The state is read and acted on synchronously (no `await` in between); a grant not yet resumed is still `queued`, and S4 covers it | `test_execution_order_by_policy` |
| `end` arrives after the partial task is created but before it runs | The `Job` is created synchronously when the partial is scheduled; `created` jobs are promoted or cancelled | `test_partial_scheduled_but_not_yet_started` |
| Own partial finishes, its slot goes to another partial before the final is submitted | Early final submission under final priority | `test_final_priority_with_inheritance_runs_own_partial_then_final_first` |
| Own partial fails while the early final is queued or already granted | The stream ends with the partial's error; the early final is cancelled, or, if already running, finishes once and its result is discarded | `test_own_partial_fails_while_early_final_is_queued` |
| Client disconnects during `_finish` with partial and final queued | Both requests are withdrawn; no decode runs; runner returns to idle | `test_disconnect_during_finish_with_early_final_queued`, `test_disconnect_after_cancellation_leaves_runner_idle` |
| Timeout while queued | The waiter is withdrawn; nothing leaks | `test_timeout_while_queued_does_not_leak` |
| Several streams end together, 1 or 2 workers, every policy | I1–I3 hold; each final decodes once | `test_final_text_and_wire_invariants_under_every_policy` |

The whole suite also passes with each flag forced on through the
environment. That includes the real-uvicorn reliability tests: disconnects
at every stage, shutdown, admission burst and recovery.

## 6. Fairness

* **Partials can wait longer.** A final can overtake partials already
  waiting. The expected cost is higher partial queue wait and staleness under
  load, and possibly first-partial latency when one wave of streams starts
  while another ends. This is measured (partial queue p95, TTFP), not assumed.
* **Starvation is bounded.** Each stream produces exactly one final.
  Admission caps open streams at `max_concurrent_streams` (8), so at most 8
  finals can be waiting ahead of a partial at any instant. New finals keep
  arriving only as streams complete, which audio duration limits.
  `test_partials_are_not_starved_once_finals_drain` checks that a partial
  runs as soon as no final is waiting.
* **Finals do not starve each other.** They are FIFO within their class.
* **No aging is implemented.** If GPU data shows unacceptable partial delay,
  aging or a cap on consecutive finals is the next step. That would need
  approval.

## 7. Correctness under cancellation and disconnects

* **Final text.** Cancellation only removes a decode of a *prefix* of the
  audio the final decodes; it never touches the buffer or the final's
  settings. By I3 the final's input is identical under every policy, and
  decoding is deterministic, so final text and WER must be identical. The
  comparison checks this directly (final text identical to baseline, per
  utterance) and fails visibly if not.
* **Wire.** A cancelled partial is never sent. A partial that started is
  always sent before the final (I2). Clients see at most one fewer
  provisional update and an unchanged final.
* **Disconnects.** Session cleanup cancels queued work (no slot taken) and
  abandons running work (slot released when its thread exits). This is
  unchanged from Phase 2 and holds under every policy (section 5).
* **TTFP survivorship.** Cancellation could, in principle, remove a stream's
  *first* partial if the stream ended before it started. TTFP would then be
  computed over fewer streams. The comparison reports
  `streams_without_nonempty_partial`. With the evaluation audio (≥ 4.2 s)
  the first partial always runs long before `end`.

## 8. Unchanged

Model (`base.en`, FP16), beams (1 partial, 5 final), partial interval
(1.0 s), evaluation audio (same five files, PCM-hash verified), measurement
code (`stress` / `load_test`), and the default wire format. `/metrics`
gains config keys and counters only.

## 9. Matched GPU comparison

`benchmarks/matched_comparison.py` runs all four policies on one commit in
one Colab session. The order is counterbalanced (base, prio, cancel, both,
then the reverse). Each run restarts the server, verifies its configuration
and records provenance. Each run executes `stress --concurrency 4 8` (c=4
again as recovery), 20 streams each. Per policy that is 80 streams at c=4
and 40 at c=8.
`benchmarks/compare_policies.py` reports per policy and concurrency: final
latency, final queue wait, own-partial wait, final decode (control), TTFP,
partial queue wait, throughput, failures, partials per stream, WER (v2),
final-text identity, per-round p95, and bootstrap intervals for the p95
differences.

### Predictions (written before the GPU run)

| Metric | Prediction | Reasoning |
|---|---|---|
| `prio`, c=8, final queue p95 | 336 → roughly 100–250 ms | A final waits for at most the running decode (≤ ~110 ms) plus finals that end at the same moment (~110–130 ms each) |
| `prio`, c=8, final latency p95 | 567 → roughly 250–400 ms | Queue reduction above plus an unchanged ~130 ms decode |
| `prio`, c=4 | small reduction, possibly within noise | Baseline queue p95 is only 77 ms |
| `cancel` alone | p95 change not detectable; removes own-partial-wait outliers | Affects about 1 stream in 20 at c=8 |
| `both` | about the same as `prio`; own-partial wait ≈ 0 | |
| TTFP p95 | unchanged, or up to about +130 ms at c=8 | First partials of a new wave can now wait behind the previous wave's finals |
| Final decode (control) | unchanged | Scheduling never changes the decode itself |
| Throughput | unchanged | Bounded by real-time pacing |
| WER / final text | identical (5.31 %, 100 % identical) | I3 |

### Proposed adoption criteria (for review)

Make `final_priority` the default only if, at c=8:
1. the final latency p95 reduction has a bootstrap interval excluding zero
   and appears in both rounds;
2. there are zero failures;
3. final text is 100 % identical and WER is unchanged;
4. TTFP p95 regresses by no more than 10 % at c=4 and c=8;
5. the final-decode control is within ±10 % of baseline.

`cancel_queued_partial_on_end` is cheap and removes wasted work. Adopt it if
it shows no regression on any metric.

### Harness validation (fake backend, not evidence about the T4)

The tooling was run end to end on the fake backend with a constant 40 ms
decode and tone audio, 8 runs. Every stress check passed, and final text
was 100 % identical across policies. The pattern matched the mechanism: at
c=8, final latency p95 fell from 192 to 105 ms under `prio` (interval
excluding zero, both rounds). `cancel` alone removed the own-partial wait
outliers with no clear p95 change. Partial queue p95 rose from 242 to
274 ms under `prio`, with TTFP unchanged. This validates the measurement
pipeline only.

## 10. Limitations

* One T4 session; Colab GPU clocks vary. Counterbalancing addresses linear
  drift, not every session effect.
* 5 utterances replayed. Streams within a run are correlated, so bootstrap
  intervals are optimistic. Use them together with the per-round values.
* Synchronized stream starts create bursty finals. This is part of the
  unchanged methodology and affects every policy equally, but real traffic
  is less synchronized, so absolute gains may be smaller in practice.
* One inference worker, as in Phase 1/2. With more workers, queueing and the
  benefit both shrink.
