# Measurement definitions

Every number EchoRelay reports, what it measures, and what it excludes. All
durations use `time.monotonic()` (client timestamps come from one client
process; server timestamps from the server process). Client and server clocks
are never subtracted from each other.

## 1. Client-side stream metrics (`benchmarks/load_test.py`)

| Metric | Start | End | Includes | Excludes |
|---|---|---|---|---|
| **Time to first partial (TTFP)** | first audio frame handed to the socket | first *nonempty* `partial` received | pacing wait until `partial_interval_s` of audio has been sent, network, queueing, partial decode | connection setup |
| **Final latency** | `end` message sent | `final` received | wait for an in-flight partial, queueing, final decode, network | audio streaming time |
| **Throughput** | first connect of the measured run | last stream finishes | — | warm-up streams |
| **Failures** | — | — | server `error` events, connection errors, `stream_timeout` (`--stream-timeout-s`, connect → final) | — |

**Pacing floor.** At real-time pacing the first partial cannot arrive before
the frame that completes `partial_interval_s` of audio is sent. With
`chunk_ms` frames that is `(ceil(interval / chunk) − 1) × chunk / speed` after
the first frame: **0.9 s** for the Phase 1 settings (100 ms frames, 1.0 s
interval, speed 1.0). `analyze_results.pacing_floor_s` computes it.

**Throughput at real-time pacing measures the client, not the server.** A
stream of duration *d* occupies a client slot for about *d* seconds, so
throughput ≈ concurrency / (mean duration + final latency). Use `--speed 0`
(unpaced) or the concurrency sweep to measure server capacity, and label such
runs accordingly.

## 2. Server-reported timing

Inside `partial` / `final` events:

| Field | Definition |
|---|---|
| `inference_ms` (always present) | model call inside the worker thread only |
| `timing.queue_ms` | decode requested → inference slot acquired |
| `timing.decode_ms` | same as `inference_ms` |
| `timing.staleness_ms` (partial) | audio received after this partial's snapshot, at the time it is sent |
| `timing.wait_partial_ms` (final) | `end` processed → this stream's in-flight partial finished |

`timing` is present only with `ECHORELAY_EVENT_TIMING=true`.

**Final latency decomposition** (server side, timing enabled; with
`FINAL_PRIORITY` the final is submitted at `end`, so `wait_partial` and
`queue` overlap and are reported separately rather than summed):
final latency ≈ `wait_partial` + `queue` + `decode` + network/scheduling.
Without timing (Phase 1), only `final latency − final decode` can be computed;
`analyze_results` reports it as the *remainder*.

**Final inference RTF** = `final.inference_ms / final.audio_duration_s`. It is
the model's speed on the full utterance and **excludes partial decodes,
queueing, buffering and network**. It is not the per-stream compute cost: in
Phase 1, partial decodes used 5–6× more inference time than the final for
11 s utterances.

## 3. `/metrics` (JSON)

Gauges (current values):

| Key | Meaning |
|---|---|
| `streams.active` / `peak_active` / `max` | open streams now / highest since start / admission limit |
| `streams.rejected` | connections refused with `server_busy` |
| `streams.buffered_audio_bytes` | PCM bytes held across all open streams; bounded by `active × max_audio_bytes` |
| `inference.in_flight` | decodes running in worker threads (≤ `max_concurrent`) |
| `inference.waiting` | callers waiting for an inference slot |
| `process.*` | CPU user/system seconds, current and max RSS, thread count |

Distributions (`distributions.<name>`) keep the most recent `metrics_window`
observations (default 10,000) for percentiles, plus a lifetime `count`, `sum`
and `max`. Memory is bounded regardless of uptime.

| Name | Unit | Definition |
|---|---|---|
| `inference.{partial,final}.queue_wait_s` | s | request → slot acquired |
| `inference.{partial,final}.decode_s` | s | model call in the worker thread |
| `inference.{partial,final}.audio_s` | s | audio passed to the decode |
| `inference.{partial,final}.rtf` | — | decode_s / audio_s |
| `event_loop.lag_s` | s | how late a 100 ms sleep wakes up; sustained values above a few ms mean something is blocking the loop |
| `stream.audio_s` | s | audio received per finished stream |
| `stream.partial_decode_s` | s | total partial decode time per stream |
| `stream.partial_amplification` | × | audio decoded by partials / audio received |
| `stream.partial_staleness_s` | s | per partial sent (see §2) |
| `stream.final_{wait_partial,queue,decode}_s` | s | see §2 |

Counters: `inference.{kind}.{completed,errors,timeouts,timeouts_while_queued,cancelled_callers,cancelled_while_queued}`,
`stream.partials_{promoted,cancelled}_at_end`, `stream.partial_at_end.<state>` (state of the
stream's partial when `end` was processed: `none`, `created`, `queued`, `running`, `done`),
`stream.partials_{scheduled,sent,deferred,skipped_empty,skipped_duplicate}`,
`streams.outcome.<outcome>`.

**Deferred partial**: the partial interval elapsed while the stream's
previous partial was still in flight. Counted once per in-flight partial.
Non-zero deferrals mean partial decoding is not keeping up.

`/metrics` has no reset. For per-configuration numbers, restart the server
(the Colab workflow does) or difference the counters.

## 4. Per-stream summary log

One `stream_summary` INFO line per finished stream with the §3 per-stream
values and the outcome. It never contains transcript text.

## 5. Resource sampling (`benchmarks/sampler.py`)

During a load test, every `--sample-interval-s` (default 1 s): server gauges
from `/metrics`, and from `nvidia-smi`: GPU utilization (%), memory used
(MiB), SM clock and max SM clock (MHz), power (W), temperature (°C).
`utilization.gpu` is the fraction of time a kernel was running, not how much
of the GPU it used. `server_cpu_cores_mean` is server CPU seconds per wall
second between the first and last successful sample.
`load_generator_cpu_cores_mean` is the client's own CPU use; if the client
and server share a VM, report it alongside the results.

## 6. Offline measurements

* **`compare_precision` aggregate RTF (headline)**: for each repeat, total
  decode seconds over all files / total audio seconds; then the mean across
  repeats. Per-repeat values and spread are reported. Per-file best-of-repeats
  RTF is a secondary, optimistic view.
* **`profile_decode`**: median of `--repeats` decodes of a fixed-length signal
  per length, with partial (greedy) and final (beam) settings, after one
  warm-up. Linear fits use lengths ≤ 30 s (one Whisper window).

## 7. WER

Corpus WER = total edits / total reference words, after normalization.

* **v2** (default since Phase 2): lowercase; typographic quotes mapped to
  ASCII; hyphens → spaces; punctuation removed except apostrophes; apostrophes
  stripped from the start and end of each word (word-internal ones kept).
* **v1** (Phase 1): as v2 without the quote mapping and edge stripping. An
  apostrophe used as a quotation mark stays on the word (`'he`, `good'`) and
  scores as an error. Kept only to reproduce Phase 1 numbers.

Result files record `wer_normalizer`. `benchmarks/rescore.py` rescores saved
transcripts under both versions without re-running a model.

## 8. Derived analysis (`benchmarks/analyze_results.py`)

* **Busy fraction** = total decode time / (wall clock × inference workers).
  Concurrency ramps down at the end of a run, so per-stream load is slightly
  underestimated.
* **Partial share** = partial decode time / all decode time.
* **TTFP remainder** = TTFP − pacing floor − first partial decode
  (queueing + network + scheduling).
* **Projections** (saturation concurrency, per-stream load by stream length)
  are models fitted to one run and are always labelled `PROJECTION`. Rows
  beyond the fitted audio range are flagged `extrapolated`. Above 30 s
  Whisper decodes in multiple windows, which the linear model does not
  capture.

## 9. Scheduling comparison (`benchmarks/compare_policies.py`)

* Groups: policy × concurrency. c=4 pools the `c4` and `recovery`
  scenarios (identical workloads); rounds are pooled, and the p95 per round
  is reported alongside.
* **Final text identical to baseline**: fraction of a policy's streams whose
  final text equals the baseline's text for the same utterance. Decoding is
  deterministic and scheduling never changes the final's input, so the
  expected value is 100 %.
* **Final decode (control)**: scheduling must not change decode time; a
  shift means GPU state differed between runs.
* **Bootstrap interval**: percentile bootstrap of the difference in p95
  (candidate − baseline), resampling streams within each arm, 2,000
  resamples, seed 0. Optimistic, because streams within a run are correlated.

## 10. Arrival schedules and workload descriptors (added for Phase 4, before data)

**Arrival schedules** (`load_test --arrival`, `stress --arrival`):
`sync` (default, unchanged) starts every first-wave stream at once;
`stagger` starts slot k's first stream k × `--stagger-s` after the run
starts (absolute offsets). Later streams start when a slot frees in both
modes. Records carry `start_delay_s` (the intentional offset) and `t_done`
(when the stream finished, failed, timed out or was cancelled). TTFP and
final latency start at the stream's own first frame and end message, so
the offset is excluded. Throughput's wall clock includes the stagger ramp;
compare it only within an arrival mode.

**Blocks:** a repeated concurrency value in `stress --concurrency` runs
one block per repeat (`c8`, `c8_2`, `c8_3`), plus `recovery`. Each block
starts from an idle server and records `/metrics` before and after.

**Level distributions** (`/metrics` → `levels`): for `streams.active`,
`inference.waiting` (queue depth) and `inference.in_flight`, the time spent
at each integer level since start. Differencing the before and after
snapshots gives a block's time-weighted distribution: mean, share at each
level, share at or above, and highest level held. The window includes the
short idle wait after the block.

**Queue depth at end** (`stream_summary`: `inference_waiting_at_end`,
`inference_in_flight_at_end`): observed when `end` is processed, before
the stream's final is submitted or its partial promoted/cancelled.

**Workload descriptors** (`benchmarks/analyze_arrival.py`, per block):

| Descriptor | Definition |
|---|---|
| Observed active (client) | streams whose [t_connect, t_done] interval covers time t; time-weighted mean over [first connect, last done]; share of time at the scheduled concurrency; mean count at each stream's end |
| End-time synchronization | per stream, other streams in the block with \|Δ t_end_sent\| ≤ 250 ms; share with ≥ 1, mean |
| Partial request time | first frame + (round(audio / chunk) − 1) × chunk / speed, for each sent partial (cancelled or skipped partials are not sent, so not counted) |
| Phase alignment | per partial request, other streams' requests within ±50 ms (mean, share ≥ 3); R = \|mean(exp(2πi · t / P))\| with P = 1.0 s / speed (1 = all requests on one phase, ≈ 0 = evenly spread) |

**Phase 4 intervals:** the block bootstrap resamples whole blocks within
each arm (10,000 resamples, seed 0). The Phase 3 stream bootstrap is
reported alongside. Both are exploratory; see the registration.
