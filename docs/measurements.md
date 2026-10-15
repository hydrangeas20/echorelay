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

**Final latency decomposition** (server side, timing enabled):
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

Counters: `inference.{kind}.{completed,errors,timeouts,timeouts_while_queued,cancelled_callers}`,
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
