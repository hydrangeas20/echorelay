# EchoRelay Phase 1 benchmark: Whisper base.en on an NVIDIA T4 (pilot)

**Status: pilot.** One Colab session, five utterances, 20 streams per
configuration. These results describe this setup; they are not statistically
conclusive comparisons of precision modes or of accuracy.

All numbers below come from the raw result files in [`raw/`](raw/)
(SHA-256 in [`raw/SHA256SUMS`](raw/SHA256SUMS)). Derived numbers come from
[`analysis.json`](analysis.json) / [`analysis.md`](analysis.md), generated
by `benchmarks/analyze_results.py`. Definitions:
[`docs/measurements.md`](../../measurements.md).

## Summary

* All 45 measured streams completed (0 failures).
* **Time to first partial is set by configuration, not the model.** p95
  ≈ 1.0 s against a 0.9 s pacing floor; the first partial decode takes
  33–39 ms (median).
* **Partial re-decoding is 83–84 % of all inference time.** Each partial
  re-decodes the whole buffer, and its cost grows with buffer length
  (≈ 31 ms + 6.2 ms per audio-second, FP16, c=4).
* **At c=4 the single inference worker was busy 28–30 % of the time.** No
  partial was deferred for lack of capacity.
* **INT8/FP16 mixed precision was slower than FP16** on this T4: offline
  aggregate RTF 0.01237 vs 0.01111, with non-overlapping repeats.
* **No accuracy difference between precisions was found.** The reported
  WER gap (7.08 % vs 6.19 %) is one quotation mark scored as an error by the
  Phase 1 WER normalizer; see [Corrections](#corrections).

## Environment

| | |
|---|---|
| GPU | NVIDIA Tesla T4, 15,360 MiB, driver 580.82.07 (Google Colab) |
| Model | Whisper `base.en` via faster-whisper 1.2.1, CTranslate2 4.8.3 |
| Precisions | `float16` (FP16); `int8_float16` (INT8/FP16 mixed: INT8-quantized weights, FP16 elsewhere) |
| Python / libs | Python 3.13.15, numpy 2.1.3, FastAPI 0.141.1, uvicorn 0.54.0, websockets 15.0.1, jiwer 4.0.0 |
| Server | 1 inference worker, max 8 streams, partial interval 1.0 s, partial beam 1 (greedy), final beam 5 |
| Client | same VM as the server; 100 ms PCM16 frames at real-time pacing; 2 warm-up streams; 120 s per-stream deadline |
| Audio | Mini LibriSpeech dev-clean-2, utterances `1272-135031-0000` … `-0004` (10.885, 11.130, 11.475, 4.755, 4.225 s; 42.47 s total; 113 reference words) |
| Date | 2026-10-15, 19:31–20:10 UTC |

## Methodology

* **Streaming** (`benchmarks/load_test.py`): each stream connects, sends
  audio in 100 ms frames paced at real time, sends `end`, and waits for the
  final. With 20 streams at concurrency 4, the 5 files are each replayed 4
  times; the c=1 pilot plays each file once.
* **Offline precision comparison** (`benchmarks/compare_precision.py`): each
  file decoded with the server's final settings (beam 5), 3 repeats after
  one warm-up per precision. The headline metric is aggregate RTF: total
  decode time / total audio, per repeat, averaged over repeats.
* **Smoke** (`benchmarks/gpu_smoke.py`): one utterance per precision; a
  go/no-go check, not a measurement.

## Results

### Streaming

| Configuration | Streams | TTFP p50 / p95 | Final latency p50 / p95 | Final RTF p50 | Throughput | WER (v1) |
|---|---:|---:|---:|---:|---:|---:|
| FP16, c=1 | 5/5 | 941 / 945 ms | 132 / 157 ms | 0.0148 | 0.115 streams/s | 7.08 % |
| FP16, c=4 | 20/20 | 942 / 1,027 ms | 119 / 146 ms | 0.0121 | 0.455 streams/s | 7.08 % |
| INT8/FP16, c=4 | 20/20 | 938 / 1,010 ms | 130 / 178 ms | 0.0122 | 0.453 streams/s | 6.19 % |

Final RTF is the final decode only. It excludes partial decodes, queueing and
network. Throughput at real-time pacing is bounded by audio duration:
4 streams / (8.49 s mean audio + ~0.15 s) ≈ 0.46 streams/s. It therefore
does not measure server capacity.

### Offline precision comparison (final settings, beam 5)

| Precision | Aggregate RTF (mean of 3) | Per repeat | Model load |
|---|---:|---|---:|
| FP16 | **0.01111** | 0.01154, 0.01062, 0.01118 | 0.9 s |
| INT8/FP16 mixed | **0.01237** | 0.01224, 0.01258, 0.01229 | 0.5 s |

The three aggregate repeats do not overlap (FP16 max 0.01154 < INT8/FP16 min
0.01224). Per file, INT8/FP16 was slower on 4 of 5 files by mean and on all 5
by best-of-repeats. On `1272-135031-0000` the means are within 0.4 ms
(124.1 vs 123.7 ms) because FP16's first repeat was slow (137.5 ms). This is
one session on one GPU.

### Where inference time goes (derived)

| Run | Partial decode cost vs buffer length | r | Worker busy | Partial share of inference |
|---|---|---:|---:|---:|
| FP16, c=1 | 51.3 ms + 5.69 ms/s | 0.61 | 8.9 % | 84 % |
| FP16, c=4 | 31.0 ms + 6.17 ms/s | 0.92 | 27.6 % | 83 % |
| INT8/FP16, c=4 | 31.5 ms + 7.22 ms/s | 0.93 | 29.9 % | 83 % |

For the three ~11 s utterances, the 10–11 partial decodes cost 5.4–6.3× the
final decode at c=4 (5.0–6.6× including the c=1 pilot; per-utterance
medians) and re-decode 5.0–5.9× the audio (per-utterance table in
[`analysis.md`](analysis.md)). Every stream received exactly
`floor(duration)` partials, so none was deferred.

**Latency decomposition (derived).**

| | c=1 | c=4 FP16 | c=4 INT8/FP16 |
|---|---:|---:|---:|
| TTFP p95 − 0.9 s floor − first decode | 4 ms | 96 ms | 74 ms |
| Final latency p95 − final decode | 1 ms | 43 ms | 86 ms |

At c=1, latency is pacing plus decode. At c=4, tens of milliseconds appear
that are not decoding. They are waiting behind other streams' partials or
for the stream's own in-flight partial. Phase 1 data cannot separate the two;
Phase 2 adds server-side timing to do so.

## Corrections

**WER.** FP16 and INT8/FP16 produced identical transcripts on four of five
files. On `1272-135031-0001` the only difference is a closing quote:
FP16 `…for good,' answered…`, INT8/FP16 `…for good," answered…`. The Phase 1
normalizer kept apostrophes, so `'he` (both runs) and `good'` (FP16 only)
counted as errors. That accounts for exactly one word of difference
(8/113 vs 7/113).

The Phase 2 normalizer (v2) strips quote-mark apostrophes. Under v2 both
precisions are **expected** to score 6/113 = 5.31 %. This is not yet
confirmed: run `benchmarks/rescore.py` with the reference transcripts (see
below).

## Limitations

* 5 utterances (42 s, 113 words), each replayed 4× at c=4. The p95 of 20
  correlated samples is close to the second-largest value. The WER sample is
  5 utterances, not 20.
* One Colab session and one T4. GPU clocks were not recorded. Single decodes
  were slower at c=1 than at c=4 (median per partial 76 vs 61 ms); a
  clock-ramp or warm-up effect is suspected but untested.
* Client and server shared a VM. Load-generator CPU was not recorded.
* Python 3.13.15 in Colab; the project's Docker image targets 3.11.
* Result files do not record the git commit; from Phase 2 the environment
  block does.
* Buffer lengths covered 1–11.5 s. Any statement about longer streams is a
  projection.

## Reproduce

On a Colab T4 runtime with the repo cloned (see
[`benchmarks/COLAB.md`](../../../benchmarks/COLAB.md)):

```bash
pip install -e ".[bench]" soundfile
python -m benchmarks.prepare_audio --out eval_audio --num 5
ECHORELAY_DEVICE=cuda ECHORELAY_COMPUTE_TYPE=float16 python -m echorelay --port 8000 &
python -m benchmarks.load_test --audio-dir eval_audio --streams 20 --concurrency 4 \
    --wer-normalizer v1 --label "T4 FP16 c=4" --output results/stream_fp16_c4.json
python -m benchmarks.compare_precision --audio-dir eval_audio \
    --compute-types float16 int8_float16 --repeats 3 --wer-normalizer v1 \
    --output results/precision_t4.json
```

`prepare_audio --num 5` should select the same five files (the first five
dev-clean-2 utterances of 3–20 s, by id). Confirm by comparing the SHA-256
values in each result file's `audio` manifest. Restart the server with
`ECHORELAY_COMPUTE_TYPE=int8_float16` for the INT8/FP16 run. `--wer-normalizer v1`
reproduces the Phase 1 scoring.

Re-derive the analysis and rescore WER from the committed files (no GPU):

```bash
python -m benchmarks.analyze_results docs/benchmarks/phase1-t4/raw/stream_*.json
python -m benchmarks.rescore --audio-dir eval_audio docs/benchmarks/phase1-t4/raw/stream_*.json \
    docs/benchmarks/phase1-t4/raw/precision_t4.json
```
