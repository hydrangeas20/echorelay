# Running the GPU benchmarks on Google Colab

Nothing in this directory has been run on a GPU yet. These are the steps to
produce the first real measurements.

## 1. Runtime

Runtime → Change runtime type → **T4 GPU** (or better). Confirm:

```bash
!nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

## 2. Install

```bash
!git clone -b feat/streaming-mvp https://github.com/hydrangeas20/echorelay.git
%cd echorelay
!pip install -q -e ".[bench]" soundfile
!python -c "import ctranslate2; print('cuda devices:', ctranslate2.get_cuda_device_count())"
```

faster-whisper ≥1.1 (CTranslate2 4.5+) needs CUDA 12 and cuDNN 9, which
current Colab images provide. If loading fails with a `libcudnn` or
`libcublas` error, install `nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*` and set
`LD_LIBRARY_PATH` as described in the faster-whisper README.

## 3. Evaluation audio (fixed, deterministic)

```bash
!python -m benchmarks.prepare_audio --out eval_audio --num 50
```

Downloads Mini LibriSpeech dev-clean-2 (~126 MB) from OpenSLR and writes the
first 50 utterances of 3–20 s with reference transcripts. Keep this
directory unchanged across runs; result files record each file's SHA-256.

## 4. FP16 vs INT8 (offline, no server)

```bash
!python -m benchmarks.compare_precision --audio-dir eval_audio \
    --compute-types float16 int8_float16 --repeats 3 \
    --output results/precision_t4.json
```

Same audio, same decoding settings (the server's final-decode settings),
same WER normalization for both compute types.

## 5. Streaming server benchmark

Start the server in the background on the GPU:

```bash
!ECHORELAY_DEVICE=cuda ECHORELAY_COMPUTE_TYPE=float16 ECHORELAY_MAX_CONCURRENT_INFERENCE=2 \
  ECHORELAY_MAX_CONCURRENT_STREAMS=16 nohup python -m echorelay --port 8000 > server.log 2>&1 &
!sleep 30 && curl -s localhost:8000/health
```

Then, for each concurrency level:

```bash
!python -m benchmarks.load_test --audio-dir eval_audio --streams 50 --concurrency 1 \
    --label "T4 fp16 c=1" --output results/stream_fp16_c1.json
!python -m benchmarks.load_test --audio-dir eval_audio --streams 50 --concurrency 4 \
    --label "T4 fp16 c=4" --output results/stream_fp16_c4.json
```

Repeat with `ECHORELAY_COMPUTE_TYPE=int8_float16` (restart the server) for
the INT8 rows. Each result JSON contains the server `/health` payload
(model, device, compute type), client environment (`nvidia-smi` GPU name),
run configuration, the audio manifest, per-stream raw timings, and the
p50/p95 summaries.

Notes:

* Load generator and server share the Colab VM; at high concurrency the
  client competes for CPU. Record this caveat with results.
* `--speed 1.0` (default) paces audio at real time, which is what TTFP is
  meant to describe. `--speed 0` measures maximum throughput instead; label
  such runs accordingly.
* TTFP has a floor of roughly `partial_interval_s` (1.0 s default) at
  real-time pacing, because the first partial is scheduled after that much
  audio has arrived.
