# Running the GPU checks and benchmarks on Google Colab

Nothing in this directory has been run on a GPU yet. Run the stages **in
order** and stop at the first failure: the smoke stages exist so that a
broken CUDA setup or model path is found in two minutes, not halfway
through a benchmark.

| Stage | What it proves | Output |
|---|---|---|
| 0. Runtime | GPU attached, CUDA visible to CTranslate2 | console |
| 1. Clone | repo available without leaking credentials | console |
| 2. Offline smoke | base.en loads + decodes 1 utterance per compute type | `results/gpu_smoke.json` |
| 3. Server smoke | one real stream end to end on the GPU server | console |
| 4. Precision comparison | FP16 vs INT8/FP16 mixed, same audio + settings | `results/precision_*.json` |
| 5. Streaming load test | TTFP, final latency, RTF, throughput, failures | `results/stream_*.json` |

## 0. Runtime

Runtime → Change runtime type → **T4 GPU** (or better).

```bash
!nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

## 1. Clone

### Public repository

```bash
!git clone -b feat/streaming-mvp https://github.com/hydrangeas20/echorelay.git
```

### Private repository (no credentials in the notebook, URL, or `.git/config`)

1. On GitHub, create a **fine-grained personal access token**:
   Settings → Developer settings → Fine-grained tokens. Repository access:
   **only `hydrangeas20/echorelay`**. Permissions: **Contents: Read-only**
   (nothing else). Expiry: as short as practical (e.g. 7 days).
2. In Colab, open the 🔑 **Secrets** panel, add a secret named `GH_TOKEN`
   with the token as its value, and enable **Notebook access** for this
   notebook only.
3. Clone with a Python cell. The token is passed to git through
   environment variables for this one process. It never appears in cell
   source, cell output, shell history, the process argument list, the remote
   URL, or `.git/config`:

```python
import base64, os, subprocess
from google.colab import userdata

token = userdata.get("GH_TOKEN")
auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
env = {
    **os.environ,
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
    "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {auth}",
}
subprocess.run(
    ["git", "clone", "-b", "feat/streaming-mvp",
     "https://github.com/hydrangeas20/echorelay.git"],
    env=env, check=True,
)
del token, auth, env
```

4. Verify nothing was persisted (both commands should print no token):

```bash
!git -C echorelay remote -v
!git -C echorelay config --list --show-origin | grep -i -E "extraheader|token|auth" || echo "no credentials stored"
```

Rules:

* Never paste the token into a cell, a `!git clone https://<token>@...` URL,
  or `git config`. A URL-embedded token is saved in `.git/config` and often
  printed in error messages.
* Do not push from Colab; the token is read-only on purpose. Download the
  `results/*.json` files (Files panel) and commit them from your machine.
* Before sharing the notebook: Edit → Clear all outputs. Revoke the token
  when the benchmarking session is done.
* The model weights come from public Hugging Face repos; no HF token is needed.

## 2. Install and offline smoke test (smallest real-model check)

```bash
%cd echorelay
!pip install -q -e ".[bench]" soundfile
!python -c "import ctranslate2; print('CTranslate2 CUDA devices:', ctranslate2.get_cuda_device_count())"
!python -m benchmarks.prepare_audio --out eval_audio --num 50
!python -m benchmarks.gpu_smoke --audio-dir eval_audio
```

`gpu_smoke` loads base.en on CUDA once per compute type (`float16`,
`int8_float16`), decodes the first evaluation file with partial and final
settings, prints hypothesis vs reference, and exits non-zero on any failure.
Its timings are a single run and are **not** benchmark results.

If loading fails with a `libcudnn` / `libcublas` error: faster-whisper ≥1.1
(CTranslate2 4.5+) needs CUDA 12 + cuDNN 9. Install
`nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*` and set `LD_LIBRARY_PATH` as
described in the faster-whisper README, then rerun the smoke test.

`prepare_audio` downloads Mini LibriSpeech dev-clean-2 (~126 MB) from OpenSLR
and writes the first 50 utterances of 3–20 s (sorted by id) with
references. Keep `eval_audio/` unchanged across all runs; every result file
records each file's SHA-256.

## 3. Server smoke test (one real stream)

```bash
!ECHORELAY_DEVICE=cuda ECHORELAY_COMPUTE_TYPE=float16 nohup python -m echorelay --port 8000 > server.log 2>&1 &
!for i in $(seq 1 60); do curl -sf localhost:8000/health && break; sleep 2; done; echo
!python clients/stream_wav.py "$(ls eval_audio/*.wav | head -1)" --speed 0
!tail -5 server.log
```

Pass criteria: `/health` reports `"device": "cuda"`, the client prints a
nonempty final transcript and exits 0, and `server.log` shows the session
completed. Only then continue.

## 4. Precision comparison (offline, no server)

```bash
!python -m benchmarks.compare_precision --audio-dir eval_audio \
    --compute-types float16 int8_float16 --repeats 3 \
    --output results/precision_t4.json
```

* `float16` = FP16. `int8_float16` = **INT8/FP16 mixed precision**
  (INT8-quantized weights, FP16 for the remaining computation). It is not a
  pure-INT8 model; report it as mixed precision.
* **Headline metric: `aggregate_rtf`** = total decode seconds over all files
  / total audio seconds, computed per repeat and averaged across repeats
  (`aggregate_rtf_per_repeat` and `aggregate_rtf_spread` show run-to-run
  variation). Per-file best-of-repeats percentiles are a secondary,
  optimistic view.
* Same audio, same decoding settings (the server's final-decode settings),
  same WER normalization for every compute type.

## 5. Streaming load test

Restart the server for each configuration under test:

```bash
!pkill -f "python -m echorelay"; sleep 2
!ECHORELAY_DEVICE=cuda ECHORELAY_COMPUTE_TYPE=float16 ECHORELAY_MAX_CONCURRENT_INFERENCE=2 \
  ECHORELAY_MAX_CONCURRENT_STREAMS=16 nohup python -m echorelay --port 8000 > server.log 2>&1 &
!for i in $(seq 1 60); do curl -sf localhost:8000/health && break; sleep 2; done; echo
```

Start with a short run to confirm the harness works on GPU, then scale:

```bash
!python -m benchmarks.load_test --audio-dir eval_audio --streams 5 --concurrency 1 \
    --stream-timeout-s 120 --label "T4 fp16 c=1 pilot" --output results/stream_fp16_pilot.json
!python -m benchmarks.load_test --audio-dir eval_audio --streams 50 --concurrency 1 \
    --stream-timeout-s 120 --label "T4 fp16 c=1" --output results/stream_fp16_c1.json
!python -m benchmarks.load_test --audio-dir eval_audio --streams 50 --concurrency 4 \
    --stream-timeout-s 120 --label "T4 fp16 c=4" --output results/stream_fp16_c4.json
```

Repeat with `ECHORELAY_COMPUTE_TYPE=int8_float16` (restart the server) for
the mixed-precision rows.

### What the numbers mean

Every result file includes a `metric_definitions` block. In short:

* **Time to first partial**: client clock, first audio frame sent → first
  nonempty partial. At real-time pacing it has a floor of about
  `partial_interval_s` (1.0 s default).
* **Final latency**: client clock, `end` sent → `final` received.
* **Final inference RTF (server-reported)**: model time for the final
  decode / audio duration. It **excludes the partial decodes run during the
  stream, time queued for an inference slot, buffering, and network time**.
  It is the model's speed on the full utterance, not the end-to-end cost of
  a stream. Use final latency and throughput for end-to-end behaviour.
* **Partial inference RTF**: same definition per partial decode, same
  exclusions.
* **Throughput**: completed streams and audio seconds per wall-clock second.
* **Failures**: each stream has an overall deadline (`--stream-timeout-s`,
  connect → final). Streams that time out, receive a server `error` event, or
  fail to connect are counted in `streams_failed`, with the category in
  `errors` and the reason per stream in `failures`. A run with failures
  still writes its results.

Caveats to record with results: the load generator and server share the
Colab VM, so at high concurrency the client competes for CPU. `--speed 1.0`
(default) paces at real time; `--speed 0` measures maximum throughput and
should be labelled as such.
