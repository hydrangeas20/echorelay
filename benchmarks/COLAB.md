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

---

# Phase 2 experiments: E7, E1, E2

Plan, predictions and pass/fail checks: [`docs/phase2-plan.md`](../docs/phase2-plan.md).
Run the cells **in order, one experiment at a time**, in a single Colab
session on a T4 runtime. Every experiment writes into its own folder under
`/content/echorelay/results/phase2/`, together with a `provenance.json`
(commit, packages, GPU, server configuration).

Set `PHASE2_SHA` to the commit you are validating. It is the head of
`feat/phase2-observability` at the time you run this; `verify_commit` refuses
anything else.

## P0. Setup (once per session)

**P0.1 GPU attached**

```bash
!nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

**P0.2 Clone the private repo without persisting credentials.** Uses the
`GH_TOKEN` Colab secret: a fine-grained, read-only token for this repository
only (see "Private repository" above).

```python
import base64, os, subprocess
from google.colab import userdata

PHASE2_SHA = "<paste the commit SHA here>"

token = userdata.get("GH_TOKEN")
auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_COUNT": "1",
       "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
       "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {auth}"}
subprocess.run(["git", "clone", "-b", "feat/phase2-observability",
                "https://github.com/hydrangeas20/echorelay.git", "/content/echorelay"],
               env=env, check=True)
del token, auth, env
subprocess.run(["git", "-C", "/content/echorelay", "checkout", "--quiet", "--detach", PHASE2_SHA], check=True)
```

**P0.3 Install, then verify the commit.**

```python
%cd /content/echorelay
!pip install -q -e ".[bench]" soundfile
from benchmarks.colab_helpers import *
verify_commit(PHASE2_SHA)
!git -C /content/echorelay config --list | grep -i -E "extraheader|token" || echo "no credentials stored"
```

**P0.4 Evaluation audio: the same five files as Phase 1.** The check
compares PCM sample hashes with the Phase 1 result manifest. `sha256sum` on
the .wav files cannot be compared because it includes the WAV header.

```python
!python -m benchmarks.prepare_audio --out /content/echorelay/eval_audio5 --num 5
verify_eval_audio("/content/echorelay/eval_audio5",
                  "/content/echorelay/docs/benchmarks/phase1-t4/raw/stream_fp16_c4.json")
```

If this fails, stop: Phase 2 would not be comparable with Phase 1.

**P0.5 GPU smoke gate**

```python
!python -m benchmarks.gpu_smoke --audio-dir /content/echorelay/eval_audio5 \
    --output /content/echorelay/results/phase2/P0_smoke/gpu_smoke.json
```

Continue only if both compute types print `[PASS]`.

**Saving results.** Run this cell after **every** experiment. A Colab
disconnect then loses at most one experiment. The zip contains everything
under `results/phase2/`.

```python
import shutil
from google.colab import files
shutil.make_archive("/content/echorelay_phase2_results", "zip", "/content/echorelay/results/phase2")
files.download("/content/echorelay_phase2_results.zip")
```

## E7. Rescore Phase 1 WER with reference transcripts (no GPU work, seconds)

```python
EXP = "/content/echorelay/results/phase2/E7_rescore"
os.makedirs(EXP, exist_ok=True)
!python -m benchmarks.rescore --audio-dir /content/echorelay/eval_audio5 \
    docs/benchmarks/phase1-t4/raw/stream_fp16_pilot.json \
    docs/benchmarks/phase1-t4/raw/stream_fp16_c4.json \
    docs/benchmarks/phase1-t4/raw/stream_int8_c4.json \
    docs/benchmarks/phase1-t4/raw/precision_t4.json \
    --output {EXP}/rescore.json 2>&1 | tee {EXP}/console.txt
save_provenance(EXP, "E7: rescore Phase 1 transcripts under WER normalizers v1 and v2")
```

How to read it:
* `WER_v1` must reproduce Phase 1 exactly: 0.0708 (FP16) and 0.0619
  (INT8/FP16). If it does not, the references differ from Phase 1; stop and
  report.
* `WER_v2` is the corrected value. The prediction is 0.0531 (6/113) for both
  precisions. Any other value is the finding.

## E1. Offline decode cost vs audio length (no server, ~5-10 min)

```python
stop_server()   # nothing else may use the GPU during the profile
!nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
EXP = "/content/echorelay/results/phase2/E1_profile_decode"
os.makedirs(EXP, exist_ok=True)
!python -m benchmarks.profile_decode --audio-dir /content/echorelay/eval_audio5 \
    --lengths 1 2 4 8 12 16 20 25 30 40 60 --compute-types float16 int8_float16 --repeats 5 \
    --output {EXP}/profile_decode_t4.json 2>&1 | tee {EXP}/console.txt
save_provenance(EXP, "E1: offline decode cost vs audio length, FP16 and INT8/FP16")
```

The `nvidia-smi` line should list no compute processes. The prediction:
partial cost ≈ 31 ms + 6.2 ms per audio-second up to 30 s (FP16), with a
step above 30 s.

## E2. Stress matrix, FP16 (~8-10 min)

**E2.1 Start the server and verify it.** Same server settings as Phase 1,
plus per-event timing.

```python
EXP = "/content/echorelay/results/phase2/E2_stress_fp16"
os.makedirs(EXP, exist_ok=True)
start_server(f"{EXP}/server.log", device="cuda", compute_type="float16", event_timing="true",
             max_concurrent_inference=1, max_concurrent_streams=8)
verify_server({"backend": "whisper", "model": "base.en", "device": "cuda", "compute_type": "float16",
               "event_timing": True, "max_concurrent_inference": 1, "max_concurrent_streams": 8,
               "partial_interval_s": 1.0, "min_partial_audio_s": 0.5, "partial_beam_size": 1,
               "final_beam_size": 5, "active_streams": 0})
```

**E2.2 Run the matrix** (c=1, 4, 8, then a burst of 12 connections against
the limit of 8, then c=1 again for recovery).

```python
!python -m benchmarks.stress --audio-dir /content/echorelay/eval_audio5 --streams 20 \
    --concurrency 1 4 8 --burst-extra 4 --label "T4 FP16" --output-dir {EXP} 2>&1 | tee {EXP}/console.txt
```

The last lines are the PASS/FAIL checks, also in `{EXP}/stress_report.json`.
A FAIL is a finding to report, not a reason to rerun.

**E2.3 Record provenance while the server is still up, stop it, analyze.**

```python
save_provenance(EXP, "E2: stress matrix FP16 with event timing")
stop_server()
!python -m benchmarks.analyze_results {EXP}/c1.json {EXP}/c4.json {EXP}/c8.json {EXP}/recovery.json \
    --output {EXP}/analysis.json --markdown {EXP}/analysis.md
```

Then run the saving cell and send the zip back.

## Later experiments (not yet scheduled)

Same pattern: a fresh `EXP` folder, `start_server` / `verify_server`, then
save provenance.

* **E3:** E2 with `compute_type="int8_float16"`.
* **E4:** `--concat-seconds 60 --streams 8 --concurrency 1 2 4 --burst-extra 0 --stream-timeout-s 600`.
* **E6:** `load_test --speed 0 --concurrency 4 --streams 40`.
* **E8:** `max_concurrent_inference=2`, `--concurrency 4 8 --burst-extra 0`.
