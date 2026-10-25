# Real-model verification (Whisper `base.en`)

> **Status: verified on an NVIDIA T4** (2026-10-25, commit `096a034`). All 8
> real-model tests passed with faster-whisper `base.en`, CUDA FP16, and the
> recorder reported `complete: true` with no problems. The evidence is
> committed unmodified in [`docs/benchmarks/whisper-t4/`](benchmarks/whisper-t4/README.md).
> One verified configuration only; see [Limitations](#limitations).

## Why this exists

The offline suite (`pytest -m "not whisper"`, run in CI) uses fake and gated
backends. Before this change, the only real-model test was a CPU `int8`
smoke test that decoded silence and checked that the result was a string. It
did not check CUDA, FP16, transcript correctness, error handling, the server,
or the WebSocket path, and it had not been confirmed as executed.

## What the suite claims, by level

`tests/test_whisper.py` (marker `whisper`, deselected by default). Each level
claims only what it exercises.

| Level | Test | Asserts |
|---|---|---|
| Adapter | `test_adapter_loads_with_requested_device_and_compute_type` | The device and compute type **CTranslate2 reports in effect** match the request (on CUDA: exactly `float16`, so a silent fallback fails). |
| Adapter | `test_adapter_decodes_silence_and_empty_audio` | Silence decodes to a string with partial and final settings; empty audio returns `""`. |
| Adapter | `test_adapter_transcribes_real_speech` | All 5 reference utterances decode to non-empty text with final and partial settings; final corpus WER ≤ 0.12. |
| Adapter | `test_adapter_final_decode_is_deterministic` | Two final decodes of the same audio give identical text. |
| Adapter | `test_adapter_wraps_decode_errors` | A failure inside faster-whisper (non-numeric input) is raised as `InferenceError`. |
| Server | `test_server_loads_real_model_from_settings` | A real uvicorn server whose lifespan builds the model from `Settings` reports `whisper` / `base.en` / device / compute type on `/health`. |
| WebSocket | `test_websocket_streams_real_speech` | A real WebSocket client streams a 10.9 s utterance in 100 ms frames at 2× real time: `ready` first, ≥1 non-empty provisional partial with consecutive `seq`, exactly one `final`, no `error`, close code 1000. The final text **equals** a direct adapter decode of the same audio (the server path does not alter the audio), and its WER ≤ 0.25. |
| Server | `test_server_shutdown_releases_inference_workers` | After a decode, stopping the server stops the thread, the inference worker threads exit, and the port stops accepting connections. |

**Fixtures.** The speech tests need the 5-utterance set from
`python -m benchmarks.prepare_audio --num 5` (Mini LibriSpeech dev-clean-2,
first 5 utterances of 3–20 s). They check each file's PCM SHA-256 against the
recorded Phase 1 precision run (`docs/benchmarks/phase1-t4/raw/precision_t4.json`),
so the WER is computed on exactly the audio used by the recorded T4 runs.

**Thresholds are regression guards, not measurements.** The bounds (0.12
corpus, 0.25 for the streamed utterance) were set before the T4 run and were
not changed after it; a broken decode path (wrong sample rate, scaling or
truncation) lands far above them. Recorded T4 FP16 values on the same audio,
under the **v2** normalizer the tests use: corpus WER 0.0531 (6 edits / 113
words, Phase 3 evidence). For the streamed utterance, 0.1154 (3 / 26) was
recorded in `gpu_smoke.json` under v1. The T4 run scored the identical
transcript 0.1154 under v2, so for this utterance the two normalizers agree.

*Correction.* The suite was written citing Phase 1's 0.0708 (8 / 113) as the
recorded corpus value. That figure was scored with the **v1** normalizer on
the same transcripts; the comparable v2 value is 0.0531. The comment above
`MAX_CORPUS_WER` in `tests/test_whisper.py` still cites 0.0708. It is left
unchanged so that the test file stays byte-identical to the verified commit.
The bound itself is unaffected.

**Not covered.** GPU memory release after shutdown (only worker threads and
the port are checked), concurrency under load (covered by the benchmarks,
not this suite), other model sizes, `int8_float16`, and GPUs other than the T4.

## How to run

On a Colab **T4 GPU** runtime. These are the steps of the verified run,
including the library-path fix it needed (see
[troubleshooting](#troubleshooting-incident-colab-cuda-libraries)).

Cell 1: clone, install and fetch the reference audio.

```bash
!nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
%cd /content
!rm -rf echorelay && git clone -b test/whisper-t4-verification https://github.com/hydrangeas20/echorelay.git
%cd /content/echorelay
!git log -1 --format='%H %s'
!pip install -q -e ".[dev]" soundfile
!python -m benchmarks.prepare_audio --out /content/eval_audio5 --num 5
```

Cell 2: expose cuBLAS 12 (and cuDNN 9, if present), run the recorder without
Colab's default `PYTHONPATH`, and download the evidence.

```python
import glob, os, subprocess

cublas = glob.glob("/usr/local/lib/python*/dist-packages/nvidia/cublas/lib/libcublas.so.12")
assert cublas, "cuBLAS 12 library not found"
dirs = [os.path.dirname(cublas[0])]
cudnn = glob.glob("/usr/local/lib/python*/dist-packages/nvidia/cudnn/lib/libcudnn.so.9")
if cudnn:
    dirs.append(os.path.dirname(cudnn[0]))
existing = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = ":".join(dirs + ([existing] if existing else []))

env = os.environ.copy()
env.pop("PYTHONPATH", None)  # Colab sets /env/python; the recorder rejects any PYTHONPATH
result = subprocess.run(
    ["python", "-m", "benchmarks.whisper_verification", "--audio-dir", "/content/eval_audio5",
     "--device", "cuda", "--out", "/content/whisper_verification"],
    cwd="/content/echorelay", env=env)
print("exit code:", result.returncode)

subprocess.run(["zip", "-qr", "whisper_verification.zip", "whisper_verification"], cwd="/content", check=True)
from google.colab import files
files.download("/content/whisper_verification.zip")
```

It passes if the recorder's last line starts with `COMPLETE: 8 passed, 0
failed, 0 errors, 0 skipped` and the exit code is 0.

Once the PR is merged, clone `main` instead of the branch.
`benchmarks/whisper_verification.py` runs `python -m pytest -m whisper -rA -s`
on `tests/test_whisper.py` with `ECHORELAY_TEST_DEVICE=cuda` and
`ECHORELAY_TEST_COMPUTE_TYPE=float16`, then writes `record.json` (UTC times,
git commit and dirty flag, `nvidia-smi` output, GPU and driver, Python and
package versions, the requested model, device and compute type, the exact
command, any `PYTEST_ADDOPTS` / `PYTHONPATH` overrides, per-test outcomes with
recorded WER, and the runtime identity observed during the run), `pytest.log`
and `junit.xml`.

**What `complete: true` requires.** The requested settings in the record are
not treated as evidence. `complete` is true only if `problems` is empty, i.e.:

* exactly the 8 expected tests ran (by name), all passed, none failed, errored
  or was skipped, and pytest exited 0;
* the identity test recorded what the loaded runtime reported, and it matches
  the request: the object is faster-whisper's `WhisperModel` over
  CTranslate2's `Whisper` (a substituted model class fails), the CTranslate2
  device and compute type (exactly `float16` on CUDA), at least one CUDA
  device, the `models--Systran--faster-whisper-base.en` weights snapshot that
  faster-whisper resolved (with the SHA-256 of its `model.bin`), and an
  English-only model;
* on CUDA, `nvidia-smi` lists a GPU containing `T4` (`--expect-gpu`);
* `PYTEST_ADDOPTS` and `PYTHONPATH` are unset, so no plugin or module was
  injected, and the checkout had no local edits.

`tests/test_whisper_verification.py` (offline, in CI) checks each of these
rules. The `model.bin` hash is recorded for traceability; there is no
committed reference hash to compare it against.

The suite also runs on CPU (`int8`) with `--device cpu`. A CPU run verifies
the adapter, server and WebSocket paths with the real model, but not CUDA or
FP16.

## Verification log

| UTC date | Commit | GPU / driver | Python | faster-whisper / CTranslate2 | Device / compute | Result | Evidence |
|---|---|---|---|---|---|---|---|
| 2026-10-25 | `096a034` | Tesla T4 / 580.82.07 | 3.13.16 | 1.2.1 / 4.8.3 | cuda / float16 (observed) | **8 passed**, 0 failed / errors / skips; `complete: true`. Corpus WER 0.0531, stream WER 0.1154 | [whisper-t4](benchmarks/whisper-t4/README.md) |

The evidence README lists the checks made on these files independently of
the recorder's verdict. They include re-deriving the results from
`junit.xml`, re-running the validator, and matching each JUnit test line to
the committed source. It also compares the transcripts with earlier T4
evidence: all five are word for word identical to Phase 1's FP16
hypotheses.

## Troubleshooting incident: Colab CUDA libraries

*Source: the operator's account of the Colab session. The failed attempts'
output is not in the repository or in the evidence ZIP. Only the final,
successful run's files are committed.*

1. **First attempt: failed.** Loading the model on CUDA failed with:

   ```
   Library libcublas.so.12 is not found or cannot be loaded
   ```

   The Colab environment had:

   ```
   PYTHONPATH: /env/python
   LD_LIBRARY_PATH: /usr/lib64-nvidia
   ```

   The system CUDA installation exposed `libcublas.so.13`. CTranslate2 4.8.3
   needs the CUDA 12 library, which was present as a pip-installed package at
   `/usr/local/lib/python3.13/dist-packages/nvidia/cublas/lib/libcublas.so.12`
   but was not on the loader's search path.

2. **Fix:** that directory was prepended to `LD_LIBRARY_PATH` in the kernel,
   so child processes inherit it. If `libcudnn.so.9` was present, its
   directory was added too. Cell 2 above does the same thing.

3. **Second attempt: tests passed, recorder rejected it.** With the library
   path fixed, all 8 real-model tests passed. The recorder still reported
   `NOT COMPLETE`, because Colab's default `PYTHONPATH=/env/python` violates
   its evidence-integrity rule (any `PYTHONPATH` could substitute modules).
   This was the rule working as designed, and it was not relaxed.

4. **Final run: complete.** The recorder was run with `PYTHONPATH` removed
   from its environment (Cell 2). It exited 0 and reported `complete: true`.
   Those are the committed files.

The record does not capture `LD_LIBRARY_PATH`, so the library-path fix
appears only in this account, not in the evidence.

## Limitations

* **One configuration.** `base.en`, CUDA FP16, one Colab T4, one run.
  Not verified: CPU `int8`, `int8_float16`, other model sizes or GPUs, and
  other driver or library versions.
* **Not covered by the suite:** GPU memory release after shutdown (only
  worker threads and the port are checked), and behaviour under concurrent
  load (that is the benchmarks' job).
* **Single-run timings.** Test durations in the evidence are a side effect,
  not measurements.
* **`model.bin` hash.** Recorded for traceability. No independent reference
  hash is committed to compare it against.
* **Not recorded by the recorder:** `LD_LIBRARY_PATH`, and the pytest plugins
  auto-loaded from the environment (they appear only in `pytest.log`).
* **Not in CI.** CI runs the offline suite, including the recorder's
  validation rules (`tests/test_whisper_verification.py`), but not the
  real-model tests. Re-verification is manual (Colab).

## Before the GPU run

These checks came before the T4 run. They were made without a GPU or model
weights, because Hugging Face was not reachable from the environment used to
write this suite:

* The suite is collected by `pytest -m whisper` (8 tests) and deselected by
  the default run.
* With faster-whisper's `WhisperModel` replaced by a stub, all 8 tests pass,
  and they fail as intended when the stub returns wrong transcripts (both
  speech tests), when CUDA is requested with no GPU (the model fixture fails;
  it does not skip), and when the audio is not the recorded set (hash check).
  Without `ECHORELAY_TEST_AUDIO_DIR`, the 3 speech tests skip with a stated
  reason; the recorder reports `complete: false`.
* With the stub, all 8 tests pass but the recorder reports `complete: false`
  and names why: the stub's model and CTranslate2 classes, the injected
  `PYTEST_ADDOPTS` / `PYTHONPATH`, and the uncommitted checkout.

None of this is evidence about the real model. It showed only that the tests
and the recorder behave as described; the T4 run above is the real-model
evidence.
