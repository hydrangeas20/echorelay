# Real-model verification (Whisper `base.en`)

> **Status: verified on an NVIDIA T4** (2026-10-25, commit `096a034`). All 8
> real-model tests passed with faster-whisper `base.en`, CUDA FP16, and the
> recorder reported `complete: true` with no problems. The evidence is
> committed unmodified in [`docs/benchmarks/whisper-t4/`](benchmarks/whisper-t4/README.md).
> One verified configuration only; see [Limitations](#limitations).
>
> **Later changes not yet re-verified on a GPU.** PR #10 changed
> `tests/test_whisper.py` (a corrected comment; the real-speech test also
> records the CUDA libraries it loaded) and `benchmarks/whisper_verification.py`
> (environment provenance, and `PYTEST_PLUGINS` treated as an override). That
> code has been tested offline only. The T4 result above applies to `096a034`;
> see [Status of later changes](#status-of-later-changes).

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
the same transcripts; the comparable v2 value is 0.0531. PR #9 left the stale
comment in place so that the test file stayed byte-identical to the verified
commit. PR #10 corrected it, explaining the v1/v2 distinction. The bound was
not changed.

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
for var in ("PYTHONPATH", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):  # the recorder rejects all three
    env.pop(var, None)                                          # (Colab sets PYTHONPATH=/env/python)
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
command, any `PYTEST_ADDOPTS` / `PYTEST_PLUGINS` / `PYTHONPATH` overrides,
per-test outcomes with recorded WER, the runtime identity observed during the
run, and environment provenance), `pytest.log` and `junit.xml`.

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
* `PYTEST_ADDOPTS`, `PYTEST_PLUGINS` and `PYTHONPATH` are unset, so no plugin
  or module was injected, and the checkout had no local edits;
* **the checkout audit finds nothing untracked that could run in the test
  process.** It runs before pytest starts, with
  `git status --porcelain --untracked-files=all --ignored=matching`, and the
  run fails if:
  * **any untracked, non-ignored file exists anywhere** in the repository.
    `git status --untracked-files=no` (the dirty check) does not see these,
    yet pytest loads any `conftest.py` from the repository root down to
    `tests/`, and the root is on the import path;
  * **any ignored file that Python or pytest would load** sits directly in
    the repository root, `tests/` or `src/`: `*.py`, `*.pyc`, `*.pyo`,
    `*.pth`, `*.so`, or a configuration file (`conftest.py`, `pytest.ini`,
    `.pytest.ini`, `tox.ini`, `setup.cfg`, `pyproject.toml`,
    `sitecustomize.py`, `usercustomize.py`). Ignored *directories*, such as
    `__pycache__/`, `.cache/` or `*.egg-info/`, are allowed;
  * the audit itself fails (fail closed);
* every provenance section (below) was collected. A section that raises is
  recorded as `collection_error` and makes the run incomplete, instead of
  aborting the record.

`PYTEST_PLUGINS`, the checkout audit and the provenance rule were added in
PR #10, so the `096a034` run predates them. `tests/test_whisper_verification.py`
(offline, in CI) checks each of these rules. The `model.bin` hash is recorded for traceability; there is no
committed reference hash to compare it against.

### Environment provenance

Since PR #10 the record has a `provenance` section. Its *values* never
change `complete`; only a failure to collect a section does (see above). It
separates what the environment *said* from what the test process *loaded*:

| Field | What it is | What it is not |
|---|---|---|
| `environment_metadata.variables` | Allowlisted variables passed to the test process: `LD_LIBRARY_PATH` (as entries), `LD_PRELOAD`, `CUDA_HOME`, `CUDA_PATH`, `CUDA_VISIBLE_DEVICES`, `CUDA_MODULE_LOADING`, `NVIDIA_VISIBLE_DEVICES`, `NVIDIA_DRIVER_CAPABILITIES`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD`, any `CT2_*` | Proof of anything loaded. Nothing outside the allowlist is read. Recognisable credentials are redacted (see *Redaction*). |
| `environment_metadata.cuda_libraries_on_search_path` | Which `LD_LIBRARY_PATH` directories contain `libcublas.so.12`, `libcublasLt.so.12`, `libcudnn.so.9` | Which copy the loader picked: it also searches `RUNPATH`, `ld.so.cache` and system directories. A directory that cannot be read appears in `search_errors`, and `ld_library_path_entries_exist` shows the error; whether it holds the library is unknown. |
| `environment_metadata.nvidia_wheels` | Installed `nvidia-*` pip distributions and versions | Proof they were used. |
| `pytest_plugins.installed_entry_points` | Installed `pytest11` entry points (plugin, module, distribution, version) in the recorder's interpreter, which is the one the tests run in: what pytest *can* auto-load | A list of what ran. |
| `pytest_plugins.header_reported` | The `plugins:` line pytest printed for this run | **A complete list of loaded plugins.** pytest lists only plugins that came from installed distributions. A plugin loaded with `-p`, `PYTEST_PLUGINS`, a `conftest.py` or a `pytest_plugins` variable does not appear in it. Those routes are covered by the override rule and the checkout audit, not by this inventory. |
| `loaded_libraries` | CUDA / CTranslate2 shared objects mapped into the pytest process after the real-speech decodes (`/proc/self/maps`, Linux) | Which library served a given call, or anything loaded later. On macOS or without `/proc` it is `available: false`. |

**Limits.**
* Provenance is recorded but **not** validated: a surprising `LD_PRELOAD` or
  an extra pytest plugin does not make a run incomplete. Requiring that
  would mean choosing an allowlist of plugins (Colab's image ships several)
  or disabling plugin auto-loading for the run. Both are possible follow-ups
  that change what counts as a valid run, so they are left for a decision.
* `loaded_libraries` comes from the tests' own process, so a run that
  substitutes the test code could falsify it. The integrity rules (no
  overrides, clean checkout, checkout audit, the exact expected tests) are
  what make it trustworthy.
* `/proc/self/maps` is parsed with the pathname taken as everything after
  the fifth field, so paths containing spaces are kept whole. A
  ` (deleted)` suffix (file replaced while mapped) is dropped.

**Redaction.** Only allowlisted variables are read; that is the main
control. Within them, redaction removes:
* URL userinfo, from `//` to the *last* `@`, so passwords containing `@` or
  `/` are removed whole. This can over-redact a path that itself contains
  `@`.
* `KEY=value` assignments whose key looks secret (`token`, `secret`,
  `passw`, `pwd`, `apikey`, `credential`, `auth`, ...), including inside
  query strings.
* Common token formats: GitHub `ghp_`/`github_pat_`, Hugging Face `hf_`,
  AWS `AKIA`/`ASIA`, `sk-`, Slack `xox?-`, GitLab `glpat-`, JWTs.

Variables whose *name* looks secret are redacted entirely. **Redaction
cannot recognise every secret**: an unknown token format in an allowlisted
variable would be recorded.
* The variables are those the recorder passes to pytest. Changes made inside
  the test process (none in this suite) would not appear.

### Reproducing the environment

To reproduce the `096a034` T4 run as closely as the evidence allows:

1. **Runtime.** A Colab T4 GPU runtime, Python 3.13 (the record shows
   3.13.16, driver 580.82.07).
2. **Packages.** Pin the libraries the record lists instead of installing
   the latest: after `pip install -e ".[dev]" soundfile`, run
   `pip install faster-whisper==1.2.1 ctranslate2==4.8.3`. Compare the
   `environment.packages` of your record with
   `docs/benchmarks/whisper-t4/raw/record.json`.
3. **Libraries.** Apply the cuBLAS 12 / cuDNN 9 `LD_LIBRARY_PATH` fix (Cell 2
   below). With the PR #10 recorder, check that `provenance.loaded_libraries`
   lists `libcublas.so.12` from the pip wheel directory, not CUDA 13's
   `libcublas.so.13`.
4. **Recorder environment.** Run it from a fresh clone with no added files
   (the checkout audit rejects untracked files), and without `PYTHONPATH`,
   `PYTEST_ADDOPTS` or `PYTEST_PLUGINS` (Cell 2 removes them).
5. **Weights.** `observed_model_path` names the Hugging Face snapshot that was
   used (`3d3d5dee…`), and `observed_model_bin_sha256` its hash. Compare them
   with the committed record.

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

The `096a034` record does not capture `LD_LIBRARY_PATH`, so for that run
the library-path fix appears only in this account, not in the evidence.
From PR #10 the recorder records it, and the libraries actually loaded.

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
* **Provenance in the `096a034` evidence.** That record has no
  `LD_LIBRARY_PATH`, plugin inventory or loaded-library list; the plugins
  appear only in its `pytest.log`. The PR #10 recorder records all three,
  but records them without validating them (see
  [Environment provenance](#environment-provenance)).
* **Not in CI.** CI runs the offline suite, including the recorder's
  validation rules (`tests/test_whisper_verification.py`), but not the
  real-model tests. Re-verification is manual (Colab).

## Status of later changes

| Change | Commit | Offline tests | Real-model run |
|---|---|---|---|
| Suite, recorder, identity checks | `096a034` | yes | **T4, 8/8, `complete: true`** |
| v1/v2 comment correction; loaded-library recording in the real-speech test; recorder provenance; `PYTEST_PLUGINS` as an override; checkout audit; fail-closed provenance collection; stronger redaction | PR #10 | yes (`tests/test_whisper_verification.py`) | **not yet re-verified** |

Re-running Cell 2 on a T4 at the PR #10 commit would re-verify the current
code and produce the first record with provenance.

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
