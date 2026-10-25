# Real-model verification (Whisper `base.en`)

> **Status: GPU verification pending.** The real-model test suite below exists
> and its harness has been exercised, but it has **not yet been run against the
> real model on an NVIDIA T4**. No real-model result is recorded in this file
> yet. Until one is, the repository's real-model evidence is the Phase 1–4 T4
> benchmark data, collected through the benchmark tooling and the server, not
> through this test suite.

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
so the WER is computed on exactly the audio whose FP16 WER was recorded.

**Thresholds are regression guards, not measurements.** Recorded T4 FP16
values on the same audio: corpus WER 0.0708 (8 edits / 113 words,
`precision_t4.json`) and 0.1154 for the streamed utterance (3 / 26,
`gpu_smoke.json`). The bounds (0.12 and 0.25) allow for library and driver
drift; a broken decode path (wrong sample rate, scaling or truncation) lands
far above them.

**Not covered.** GPU memory release after shutdown (only worker threads and
the port are checked), concurrency under load (covered by the benchmarks,
not this suite), other model sizes, `int8_float16`, and GPUs other than the T4.

## How to run

On a Colab T4 runtime, one cell:

```bash
!nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
%cd /content
!rm -rf echorelay && git clone -b test/whisper-t4-verification https://github.com/hydrangeas20/echorelay.git
%cd /content/echorelay
!git log -1 --format='%H %s'
!pip install -q -e ".[dev]" soundfile
!python -m benchmarks.prepare_audio --out /content/eval_audio5 --num 5
!python -m benchmarks.whisper_verification --audio-dir /content/eval_audio5 --device cuda --out /content/whisper_verification
!cd /content && zip -qr whisper_verification.zip whisper_verification
from google.colab import files; files.download("/content/whisper_verification.zip")
```

Once the PR is merged, clone `main` instead of the branch.
`benchmarks/whisper_verification.py` runs `python -m pytest -m whisper -rA -s`
on `tests/test_whisper.py` with `ECHORELAY_TEST_DEVICE=cuda` and
`ECHORELAY_TEST_COMPUTE_TYPE=float16`, then writes `record.json` (UTC times,
git commit and dirty flag, `nvidia-smi` output, GPU and driver, Python and
package versions, model, device, compute type, the exact command, any
`PYTEST_ADDOPTS` / `PYTHONPATH` overrides, and per-test outcomes with recorded
WER), `pytest.log` and `junit.xml`. `complete` is true only if every collected
test passed and none was skipped.

The suite also runs on CPU (`ECHORELAY_TEST_DEVICE=cpu`, `int8`, the default)
with the same commands minus `--device cuda`. A CPU run verifies the adapter,
server and WebSocket paths with the real model, but not CUDA or FP16.

## Verification log

| UTC date | Commit | GPU / driver | Python | faster-whisper / CTranslate2 | Device / compute | Result |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | **Pending: not yet run on the real model** |

## What was verified so far

Without a GPU or model weights (Hugging Face is not reachable from the
environment used to write this suite):

* The suite is collected by `pytest -m whisper` (8 tests) and deselected by
  the default run.
* With faster-whisper's `WhisperModel` replaced by a stub, all 8 tests pass,
  and they fail as intended when the stub returns wrong transcripts (both
  speech tests), when CUDA is requested with no GPU (the model fixture fails;
  it does not skip), and when the audio is not the recorded set (hash check).
  Without `ECHORELAY_TEST_AUDIO_DIR`, the 3 speech tests skip with a stated
  reason and the recorder reports `complete: false`.

None of this is evidence about the real model; it only shows that the tests
and the recorder behave as described.
