# Real-model verification on an NVIDIA T4

The real-model test suite (`tests/test_whisper.py`, `pytest -m whisper`) run
against faster-whisper `base.en` with CUDA FP16 on a Colab Tesla T4, recorded
by `benchmarks/whisper_verification.py`. Suite design, claims and limits:
[docs/whisper-verification.md](../../whisper-verification.md).

## Result

| | |
|---|---|
| Run (UTC) | 2026-10-25 04:06:41 → 04:06:52 |
| Commit | `096a0347b620e4e5a8696cd6288379c3a3ccae77` (clean checkout) |
| GPU / driver | Tesla T4, 15360 MiB / 580.82.07 (`nvidia-smi` reports CUDA 13.0) |
| Python / platform | 3.13.16 / Linux-6.6.122+-x86_64-with-glibc2.39 |
| faster-whisper / CTranslate2 | 1.2.1 / 4.8.3 |
| Observed runtime | `faster_whisper.transcribe.WhisperModel` over `ctranslate2._ext.Whisper`; device `cuda`, compute type `float16`; 1 CUDA device; English-only |
| Weights | `models--Systran--faster-whisper-base.en`, snapshot `3d3d5dee26484f91867d81cb899cfcf72b96be6c`; `model.bin` SHA-256 `2a166925…e93e2ef` |
| Tests | **8 passed, 0 failed, 0 errors, 0 skipped** (exactly the 8 expected tests); pytest exit 0 |
| Recorder verdict | `complete: true`, `problems: []`; no `PYTEST_ADDOPTS` / `PYTHONPATH` override |
| Corpus WER, final settings (5 files) | 0.0531 (bound 0.12) |
| Corpus WER, partial settings | 0.0531 |
| WebSocket stream WER (1272-135031-0000) | 0.1154 (bound 0.25); 10 partials, then one final |

WER uses the v2 normalizer (the default since Phase 2, see
[measurements](../../measurements.md#7-wer)). The bounds are regression
guards set before this run and were not changed after it.

## Files

`raw/` holds the recorder's output **unmodified**, as returned from the Colab
session (`whisper_verification_final.zip`, SHA-256 `1a2bbf4e…bd544c8`). Hashes
are in `raw/SHA256SUMS`, checked by `python -m benchmarks.verify_evidence`.

| File | Contents |
|---|---|
| `raw/record.json` | Environment, `nvidia-smi`, requested settings, observed identity, per-test outcomes, verdict |
| `raw/pytest.log` | Complete pytest output, including all five transcripts |
| `raw/junit.xml` | Per-test results and recorded properties |

The only host-specific values are Colab paths (`/content`, `/root/.cache`) and
the JUnit `hostname` (an ephemeral container ID). No credentials or personal
data are present. Model weights and audio are not committed.

## Independent checks on the evidence

Done on the files above, without trusting the recorder's own verdict:

* `junit.xml` re-parsed with the recorder's parser reproduces
  `record.json`'s `results` exactly; the validator re-run on those raw inputs
  reports no problems.
* Every JUnit test `line` matches the `def` of that test in
  `tests/test_whisper.py` at the recorded commit, so the suite that ran is the
  committed one. Both `tests/test_whisper.py` and
  `benchmarks/whisper_verification.py` were unchanged from that commit
  through the merge of PR #9 (`be98cd1`). PR #10 later changed both (a
  corrected comment, provenance recording); see
  [the current status](../../whisper-verification.md#status-of-later-changes).
  This result applies to `096a034`.
* `pytest.log` agrees (`8 passed`, rootdir `/content/echorelay`), and its
  timestamps fall inside the recorded run window.
* The commit exists on `test/whisper-t4-verification` (PR #9).

## Comparison with recorded T4 evidence

* **Transcripts:** all five final transcripts are word for word identical to
  the FP16 hypotheses recorded in Phase 1
  (`../phase1-t4/raw/precision_t4.json`). That run used the same GPU, driver and
  library versions.
* **WER:** 0.0531 is exactly the v2 corpus WER committed for the same five
  files in Phase 3 (for example `../scheduling-t4/raw/04_r1_both/c4.json`).
  Phase 1 reported 0.0708 for the same transcripts because it used the v1
  normalizer. The difference is scoring, not decoding.
* The streamed utterance's transcript and WER (0.1154) match Phase 1's
  `gpu_smoke.json`.

## Run conditions not captured by the record

These come from the Colab session (the operator's account), not from the
files above. See [troubleshooting](../../whisper-verification.md#troubleshooting-incident-colab-cuda-libraries):

* `LD_LIBRARY_PATH` was extended to the pip-installed cuBLAS 12 (and cuDNN 9,
  if present) directories before the run. The recorder at `096a034` did not
  record `LD_LIBRARY_PATH`; it does from PR #10 onwards.
* The recorder was started with Colab's default `PYTHONPATH=/env/python`
  removed from its environment, so the record shows no override.
* pytest auto-loaded the plugins installed on the Colab image (`typeguard`,
  `anyio`, `langsmith`, `platformdirs`, besides `asyncio`); see `pytest.log`.
  None is injected by an override, and none is used by the suite.
