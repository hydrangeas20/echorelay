#!/usr/bin/env python3
"""Offline FP16 vs INT8/FP16 mixed-precision comparison for Whisper base.en on GPU.

    python -m benchmarks.compare_precision --audio-dir eval_audio \\
        --compute-types float16 int8_float16 --output results/precision.json

Controls that keep the comparison fair:

* identical evaluation audio (file list + SHA-256 recorded),
* identical decoding settings: ``WhisperTranscriber.decode_options(final=True)``,
  the same settings the server uses for final transcripts,
* identical WER normalization (``benchmarks.common.normalize_text``),
* one warm-up decode per compute type (excluded), then ``--repeats`` timed
  passes over every file; per-file time is ``time.monotonic()`` around the
  full decode (generator fully consumed).

Headline metric: **aggregate RTF** = (sum over files of decode seconds) /
(sum of audio seconds), computed per repeat and averaged across repeats.
Per-file best-of-repeats RTF percentiles are reported as a secondary,
optimistic view (minimum time hides variance).

Precision labels: ``float16`` is FP16. ``int8_float16`` is NOT pure INT8: it
is CTranslate2 mixed precision, with INT8-quantized weights and FP16 for the
remaining computation. ``int8`` / ``int8_float32`` are listed for
completeness (CPU-oriented).

Each compute type is loaded, measured, and released before the next.
Refuses to run without a CUDA device unless ``--allow-cpu`` is given; CPU
results are marked ``valid_for_gpu_comparison: false``.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from datetime import datetime, timezone

from .common import (
    DEFAULT_NORMALIZER,
    NORMALIZERS,
    audio_manifest,
    corpus_wer,
    environment,
    load_audio_dir,
    percentiles,
    write_json,
)


def cuda_devices() -> int:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count()
    except Exception:
        return 0


PRECISION_LABELS = {
    "float16": "FP16",
    "bfloat16": "BF16",
    "float32": "FP32",
    "int8_float16": "INT8/FP16 mixed precision (INT8-quantized weights, FP16 elsewhere)",
    "int8_bfloat16": "INT8/BF16 mixed precision (INT8-quantized weights, BF16 elsewhere)",
    "int8_float32": "INT8/FP32 mixed precision (INT8-quantized weights, FP32 elsewhere)",
    "int8": "INT8-quantized weights (CTranslate2 'int8'; non-quantized ops in the device default)",
}


def precision_label(compute_type: str) -> str:
    return PRECISION_LABELS.get(compute_type, compute_type)


def rtf_stats(per_file_s: dict[str, list[float]], durations: dict[str, float]) -> dict:
    """RTF summaries from per-file decode times (one entry per repeat).

    ``aggregate_rtf`` (headline): for each repeat, total decode seconds over
    all files / total audio seconds; then the mean across repeats. Long files
    weigh in proportion to their length, as they would in production.
    """
    total_audio = sum(durations.values())
    repeats = min(len(v) for v in per_file_s.values())
    per_repeat = [sum(per_file_s[n][r] for n in per_file_s) / total_audio for r in range(repeats)]
    best_per_file = [min(per_file_s[n]) / durations[n] for n in per_file_s]
    return {
        "aggregate_rtf": sum(per_repeat) / len(per_repeat),
        "aggregate_rtf_per_repeat": per_repeat,
        "aggregate_rtf_spread": max(per_repeat) - min(per_repeat),
        "rtf_per_file_best_of_repeats": percentiles(best_per_file),
    }


def measure(compute_type: str, device: str, utts, model_name: str, repeats: int, beam: int,
            wer_normalizer: str = DEFAULT_NORMALIZER) -> dict:
    from echorelay.inference.whisper import WhisperTranscriber

    model = WhisperTranscriber(model_name, device, compute_type, final_beam_size=beam)
    t0 = time.monotonic()
    model.load()
    load_s = time.monotonic() - t0
    options = model.decode_options(final=True)
    model.transcribe(utts[0].float32(), final=True)  # warm-up, excluded

    per_file_s: dict[str, list[float]] = {u.name: [] for u in utts}
    hyps: dict[str, str] = {}
    nondeterministic: list[str] = []
    for _ in range(repeats):
        for u in utts:
            audio = u.float32()
            t = time.monotonic()
            text = model.transcribe(audio, final=True).text
            per_file_s[u.name].append(time.monotonic() - t)
            if u.name in hyps and hyps[u.name] != text and u.name not in nondeterministic:
                nondeterministic.append(u.name)
            hyps.setdefault(u.name, text)  # WER uses the first repeat's hypothesis

    refs = [(u.reference, hyps[u.name]) for u in utts if u.reference]
    result = {
        "compute_type": compute_type,
        "precision": precision_label(compute_type),
        "device": device,
        "model": model_name,
        "decode_options": options,
        "model_load_s": load_s,
        "repeats": repeats,
        **rtf_stats(per_file_s, {u.name: u.duration_s for u in utts}),
        "wer": corpus_wer([r for r, _ in refs], [h for _, h in refs], wer_normalizer),
        "wer_normalizer": wer_normalizer,
        "wer_num_utterances": len(refs),
        "nondeterministic_files": nondeterministic,
        "hypotheses": hyps,
        "per_file_seconds": per_file_s,
    }
    del model
    gc.collect()
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--model", default="base.en")
    p.add_argument("--compute-types", nargs="+", default=["float16", "int8_float16"],
                   help="CTranslate2 compute types; int8_float16 is INT8/FP16 mixed precision")
    p.add_argument("--beam-size", type=int, default=5, help="must match server final_beam_size")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--allow-cpu", action="store_true", help="harness check only; not a GPU result")
    p.add_argument("--wer-normalizer", choices=NORMALIZERS, default=DEFAULT_NORMALIZER)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    n_gpu = cuda_devices()
    if n_gpu == 0 and not a.allow_cpu:
        sys.exit("no CUDA device visible to CTranslate2; refusing to produce GPU results "
                 "(use --allow-cpu only to smoke-test the harness)")
    device = "cuda" if n_gpu else "cpu"
    utts = load_audio_dir(a.audio_dir)

    runs = []
    for ct in a.compute_types:
        print(f"measuring {ct} [{precision_label(ct)}] on {device} ...", flush=True)
        r = measure(ct, device, utts, a.model, a.repeats, a.beam_size, a.wer_normalizer)
        runs.append(r)
        best = r["rtf_per_file_best_of_repeats"]
        print(f"  aggregate RTF (mean of {r['repeats']} repeats) = {r['aggregate_rtf']:.4f} "
              f"[spread {r['aggregate_rtf_spread']:.4f}]   WER = {r['wer']}")
        print(f"  secondary: per-file best-of-repeats RTF p50={best['p50']:.4f} p95={best['p95']:.4f}")

    write_json(a.output, {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "valid_for_gpu_comparison": device == "cuda",
        "headline_metric": "aggregate_rtf: total decode s / total audio s per repeat, mean over repeats",
        "environment": environment(),
        "audio": audio_manifest(utts),
        "runs": runs,
    })
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
