#!/usr/bin/env python3
"""Offline FP16 vs INT8 comparison for Whisper base.en on GPU.

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

from .common import audio_manifest, corpus_wer, environment, load_audio_dir, percentiles, write_json


def cuda_devices() -> int:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count()
    except Exception:
        return 0


def measure(compute_type: str, device: str, utts, model_name: str, repeats: int, beam: int) -> dict:
    from echorelay.inference.whisper import WhisperTranscriber

    model = WhisperTranscriber(model_name, device, compute_type, final_beam_size=beam)
    t0 = time.monotonic()
    model.load()
    load_s = time.monotonic() - t0
    options = model.decode_options(final=True)
    model.transcribe(utts[0].float32(), final=True)  # warm-up, excluded

    per_file_s: dict[str, list[float]] = {u.name: [] for u in utts}
    hyps: dict[str, str] = {}
    for _ in range(repeats):
        for u in utts:
            audio = u.float32()
            t = time.monotonic()
            text = model.transcribe(audio, final=True).text
            per_file_s[u.name].append(time.monotonic() - t)
            hyps.setdefault(u.name, text)
            if hyps[u.name] != text:
                hyps[u.name + "#nondeterministic"] = text

    total_audio = sum(u.duration_s for u in utts)
    rtf = [min(per_file_s[u.name]) / u.duration_s for u in utts]
    refs = [(u.reference, hyps[u.name]) for u in utts if u.reference]
    result = {
        "compute_type": compute_type,
        "device": device,
        "model": model_name,
        "decode_options": options,
        "model_load_s": load_s,
        "repeats": repeats,
        "rtf_per_file_best_of_repeats": percentiles(rtf),
        "aggregate_rtf_mean_of_repeats": sum(sum(v) / len(v) for v in per_file_s.values()) / total_audio,
        "wer": corpus_wer([r for r, _ in refs], [h for _, h in refs]),
        "wer_num_utterances": len(refs),
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
    p.add_argument("--compute-types", nargs="+", default=["float16", "int8_float16"])
    p.add_argument("--beam-size", type=int, default=5, help="must match server final_beam_size")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--allow-cpu", action="store_true", help="harness check only; not a GPU result")
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
        print(f"measuring compute_type={ct} on {device} ...", flush=True)
        r = measure(ct, device, utts, a.model, a.repeats, a.beam_size)
        runs.append(r)
        print(f"  RTF p50={r['rtf_per_file_best_of_repeats']['p50']:.4f} "
              f"p95={r['rtf_per_file_best_of_repeats']['p95']:.4f}  WER={r['wer']}")

    write_json(a.output, {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "valid_for_gpu_comparison": device == "cuda",
        "environment": environment(),
        "audio": audio_manifest(utts),
        "runs": runs,
    })
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
