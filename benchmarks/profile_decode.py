#!/usr/bin/env python3
"""Offline decode cost as a function of audio length (no server, no queueing).

    python -m benchmarks.profile_decode --audio-dir eval_audio \\
        --lengths 1 2 4 8 12 16 20 25 30 40 60 --compute-types float16 int8_float16 \\
        --output results/profile_decode_t4.json

For each compute type and each target length L, a test signal is built
deterministically by concatenating the sorted evaluation files and cutting
the result to exactly L seconds (the cut can fall mid-word; that is fine for
cost, and this tool does not score accuracy). Each length is decoded with
the server's partial settings (greedy) and final settings (beam search),
``--repeats`` times after one warm-up per compute type; time is
``time.monotonic()`` around the full decode (generator consumed).

This isolates the cost curve behind repeated full-buffer partial decoding:
Phase 1 streaming data covered only 1-11.5 s of buffered audio. Lengths
above 30 s exercise faster-whisper's multi-window decoding.

Refuses to run without CUDA unless ``--allow-cpu`` (results then marked
``valid_for_gpu: false``).
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from datetime import datetime, timezone

import numpy as np

from .common import SAMPLE_RATE, environment, load_audio_dir, write_json
from .compare_precision import cuda_devices, precision_label


def build_signal(utts, length_s: float) -> np.ndarray:
    """Concatenate sorted utterances cyclically and cut to exactly length_s."""
    need = int(round(length_s * SAMPLE_RATE))
    parts, have, i = [], 0, 0
    while have < need:
        a = utts[i % len(utts)].float32()
        parts.append(a)
        have += a.size
        i += 1
    return np.concatenate(parts)[:need]


def fit_linear(lengths: list[float], ms: list[float], max_len: float | None = None) -> dict:
    x = np.array(lengths, dtype=float)
    y = np.array(ms, dtype=float)
    if max_len is not None:
        keep = x <= max_len
        x, y = x[keep], y[keep]
    if x.size < 3:
        return {"n": int(x.size)}
    slope, icpt = np.polyfit(x, y, 1)
    return {"n": int(x.size), "intercept_ms": float(icpt), "slope_ms_per_audio_s": float(slope),
            "r": float(np.corrcoef(x, y)[0, 1]) if np.ptp(y) > 0 else None, "max_length_s": float(x.max())}


def profile(compute_type: str, device: str, model_name: str, utts, lengths, repeats: int) -> dict:
    from echorelay.inference.whisper import WhisperTranscriber

    model = WhisperTranscriber(model_name, device, compute_type)
    model.load()
    model.transcribe(build_signal(utts, 5.0), final=True)  # warm-up, excluded
    rows = []
    for length in lengths:
        audio = build_signal(utts, length)
        row = {"length_s": length}
        for final in (False, True):
            times, words = [], 0
            for _ in range(repeats):
                t0 = time.monotonic()
                text = model.transcribe(audio, final=final).text
                times.append(time.monotonic() - t0)
                words = len(text.split())
            key = "final" if final else "partial"
            row[key] = {"median_ms": float(np.median(times) * 1000), "min_ms": float(min(times) * 1000),
                        "max_ms": float(max(times) * 1000), "rtf": float(np.median(times) / length),
                        "words": words}
        rows.append(row)
        print(f"  {length:6.1f}s  partial {row['partial']['median_ms']:7.1f} ms   "
              f"final {row['final']['median_ms']:7.1f} ms", flush=True)
    result = {
        "compute_type": compute_type,
        "precision": precision_label(compute_type),
        "decode_options": {"partial": model.decode_options(final=False),
                           "final": model.decode_options(final=True)},
        "rows": rows,
        "fit_partial_le_30s": fit_linear([r["length_s"] for r in rows],
                                         [r["partial"]["median_ms"] for r in rows], 30.0),
        "fit_final_le_30s": fit_linear([r["length_s"] for r in rows],
                                       [r["final"]["median_ms"] for r in rows], 30.0),
    }
    del model
    gc.collect()
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--model", default="base.en")
    p.add_argument("--lengths", type=float, nargs="+",
                   default=[1, 2, 4, 8, 12, 16, 20, 25, 30, 40, 60])
    p.add_argument("--compute-types", nargs="+", default=["float16", "int8_float16"])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--output", required=True)
    a = p.parse_args()

    n_gpu = cuda_devices()
    if n_gpu == 0 and not a.allow_cpu:
        sys.exit("no CUDA device visible to CTranslate2; refusing to produce GPU results")
    device = "cuda" if n_gpu else "cpu"
    utts = load_audio_dir(a.audio_dir)
    if sum(u.duration_s for u in utts) < max(a.lengths) / 4:
        print("note: evaluation audio is short; long signals repeat the same files", flush=True)

    runs = []
    for ct in a.compute_types:
        print(f"profiling {ct} [{precision_label(ct)}] on {device}", flush=True)
        runs.append(profile(ct, device, a.model, utts, a.lengths, a.repeats))
    write_json(a.output, {
        "kind": "decode_profile",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "valid_for_gpu": device == "cuda",
        "model": a.model, "repeats": a.repeats,
        "audio_source": {"dir_files": [u.name for u in utts], "sha256": [u.sha256 for u in utts]},
        "environment": environment(),
        "runs": runs,
    })
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
