#!/usr/bin/env python3
"""Smallest real-model check before any benchmark: does base.en load and
transcribe ONE utterance on this device, with each compute type?

    python -m benchmarks.gpu_smoke --audio-dir eval_audio            # CUDA, fp16 + int8_float16
    python -m benchmarks.gpu_smoke --audio-dir eval_audio --device cpu --compute-types int8

This is a go/no-go check, not a measurement: it decodes a single file once
per compute type (after one warm-up), so its timings are reported only as a
sanity signal and are labelled ``kind: smoke`` in the output JSON. Exits
non-zero if CUDA is missing (for --device cuda), the model fails to load or
decode, or a reference exists and the transcript is empty.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime, timezone

from .common import corpus_wer, environment, load_audio_dir, write_json
from .compare_precision import cuda_devices, precision_label


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio-dir", required=True, help="uses the first .wav (sorted)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--compute-types", nargs="+", default=None,
                   help="default: float16 int8_float16 on cuda, int8 on cpu")
    p.add_argument("--model", default="base.en")
    p.add_argument("--output", default="results/gpu_smoke.json")
    a = p.parse_args()

    env = environment()
    print(f"GPUs: {env['gpus'] or 'none'}; CTranslate2 CUDA devices: {cuda_devices()}")
    if a.device == "cuda" and cuda_devices() == 0:
        sys.exit("FAIL: no CUDA device visible to CTranslate2 (check the runtime type / CUDA libs)")
    compute_types = a.compute_types or (["float16", "int8_float16"] if a.device == "cuda" else ["int8"])

    utt = load_audio_dir(a.audio_dir)[0]
    print(f"audio: {utt.name} ({utt.duration_s:.2f}s) reference={'yes' if utt.reference else 'no'}")

    from echorelay.inference.whisper import WhisperTranscriber

    checks, ok = [], True
    for ct in compute_types:
        check = {"compute_type": ct, "precision": precision_label(ct), "ok": False}
        try:
            model = WhisperTranscriber(a.model, a.device, ct)
            t0 = time.monotonic()
            model.load()
            check["model_load_s"] = time.monotonic() - t0
            audio = utt.float32()
            model.transcribe(audio, final=True)  # warm-up
            for final in (False, True):
                t0 = time.monotonic()
                text = model.transcribe(audio, final=final).text
                key = "final" if final else "partial_settings"
                check[f"{key}_text"] = text
                check[f"{key}_decode_s"] = time.monotonic() - t0
            if utt.reference:
                check["wer"] = corpus_wer([utt.reference], [check["final_text"]])
            check["ok"] = bool(check["final_text"].strip()) or not utt.reference
            del model
        except Exception as exc:
            check["error"] = f"{type(exc).__name__}: {exc}"
            check["traceback"] = traceback.format_exc()
        ok &= check["ok"]
        status = "PASS" if check["ok"] else "FAIL"
        print(f"[{status}] {ct} ({check['precision']})")
        if "error" in check:
            print(f"    {check['error']}")
        else:
            print(f"    load {check['model_load_s']:.1f}s, final decode {check['final_decode_s']:.3f}s "
                  f"(single run, not a benchmark), WER={check.get('wer')}")
            print(f"    hyp: {check['final_text']}")
            if utt.reference:
                print(f"    ref: {utt.reference}")
        checks.append(check)

    write_json(a.output, {
        "kind": "smoke",
        "note": "single-utterance go/no-go check; timings are not benchmark measurements",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": a.device, "model": a.model,
        "audio": {"name": utt.name, "duration_s": utt.duration_s, "sha256": utt.sha256},
        "environment": env,
        "checks": checks,
    })
    print(f"wrote {a.output}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
