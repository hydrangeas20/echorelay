#!/usr/bin/env python3
"""Recompute WER for saved result files without re-running any model.

    python -m benchmarks.rescore --audio-dir eval_audio results/stream_fp16_c4.json \\
        results/stream_int8_c4.json results/precision_t4.json

Reads the transcripts stored in load_test results (``streams[].final_text``)
or compare_precision results (``runs[].hypotheses``), loads references from
``<audio-dir>/<utterance>.txt``, and reports corpus WER under every
normalizer version, plus the hypothesis words that differ between versions.
Use it to correct Phase 1 WER, which was scored with the v1 normalizer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import NORMALIZERS, corpus_wer, normalize_text


def load_references(audio_dir: str | Path) -> dict[str, str]:
    return {p.stem: p.read_text().strip() for p in sorted(Path(audio_dir).glob("*.txt"))}


def transcripts(result: dict) -> dict[str, list[tuple[str, str]]]:
    """Map label -> [(utterance, hypothesis), ...] for either result format."""
    if "streams" in result:
        label = result.get("label") or result["server"]["health"]["model"]["compute_type"]
        return {label: [(s["utterance"], s["final_text"] or "") for s in result["streams"] if s["ok"]]}
    if "runs" in result:
        return {r["compute_type"]: sorted(r["hypotheses"].items()) for r in result["runs"]}
    raise ValueError("unrecognised result file (expected 'streams' or 'runs')")


def rescore(result: dict, refs: dict[str, str]) -> dict:
    out = {}
    for label, pairs in transcripts(result).items():
        pairs = [(u, h) for u, h in pairs if u in refs]
        missing = sorted({u for u, _ in transcripts(result)[label]} - set(refs))
        entry = {
            "num_scored": len(pairs),
            "num_reference_words": sum(len(normalize_text(refs[u]).split()) for u, _ in pairs),
            "missing_references": missing,
        }
        for v in NORMALIZERS:
            entry[f"wer_{v}"] = corpus_wer([refs[u] for u, _ in pairs], [h for _, h in pairs], v)
        changed = sorted({
            (u, normalize_text(h, "v1"), normalize_text(h, "v2"))
            for u, h in pairs if normalize_text(h, "v1") != normalize_text(h, "v2")
        })
        entry["hypotheses_changed_by_v2"] = [
            {"utterance": u, "v1": a, "v2": b} for u, a, b in changed
        ]
        out[label] = entry
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+")
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--output", default=None)
    a = p.parse_args()

    refs = load_references(a.audio_dir)
    report = {}
    for path in a.results:
        report[path] = rescore(json.loads(Path(path).read_text()), refs)
        for label, e in report[path].items():
            print(f"{path} [{label}] n={e['num_scored']} words={e['num_reference_words']} "
                  + " ".join(f"WER_{v}={e[f'wer_{v}']:.4f}" for v in NORMALIZERS if e[f"wer_{v}"] is not None))
            if e["missing_references"]:
                print(f"   missing references: {e['missing_references']}")
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
