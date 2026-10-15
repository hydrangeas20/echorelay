"""Shared benchmark utilities: evaluation audio, environment capture, stats, WER.

Evaluation audio layout (``--audio-dir``)::

    utt1.wav   16 kHz, mono, 16-bit PCM (required format; no resampling here)
    utt1.txt   reference transcript (optional; enables WER)

``benchmarks/prepare_audio.py`` creates such a directory from Mini LibriSpeech.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Utterance:
    name: str
    pcm16: bytes
    reference: str | None
    sha256: str

    @property
    def duration_s(self) -> float:
        return len(self.pcm16) / 2 / SAMPLE_RATE

    def float32(self) -> np.ndarray:
        return np.frombuffer(self.pcm16, dtype="<i2").astype(np.float32) / 32768.0


def load_audio_dir(path: str | Path) -> list[Utterance]:
    root = Path(path)
    wavs = sorted(root.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"no .wav files in {root}")
    out = []
    for wav in wavs:
        with wave.open(str(wav), "rb") as wf:
            fmt = (wf.getframerate(), wf.getnchannels(), wf.getsampwidth())
            if fmt != (SAMPLE_RATE, 1, 2):
                raise SystemExit(
                    f"{wav}: expected 16000 Hz mono 16-bit, got {fmt}. "
                    "Convert with ffmpeg -ar 16000 -ac 1 -sample_fmt s16 before benchmarking."
                )
            pcm = wf.readframes(wf.getnframes())
        ref_path = wav.with_suffix(".txt")
        ref = ref_path.read_text().strip() if ref_path.exists() else None
        out.append(Utterance(wav.stem, pcm, ref, hashlib.sha256(pcm).hexdigest()))
    return out


def concatenate(utts: list[Utterance], target_s: float) -> list[Utterance]:
    """Build long utterances (>= target_s) by joining whole evaluation files.

    Deterministic: utterance i starts at file i (sorted order) and appends
    the following files cyclically until it reaches ``target_s``. Whole files
    are joined (no truncation) so the concatenated reference stays exact.
    """
    out = []
    n = len(utts)
    for i in range(n):
        parts, dur, j = [], 0.0, i
        while dur < target_s:
            parts.append(utts[j % n])
            dur += utts[j % n].duration_s
            j += 1
        pcm = b"".join(u.pcm16 for u in parts)
        refs = [u.reference for u in parts]
        ref = " ".join(refs) if all(r is not None for r in refs) else None
        out.append(Utterance(f"concat{target_s:g}s_{i:02d}", pcm, ref, hashlib.sha256(pcm).hexdigest()))
    return out


def audio_manifest(utts: list[Utterance]) -> dict:
    return {
        "num_files": len(utts),
        "total_audio_s": round(sum(u.duration_s for u in utts), 3),
        "files": [
            {"name": u.name, "duration_s": round(u.duration_s, 3), "sha256": u.sha256,
             "has_reference": u.reference is not None}
            for u in utts
        ],
    }


def _pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment() -> dict:
    env: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "packages": {p: _pkg_version(p) for p in
                     ("faster-whisper", "ctranslate2", "numpy", "fastapi", "uvicorn", "websockets", "jiwer")},
        "gpus": [],
    }
    if shutil.which("nvidia-smi"):
        try:
            q = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10, check=True,
            )
            env["gpus"] = [line.strip() for line in q.stdout.splitlines() if line.strip()]
        except (subprocess.SubprocessError, OSError) as exc:
            env["gpus_error"] = str(exc)
    try:
        import ctranslate2

        env["ctranslate2_cuda_devices"] = ctranslate2.get_cuda_device_count()
    except Exception:  # pragma: no cover
        pass
    return env


def percentiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "p50": None, "p95": None, "mean": None, "min": None, "max": None}
    a = np.asarray(values, dtype=float)
    return {
        "n": int(a.size),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "mean": float(a.mean()),
        "min": float(a.min()),
        "max": float(a.max()),
    }


_PUNCT = re.compile(r"[^\w\s']")
_SPACE = re.compile(r"\s+")
_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})

DEFAULT_NORMALIZER = "v2"
NORMALIZERS = ("v1", "v2")


def normalize_text(text: str, version: str = DEFAULT_NORMALIZER) -> str:
    """WER normalization used by every benchmark in this repo.

    v1 (Phase 1): lowercase; hyphens -> spaces; drop punctuation except
    apostrophes; collapse whitespace. Bug: an apostrophe used as a quotation
    mark stays attached to the word ("'he", "good'") and scores as an error.

    v2 (default): as v1, plus typographic quotes mapped to ASCII and
    apostrophes stripped from the start and end of each word. Word-internal
    apostrophes are kept ("it's", "o'er"). A trailing possessive apostrophe
    ("students'") is stripped on both reference and hypothesis.

    Deliberately simple and fixed so every configuration is scored
    identically. No number or spelling normalization.
    """
    if version not in NORMALIZERS:
        raise ValueError(f"unknown normalizer {version!r}; expected one of {NORMALIZERS}")
    if version == "v2":
        text = text.translate(_QUOTES)
    text = text.lower().replace("-", " ")
    text = _PUNCT.sub("", text)
    if version == "v2":
        text = " ".join(w.strip("'") for w in text.split())
    return _SPACE.sub(" ", text).strip()


def corpus_wer(
    references: list[str], hypotheses: list[str], normalizer: str = DEFAULT_NORMALIZER
) -> float | None:
    """Corpus-level WER (total edits / total reference words) after normalization."""
    pairs = [(normalize_text(r, normalizer), normalize_text(h, normalizer))
             for r, h in zip(references, hypotheses)]
    pairs = [(r, h) for r, h in pairs if r]
    if not pairs:
        return None
    import jiwer

    return float(jiwer.wer([r for r, _ in pairs], [h for _, h in pairs]))


def write_json(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2) + "\n")
