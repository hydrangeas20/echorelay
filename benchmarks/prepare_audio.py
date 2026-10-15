#!/usr/bin/env python3
"""Build a fixed evaluation set from Mini LibriSpeech dev-clean-2 (OpenSLR 31).

    pip install soundfile
    python -m benchmarks.prepare_audio --out eval_audio --num 50

Selects the first ``--num`` utterances (sorted by utterance id) whose
duration lies in [--min-s, --max-s], writes ``<id>.wav`` (16 kHz mono
PCM16) and ``<id>.txt`` (reference). Selection is deterministic, so every
run and every precision is evaluated on the same files.
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
import wave
from pathlib import Path

URL = "https://www.openslr.org/resources/31/dev-clean-2.tar.gz"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="eval_audio")
    p.add_argument("--num", type=int, default=50)
    p.add_argument("--min-s", type=float, default=3.0)
    p.add_argument("--max-s", type=float, default=20.0)
    p.add_argument("--cache", default=".cache/dev-clean-2.tar.gz")
    a = p.parse_args()

    import numpy as np
    import soundfile as sf

    cache = Path(a.cache)
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, cache)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(cache) as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        refs: dict[str, str] = {}
        for name, m in members.items():
            if name.endswith(".trans.txt"):
                for line in tar.extractfile(m).read().decode().splitlines():
                    utt_id, text = line.split(" ", 1)
                    refs[utt_id] = text
        flacs = {Path(n).stem: m for n, m in members.items() if n.endswith(".flac")}
        written = 0
        for utt_id in sorted(flacs):
            if written >= a.num:
                break
            data, sr = sf.read(tar.extractfile(flacs[utt_id]), dtype="int16")
            assert sr == 16000 and data.ndim == 1, (utt_id, sr, data.shape)
            dur = len(data) / sr
            if not a.min_s <= dur <= a.max_s or utt_id not in refs:
                continue
            with wave.open(str(out / f"{utt_id}.wav"), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(np.asarray(data, dtype="<i2").tobytes())
            (out / f"{utt_id}.txt").write_text(refs[utt_id] + "\n")
            written += 1
    print(f"wrote {written} utterances to {out}")


if __name__ == "__main__":
    main()
