"""Committed benchmark evidence matches its SHA-256 manifests, and every
listed file is actually tracked by git (an ignored local copy is not
evidence: a fresh clone would not have it)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from benchmarks.verify_evidence import verify

ROOT = Path(__file__).resolve().parents[1]


def test_committed_evidence_matches_manifests():
    manifests, files, problems = verify(ROOT)
    assert manifests >= 1 and files > 0
    assert problems == [], "\n".join(problems[:25])


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_verifier_flags_missing_mismatched_untracked_and_unlisted(tmp_path):
    raw = tmp_path / "docs" / "benchmarks" / "x" / "raw"
    raw.mkdir(parents=True)
    (raw / "a.json").write_text("a")
    (raw / "b.json").write_text("b")
    (raw / "c.log").write_text("c")  # will stay untracked, like an ignored log
    (raw / "extra.json").write_text("e")  # tracked but not listed
    lines = [
        "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb  ./a.json",  # sha256("a")
        "0" * 64 + "  ./b.json",
        "2e7d2c03a9507ae265ecf5b5356885a53393a2029d241394997265a1a25aefc6  ./c.log",  # sha256("c")
        "0" * 64 + "  ./gone.json",
    ]
    (raw / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(git + ["init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(git + ["add", "docs/benchmarks/x/raw/SHA256SUMS", "docs/benchmarks/x/raw/a.json",
                          "docs/benchmarks/x/raw/b.json", "docs/benchmarks/x/raw/extra.json"],
                   cwd=tmp_path, check=True)
    _, files, problems = verify(tmp_path)
    assert files == 4
    assert sorted(problems) == sorted([
        "checksum mismatch: docs/benchmarks/x/raw/b.json",
        "not tracked by git (present only locally): docs/benchmarks/x/raw/c.log",
        "missing: docs/benchmarks/x/raw/gone.json",
        "tracked but not in docs/benchmarks/x/raw/SHA256SUMS: docs/benchmarks/x/raw/extra.json",
    ])
