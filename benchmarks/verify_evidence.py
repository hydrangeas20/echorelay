"""Verify committed benchmark evidence against its SHA-256 manifests.

For every ``docs/benchmarks/**/SHA256SUMS`` under the repository root:

* every listed file must exist and match its SHA-256;
* in a git checkout, every listed file must also be tracked by git. A file
  that is present only as an ignored local copy would pass a plain
  ``sha256sum -c`` but is missing from every fresh clone;
* every tracked file in the manifest's directory must be listed.

Exit status 1 on any problem. Run from a fresh clone to check what was
actually committed:

    python -m benchmarks.verify_evidence
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _entries(manifest: Path) -> list[tuple[str, str]]:
    out = []
    for line in manifest.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out.append((digest, name.lstrip("*").removeprefix("./")))
    return out


def _tracked(root: Path) -> set[str] | None:
    """Repository-relative paths tracked by git, or None outside a git checkout."""
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root,
                             capture_output=True, text=True, check=True).stdout.strip()
        if Path(top).resolve() != root.resolve():
            return None
        out = subprocess.run(["git", "ls-files", "-z"], cwd=root,
                             capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {p.decode() for p in out.split(b"\0") if p}


def verify(root: Path) -> tuple[int, int, list[str]]:
    """Returns (manifests checked, files checked, problems)."""
    root = root.resolve()
    tracked = _tracked(root)
    manifests = sorted((root / "docs" / "benchmarks").glob("**/SHA256SUMS"))
    problems: list[str] = []
    files = 0
    for m in manifests:
        base = m.parent
        rel_base = base.relative_to(root).as_posix()
        listed = set()
        for digest, name in _entries(m):
            files += 1
            path = base / name
            rel = (path.relative_to(root)).as_posix()
            listed.add(rel)
            if not path.is_file():
                problems.append(f"missing: {rel}")
                continue
            if _sha256(path) != digest:
                problems.append(f"checksum mismatch: {rel}")
            if tracked is not None and rel not in tracked:
                problems.append(f"not tracked by git (present only locally): {rel}")
        if tracked is not None:
            for rel in sorted(t for t in tracked if t.startswith(rel_base + "/")):
                if rel != f"{rel_base}/SHA256SUMS" and rel not in listed:
                    problems.append(f"tracked but not in {rel_base}/SHA256SUMS: {rel}")
    return len(manifests), files, problems


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path)
    a = p.parse_args()
    n_manifests, n_files, problems = verify(a.root)
    for line in problems:
        print(line)
    status = "FAIL" if problems or not n_manifests else "OK"
    print(f"{status}: {n_manifests} manifests, {n_files} files, {len(problems)} problems")
    sys.exit(1 if status == "FAIL" else 0)


if __name__ == "__main__":
    main()
