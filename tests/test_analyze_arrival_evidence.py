"""The Phase 4 analysis fails explicitly when required evidence is missing,
instead of silently producing results without end states, blocks or WER."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks import analyze_arrival as aa

ROOT = Path(__file__).resolve().parents[1]
LINE = ("2026-01-01 00:00:00,000 INFO echorelay.transport: stream_summary session={sid} "
        "outcome=completed partial_state_at_end=done partial_promoted_at_end=False "
        "partial_cancelled_at_end=False inference_waiting_at_end=0")


def make_root(tmp_path: Path, *, log: bool = True, summarized: bool = True, blocks=aa.BLOCKS) -> Path:
    root = tmp_path / "results"
    run = root / "01_r1_sync_base"
    run.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps(
        {"order": [{"dir": run.name, "arrival": "sync", "policy": "base", "round": 1}]}))
    sids = []
    for name in blocks:
        sid = f"{name}x"
        sids.append(sid)
        (run / f"{name}.json").write_text(json.dumps(
            {"streams": [{"ready": {"session_id": sid}, "utterance": "u1"}]}))
    if log:
        (run / "server.log").write_text(
            "\n".join(LINE.format(sid=s) for s in (sids if summarized else sids[1:])) + "\n")
    return root


def test_complete_evidence_loads_with_end_states(tmp_path):
    _, blocks = aa.load(make_root(tmp_path))
    assert len(blocks) == len(aa.BLOCKS)
    assert all(s["_end"]["state"] == "done" for b in blocks for s in b["streams"])


def test_missing_server_log_fails(tmp_path):
    with pytest.raises(aa.MissingEvidence, match="server log not found"):
        aa.load(make_root(tmp_path, log=False))


def test_stream_without_summary_line_fails(tmp_path):
    with pytest.raises(aa.MissingEvidence, match="no stream_summary for 1 session"):
        aa.load(make_root(tmp_path, summarized=False))


def test_stream_that_never_got_a_session_needs_no_summary(tmp_path):
    root = make_root(tmp_path)
    f = root / "01_r1_sync_base" / "c8.json"
    blk = json.loads(f.read_text())
    blk["streams"].append({"ready": None, "utterance": "u1", "error": "server_busy"})
    f.write_text(json.dumps(blk))
    _, blocks = aa.load(root)
    assert "_end" not in blocks[0]["streams"][-1]


def test_missing_block_file_fails(tmp_path):
    with pytest.raises(aa.MissingEvidence, match="block result not found"):
        aa.load(make_root(tmp_path, blocks=aa.BLOCKS[:3]))


def test_references_must_exist_and_cover_every_utterance(tmp_path):
    root = make_root(tmp_path)
    with pytest.raises(aa.MissingEvidence, match="--audio-dir not found"):
        aa.load_references(tmp_path / "nope", root)
    refs = tmp_path / "refs"
    refs.mkdir()
    with pytest.raises(aa.MissingEvidence, match=r"no reference transcript .* \['u1'\]"):
        aa.load_references(refs, root)
    (refs / "u1.txt").write_text("hello world\n")
    assert aa.load_references(refs, root) == {"u1": "hello world"}


def test_cli_exits_nonzero_on_missing_evidence(tmp_path):
    root = make_root(tmp_path, log=False)
    r = subprocess.run([sys.executable, "-m", "benchmarks.analyze_arrival", str(root)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0
    assert "incomplete evidence: server log not found" in r.stderr
