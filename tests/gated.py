"""A transcriber whose decodes block until the test releases them, so tests
can assert exact execution order without depending on timing."""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np

from echorelay.inference.base import ModelInfo, TranscriptionResult


def label(audio: np.ndarray, final: bool) -> str:
    """Decodes are identified by kind + sample count: P16000, F16000, ..."""
    return f"{'F' if final else 'P'}{audio.size}"


class GatedTranscriber:
    def __init__(self) -> None:
        self.info = ModelInfo(backend="gated", model="gated", device="cpu", compute_type="none")
        self._lock = threading.Lock()
        self.started: list[str] = []
        self._gates: list[threading.Event] = []
        self._released = 0
        self.fail: set[str] = set()  # decode names that raise when released

    def load(self) -> None:
        pass

    def transcribe(self, audio: np.ndarray, *, final: bool) -> TranscriptionResult:
        gate = threading.Event()
        name = label(audio, final)
        with self._lock:
            self.started.append(name)
            self._gates.append(gate)
        if not gate.wait(timeout=20):
            raise RuntimeError(f"test never released decode {name}")
        if name in self.fail:
            raise RuntimeError(f"injected failure in {name}")
        return TranscriptionResult(text=name)

    def release_next(self) -> None:
        """Let the oldest running decode finish."""
        with self._lock:
            gate = self._gates[self._released]
            self._released += 1
        gate.set()

    def release_all(self) -> None:
        with self._lock:
            pending = self._gates[self._released:]
            self._released = len(self._gates)
        for g in pending:
            g.set()


async def until(predicate, timeout: float = 5.0, what: str = "condition") -> None:
    """Wait for state, not time: polls a predicate; fails loudly on timeout."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.001)
