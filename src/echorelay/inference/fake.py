"""Deterministic fake adapter for tests and model-free demos.

Emits one word per ``seconds_per_word`` of audio (``w1 w2 w3 ...``) so tests
can assert on transcripts without a model. Can inject latency and failures.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from ..config import SAMPLE_RATE_HZ
from .base import InferenceError, ModelInfo, TranscriptionResult


class FakeTranscriber:
    def __init__(
        self,
        *,
        seconds_per_word: float = 0.5,
        delay_s: float = 0.0,
        fail_on_call: int | None = None,
        fail_on_final: bool = False,
    ) -> None:
        self.info = ModelInfo(backend="fake", model="fake", device="cpu", compute_type="none")
        self.seconds_per_word = seconds_per_word
        self.delay_s = delay_s
        self.fail_on_call = fail_on_call  # 1-based call index that raises
        self.fail_on_final = fail_on_final
        self.calls: list[tuple[int, bool]] = []  # (num_samples, final)
        self.loaded = False
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def load(self) -> None:
        self.loaded = True

    def transcribe(self, audio: np.ndarray, *, final: bool) -> TranscriptionResult:
        with self._lock:
            self.calls.append((int(audio.size), final))
            call_no = len(self.calls)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay_s:
                time.sleep(self.delay_s)
            if (self.fail_on_call is not None and call_no == self.fail_on_call) or (
                final and self.fail_on_final
            ):
                raise InferenceError(f"injected failure on call {call_no}")
            n_words = int((audio.size / SAMPLE_RATE_HZ) / self.seconds_per_word)
            return TranscriptionResult(text=" ".join(f"w{i + 1}" for i in range(n_words)))
        finally:
            with self._lock:
                self.active -= 1
