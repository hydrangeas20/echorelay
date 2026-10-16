"""Inference adapter interface.

Adapters are synchronous and are always called from a worker thread by
:class:`echorelay.inference.runner.InferenceRunner`, never on the event loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class TranscriptionResult:
    text: str


@dataclass(frozen=True)
class ModelInfo:
    backend: str
    model: str
    device: str
    compute_type: str
    extra: dict[str, str] = field(default_factory=dict)


class InferenceError(Exception):
    """Raised by adapters when decoding fails."""


@runtime_checkable
class Transcriber(Protocol):
    info: ModelInfo

    def load(self) -> None:
        """Load weights / warm up. Called once, off the event loop."""

    def transcribe(self, audio: np.ndarray, *, final: bool) -> TranscriptionResult:
        """Decode a float32 mono 16 kHz array.

        ``final`` selects final-quality decoding settings; partial decodes may
        use cheaper settings.
        """
