from __future__ import annotations

from ..config import Settings
from .base import InferenceError, ModelInfo, Transcriber, TranscriptionResult
from .fake import FakeTranscriber
from .runner import InferenceRunner, InferenceTimeout, TimedResult
from .scheduler import Job, SlotScheduler


def build_transcriber(settings: Settings) -> Transcriber:
    if settings.backend == "fake":
        return FakeTranscriber(delay_s=settings.fake_delay_s)
    from .whisper import WhisperTranscriber

    return WhisperTranscriber(
        settings.model_name,
        settings.device,
        settings.compute_type,
        num_workers=settings.max_concurrent_inference,
        cpu_threads=settings.cpu_threads,
        model_dir=settings.model_dir,
        language=settings.language,
        partial_beam_size=settings.partial_beam_size,
        final_beam_size=settings.final_beam_size,
    )


__all__ = [
    "FakeTranscriber",
    "InferenceError",
    "InferenceRunner",
    "InferenceTimeout",
    "Job",
    "SlotScheduler",
    "ModelInfo",
    "TimedResult",
    "Transcriber",
    "TranscriptionResult",
    "build_transcriber",
]
