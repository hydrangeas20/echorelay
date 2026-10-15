"""faster-whisper adapter."""

from __future__ import annotations

import logging
import threading

import numpy as np

from .base import InferenceError, ModelInfo, TranscriptionResult

log = logging.getLogger(__name__)


class WhisperTranscriber:
    def __init__(
        self,
        model_name: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
        *,
        num_workers: int = 1,
        cpu_threads: int = 0,
        model_dir: str | None = None,
        language: str = "en",
        partial_beam_size: int = 1,
        final_beam_size: int = 5,
    ) -> None:
        self.info = ModelInfo(
            backend="whisper",
            model=model_name,
            device=device,
            compute_type=compute_type,
        )
        self._num_workers = num_workers
        self._cpu_threads = cpu_threads
        self._model_dir = model_dir
        self._language = language
        self._partial_beam = partial_beam_size
        self._final_beam = final_beam_size
        self._model = None
        self._load_lock = threading.Lock()

    def load(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            from faster_whisper import WhisperModel  # heavy import, deferred

            log.info(
                "loading faster-whisper model=%s device=%s compute_type=%s workers=%d",
                self.info.model, self.info.device, self.info.compute_type, self._num_workers,
            )
            self._model = WhisperModel(
                self.info.model,
                device=self.info.device,
                compute_type=self.info.compute_type,
                num_workers=self._num_workers,
                cpu_threads=self._cpu_threads,
                download_root=self._model_dir,
            )

    def decode_options(self, *, final: bool) -> dict:
        """Decoding settings. Exposed so benchmarks can reuse them verbatim."""
        return dict(
            language=self._language,
            beam_size=self._final_beam if final else self._partial_beam,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
        )

    def transcribe(self, audio: np.ndarray, *, final: bool) -> TranscriptionResult:
        if self._model is None:
            self.load()
        if audio.size == 0:
            return TranscriptionResult(text="")
        try:
            segments, _info = self._model.transcribe(audio, **self.decode_options(final=final))
            # ``segments`` is a lazy generator; decoding happens while iterating.
            text = " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as exc:  # pragma: no cover - depends on model runtime
            raise InferenceError(f"faster-whisper decode failed: {exc}") from exc
        return TranscriptionResult(text=text)
