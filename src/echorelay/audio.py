"""Audio buffering for a single stream.

Frames are raw 16 kHz mono signed 16-bit little-endian PCM. The buffer
enforces a total size limit and converts to the float32 [-1, 1] array that
faster-whisper expects.
"""

from __future__ import annotations

import numpy as np

from .config import SAMPLE_RATE_HZ, SAMPLE_WIDTH_BYTES


class AudioLimitExceeded(Exception):
    pass


class InvalidAudioFrame(Exception):
    pass


class AudioBuffer:
    def __init__(self, max_bytes: int) -> None:
        self._buf = bytearray()
        self._max_bytes = max_bytes

    def append(self, frame: bytes) -> None:
        if len(frame) % SAMPLE_WIDTH_BYTES:
            raise InvalidAudioFrame(
                f"frame length {len(frame)} is not a multiple of {SAMPLE_WIDTH_BYTES} bytes"
            )
        if len(self._buf) + len(frame) > self._max_bytes:
            raise AudioLimitExceeded(
                f"stream audio would exceed {self._max_bytes} bytes "
                f"({self._max_bytes / SAMPLE_WIDTH_BYTES / SAMPLE_RATE_HZ:.1f} s)"
            )
        self._buf.extend(frame)

    @property
    def num_bytes(self) -> int:
        return len(self._buf)

    @property
    def num_samples(self) -> int:
        return len(self._buf) // SAMPLE_WIDTH_BYTES

    @property
    def duration_s(self) -> float:
        return self.num_samples / SAMPLE_RATE_HZ

    def snapshot(self) -> np.ndarray:
        """Return a float32 copy of all buffered audio.

        ``bytes(...)`` copies, so the event loop may keep appending while a
        worker thread decodes the snapshot.
        """
        pcm = np.frombuffer(bytes(self._buf), dtype="<i2")
        return pcm.astype(np.float32) / 32768.0

    def clear(self) -> None:
        self._buf = bytearray()
