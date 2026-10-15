"""Runtime configuration.

All settings can be overridden with environment variables prefixed with
``ECHORELAY_`` (for example ``ECHORELAY_BACKEND=fake``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Wire-format constants. These are part of the protocol, not tunables.
SAMPLE_RATE_HZ = 16_000
SAMPLE_WIDTH_BYTES = 2  # signed 16-bit little-endian PCM
CHANNELS = 1
ENCODING = "pcm_s16le"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ECHORELAY_", extra="ignore")

    # --- inference backend -------------------------------------------------
    backend: Literal["whisper", "fake"] = "whisper"
    model_name: str = "base.en"
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    # CTranslate2 compute type: int8 (CPU default), float16 / int8_float16 (GPU).
    compute_type: str = "int8"
    cpu_threads: int = Field(default=0, ge=0)  # 0 = CTranslate2 default
    model_dir: str | None = None  # download/cache directory for model weights
    language: str = "en"
    partial_beam_size: int = Field(default=1, ge=1)  # greedy for fast provisional decodes
    final_beam_size: int = Field(default=5, ge=1)

    # --- resource limits ---------------------------------------------------
    max_concurrent_streams: int = Field(default=8, ge=1)
    # Number of inference jobs that may run at once across all streams.
    # Also used as the CTranslate2 ``num_workers`` value.
    max_concurrent_inference: int = Field(default=1, ge=1)
    max_audio_seconds: float = Field(default=120.0, gt=0)
    max_frame_bytes: int = Field(default=256 * 1024, gt=0)
    idle_timeout_s: float = Field(default=15.0, gt=0)
    inference_timeout_s: float = Field(default=30.0, gt=0)

    # --- partial hypotheses ------------------------------------------------
    partial_interval_s: float = Field(default=1.0, gt=0)
    min_partial_audio_s: float = Field(default=0.5, ge=0)

    # --- observability -----------------------------------------------------
    # Add a "timing" object (queue wait, decode, wait-for-partial) to partial
    # and final events. Off by default so the Phase 1 wire format is unchanged.
    event_timing: bool = False
    metrics_window: int = Field(default=10_000, ge=100)  # observations kept per distribution

    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check(self) -> "Settings":
        if self.max_frame_bytes % SAMPLE_WIDTH_BYTES:
            raise ValueError("max_frame_bytes must be a multiple of the sample width (2)")
        return self

    @property
    def max_audio_bytes(self) -> int:
        return int(self.max_audio_seconds * SAMPLE_RATE_HZ) * SAMPLE_WIDTH_BYTES


@lru_cache
def get_settings() -> Settings:
    return Settings()
