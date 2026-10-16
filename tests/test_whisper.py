"""Real-model smoke test. Deselected by default; run with ``pytest -m whisper``.

Requires network access to Hugging Face (or a pre-populated
``ECHORELAY_MODEL_DIR``) to load the ``base.en`` weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from echorelay.inference.whisper import WhisperTranscriber

pytestmark = pytest.mark.whisper


def test_whisper_base_en_loads_and_decodes():
    import os

    model = WhisperTranscriber("base.en", "cpu", "int8", model_dir=os.getenv("ECHORELAY_MODEL_DIR"))
    model.load()
    silence = np.zeros(16000 * 2, dtype=np.float32)
    for final in (False, True):
        result = model.transcribe(silence, final=final)
        assert isinstance(result.text, str)
    assert model.transcribe(np.zeros(0, dtype=np.float32), final=True).text == ""
