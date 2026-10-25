"""Real-model verification. Deselected by default; run with ``pytest -m whisper``.

Three levels, each claiming only what it exercises:

* adapter   ``WhisperTranscriber`` on its own: load, device and compute type
            actually in effect, decoding real speech, error wrapping.
* server    the FastAPI app building and loading the model from ``Settings``
            and reporting it on ``/health``.
* websocket a real uvicorn server and a real WebSocket client streaming
            real speech: partials, one final, clean close, then shutdown.

Configuration (environment variables):

``ECHORELAY_TEST_DEVICE``        ``cpu`` (default) or ``cuda``. With ``cuda``,
                                 a missing GPU is a failure, not a skip.
``ECHORELAY_TEST_COMPUTE_TYPE``  default ``int8`` on CPU, ``float16`` on CUDA.
``ECHORELAY_TEST_AUDIO_DIR``     the 5-utterance set from
                                 ``python -m benchmarks.prepare_audio --num 5``.
                                 Speech tests skip without it (reported by -rs).
``ECHORELAY_MODEL_DIR``          optional weights cache.

The speech tests require exactly the five Mini LibriSpeech dev-clean-2 files
of the recorded Phase 1 precision run (checked by PCM SHA-256), so their WER
is comparable to the recorded T4 FP16 values (see the bounds below). The WER
bounds are regression guards, not measurements.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pytest

from echorelay.config import Settings
from echorelay.inference.base import InferenceError
from echorelay.inference.whisper import WhisperTranscriber

pytestmark = pytest.mark.whisper

REPO = Path(__file__).resolve().parents[1]
RECORDED = REPO / "docs/benchmarks/phase1-t4/raw/precision_t4.json"
MODEL = "base.en"
DEVICE = os.getenv("ECHORELAY_TEST_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("ECHORELAY_TEST_COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
MODEL_DIR = os.getenv("ECHORELAY_MODEL_DIR")
AUDIO_DIR = os.getenv("ECHORELAY_TEST_AUDIO_DIR")
# WER here is scored with the v2 normalizer (benchmarks.common default since
# Phase 2: typographic quotes mapped, edge punctuation stripped). Recorded T4
# FP16 corpus WER on these five files under v2 is 0.0531 (6 edits / 113 words:
# docs/benchmarks/scheduling-t4 and the real-model run in
# docs/benchmarks/whisper-t4). Phase 1 reported 0.0708 (8 / 113) for the SAME
# transcripts because it scored them with v1; the difference is scoring, not
# decoding. 0.12 allows several extra word errors for library/driver drift; a
# broken decode path (wrong sample rate, scaling, truncated audio) lands far
# above it.
MAX_CORPUS_WER = 0.12
# Recorded FP16 WER of the streamed utterance (1272-135031-0000) is 0.1154
# (3 edits / 26 words): v1 in docs/benchmarks/phase1-t4/raw/gpu_smoke.json and
# v2 in docs/benchmarks/whisper-t4 agree for this utterance.
MAX_STREAM_WER = 0.25


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def model() -> WhisperTranscriber:
    if DEVICE == "cuda":
        import ctranslate2

        assert ctranslate2.get_cuda_device_count() >= 1, (
            "ECHORELAY_TEST_DEVICE=cuda but CTranslate2 sees no CUDA device")
    m = WhisperTranscriber(MODEL, DEVICE, COMPUTE_TYPE, model_dir=MODEL_DIR)
    m.load()
    return m


@pytest.fixture(scope="module")
def utterances():
    if not AUDIO_DIR:
        pytest.skip("set ECHORELAY_TEST_AUDIO_DIR (python -m benchmarks.prepare_audio --num 5)")
    from benchmarks.common import load_audio_dir

    utts = load_audio_dir(AUDIO_DIR)
    recorded = json.loads(RECORDED.read_text())["audio"]["files"]
    assert [(u.name, u.sha256) for u in utts] == [(f["name"], f["sha256"]) for f in recorded], (
        "audio set differs from the recorded Phase 1 set; WER would not be comparable")
    assert all(u.reference for u in utts), "every utterance needs its .txt reference"
    return utts


def _wer(refs: list[str], hyps: list[str]) -> float:
    from benchmarks.common import corpus_wer

    return corpus_wer(refs, hyps)


# ------------------------------------------------------------- adapter level
def _class_name(obj) -> str:
    return f"{type(obj).__module__}.{type(obj).__qualname__}"


def _observed(model: WhisperTranscriber) -> dict[str, str]:
    """What the loaded runtime reports about itself, independent of the
    settings passed in. Recorded as JUnit properties for the verifier."""
    import hashlib

    import ctranslate2
    from faster_whisper.utils import download_model

    ct2 = model._model.model  # faster_whisper.WhisperModel -> ctranslate2.models.Whisper
    # Same resolution WhisperModel performed at load time (download_root=MODEL_DIR);
    # local_files_only, so this reads the cached snapshot and never downloads.
    path = Path(download_model(MODEL, cache_dir=MODEL_DIR, local_files_only=True))
    digest = hashlib.sha256()
    with open(path / "model.bin", "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return {
        "observed_whisper_class": _class_name(model._model),
        "observed_ct2_class": _class_name(ct2),
        "observed_ct2_device": ct2.device,
        "observed_ct2_compute_type": ct2.compute_type,
        "observed_ct2_is_multilingual": str(ct2.is_multilingual),
        "observed_cuda_devices": str(ctranslate2.get_cuda_device_count()),
        "observed_model_path": str(path),
        "observed_model_bin_sha256": digest.hexdigest(),
    }


def test_adapter_loads_with_requested_device_and_compute_type(model, record_property):
    """The device and compute type CTranslate2 actually uses, not only the
    values passed in."""
    for key, value in _observed(model).items():  # recorded before asserting
        record_property(key, value)
    assert model.info.backend == "whisper" and model.info.model == MODEL
    assert (model.info.device, model.info.compute_type) == (DEVICE, COMPUTE_TYPE)
    ct2 = model._model.model  # faster_whisper.WhisperModel -> ctranslate2.models.Whisper
    assert ct2.device == DEVICE
    if DEVICE == "cuda":
        assert ct2.compute_type == COMPUTE_TYPE  # no silent fallback (e.g. to float32)
    else:
        assert ct2.compute_type.startswith(COMPUTE_TYPE)  # CPU may report int8_float32


def test_adapter_decodes_silence_and_empty_audio(model):
    silence = np.zeros(16000 * 2, dtype=np.float32)
    for final in (False, True):
        assert isinstance(model.transcribe(silence, final=final).text, str)
    assert model.transcribe(np.zeros(0, dtype=np.float32), final=True).text == ""


def _loaded_libraries() -> list[str] | str:
    """CUDA-related shared objects mapped into this process (Linux). Evidence
    of what a decode actually loaded, unlike LD_LIBRARY_PATH, which only says
    where the loader would look."""
    from benchmarks.whisper_verification import parse_loaded_libraries

    try:
        with open("/proc/self/maps") as maps:
            return parse_loaded_libraries(maps)
    except OSError as exc:
        return f"unavailable: {exc.strerror or exc}"


def test_adapter_transcribes_real_speech(model, utterances, record_property):
    finals = [model.transcribe(u.float32(), final=True).text for u in utterances]
    partial_settings = [model.transcribe(u.float32(), final=False).text for u in utterances]
    record_property("observed_loaded_libraries", json.dumps(_loaded_libraries()))  # before asserting
    assert all(t.strip() for t in finals), finals
    assert all(t.strip() for t in partial_settings), partial_settings
    wer = _wer([u.reference for u in utterances], finals)
    record_property("corpus_wer_final", wer)
    record_property("corpus_wer_partial_settings",
                    _wer([u.reference for u in utterances], partial_settings))
    print(f"\n{DEVICE}/{COMPUTE_TYPE} final corpus WER = {wer:.4f} over {len(utterances)} files")
    for u, t in zip(utterances, finals):
        print(f"  {u.name}: {t}")
    assert wer <= MAX_CORPUS_WER


def test_adapter_final_decode_is_deterministic(model, utterances):
    audio = utterances[0].float32()
    assert model.transcribe(audio, final=True).text == model.transcribe(audio, final=True).text


def test_adapter_wraps_decode_errors(model):
    """A failure inside faster-whisper surfaces as InferenceError, which the
    session maps to an ``inference_failed`` error event."""
    bad = np.array(["not audio"] * 16000, dtype=object)
    with pytest.raises(InferenceError, match="faster-whisper decode failed"):
        model.transcribe(bad, final=True)


# -------------------------------------------------------------- server level
def _settings(**overrides) -> Settings:
    base = {"backend": "whisper", "model_name": MODEL, "device": DEVICE, "compute_type": COMPUTE_TYPE,
            "model_dir": MODEL_DIR, "idle_timeout_s": 30.0, "inference_timeout_s": 60.0}
    base.update(overrides)
    return Settings(**base)


def _infer_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("infer")]


@pytest.fixture(scope="module")
def live_server():
    """A real uvicorn server whose lifespan builds and loads the real model
    from Settings. Yields its address; stopped (and checked) at teardown."""
    import uvicorn

    from echorelay.app import create_app

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    app = create_app(_settings())
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 600  # first load may download the weights
    while not server.started:
        assert thread.is_alive(), "server exited during start-up (model load failed?)"
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.05)
    yield {"addr": f"127.0.0.1:{port}", "server": server, "thread": thread}
    server.should_exit = True
    thread.join(timeout=30)


def test_server_loads_real_model_from_settings(live_server, record_property):
    body = httpx.get(f"http://{live_server['addr']}/health", timeout=10).json()
    record_property("reported_health_model", json.dumps(body.get("model")))  # self-reported config
    assert body["status"] == "ok"
    assert body["model"]["backend"] == "whisper"
    assert body["model"]["name"] == MODEL
    assert body["model"]["device"] == DEVICE
    assert body["model"]["compute_type"] == COMPUTE_TYPE


# ----------------------------------------------------------- websocket level
async def _stream(addr: str, pcm16: bytes, *, frame_s: float = 0.1, speed: float = 2.0):
    """Send PCM in frame_s frames at `speed` x real time, then end; collect events."""
    import websockets

    step = int(frame_s * 16000) * 2
    events: list[dict] = []
    async with websockets.connect(f"ws://{addr}/ws/transcribe", max_size=None) as ws:

        async def receive():
            async for msg in ws:
                events.append(json.loads(msg))

        reader = asyncio.create_task(receive())
        for i in range(0, len(pcm16), step):
            await ws.send(pcm16[i:i + step])
            await asyncio.sleep(frame_s / speed)
        await ws.send(json.dumps({"type": "end"}))
        await asyncio.wait_for(reader, timeout=120)
        return events, ws.close_code


def test_websocket_streams_real_speech(live_server, model, utterances, record_property):
    utt = utterances[0]  # 10.9 s
    events, close_code = asyncio.run(_stream(live_server["addr"], utt.pcm16))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "ready" and kinds[-1] == "final", kinds
    assert "error" not in kinds, events
    assert close_code == 1000
    partials = [e for e in events if e["type"] == "partial"]
    finals = [e for e in events if e["type"] == "final"]
    assert len(finals) == 1
    assert partials and all(p["provisional"] and p["text"].strip() for p in partials)
    assert [p["seq"] for p in partials] == list(range(1, len(partials) + 1))
    text = finals[0]["text"]
    # Same audio, same model and decode settings as a direct adapter call:
    # the server path must not alter the audio (framing, scaling, truncation).
    assert text == model.transcribe(utt.float32(), final=True).text
    wer = _wer([utt.reference], [text])
    record_property("websocket_wer", wer)
    print(f"\nwebsocket: {len(partials)} partials; final WER {wer:.4f}: {text}")
    assert wer <= MAX_STREAM_WER


def test_server_shutdown_releases_inference_workers(live_server):
    """Runs last in this module (pytest keeps file order): stops the server
    and checks that the lifespan shut the inference pool down."""
    silence = np.zeros(16000, dtype="<i2").tobytes()  # one decode, so workers exist
    events, close_code = asyncio.run(_stream(live_server["addr"], silence, speed=10.0))
    assert events[-1]["type"] == "final" and close_code == 1000, events
    assert _infer_threads(), "expected inference worker threads while serving"
    live_server["server"].should_exit = True
    live_server["thread"].join(timeout=30)
    assert not live_server["thread"].is_alive(), "server did not stop"
    deadline = time.monotonic() + 30
    while _infer_threads():
        assert time.monotonic() < deadline, f"workers still alive: {_infer_threads()}"
        time.sleep(0.1)
    with socket.socket() as s:
        host, port = live_server["addr"].split(":")
        assert s.connect_ex((host, int(port))) != 0, "port still accepting connections"
