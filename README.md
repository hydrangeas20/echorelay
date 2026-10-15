# EchoRelay

Streaming speech-to-text over WebSockets, built on FastAPI and
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper `base.en`).

Clients stream 16 kHz mono PCM16 audio as binary frames, receive
**provisional** partial transcripts while they speak, and get one **final**
transcript after sending an explicit end-of-stream message.

> Whisper is window-based, not a streaming model. EchoRelay approximates
> streaming by periodically re-decoding the buffered audio. Partials are
> labelled `"provisional": true` and may change or disappear; only the
> `final` event is a committed transcript.

## Layout

```
src/echorelay/
  config.py          settings (env vars, ECHORELAY_*) and wire-format constants
  protocol.py        message shapes, error codes, close codes
  audio.py           bounded PCM16 buffer -> float32
  session.py         per-stream state machine: buffering, partial scheduling, final
  transport.py       WebSocket endpoint, admission control, error reporting, cleanup
  app.py             FastAPI app factory, /health, model lifecycle
  inference/
    base.py          Transcriber interface
    whisper.py       faster-whisper adapter
    fake.py          deterministic fake adapter (tests, model-free demos)
    runner.py        thread-pool execution, bounded concurrency, timeouts
clients/stream_wav.py   demo client
benchmarks/             load test, FP16 vs INT8/FP16 mixed-precision comparison, GPU smoke test (see COLAB.md)
docs/protocol.md        WebSocket protocol specification
tests/                  pytest suite
```

## Quick start (local, CPU)

Requires Python 3.11+.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run the test suite (uses the fake adapter; no model download)
pytest

# Run the server with the real model (downloads base.en, ~140 MB, on first start)
python -m echorelay --port 8000

# In another terminal
curl localhost:8000/health
python clients/stream_wav.py path/to/speech.wav          # real-time pacing
python clients/stream_wav.py path/to/speech.wav --speed 0  # as fast as possible
```

Without network access to Hugging Face, or to demo the protocol only, run
the server with the fake adapter (it emits `w1 w2 ...`, one word per 0.5 s):

```bash
ECHORELAY_BACKEND=fake python -m echorelay --port 8000
python clients/stream_wav.py --synthetic 3
```

Real-model smoke test (needs Hugging Face access): `pytest -m whisper`.

### Docker

```bash
docker build -t echorelay .
docker run --rm -p 8000:8000 -v echorelay-models:/models echorelay
# or bake the model into the image:
docker build --build-arg PRELOAD_MODEL=1 -t echorelay .
```

## Configuration

Environment variables (prefix `ECHORELAY_`):

| Variable | Default | Meaning |
|---|---|---|
| `BACKEND` | `whisper` | `whisper` or `fake` |
| `MODEL_NAME` | `base.en` | faster-whisper model |
| `DEVICE` | `cpu` | `cpu`, `cuda`, `auto` |
| `COMPUTE_TYPE` | `int8` | CTranslate2 type; `float16` or `int8_float16` (INT8/FP16 mixed) on GPU |
| `MODEL_DIR` | – | weights cache directory |
| `MAX_CONCURRENT_STREAMS` | `8` | open streams; extra connections get `server_busy` |
| `MAX_CONCURRENT_INFERENCE` | `1` | decodes running at once (thread pool + model workers) |
| `MAX_AUDIO_SECONDS` | `120` | per-stream audio limit |
| `MAX_FRAME_BYTES` | `262144` | per-frame limit |
| `IDLE_TIMEOUT_S` | `15` | close a stream with no frames for this long |
| `INFERENCE_TIMEOUT_S` | `30` | queueing + decode limit per job |
| `PARTIAL_INTERVAL_S` | `1.0` | new audio between partial decodes |
| `MIN_PARTIAL_AUDIO_S` | `0.5` | audio required before the first partial |
| `PARTIAL_BEAM_SIZE` / `FINAL_BEAM_SIZE` | `1` / `5` | greedy partials, beam-search final |

## Design notes

* **Event loop never runs inference.** Decodes run in a dedicated thread pool
  sized to `MAX_CONCURRENT_INFERENCE`; a semaphore bounds in-flight jobs and
  is released only when the worker thread finishes, so a timed-out decode
  cannot oversubscribe the model.
* **One partial in flight per stream.** Audio arriving during a partial
  decode is buffered and covered by the next one, so slow inference degrades
  partial frequency instead of growing a queue.
* **Partials re-decode the full buffer.** Simple and accurate for short
  utterances; cost grows with stream length (see limitations).
* **Ordering.** The final decode waits for an in-flight partial, so every
  partial precedes the final.
* **Failure is explicit.** Every failure ends with an `error` event and a
  specific WebSocket close code; see [docs/protocol.md](docs/protocol.md).

## Known limitations

* Partial decoding cost grows linearly with stream length (whole buffer
  re-decoded each interval); `MAX_AUDIO_SECONDS` caps it. A sliding window
  with committed-prefix agreement (e.g. LocalAgreement) is the next step.
* No VAD or endpointing: the client must send `end`.
* Only 16 kHz mono PCM16 is accepted; no server-side resampling.
* A decode already running in a worker thread cannot be interrupted on
  timeout or disconnect; it finishes and its result is discarded.
* No authentication, TLS termination, or rate limiting per client.
* Single process; one model instance.

## Benchmarks

See [benchmarks/COLAB.md](benchmarks/COLAB.md). No GPU numbers have been
recorded yet.
