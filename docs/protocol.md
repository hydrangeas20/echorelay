# EchoRelay WebSocket protocol (v1)

Endpoint: `GET /ws/transcribe` (WebSocket upgrade). One stream per connection.

## Audio format

Binary frames carry raw audio only, no headers:

| Property | Value |
|---|---|
| Encoding | signed 16-bit little-endian PCM (`pcm_s16le`) |
| Sample rate | 16000 Hz |
| Channels | 1 (mono) |

Each binary frame must contain a whole number of samples (even byte length).
Frame size is the client's choice (100 ms = 3200 bytes is a good default) up
to `max_frame_bytes`. Empty binary frames are ignored.

## Sequence

```
client                                server
  | ---- WebSocket upgrade ------------> |
  | <--- {"type":"ready", ...} --------- |
  | ---- binary PCM16 frame -----------> |
  | ---- binary PCM16 frame -----------> |
  | <--- {"type":"partial", ...} ------- |   provisional, zero or more
  | ---- ...                             |
  | ---- {"type":"end"} ---------------> |   text frame, explicit end of stream
  | <--- {"type":"final", ...} --------- |   exactly one on success
  | <--- close 1000 -------------------- |
```

The client may start sending audio immediately; it does not have to wait for
`ready`. Audio sent after `end` is not processed.

## Client → server

* **Binary frame**: audio, as above.
* **Text frame** `{"type": "end"}`: no more audio. This is the only valid text
  message; anything else is a protocol error.

## Server → client

### `ready`
Sent once after the connection is accepted.
```json
{"type": "ready", "protocol_version": "1", "session_id": "3f2a9c1b7d4e",
 "audio": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1},
 "limits": {"max_audio_seconds": 120.0, "max_frame_bytes": 262144,
            "idle_timeout_s": 15.0, "partial_interval_s": 1.0}}
```

### `partial` — provisional hypothesis
```json
{"type": "partial", "provisional": true, "seq": 3, "text": "the quick brown",
 "audio_duration_s": 3.0, "inference_ms": 84.2}
```
* Whisper is window-based, not natively streaming. A partial is the result of
  re-decoding **all audio received so far** with cheaper (greedy) settings.
* **Partials are not committed words.** Any word in a partial may change or
  disappear in the next partial or in the final. Each partial replaces the
  previous one entirely; clients should overwrite, not append.
* Scheduled every `partial_interval_s` of newly received audio, once at least
  `min_partial_audio_s` is buffered. At most one partial decode per stream is
  in flight; audio arriving meanwhile is folded into the next one.
* Empty hypotheses and exact repeats of the previous partial are not sent.
* `seq` increases by 1 per partial sent. `audio_duration_s` is the audio the
  hypothesis covers. `inference_ms` is model time in the worker thread.

### `final` — committed transcript
```json
{"type": "final", "provisional": false, "text": "The quick brown fox.",
 "audio_duration_s": 4.2, "inference_ms": 230.5}
```
Sent once after `end`, decoded over the full stream with final-quality
settings (beam search). Always after every partial. The server then closes
with code 1000. An empty stream produces `"text": ""`.

### Optional `timing` object

When the server runs with `ECHORELAY_EVENT_TIMING=true` (default **off**, so
the v1 wire format above is unchanged), `partial` and `final` events carry an
extra `timing` object. Clients must ignore it if they don't use it.

```json
{"type": "partial", ..., "timing": {"queue_ms": 12.4, "decode_ms": 61.0, "staleness_ms": 100.0}}
{"type": "final",   ..., "timing": {"wait_partial_ms": 35.2, "queue_ms": 0.1, "decode_ms": 118.7}}
```

| Field | Meaning |
|---|---|
| `queue_ms` | decode requested → inference slot acquired |
| `decode_ms` | model call inside the worker thread (equals `inference_ms`) |
| `staleness_ms` | audio received after this partial's snapshot was taken, at send time |
| `wait_partial_ms` | `end` processed → in-flight partial for this stream finished |

Definitions and measurement boundaries: [measurements.md](measurements.md).

### `error`
```json
{"type": "error", "code": "audio_too_large", "message": "..."}
```
Always the last message; the server closes immediately afterwards.

## Error codes

| `code` | Close code | Cause |
|---|---|---|
| `malformed_message` | 1003 | Text frame is not JSON, not an object, lacks `type`, or `type` is not `end` |
| `invalid_audio_frame` | 1003 | Binary frame length is odd (not whole PCM16 samples) |
| `frame_too_large` | 1009 | Single frame exceeds `max_frame_bytes` |
| `audio_too_large` | 1009 | Stream audio would exceed `max_audio_seconds` |
| `idle_timeout` | 1008 | No frame received for `idle_timeout_s` |
| `send_timeout` | 1008 | Client did not accept a server message within `send_timeout_s` (stopped reading) |
| `server_busy` | 1013 | `max_concurrent_streams` already active; retry later |
| `inference_error` | 1011 | Model raised during a partial or final decode |
| `inference_timeout` | 1011 | Queueing + decode exceeded `inference_timeout_s` |
| `internal_error` | 1011 | Unexpected server bug |

A failed partial decode ends the stream (it is not silently retried): an
inference failure usually means the final would fail too, and failing fast
frees resources.

## Disconnects

If the client disconnects without `end`, the server cancels the stream's
pending partial task, drops its buffer, and releases its stream slot. A
decode already running in a worker thread cannot be interrupted; it finishes,
its result is discarded, and its inference slot is then released.
