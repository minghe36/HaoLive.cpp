# Protocol and integration reference

[English README](../README.md) | [中文 README](../README.zh-CN.md)

## Audio and transcript contract

| Property           | Contract                                                                    |
| ------------------ | --------------------------------------------------------------------------- |
| Transport input    | 16 kHz mono PCM16LE, or RIFF WAV containing that format for HTTP            |
| Embedded API input | 16 kHz mono normalized float32 PCM                                          |
| Append ordering    | Contiguous, nonoverlapping audio within an utterance                        |
| Partial text       | Complete current hypothesis; replace the preceding text                     |
| Unchanged result   | `updated=false` keeps the preceding hypothesis visible                      |
| Final result       | Fixed text; session is released and rejects further audio                   |
| Retries            | Do not retry an uncertain append; create a new session instead              |
| Endpointing        | Caller decides when to finalize; VAD does not alter ASR input automatically |

An HTTP result has this shape:

```json
{
  "api_version": 1,
  "session_id": "1",
  "model": "xasr",
  "sequence": 1,
  "type": "partial",
  "text": "Current complete transcript",
  "updated": true,
  "audio_ms": 800
}
```

| Field         | Meaning                                       |
| ------------- | --------------------------------------------- |
| `api_version` | HTTP protocol version                         |
| `session_id`  | Utterance identifier                          |
| `model`       | GGUF architecture name                        |
| `sequence`    | Inference-round sequence within the utterance |
| `type`        | `partial` or `final`                          |
| `text`        | Full text, never a string delta               |
| `updated`     | Whether this result changes the hypothesis    |
| `audio_ms`    | Audio duration processed for this result      |

Confucius and Moonshine partials may revise earlier words. Filter results from old sessions and old sequences; several token callbacks in the current round may share a sequence.

## WebSocket

The HTTP listener and WebSocket listener use separate ports. Read the dynamically allocated `websocket_port` from `GET /health`; the WebSocket listener binds to loopback.

| Stage    | Wire message                                                                                                      |
| -------- | ----------------------------------------------------------------------------------------------------------------- |
| Connect  | `ws://127.0.0.1:<websocket_port>/v1/realtime`                                                                     |
| Headers  | `X-Session-Id`: nonempty, at most 64 bytes; `Authorization: Bearer <token>` when configured                       |
| Ready    | `{"event":"ready","api_version":1,"session_id":"..."}`                                                            |
| Audio    | Binary PCM16LE frames                                                                                             |
| Progress | JSON `event: "partial"` / `event: "result"`, with `session_id`, `sequence`, `text`, `type`, `updated`, `audio_ms` |
| Finalize | Text frame `{"event":"final"}`, ordered after all audio                                                           |
| Complete | Wait for `type: "final"`, then close; use a new connection for the next utterance                                 |
| Failure  | `event: "error"` with a structured `error` object                                                                 |

The reader continues accepting audio while the worker performs inference. Pending audio can be coalesced without dropping samples. Clients should also send and receive independently. A fast reader cannot make a slow model process audio faster; accumulated work still affects latency.

### Timing fields

WebSocket transcript messages also include a `timing` object:

| Field                      | Unit / meaning                                                                            |
| -------------------------- | ----------------------------------------------------------------------------------------- |
| `serverReceivedAtMs`       | Unix milliseconds when the earliest pending audio batch arrived                           |
| `serverLatestReceivedAtMs` | Unix milliseconds when the latest pending batch arrived                                   |
| `serverEmittedAtMs`        | Unix milliseconds at response emission                                                    |
| `queueMs`                  | Monotonic duration from pending-batch arrival to dequeue                                  |
| `lockWaitMs`               | Time waiting for the shared inference lock                                                |
| `sessionSetupMs`           | Session creation/check duration                                                           |
| `inferenceMs`              | Elapsed inference time at emission; token callbacks may arrive before the round completes |

These fields measure server stages, not native audio capture or physical display latency. Scheduling at 160 ms does not guarantee a first word at 160 ms.

## HTTP and SSE

Create a session with `POST /v1/sessions`, append PCM/WAV with `POST /v1/sessions/{id}/audio`, and use `final=true` to flush the last audio and release it. The final request may have an empty body. Cancel with `DELETE /v1/sessions/{id}`.

Confucius supports `stream=true` on an audio request. Its `text/event-stream` response emits:

| SSE event | Meaning                                                  |
| --------- | -------------------------------------------------------- |
| `partial` | Full hypothesis as text tokens become available          |
| `result`  | Exactly one successful completion event for this request |
| `error`   | Inference failure; the session is removed                |

Other adapters currently return their regular chunk results without token callbacks. Confucius partial generation yields after 32 new tokens, or a lower configured limit; final uses the full `--max-tokens` budget and reports exhaustion.

| HTTP status | Meaning                                        |
| ----------- | ---------------------------------------------- |
| 400         | Invalid parameters or audio                    |
| 401         | Missing or invalid bearer token                |
| 404         | Session missing, finalized or expired          |
| 413         | Audio or payload limit exceeded                |
| 415         | Unsupported audio format                       |
| 429         | Session limit reached                          |
| 500         | Inference failure                              |
| 503         | Requested ASR or VAD capability was not loaded |

Errors use `{"error":{"code":"...","message":"..."}}`. Enabling `--token` requires the bearer header on all HTTP requests, including health checks.

## Independent FireRed VAD

Run VAD alone, or add `--vad-model` to an ASR launch:

```sh
./haolive-server --vad-model ./models/firered-vad.gguf --backend cpu

curl -H "Content-Type: audio/wav" --data-binary @speech.wav \
  'http://127.0.0.1:9797/v1/vad?threshold=0.5&min_speech_ms=100'
```

FireRed VAD runs on CPU and does not create an ASR engine. Combined GPU ASR and CPU VAD can coexist.

| Parameter / field                      | Contract                                                       |
| -------------------------------------- | -------------------------------------------------------------- |
| `threshold`                            | Default 0.5; must be strictly between 0 and 1                  |
| `min_speech_ms`                        | Default 100; allowed range 1–30,000 ms                         |
| `is_speech`                            | True when enough consecutive 10 ms frames meet the threshold   |
| `sufficient_audio`                     | At least one complete 25 ms frame                              |
| `max_probability` / `mean_probability` | Model frame scores, not calibrated whole-clip confidence       |
| `speech_ms`                            | All above-threshold frames × 10 ms, including disjoint regions |
| `audio_ms` / `frames`                  | Input duration and evaluated frame count                       |

Requests are stateless windows; overlapping VAD windows are allowed. ASR appends must remain nonoverlapping. Audio shorter than 25 ms reports insufficient audio; no fake frame is padded. The last incomplete frame is not evaluated.

A caller can submit 1–2 second rolling windows every 200–500 ms. Short windows can miss speech because of limited context or the consecutive-duration rule. VAD detects acoustic activity; it does not determine whether speech is live, recorded or synthetic. See [VAD validation](vad.md).

## Embedding

```cpp
haolive::Options options;
options.model_path = model_path; // Supplied by the caller.
options.backend = "cpu";
auto engine = haolive::load(options);
auto session = engine->create_session();
auto partial = session->accept(samples.data(), samples.size(), false);
auto final = session->accept(nullptr, 0, true);
```

| API            | Ownership / usage                                                                                    |
| -------------- | ---------------------------------------------------------------------------------------------------- |
| C++ ASR        | Engine must outlive sessions; inference on one engine is serialized                                  |
| C ABI ASR      | Sessions retain the engine; no exceptions cross the ABI; inspect `hl_last_error()`                   |
| C++ VAD        | `load_vad(options)` → `detect(samples, count, params)`; concurrent calls are safe                    |
| C ABI VAD      | `hl_default_vad_options` → `hl_vad_create` → `hl_vad_detect` → `hl_vad_free`; no ASR engine required |
| Shared library | Optional FFI artifact built with `HAOLIVE_BUILD_SHARED=ON`; not needed by the server                 |

Read the [C++ header](../include/haolive/haolive.hpp) and [C header](../include/haolive/haolive.h) for complete signatures, borrowed-string ownership and release functions. Keep a detector alive until its concurrent calls have finished.

## Host process lifecycle

| Integration step  | Behavior                                                                                          |
| ----------------- | ------------------------------------------------------------------------------------------------- |
| Launch            | Supply a model path, backend, fresh bearer token and `--port 0`                                   |
| Handshake         | stdout begins with `{"event":"ready","api_version":1,"port":<assigned port>}` after bind succeeds |
| Verify            | Call authenticated `/health` to check API version and loaded capabilities                         |
| Logs              | Read stderr for diagnostics; it is not the handshake channel                                      |
| Parent monitoring | On POSIX, pass `--parent-pid` with the direct parent PID to exit when that parent disappears      |

Model weights are supplied by the host; the engine does not download them. An ASR-disabled server returns 503 for ASR requests; a VAD-disabled server returns 503 for VAD requests.
