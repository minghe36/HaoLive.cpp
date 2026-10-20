# MVP acceptance record

Verified on 2026-10-19, macOS arm64, Apple M4 Max. CPU and Metal were compiled; Vulkan/CUDA were not compiled.

## Automated checks

- CTest: `unit` and `c_header` passed. Unit executable: `904 checks passed`.
- Eight real-model HTTP runs passed, 59 request checks each (472 total), plus semantic assertions. Both English and Chinese were tested on both models/backends.
- Installed shared library was loaded directly with Python ctypes. A copy of the installed server binary, placed alone in a temporary directory, started and performed Chinese Metal inference successfully. Dynamic dependencies contain only macOS system libraries/frameworks.
- Explicit Vulkan/CUDA requests were rejected as unavailable in the Mac binary.
- The shared-library C ABI model checks passed on all four model/backend combinations and exercise engine/session ownership, invalid input, empty finalization and rejection after finalization on both models/backends.
- HTTP checks include nonempty real speech, partial/final, repeated-call determinism, interleaved-session isolation, PCM/WAV parity, cancellation, authentication, size/session limits and malformed input.

| Model | Backend | Audio | Final text |
| --- | --- | --- | --- |
| x-asr-zh-en-q8_0 | cpu | en, 5 s | And so my fellow americans asked not |
| x-asr-zh-en-q8_0 | cpu | zh, 5 s | 正是因为存在绝对正义，所以我们接受现实的相对正义 |
| x-asr-zh-en-q8_0 | metal | en, 5 s | And so my fellow americans ask not |
| x-asr-zh-en-q8_0 | metal | zh, 5 s | 正是因为存在绝对正义，所以我们接受现实的相对正义 |
| confucius4-r2t2-q4_k | cpu | en, 5 s | And so, my fellow Americans, ask not. |
| confucius4-r2t2-q4_k | cpu | zh, 5 s | 正是因为存在绝对正义，所以我们接受现实的相对正义。 |
| confucius4-r2t2-q4_k | metal | en, 5 s | And so, my fellow Americans, ask not. |
| confucius4-r2t2-q4_k | metal | zh, 5 s | 正是因为存在绝对正义，所以我们接受现实的相对正义。 |

All eight runs produced the same final for offline and 500 ms chunked input within that backend. X-ASR CPU/Metal differed on English `asked` versus `ask`; floating-point backend parity is not promised. The CPU X-ASR English output matches the existing reference executable. Confucius Chinese output matches the reference executable with Chinese conditioning.

## Fixed regressions

1. Confucius early EOS when generating its own language header: prefill `<asr_text>` without an auxiliary LID model; require nonempty real bilingual ASR output.
2. X-ASR reused positional-input storage: upload constant positional data for every encoder run. Test repeat inference, interleaved sessions and offline/chunked final parity.
3. Metal missing high-precision F32 matrix kernel: use default backend precision for attention scores. Verified with actual inference on Apple M4 Max.
4. R2T2 pipe chunk delimiters: stop generation at the boundary and omit it from displayed text. Unit and real partial-output checks.

## Scope and limits

- Native X-ASR streaming is chunk invariant for the checked fixtures. Confucius uses a smaller fixed-interval prefix/rollback policy rather than the upstream adaptive R2T2 scheduler; partial text is revisable and arbitrary chunk schedules are not guaranteed identical.
- No VAD endpointing or automatic long-session segmentation. Clients explicitly finalize each utterance (maximum 30 seconds) and start the next session.
- Audio input is PCM16LE mono 16 kHz or WAV containing that format. No compressed codecs, resampling or microphone/device capture in this library.
- No WebSocket/OpenAI-compatible protocol in this MVP. HTTP sessions, C++ and C ABI share the same model-independent lifecycle.
- Vulkan/CUDA build options exist but have not been compiled or tested in this Mac task. Windows/Linux builds and other GPU hardware have not been validated.
- This is a functional MVP acceptance record, not a WER/CER benchmark or a general model-quality claim. Timings in local JSON reports are diagnostic, not a controlled performance benchmark.

## Input identity

Audio fixtures are the first 5 seconds of `samples/jfk.wav` and `samples/paraformer_zh.wav` from the provided reference checkout. They are not redistributed here. SHA-256 of decoded PCM bytes:

- `jfk.wav`: `329d61c2b83e8e393b0d728ef43bc6c51aa5f798def1162f2dc60894651cba95`
- `paraformer_zh.wav`: `dde1270fc7abae5efbc91a67e16025ab28a614dd2cc4b2c1441c4f6fef77e66b`

## Additional streaming models and single-file delivery

Added Moonshine Streaming Medium Q4_K and Nemotron 3.5 ASR Streaming 0.6B
Q8_0, without changing HTTP/C++/C ABI v1. Moonshine's compact 32768-token
vocabulary is embedded; no `tokenizer.bin` is opened at runtime. The shared
library is optional and disabled by default for fresh builds. All license
notices are accessible through `haolive-server --licenses`.

CTest now passes four targets with the optional shared library enabled (1071
unit assertions, embedded license CLI, 15 benchmark tests and the C header).
Fresh standalone builds disable the shared library and run three targets.

Real-model checks cover both new models on CPU and Metal. Moonshine uses
English speech; Nemotron uses both English and Chinese. Each HTTP run checks
68 requests plus semantic assertions, including repeated inference, interleaved
sessions, offline/500 ms/irregular chunk final parity, nonempty final bodies,
one-sample audio, PCM/WAV, authentication and resource bounds. Shared-library
C ABI lifecycle checks pass for each new model/backend pair.

| Model | Backend | Speech | Offline and streamed final |
| --- | --- | --- | --- |
| Moonshine Medium | CPU / Metal | English, 5 s | And so my fellow Americans, ask not. |
| Nemotron 3.5 | CPU / Metal | English, 5 s | And so my fellow Americans ask not. |
| Nemotron 3.5 | CPU / Metal | Chinese, 5 s | 正是因为存在绝对正义, 所以我们接受县市的相对正义。 |

The Chinese fixture actually says “现实”, not “县市”. The reference Nemotron
executable produces the same substitution. This is a recorded model-quality
limitation; nonempty text does not establish general recognition accuracy.
The reference Moonshine executable returns the same English words with a
comma at the end, while this runtime returns a period. Exact punctuation
parity with the reference runtime is not promised.

`tests/standalone.py` copies only the executable into an empty temporary
directory, presents each selected weight as `model.gguf` with no companion
files, clears inherited runtime search configuration, and runs actual speech
inference for all four models on CPU and Metal. `otool -L` confirms only
macOS system frameworks/libraries, with no build-directory or third-party
shared library paths. The binary includes Metal shaders, vocabulary and
license notices. GGUF weights remain caller-supplied data. This verifies
single-file execution on this host, not other macOS versions or architectures.

The final standalone report is `build/standalone-four-models-final.json`.
Server SHA-256: `f102153ad93efc8f9913df062f53144632266f7adc29e4b73cb62c5390149194`.
The new-model HTTP reports contain 408 request checks in six runs; the
existing models also retain their Chinese CPU/Metal HTTP regression coverage.

The deliverable is `build/single-file/haolive-server` (4800960 bytes, about
4.58 MiB). `build/HaoLive-macos-arm64-single-file.tar.gz` contains exactly one
entry named `haolive-server`; checksums are in `build/SHA256SUMS-single-file`.
The optional shared-library development build is not part of this package.

## Independent FireRed VAD addition

The subsequent single-file build adds independent CPU VAD and updates the
deliverable at the same path. See [VAD acceptance and latency](vad.md) for
the current executable hash, 1738 unit assertions, real C ABI/HTTP checks and
small-window limitations. Earlier four-model benchmark hashes above identify
that earlier release; ASR performance was not re-benchmarked in this VAD task.

## Token progress regression (2026-10-20)

`tests/streaming.py` accepts `--server`, `--model`, `--audio` and `--backend`.
It verifies multiple hypotheses arrive before the HTTP request completes,
request sequences, continued speech across requests, final flushing and SSE
error frames. CPU and Metal passed with the supplied Chinese speech WAV.
`haolive-qwen-streaming-tests MODEL BACKEND SPEECH.wav` checks a deliberately
one-token budget: partials yield without closing the session, callbacks run
before return, and final budget exhaustion remains an error.

The default step is now 320 ms. Qwen partial generation is capped at 32 new
tokens per call, control delimiters are excluded from future prefixes, and
incomplete UTF-8 suffixes are withheld until complete. JSON callers retain the
full-hypothesis replacement contract; SSE callers opt in with `stream=true`.
This does not add incremental reuse of the Qwen audio encoder.

## 160 ms live defaults (2026-10-20)

X-ASR selects the 160 ms metadata entry: 8 encoder frames, 29 input feature frames and matching left context (96 in the tested Q8_0 GGUF). Unit tests also exercise reordered metadata, incompatible downsampling and missing/mismatched context entries. Confucius defaults to a 160 ms new-audio inference step in the C++ API, C ABI and server CLI.

The macOS 13.0 arm64 static build passed all three CTest suites (1766 unit checks). `haolive-xasr-streaming-tests` passed CPU and Metal on eight-second Chinese and English speech fixtures, comparing offline final text with 160 ms and irregular interleaved-session final text. The Chinese fixture produced 34 live updates; the English ali-youtube fixture produced 32. A separate English presentation fixture produced no partial text in its first eight seconds and is unsuitable for asserting multiple live updates in that interval; this does not establish recognition quality for that clip.

Confucius passed the partial token-budget/finalization regression and the paced full-duplex WebSocket test with 160 ms PCM and step size. The eight-second Chinese paced fixture produced 34 text updates for each model. These tests cover inference and loopback transport, not OS audio capture or displayed-frame latency. They do not establish a 160 ms end-to-end latency guarantee.
