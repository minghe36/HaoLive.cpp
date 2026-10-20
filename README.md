# HaoLive.cpp

**Lightweight, local ASR inference.**

[English](README.md) | [简体中文](README.zh-CN.md)

A C++ engine dedicated to real-time ASR inference, only providing the most useful real-time ASR models. It has no dependency on binary programs, is small in size, easy to use, and offers low-latency local inference. Suitable for real-time subtitles, real-time translation, and other scenarios

| Capability               | Included                                                                                       |
| ------------------------ | ---------------------------------------------------------------------------------------------- |
| Recognition              | Four GGUF model adapters with partial and final transcripts                                    |
| Transport                | Bidirectional WebSocket, HTTP sessions, optional token streaming over SSE                      |
| Integration              | C++ API, C ABI, standalone server process                                                      |
| Voice activity detection | Independent CPU FireRed VAD                                                                    |
| Distribution             | CPU/macOS/Vulkan: single executable; CUDA: executable and runtime DLLs; external model weights |
| Runtime dependencies     | No Python, FFmpeg, ONNX Runtime or LibTorch; GPU variants need a vendor driver                 |

## Download

Get prebuilt binaries from [GitHub Releases](https://github.com/minghe36/HaoLive.cpp/releases/latest). Each archive has a separate SHA-256 checksum file. CPU, macOS and Vulkan packages contain one executable; CUDA packages also include the required NVIDIA runtime DLLs and license.

| Platform              | Archive                                     | ASR backends | Requirements                                         |
| --------------------- | ------------------------------------------- | ------------ | ---------------------------------------------------- |
| macOS Apple Silicon   | `HaoLive-<version>-macos-arm64.tar.gz`      | CPU, Metal   | macOS 13+                                            |
| macOS Intel           | `HaoLive-<version>-macos-x86_64.tar.gz`     | CPU, Metal   | macOS 13+; SSE4.2, AVX2, FMA, F16C, BMI2             |
| Windows x64 CPU       | `HaoLive-<version>-windows-x64-cpu.zip`     | CPU          | x64 CPU with SSE4.2, AVX2, FMA, F16C, BMI2           |
| Windows x64 Vulkan    | `HaoLive-<version>-windows-x64-vulkan.zip`  | CPU, Vulkan  | Vulkan-capable GPU and vendor driver                 |
| Windows x64 CUDA 12.6 | `HaoLive-<version>-windows-x64-cuda126.zip` | CPU, CUDA    | Supported NVIDIA GPU and CUDA 12.6-compatible driver |

Windows executables statically link the MSVC runtime. Keep CUDA package DLLs beside the executable. Vulkan uses the loader supplied by the GPU driver. Metal shaders, the supported Moonshine tokenizer and third-party license texts are embedded. Model weights are not included.

## Quick start

Download and extract the executable, then download an [X-ASR model](#models). Run these commands from the extracted directory.

**macOS**

```sh
./haolive-server --model ./models/x-asr-zh-en-q8_0.gguf --backend cpu
```

**Windows PowerShell**

```powershell
.\haolive-server.exe --model .\models\x-asr-zh-en-q8_0.gguf --backend cpu
```

Model paths above are examples; pass the actual path to your downloaded GGUF file. The server listens on `http://127.0.0.1:9797` by default.

Send a supported WAV file from another terminal:

```sh
curl -H "Content-Type: audio/wav" --data-binary @speech.wav \
  http://127.0.0.1:9797/v1/transcribe
```

In Windows PowerShell, use `curl.exe` for the same HTTP request.

| Audio property            | Required format                                                                  |
| ------------------------- | -------------------------------------------------------------------------------- |
| Sample rate               | 16,000 Hz                                                                        |
| Channels                  | Mono                                                                             |
| HTTP / WebSocket encoding | PCM16 little-endian; HTTP also accepts RIFF WAV containing this format           |
| C++ / C ABI encoding      | Normalized float32 PCM in `[-1, 1]`                                              |
| Conversion                | Resampling, mixing and compressed-media decoding are the caller's responsibility |

Useful commands: `--help`, `--version`, `--list-backends`, `--licenses`.

## Models

The following models have been tested by Haoshu and are recommended for use.

Weights are available from the [cstr Hugging Face model listing](https://huggingface.co/cstr/models). The adapter is selected from GGUF architecture metadata, not the filename. One server process loads one ASR model; use separate processes and ports for multiple models.

| Model                                                                                               | Supported languages                                                                                                                                            | Model size | Quantization | Download                                                                                                                                      |
| --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| [X-ASR zh/en](https://huggingface.co/cstr/x-asr-zh-en-GGUF)                                         | Chinese, English                                                                                                                                               | ≈ 168 MB   | Q8_0         | [GGUF](https://huggingface.co/cstr/x-asr-zh-en-GGUF/resolve/main/x-asr-zh-en-q8_0.gguf?download=true)                                         |
| [Confucius4-R2T2](https://huggingface.co/cstr/confucius4-r2t2-GGUF)                                 | Optimized for Chinese and English; [additional languages](https://huggingface.co/netease-youdao/Confucius4-R2T2#supported-languages)                           | ≈ 1.49 GB  | Q4_K         | [GGUF](https://huggingface.co/cstr/confucius4-r2t2-GGUF/resolve/main/confucius4-r2t2-q4_k.gguf?download=true)                                 |
| [Moonshine Streaming Medium — English](https://huggingface.co/cstr/moonshine-streaming-medium-GGUF) | English                                                                                                                                                        | ≈ 191 MB   | Q4_K         | [GGUF](https://huggingface.co/cstr/moonshine-streaming-medium-GGUF/resolve/main/moonshine-streaming-medium-q4_k.gguf?download=true)           |
| [Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/cstr/nemotron-3.5-asr-streaming-0.6b-GGUF) | Multilingual: Chinese, English, Japanese, Korean and more; [language tiers](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b#supported-languages) | ≈ 686 MB   | Q8_0         | [GGUF](https://huggingface.co/cstr/nemotron-3.5-asr-streaming-0.6b-GGUF/resolve/main/nemotron-3.5-asr-streaming-0.6b-q8_0.gguf?download=true) |

Model names link to their Hugging Face pages. Sizes are approximate GGUF download sizes (decimal MB/GB), not runtime memory usage.

Language coverage follows the model cards. Some Nemotron languages require fine-tuning; see the linked language tiers.

| Adapter   | Streaming behavior                                     | Important detail                                                                                 |
| --------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| X-ASR     | Native 160 ms Zipformer2 / Transducer chunks           | Reads matching geometry and left-context caches from GGUF                                        |
| Confucius | Default 160 ms scheduling; token callbacks             | Re-encodes accumulated audio, rolls back four hypothesis tokens; partial budget is 32 new tokens |
| Moonshine | Cumulative waveform inference at `--step-ms` intervals | Recomputes encoder state; partial text can change                                                |
| Nemotron  | Cached FastConformer / RNN-T chunks                    | Schedule is bounded by the model chunk size; holds one chunk until the STFT boundary is stable   |

Scheduling intervals are not first-word latency guarantees. For slower models, a larger `--step-ms` can reduce repeated work. Confucius uses a smaller policy than the upstream adaptive R2T2 scheduler; arbitrary chunk schedules need not produce identical final text. See [architecture](docs/architecture.md) for adapter details.

### Voice activity detection model

| Model                                                       | Purpose                                     | Model size | Precision | Download                                                                                                     |
| ----------------------------------------------------------- | ------------------------------------------- | ---------- | --------- | ------------------------------------------------------------------------------------------------------------ |
| [FireRed VAD](https://huggingface.co/cstr/firered-vad-GGUF) | Language-agnostic speech activity detection | ≈ 2.36 MB  | F32       | [firered-vad.gguf](https://huggingface.co/cstr/firered-vad-GGUF/resolve/main/firered-vad.gguf?download=true) |

FireRed VAD runs on CPU and detects whether audio contains speech; it does not produce a transcript. Use it independently or load it alongside any ASR model with `--vad-model`. The caller decides how to use the detection result; the server does not automatically split ASR utterances or discard audio.

```sh
# VAD only
./haolive-server --vad-model ./models/firered-vad.gguf --backend cpu

# ASR and VAD together
./haolive-server --model ./models/x-asr-zh-en-q8_0.gguf \
  --vad-model ./models/firered-vad.gguf --backend cpu
```

Query `POST /v1/vad` for speech detection. See the [VAD API reference](docs/protocol.md#independent-firered-vad) for parameters and response fields.

## Server options

| Option              | Default              | Purpose                                                                       |
| ------------------- | -------------------- | ----------------------------------------------------------------------------- |
| `--model`           | None                 | ASR GGUF file; required unless using VAD only                                 |
| `--vad-model`       | None                 | Optional FireRed VAD GGUF file                                                |
| `--backend`         | `cpu`                | Requested ASR backend; `--list-backends` lists devices available on this host |
| `--host` / `--port` | `127.0.0.1` / `9797` | HTTP listener; port `0` requests an OS-assigned port                          |
| `--threads`         | `4`                  | CPU worker threads                                                            |
| `--step-ms`         | `160`                | Model-dependent partial inference scheduling                                  |
| `--max-audio-ms`    | `30000`              | Per-utterance audio limit                                                     |
| `--max-tokens`      | `512`                | Generation limit; final decoding reports budget exhaustion                    |
| `--token`           | None                 | Require `Authorization: Bearer <token>`                                       |
| `--parent-pid`      | None                 | POSIX only: exit when the direct parent exits                                 |

On supported macOS hardware, pass `--backend metal` to request GPU inference. An unavailable requested backend is an error; individual unsupported GPU operations may run on CPU. Windows GPU variants are compiled in CI, which has no physical GPU for inference validation. `--list-compiled-backends` reports included backends; `--list-backends` reports devices available on this host. CUDA 12.6 includes SM 75/86/89 native code and SM 80 PTX for forward JIT compatibility; newer GPUs require separate validation.

## APIs

| Interface         | Entry point                                                            | Use                                                                        |
| ----------------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Health            | `GET /health`                                                          | API version, loaded capabilities and WebSocket port                        |
| Models            | `GET /v1/models`                                                       | Loaded model metadata                                                      |
| One-shot ASR      | `POST /v1/transcribe`                                                  | Transcribe one PCM/WAV request                                             |
| HTTP sessions     | `POST /v1/sessions`                                                    | Create a streaming utterance                                               |
| Append / finalize | `POST /v1/sessions/{id}/audio`                                         | Append nonoverlapping audio; `final=true` flushes and releases the session |
| Cancel            | `DELETE /v1/sessions/{id}`                                             | Release an unfinished HTTP session                                         |
| WebSocket         | `/v1/realtime` on the port returned by `/health`                       | Continuous bidirectional audio and transcript updates                      |
| Independent VAD   | `POST /v1/vad`                                                         | Detect speech without an ASR session                                       |
| Embedded APIs     | [C++](include/haolive/haolive.hpp), [C ABI](include/haolive/haolive.h) | Integrate without an HTTP server                                           |

### Real-time WebSocket

| Step        | Client behavior                                                                                                             |
| ----------- | --------------------------------------------------------------------------------------------------------------------------- |
| 1. Discover | Read `websocket_port` from authenticated `GET /health`                                                                      |
| 2. Connect  | Open `ws://127.0.0.1:<websocket_port>/v1/realtime`; supply the bearer token when configured and `X-Session-Id` (1–64 bytes) |
| 3. Stream   | Send binary PCM16LE frames; receive JSON updates concurrently                                                               |
| 4. Finalize | After all audio, send `{"event":"final"}`; wait for `type: "final"`, then close                                             |
| 5. Continue | Open a new connection for the next utterance                                                                                |

Receive and send independently: inference does not block the server's audio reader. Updates contain the **complete current utterance**; replace the previous hypothesis instead of appending it. Keep the current text when `updated=false`. Token callbacks in one inference round share a sequence number.

### HTTP sessions

```sh
curl -X POST http://127.0.0.1:9797/v1/sessions

# Replace SESSION_ID with the returned session_id.
curl -H "Content-Type: audio/pcm" --data-binary @chunk.pcm \
  http://127.0.0.1:9797/v1/sessions/SESSION_ID/audio

curl -H "Content-Type: audio/pcm" --data-binary '' \
  'http://127.0.0.1:9797/v1/sessions/SESSION_ID/audio?final=true'
```

Confucius also supports `stream=true` on audio requests for SSE token progress. Filter stale updates using session ID and sequence. Do not automatically retry audio appends: this can duplicate samples. The protocol is not an OpenAI-compatible API.

| Limit / rule          | Value                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------ |
| Concurrent utterances | Up to 8 HTTP sessions; up to 8 WebSocket connections                                       |
| Audio per utterance   | Up to 30 seconds, bounded by `--max-audio-ms`                                              |
| Payload               | Up to 1 MiB per HTTP request / WebSocket input or coalesced fragment                       |
| HTTP idle cleanup     | After 120 seconds, lazily on a subsequent session operation                                |
| Endpointing           | Caller sends final and starts a new utterance; no automatic VAD endpointing                |
| Transcript contract   | Full-text partials may be revised; final text is fixed                                     |
| Errors                | Structured code/message; HTTP 400, 401, 404, 413, 415, 429, 500 or capability-specific 503 |

Response fields, SSE events, VAD examples and process handshakes: [protocol reference](docs/protocol.md).

## Build from source

Dependencies are vendored and pinned; model files remain external. Release scripts configure, compile, run CTest, audit the binary, and write archives plus checksums to `dist/`.

| Target      | Prerequisites                                           | Build command                                                |
| ----------- | ------------------------------------------------------- | ------------------------------------------------------------ |
| macOS arm64 | CMake 3.24+, Python 3.10+, Xcode Command Line Tools     | `./scripts/build-macos.sh --arch arm64`                      |
| macOS Intel | Same; target architecture must be runnable for tests    | `./scripts/build-macos.sh --arch x86_64`                     |
| Windows x64 | CMake 3.24+, Python 3.10+, Visual Studio 2022 C++ tools | `./scripts/build-windows.ps1 -Architecture x64 -Variant cpu` |

For Windows GPU builds, use `-Variant vulkan` with the Vulkan SDK, or `-Variant cuda126` with CUDA Toolkit 12.6.x and the VS 2019 v142 toolset installed under Visual Studio 2022. Set `VULKAN_SDK` or `CUDA_PATH` to the actual SDK location.

For a development build:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --output-on-failure
```

| CMake switch                            | Purpose                                                                                |
| --------------------------------------- | -------------------------------------------------------------------------------------- |
| `HAOLIVE_METAL=ON/OFF`                  | Metal backend; enabled by default on macOS                                             |
| `HAOLIVE_VULKAN=ON` / `HAOLIVE_CUDA=ON` | GPU backends requiring their SDK/toolkit; enabled by the corresponding Windows variant |
| `HAOLIVE_BUILD_SHARED=ON`               | Build an optional C ABI shared library; server remains independent                     |
| `HAOLIVE_BUILD_TESTS=ON/OFF`            | Enable or disable test targets                                                         |

Static library: `libhaolive.a` / `haolive.lib`. Link through CMake `add_subdirectory(...)` and `target_link_libraries(your_app PRIVATE haolive)`. The engine must outlive C++ sessions; C ABI sessions retain engine ownership. Calls on one engine are serialized, and sessions keep independent utterance state.

## Development and documentation

| Topic                                     | Entry point                                                                    |
| ----------------------------------------- | ------------------------------------------------------------------------------ |
| Protocol, VAD and host process lifecycle  | [Protocol reference](docs/protocol.md)                                         |
| Model adapters and engine ownership       | [Architecture](docs/architecture.md)                                           |
| Build, versioning and GitHub Releases     | [Release guide](docs/releasing.md)                                             |
| Benchmark commands and metric definitions | [Benchmark guide](docs/benchmark.md)                                           |
| Model/runtime acceptance and known limits | [Validation](docs/validation.md)                                               |
| Benchmark results                         | [Summary](docs/benchmark-results.md), [detailed report (Chinese)](测试报告.md) |
| Python streaming client                   | [examples/stream.py](examples/stream.py)                                       |

Some detailed guides are currently in Chinese. Python is needed for scripts and tests, not for running the released server. Real-model tests require caller-supplied GGUF weights and speech audio; a successful build alone does not establish recognition quality.

## License and acknowledgements

HaoLive.cpp is [MIT licensed](LICENSE). Third-party sources, adapted equations, attribution and license texts are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and available through `haolive-server --licenses`.

Thanks to [CrispASR](https://github.com/CrispStrobe/CrispASR) and its contributors for their open-source speech work. CrispASR's model implementations informed the adapters in this project; its executable is not a runtime dependency. Thanks also to ggml, IXWebSocket, cpp-httplib and nlohmann/json.
