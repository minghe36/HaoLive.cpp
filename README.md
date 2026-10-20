# HaoLive.cpp

轻量实时 ASR 推理库与 HTTP server，C++17 / MIT。依赖 ggml、静态链接的 IXWebSocket 和两个单头文件
HTTP/JSON 库；不依赖 CrispASR、FFmpeg、Python、ONNX Runtime 或 LibTorch。
Python 仅用于可选的测试、基准测试和示例客户端。

支持的 MVP 模型：

| GGUF architecture | 模型 | 推理方式 |
| --- | --- | --- |
| `xasr` | X-ASR zh/en Q8_0 | 原生 Zipformer2 分块编码与 Transducer 解码 |
| `qwen3asr` | Confucius4-R2T2 Q4_K | Qwen3 音频编码、KV 解码、最近 4 token 回退的增量假设 |
| `moonshine_streaming` | Moonshine Streaming Medium Q4_K（英文） | 波形前端、滑动窗口编码器、带 KV 缓存的自回归解码；累计音频重算 partial |
| `nemotron` | Nemotron 3.5 ASR Streaming 0.6B Q8_0 | 因果下采样、FastConformer 原生分块缓存与 RNN-T 状态；自动语言 prompt |

模型类型读取 GGUF 元数据，不依赖文件名。一次 server 进程加载一个模型；启动另一个
进程并使用不同端口即可同时提供多个模型。模型路径由启动参数提供，不自动下载。

## 模型下载

以下权重来自 Hugging Face 的 [cstr 模型列表](https://huggingface.co/cstr/models)，
量化版本与本文启动示例一致。下载后放入 `models/`，或通过 `--model` 指定实际文件路径。

| 模型 | GGUF 文件下载 | 模型主页 |
| --- | --- | --- |
| X-ASR zh/en Q8_0 | [x-asr-zh-en-q8_0.gguf](https://huggingface.co/cstr/x-asr-zh-en-GGUF/resolve/main/x-asr-zh-en-q8_0.gguf?download=true) | [cstr/x-asr-zh-en-GGUF](https://huggingface.co/cstr/x-asr-zh-en-GGUF) |
| Confucius4-R2T2 Q4_K | [confucius4-r2t2-q4_k.gguf](https://huggingface.co/cstr/confucius4-r2t2-GGUF/resolve/main/confucius4-r2t2-q4_k.gguf?download=true) | [cstr/confucius4-r2t2-GGUF](https://huggingface.co/cstr/confucius4-r2t2-GGUF) |
| Moonshine Streaming Medium Q4_K | [moonshine-streaming-medium-q4_k.gguf](https://huggingface.co/cstr/moonshine-streaming-medium-GGUF/resolve/main/moonshine-streaming-medium-q4_k.gguf?download=true) | [cstr/moonshine-streaming-medium-GGUF](https://huggingface.co/cstr/moonshine-streaming-medium-GGUF) |
| Nemotron 3.5 ASR Streaming 0.6B Q8_0 | [nemotron-3.5-asr-streaming-0.6b-q8_0.gguf](https://huggingface.co/cstr/nemotron-3.5-asr-streaming-0.6b-GGUF/resolve/main/nemotron-3.5-asr-streaming-0.6b-q8_0.gguf?download=true) | [cstr/nemotron-3.5-asr-streaming-0.6b-GGUF](https://huggingface.co/cstr/nemotron-3.5-asr-streaming-0.6b-GGUF) |

## 构建（macOS）

需要 CMake 3.24+ 和 Xcode Command Line Tools。依赖源码已固定版本并包含在仓库中，
构建过程不需要下载依赖。模型权重不纳入 Git。

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DHAOLIVE_METAL=ON -DHAOLIVE_VULKAN=OFF -DHAOLIVE_CUDA=OFF \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0
cmake --build build -j 8
ctest --test-dir build --output-on-failure
```

交付程序为 **`build/bin/haolive-server` 单文件**。复制该文件和所选 GGUF 权重即可运行，
不需要安装第三方动态库、Python、FFmpeg 或附带 shader/tokenizer。Moonshine 的词表已
嵌入程序，模型目录中的 `tokenizer.bin` 不再参与运行。macOS 自带的系统框架仍由系统提供。
开源许可文本也嵌入二进制，可运行 `haolive-server --licenses` 查看。
本机已验证的单文件副本为 `build/single-file/haolive-server`，压缩包为
`build/HaoLive-macos-arm64-single-file.tar.gz`（仅含该可执行文件）。

开发时同时生成静态库 `build/libhaolive.a`，公开头文件位于 `include/haolive/`。
C ABI/FFI 共享库为可选开发产物，通过 `-DHAOLIVE_BUILD_SHARED=ON` 构建，server 不依赖它。
server 静态链接第三方库，Metal shader 嵌入二进制。运行不需要构建目录中的 dylib 或
shader 文件；需要操作系统框架。可使用 `cmake --install build --prefix dist` 打包。

纯 CPU 构建：增加 `-DHAOLIVE_METAL=OFF`。跨平台构建开关为
`HAOLIVE_VULKAN`、`HAOLIVE_CUDA`，默认关闭；需要对应平台 SDK/toolkit。
本次 Mac MVP 验收范围是 CPU/Metal，未验证 Vulkan/CUDA 的构建与运行。

## 平台构建与版本发布

macOS 使用 `./scripts/build-macos.sh`，Windows 使用 `./scripts/build-windows.ps1`。
脚本自动编译、运行测试并打包到 `dist/`，产物包含版本号和平台架构，并生成 SHA-256 校验文件。

```sh
# 创建版本提交和 annotated tag，推送后由 GitHub Actions 构建并发布
python3 scripts/release.py create 0.1.1 --notes-file /path/to/release-notes.md
```

版本号统一读取 `VERSION`，更新日志位于 `changelog/<版本号>.md`。上述版本为调用示例。
工作流在 macOS arm64、macOS Intel 和 Windows x64 全部构建成功后，上传二进制程序包
并公开 GitHub Release。完整参数、前置条件和失败恢复见 [构建与发布](docs/releasing.md)。

## 启动

```sh
./build/bin/haolive-server --list-backends
./build/bin/haolive-server --model ./models/x-asr-zh-en-q8_0.gguf --backend metal
# 或：
./build/bin/haolive-server --model ./models/confucius4-r2t2-q4_k.gguf --backend cpu
./build/bin/haolive-server --model ./models/moonshine-streaming-medium-q4_k.gguf --backend metal
./build/bin/haolive-server --model ./models/nemotron-3.5-asr-streaming-0.6b-q8_0.gguf --backend metal
```

默认监听 `127.0.0.1:8080`。可配置 `--host`、`--port`、`--threads`、`--step-ms`、
`--max-audio-ms`、`--max-tokens`、`--token`。指定的 backend 不可用时启动失败，
不会静默换成其他 GPU。GPU 不支持的个别算子由 ggml 调度到 CPU。

音频输入统一为 **16 kHz、单声道、PCM16 little-endian**，支持原始 PCM 或包含
该格式音频的 RIFF WAV。采集端负责重采样/混音。服务端不解码压缩音视频。

## 双向实时音频

桌面实时采集使用长连接 WebSocket，不逐块等待 HTTP 推理请求。
认证 `/health` 返回 `websocket_port`（由操作系统分配）；连接
`ws://127.0.0.1:<websocket_port>/v1/realtime`，请求头携带
`Authorization: Bearer <token>` 和长度不超过 64 的 `X-Session-Id`。
鉴权后服务端返回 `{"event":"ready","api_version":1,"session_id":"..."}`。

客户端持续发送二进制 PCM16LE 帧；采集读入与推理运行在不同线程，推理期间
继续接收后续音频并合并待处理样本。客户端读取与写入也应独立运行。字幕为
`partial` / `result` JSON 帧，字段包含完整 `text`、`session_id`、`sequence`、
`audio_ms`、`type` 和 `updated`。同一轮 token 回调共享 sequence，不能当作
字符串 delta 拼接。`updated=false` 结果不清空已有草稿。

一句结束后，在同一连接、所有 PCM 之后发送文本帧 `{"event":"final"}`；
收到 `type=final` 后关闭连接，下一句建立新连接。失败以 `event=error` 返回。
每句最多 30 秒音频，最多 8 个连接；输入帧及合并分片上限 1 MiB。
长时间推理不会阻止接收后续 PCM，但模型处理速度仍决定实际字幕延迟。

## HTTP API v1

```sh
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models
curl -H 'Content-Type: audio/wav' --data-binary @speech.wav \
  http://127.0.0.1:8080/v1/transcribe
```

实时输入使用持久会话，按顺序上传 PCM 分块。HTTP keep-alive 可复用连接。默认每 320 ms 新音频触发一次 partial 推理。
Confucius 可在 audio 请求上增加 `stream=true`，以 `text/event-stream` 接收
解码过程中的 `partial`（完整当前假设）、本轮完成时的 `result` 或失败时的
`error` 事件；草稿按 token 生成即推送。其它模型目前按原生分块返回 result。
客户端应以 session_id 和 sequence 过滤过期进度，partial 不能当作追加字符串。
Confucius 单轮草稿最多生成 32 个新 token 后返回，后续音频继续识别；final
仍使用完整 max-tokens 预算并在耗尽时报告错误。

```sh
# 1. 创建会话，读取返回的 session_id
curl -X POST http://127.0.0.1:8080/v1/sessions

# 2. 重复提交连续、不重叠的音频分块（替换 SESSION_ID）
curl -H 'Content-Type: audio/pcm' --data-binary @chunk.pcm \
  http://127.0.0.1:8080/v1/sessions/SESSION_ID/audio

# 3. 一句结束时显式 flush，可同时提交最后一块音频，也可提交空 body
curl -H 'Content-Type: audio/pcm' --data-binary '' \
  'http://127.0.0.1:8080/v1/sessions/SESSION_ID/audio?final=true'
```

四个模型共用响应结构：

```json
{
  "api_version": 1,
  "session_id": "...",
  "model": "xasr",
  "sequence": 1,
  "type": "partial",
  "text": "当前完整识别文本",
  "updated": true,
  "audio_ms": 800
}
```

`text` 始终为本句完整文本，客户端**替换**上一条 partial，不拼接。
Confucius、Moonshine partial 允许修订；final 后该句才固定。`sequence` 为会话内递增响应序号。
`updated=false` 表示本次没有新文本；不代表该音频未接收。不要自动重试同一音频请求，
否则会重复输入；连接结果不确定时重建会话。final 响应后 server 释放会话。

`DELETE /v1/sessions/{id}` 取消会话。默认最多 8 个会话，每句最多 30 秒，
120 秒无操作后在下一次会话请求时清理。请求体最多 1 MiB。
长时间采集由客户端划分语句、显式 final 并创建下一句；MVP 无 VAD 自动断句。
Confucius 默认每累计 160 ms 新音频触发一次推理，X-ASR 按模型原生 160 ms 块推进。
X-ASR 从 GGUF 的 `xasr.chunk_ms` 中选择 160 ms，并读取对应的 `xasr.left_context_frames`；当前权重为 8 个编码帧、29 帧特征窗口和 96 帧左上下文，各层缓存按下采样比例构建。缺失该档位或尺寸不兼容时明确报错。Confucius 的 C++、C ABI 与 Server 默认 `step_ms` 均为 160；该参数是推理调度间隔，不是端到端出字延迟保证。
Confucius 会重新编码当前句累计音频，并回退最近 4 token；这不是完整复制上游
R2T2 调度器，也不承诺任意分块方式得到完全相同的 final。
Moonshine 默认每累计 800 ms 重算本句，partial 可修订，final 重新计算全部音频；此版
未实现增量复用 Moonshine encoder。内置词表对应本次指定的 Medium 英文 checkpoint；
其他 Moonshine 变体需要在 GGUF 中携带 `tokenizer.ggml.tokens`，或显式增加配套词表。
Nemotron 使用首个 attention preset（当前权重 L=56、R=3，每块 320 ms），会话保留
每层 attention/卷积历史和两层 LSTM 状态。默认按 `--step-ms 800` 调度，非 final
保留一块等待 STFT 右边界稳定，final 排空尾部。语言 prompt 从权重表读取 `auto`，
无需额外语言检测模型。词数限制通过 `--max-tokens` 统一控制。

配置 `--token SECRET` 后，所有请求都要加 `Authorization: Bearer SECRET`。
错误使用 `{"error":{"code":"...","message":"..."}}`，常见状态码为
400 参数/音频错误、401 未授权、404 会话不存在/已结束、413 音频上限、
415 不支持的音频、429 会话上限、500 推理失败。推理失败的会话会被释放。
本版提供 HTTP 分块会话 API，未实现 WebSocket/OpenAI 协议兼容层。

## 独立 VAD：快速判断是否有语音

使用 `firered-vad.gguf`，无需加载任何 ASR 模型。模型在启动/创建 detector 时加载一次，
后续请求复用权重。FireRed VAD 固定走 CPU；macOS 使用系统 Accelerate，不新增第三方依赖。
可单独启动，也可在原 ASR 启动命令中追加 `--vad-model`：

```sh
./build/single-file/haolive-server --vad-model ./models/firered-vad.gguf

curl -H 'Content-Type: audio/wav' --data-binary @speech.wav \
  'http://127.0.0.1:8080/v1/vad?threshold=0.5&min_speech_ms=100'
```

输入同样支持 16 kHz 单声道 PCM16LE 或该格式的 WAV，每次最多 30 秒。
VAD 是**独立、无状态的音频窗口判断**，不创建 ASR 会话，也不修改 ASR 的音频或文本。
用于实时输入时，建议调用方每隔 200–500 ms 提交最近 1–2 秒的窗口；相邻 VAD 窗口可以重叠。
ASR 会话仍接收连续、不重叠的分块。本版不自动根据 VAD 丢弃 ASR 输入或结束语句。

响应示例：

```json
{
  "api_version": 1,
  "model": "firered-vad",
  "backend": "cpu",
  "is_speech": true,
  "sufficient_audio": true,
  "max_probability": 0.9984,
  "mean_probability": 0.6827,
  "speech_ms": 670,
  "audio_ms": 1000,
  "frames": 98
}
```

- `is_speech`：存在连续至少 `min_speech_ms` 的帧，且每帧概率均不小于 `threshold`。
  默认阈值 0.5、连续 100 ms；阈值必须在 (0,1)，持续时间为 1–30000 ms。
- `max_probability` / `mean_probability`：帧级语音概率的最大值/均值，不是校准过的整段准确率。
- `speech_ms`：所有高于阈值的帧数乘以 10 ms，可包含不连续的片段；`is_speech` 使用连续帧判断。
- `sufficient_audio`：是否含至少一帧完整的 25 ms 音频。空输入或不足 25 ms 返回 false，
  不补零伪造完整帧。帧移 10 ms，尾部不足一帧的样本不参与推理。

短窗口可能因上下文不足或语音持续时间未达标返回 false。该接口检测语音活动，不能区分
真人、录音和合成语音，也不替代说话人识别。没有启用 VAD 时 `/v1/vad` 返回
503 `vad_unavailable`；VAD-only 模式下 ASR 接口返回 503 `asr_unavailable`。
`/health` 和 `/v1/models` 会额外报告 VAD 是否启用。鉴权和请求体上限沿用现有 server 设置。

C++ 独立方法：

```cpp
haolive::VadOptions options;
options.model_path = vad_model_path; // 来自调用方配置
const auto vad = haolive::load_vad(options);
const auto result = vad->detect(samples.data(), samples.size());
if (result.is_speech) { /* 当前窗口有语音 */ }
// 可选参数：vad->detect(samples.data(), samples.size(), {0.5f, 100});
```

库接口输入为归一化 float32 单声道 16 kHz。C ABI 使用
`hl_default_vad_options` → `hl_vad_create` → `hl_vad_detect` → `hl_vad_free`，
不需要 `hl_engine`。`hl_vad_detect` 的 params 可以为 NULL 使用默认值；结果结构不含借用字符串。
同一个 detector 支持并发 detect，释放时需确保没有正在执行的调用。

真实模型验收与 VAD 延迟测量：

```sh
./build-single/haolive-vad-tests ./models/firered-vad.gguf ./speech-en.wav ./speech-zh.wav
python3 tests/vad_integration.py --server ./build/single-file/haolive-server \
  --model ./models/firered-vad.gguf --audio ./speech-en.wav --audio ./speech-zh.wav \
  --iterations 10 --output ./build/vad-report.json
```

脚本将二进制复制到临时目录后测试独立运行，验证语音/静音/白噪声/纯音、输入校验、
并发隔离、PCM/WAV 一致性，并测量 0.5/1/5 秒窗口（预热 1 次，正式默认 10 次）的 HTTP 延迟。
可增加 `--asr-model MODEL.gguf --backend metal` 验证 ASR 和 VAD 同时启用。

## 嵌入其他程序

C++ 接口见 `include/haolive/haolive.hpp`；C ABI 见 `include/haolive/haolive.h`。
使用 C ABI 可以为 Rust、Python、Go 等编写绑定，调用流程统一：

```cpp
haolive::Options options;
options.model_path = model_path; // 来自调用者配置
options.backend = "metal";
auto engine = haolive::load(options);
auto session = engine->create_session();
auto partial = session->accept(samples.data(), samples.size(), false);
auto final = session->accept(nullptr, 0, true);
```

库接口音频为归一化 `float32 [-1,1]`、16 kHz 单声道。调用同步返回，调用方管理音频
采集与背压。每个 engine 内部串行推理，会话音频、编码缓存和文本独立。C++ engine
必须比 session 活得更久；C ABI session 会保留 engine 生命周期。C ABI 不抛异常，
使用返回值与 `hl_last_error()` 获取错误；字符串所有权见头文件。

共享库可直接由 FFI 加载，例如 Python 的 `ctypes.CDLL(".../libhaolive.dylib")`。
分块客户端示例：`python3 examples/stream.py ./speech.wav --url http://127.0.0.1:8080`。

从源码嵌入时通过 CMake `add_subdirectory(...)` 和 `target_link_libraries(your_app PRIVATE haolive)`
连接，ggml 的依赖与框架会一同传递。扩展模型实现见 [架构说明](docs/architecture.md)。

## 测试

```sh
ctest --test-dir build --output-on-failure
# C ABI 生命周期测试需配置 HAOLIVE_BUILD_SHARED=ON
./build/haolive-model-tests ./models/x-asr-zh-en-q8_0.gguf cpu
python3 tests/integration.py --server build/bin/haolive-server \
  --model ./models/x-asr-zh-en-q8_0.gguf --backend cpu \
  --audio ./speech.wav --expected '期望文本的一部分'
```

真实模型测试接收外部 WAV，不依赖 CrispASR 的安装位置。对四个模型分别以 `cpu` /
`metal` 执行；覆盖识别、partial/final、WAV/PCM 一致性、会话隔离、取消、鉴权、
输入校验和资源上限。回归必须验证真实语音有正文，防止把空文本误报为成功。
可用 `python3 tests/standalone.py --server build/bin/haolive-server --model ./models/moonshine-streaming-medium-q4_k.gguf --audio ./speech.wav` 验证程序复制到空目录后独立推理；测试同时审计动态依赖。
MVP 的测试记录和限制见 [验收记录](docs/validation.md)。

## Bench 基准测试

`bench/bench.py` 使用 Python 标准库，自动启动并回收 server，顺序测试模型与 backend。
默认每项 warmup 1 次、正式测量 3 次，输出完整 `results.json`、汇总 `summary.csv`
和 server 日志。输入音频格式与 server 相同，每段最多 30 秒。

```sh
python3 bench/bench.py --server ./build/bin/haolive-server \
  --model ./models/x-asr-zh-en-q8_0.gguf \
  --model ./models/confucius4-r2t2-q4_k.gguf \
  --model ./models/moonshine-streaming-medium-q4_k.gguf \
  --model ./models/nemotron-3.5-asr-streaming-0.6b-q8_0.gguf \
  --backend cpu --backend metal \
  --audio ./speech.wav --seconds 5 \
  --iterations 3 --warmup 1 --chunk-ms 160 \
  --output ./build/bench-001
```

`--audio` 可重复添加。默认同时测离线与流式；流式按音频时间模拟实时输入。
吞吐测试使用 `--stream-mode fast`，只测某一项使用 `--mode offline` 或 `--mode stream`。
输出目录必须为新目录，避免覆盖已有结果。

指标包括启动就绪时间、RTF、首次非空文本/partial 延迟、final 请求耗时、采集结束到
final 的延迟、HTTP 请求 p50/p95 和输入积压。计时定义及用法见
[基准测试说明](docs/benchmark.md)。

依赖来源与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 独立宿主进程集成

宿主可传 `--port 0`，由系统直接分配并绑定端口。绑定完成后 stdout 首行输出
`{"event":"ready","api_version":1,"port":实际端口}`；宿主仍需调用带鉴权的
`/health` 确认能力与 API 版本。使用 `--token` 设置每次启动独立的随机令牌。
POSIX 系统可同时传 `--parent-pid PID`（直接父进程 PID），父进程崩溃或被杀后
server 自动退出；独立命令行使用时无需该参数。stderr 为诊断日志，不要把它当作握手。
macOS app 集成构建指定 deployment target 13.0；实际旧系统运行仍需单独验收。

## 开源致谢

感谢 [CrispASR](https://github.com/CrispStrobe/CrispASR) 项目及其贡献者在开源语音识别领域的工作与分享。
