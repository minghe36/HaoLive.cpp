# HaoLive.cpp

**轻量、在本地运行的 ASR 推理引擎，单文件交付。**

[English](README.md) | [简体中文](README.zh-CN.md)

专注于实时 ASR 推理的 C++ 引擎，仅提供最实用的实时 ASR 模型。不依赖外部二进制程序，体积小、易于使用，提供低延迟的本地推理，适用于实时字幕、实时翻译等场景。

| 能力         | 内容                                                        |
| ------------ | ----------------------------------------------------------- |
| 语音识别     | 四种 GGUF 模型适配器，支持中间文本与最终结果                |
| 通信协议     | 双向 WebSocket、HTTP 会话、可选 SSE token 流                |
| 集成方式     | C++ API、C ABI、独立 Server 进程                            |
| 语音活动检测 | 独立运行的 CPU FireRed VAD                                  |
| 交付形式     | 单个可执行文件，模型权重单独提供                            |
| 运行依赖     | 仅操作系统库；无需 Python、FFmpeg、ONNX Runtime 或 LibTorch |

## 下载

从 [GitHub Releases](https://github.com/minghe36/HaoLive.cpp/releases/latest) 下载预编译程序。每个压缩包仅含一个可执行文件，并提供独立的 SHA-256 校验文件。

| 平台                | 压缩包                                  | ASR 后端   | 运行要求                                      |
| ------------------- | --------------------------------------- | ---------- | --------------------------------------------- |
| macOS Apple Silicon | `HaoLive-<version>-macos-arm64.tar.gz`  | CPU、Metal | macOS 13+                                     |
| macOS Intel         | `HaoLive-<version>-macos-x86_64.tar.gz` | CPU、Metal | macOS 13+；SSE4.2、AVX2、FMA、F16C、BMI2      |
| Windows x64         | `HaoLive-<version>-windows-x64.zip`     | CPU        | 支持 SSE4.2、AVX2、FMA、F16C、BMI2 的 x64 CPU |

Windows 程序静态链接 MSVC 运行库。Metal shader、所支持的 Moonshine 词表和第三方许可文本均嵌入程序，模型权重不包含在压缩包中。

## 快速开始

下载并解压程序，再下载 [X-ASR 模型](#模型)。在解压目录执行：

**macOS**

```sh
./haolive-server --model ./models/x-asr-zh-en-q8_0.gguf --backend cpu
```

**Windows PowerShell**

```powershell
.\haolive-server.exe --model .\models\x-asr-zh-en-q8_0.gguf --backend cpu
```

以上模型路径仅为示例，请传入实际 GGUF 文件位置。Server 默认监听 `http://127.0.0.1:9797`。

在另一个终端提交符合要求的 WAV 文件：

```sh
curl -H "Content-Type: audio/wav" --data-binary @speech.wav \
  http://127.0.0.1:9797/v1/transcribe
```

Windows PowerShell 中请使用 `curl.exe` 执行同一 HTTP 请求。

| 音频属性              | 要求                                                      |
| --------------------- | --------------------------------------------------------- |
| 采样率                | 16,000 Hz                                                 |
| 声道                  | 单声道                                                    |
| HTTP / WebSocket 编码 | PCM16 little-endian；HTTP 也接受包含该格式音频的 RIFF WAV |
| C++ / C ABI 编码      | 归一化到 `[-1, 1]` 的 float32 PCM                         |
| 格式转换              | 重采样、混音、压缩音视频解码由调用方完成                  |

常用命令：`--help`、`--version`、`--list-backends`、`--licenses`。

## 模型

以下模型均经过浩叔测试，推荐使用。

权重来自 [cstr Hugging Face 模型列表](https://huggingface.co/cstr/models)。引擎通过 GGUF 架构元数据选择适配器，不依赖文件名。每个 Server 进程加载一个 ASR 模型；同时使用多个模型时，请启动多个进程并分配不同端口。

| 模型                                                                                                | 支持语言                                                                                                                            | 模型大小  | 量化版本 | 下载                                                                                                                                          |
| --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | --------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| [X-ASR 中英双语](https://huggingface.co/cstr/x-asr-zh-en-GGUF)                                      | 中文、英文                                                                                                                          | ≈ 168 MB  | Q8_0     | [GGUF](https://huggingface.co/cstr/x-asr-zh-en-GGUF/resolve/main/x-asr-zh-en-q8_0.gguf?download=true)                                         |
| [Confucius4-R2T2](https://huggingface.co/cstr/confucius4-r2t2-GGUF)                                 | 重点优化中文、英文；兼容[其他语言](https://huggingface.co/netease-youdao/Confucius4-R2T2#supported-languages)                       | ≈ 1.49 GB | Q4_K     | [GGUF](https://huggingface.co/cstr/confucius4-r2t2-GGUF/resolve/main/confucius4-r2t2-q4_k.gguf?download=true)                                 |
| [Moonshine Streaming Medium（英文）](https://huggingface.co/cstr/moonshine-streaming-medium-GGUF)   | 英文                                                                                                                                | ≈ 191 MB  | Q4_K     | [GGUF](https://huggingface.co/cstr/moonshine-streaming-medium-GGUF/resolve/main/moonshine-streaming-medium-q4_k.gguf?download=true)           |
| [Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/cstr/nemotron-3.5-asr-streaming-0.6b-GGUF) | 多语言：中文、英文、日语、韩语等；详见[语言分级](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b#supported-languages) | ≈ 686 MB  | Q8_0     | [GGUF](https://huggingface.co/cstr/nemotron-3.5-asr-streaming-0.6b-GGUF/resolve/main/nemotron-3.5-asr-streaming-0.6b-q8_0.gguf?download=true) |

点击模型名称可打开 Hugging Face 模型主页。大小为 GGUF 下载文件的近似值，使用十进制 MB/GB，不代表运行内存占用。

语言范围依据模型卡标注。Nemotron 的部分语言需要微调，具体请查看链接中的语言分级说明。

| 适配器    | 流式方式                                | 实现要点                                                                |
| --------- | --------------------------------------- | ----------------------------------------------------------------------- |
| X-ASR     | 原生 160ms Zipformer2 / Transducer 分块 | 从 GGUF 读取匹配的几何配置及左上下文缓存                                |
| Confucius | 默认 160ms 调度，支持 token 回调        | 重编码累计音频，回退最近 4 个假设 token；单轮草稿最多生成 32 个新 token |
| Moonshine | 按 `--step-ms` 间隔处理累计波形         | 重算编码器状态，中间文本允许修订                                        |
| Nemotron  | 带缓存的 FastConformer / RNN-T 分块     | 调度受模型分块大小约束，保留一块等待 STFT 右边界稳定                    |

调度间隔不等于首字上屏延迟。对于较慢模型，增大 `--step-ms` 可以减少重复计算。Confucius 使用比上游自适应 R2T2 调度器更精简的策略，不保证任意分块方式均得到相同最终文本。适配器细节见 [架构说明](docs/architecture.md)。

### 语音活动检测模型

| 模型                                                        | 用途                   | 模型大小  | 精度 | 下载                                                                                                         |
| ----------------------------------------------------------- | ---------------------- | --------- | ---- | ------------------------------------------------------------------------------------------------------------ |
| [FireRed VAD](https://huggingface.co/cstr/firered-vad-GGUF) | 不限语言的语音活动检测 | ≈ 2.36 MB | F32  | [firered-vad.gguf](https://huggingface.co/cstr/firered-vad-GGUF/resolve/main/firered-vad.gguf?download=true) |

FireRed VAD 使用 CPU 判断音频中是否存在语音，不生成转录文字。既可独立运行，也可通过 `--vad-model` 与任意 ASR 模型一起加载。调用方决定如何使用检测结果，Server 不会自动切分 ASR 语句或丢弃音频。

```sh
# 仅运行 VAD
./haolive-server --vad-model ./models/firered-vad.gguf --backend cpu

# 同时运行 ASR 和 VAD
./haolive-server --model ./models/x-asr-zh-en-q8_0.gguf \
  --vad-model ./models/firered-vad.gguf --backend cpu
```

通过 `POST /v1/vad` 调用语音检测，参数和返回字段见 [VAD 接口参考（英文）](docs/protocol.md#independent-firered-vad)。

## Server 参数

| 参数                | 默认值               | 用途                                                         |
| ------------------- | -------------------- | ------------------------------------------------------------ |
| `--model`           | 无                   | ASR GGUF 文件；仅使用 VAD 时可省略                           |
| `--vad-model`       | 无                   | 可选的 FireRed VAD GGUF 文件                                 |
| `--backend`         | `cpu`                | 请求使用的 ASR 后端；通过 `--list-backends` 查看本机可用设备 |
| `--host` / `--port` | `127.0.0.1` / `9797` | HTTP 监听地址；端口 `0` 表示由系统分配                       |
| `--threads`         | `4`                  | CPU 工作线程数                                               |
| `--step-ms`         | `160`                | 中间结果的推理调度间隔，具体行为由模型决定                   |
| `--max-audio-ms`    | `30000`              | 每句话的音频时长上限                                         |
| `--max-tokens`      | `512`                | 生成上限，最终解码耗尽预算时报错                             |
| `--token`           | 无                   | 启用 `Authorization: Bearer <token>` 鉴权                    |
| `--parent-pid`      | 无                   | 仅 POSIX：直接父进程退出后自动退出                           |

macOS 硬件支持时，可传入 `--backend metal` 使用 GPU。请求的后端不可用会报错，个别 GPU 不支持的算子可调度到 CPU。项目提供 CUDA/Vulkan 构建开关，但当前发布流程未验证这两个后端。

## 接口

| 接口        | 入口                                                                   | 用途                                            |
| ----------- | ---------------------------------------------------------------------- | ----------------------------------------------- |
| 健康检查    | `GET /health`                                                          | API 版本、已加载能力及 WebSocket 端口           |
| 模型信息    | `GET /v1/models`                                                       | 当前加载模型的元数据                            |
| 单次转录    | `POST /v1/transcribe`                                                  | 转录一次 PCM/WAV 请求                           |
| HTTP 会话   | `POST /v1/sessions`                                                    | 创建流式语句会话                                |
| 追加 / 结束 | `POST /v1/sessions/{id}/audio`                                         | 追加不重叠音频；`final=true` 刷新尾部并释放会话 |
| 取消会话    | `DELETE /v1/sessions/{id}`                                             | 释放未结束的 HTTP 会话                          |
| WebSocket   | `/health` 返回端口上的 `/v1/realtime`                                  | 双向持续发送音频、接收转录更新                  |
| 独立 VAD    | `POST /v1/vad`                                                         | 不创建 ASR 会话的语音检测                       |
| 嵌入式 API  | [C++](include/haolive/haolive.hpp)、[C ABI](include/haolive/haolive.h) | 无需 HTTP Server 即可集成                       |

### 实时 WebSocket

| 步骤        | 客户端行为                                                                                                      |
| ----------- | --------------------------------------------------------------------------------------------------------------- |
| 1. 获取端口 | 调用带鉴权的 `GET /health`，读取 `websocket_port`                                                               |
| 2. 建立连接 | 连接 `ws://127.0.0.1:<websocket_port>/v1/realtime`；按配置提供 bearer token，并传入 `X-Session-Id`（1–64 字节） |
| 3. 持续收发 | 发送 PCM16LE 二进制帧，同时接收 JSON 更新                                                                       |
| 4. 结束语句 | 所有音频发送完成后发送 `{"event":"final"}`；等待 `type: "final"`，再关闭连接                                    |
| 5. 下一句   | 建立新的连接                                                                                                    |

发送与接收应独立执行，Server 的音频读取不会等待推理结束。每条更新包含**当前语句的完整文本**，客户端应替换上一条假设，不要追加拼接。`updated=false` 时保留已有文本；同一轮推理内的 token 回调共享 sequence。

### HTTP 会话

```sh
curl -X POST http://127.0.0.1:9797/v1/sessions

# 将 SESSION_ID 替换为返回的 session_id。
curl -H "Content-Type: audio/pcm" --data-binary @chunk.pcm \
  http://127.0.0.1:9797/v1/sessions/SESSION_ID/audio

curl -H "Content-Type: audio/pcm" --data-binary '' \
  'http://127.0.0.1:9797/v1/sessions/SESSION_ID/audio?final=true'
```

Confucius 的音频请求还支持 `stream=true`，通过 SSE 接收 token 进度。请按会话 ID 和 sequence 过滤过期更新。不要自动重试音频追加请求，否则可能重复输入采样。本协议不兼容 OpenAI API。

| 限制 / 规则   | 内容                                                                              |
| ------------- | --------------------------------------------------------------------------------- |
| 并发语句      | 最多 8 个 HTTP 会话，最多 8 个 WebSocket 连接                                     |
| 单句音频      | 最多 30 秒，同时受 `--max-audio-ms` 限制                                          |
| 请求大小      | HTTP 单次请求、WebSocket 输入帧或合并分片最多 1 MiB                               |
| HTTP 空闲清理 | 空闲 120 秒后，在后续会话操作中惰性清理                                           |
| 断句          | 调用方显式发送 final 并开始下一句；无自动 VAD 断句                                |
| 文本约定      | 中间结果为可修订的完整文本，最终文本固定                                          |
| 错误返回      | 结构化 code/message；HTTP 400、401、404、413、415、429、500，能力未启用时返回 503 |

响应字段、SSE 事件、VAD 示例和进程握手见 [协议参考（英文）](docs/protocol.md)。

## 从源码构建

依赖源码已固定版本并随仓库提供，模型文件单独下载。发布脚本依次配置、编译、运行 CTest、检查二进制依赖，并将压缩包及校验文件输出到 `dist/`。

| 目标平台    | 前置环境                                               | 构建命令                                        |
| ----------- | ------------------------------------------------------ | ----------------------------------------------- |
| macOS arm64 | CMake 3.24+、Python 3.10+、Xcode Command Line Tools    | `./scripts/build-macos.sh --arch arm64`         |
| macOS Intel | 同上；测试环境需能运行目标架构                         | `./scripts/build-macos.sh --arch x86_64`        |
| Windows x64 | CMake 3.24+、Python 3.10+、Visual Studio 2022 C++ 工具 | `./scripts/build-windows.ps1 -Architecture x64` |

开发构建：

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --output-on-failure
```

| CMake 开关                              | 用途                                      |
| --------------------------------------- | ----------------------------------------- |
| `HAOLIVE_METAL=ON/OFF`                  | Metal 后端，macOS 默认开启                |
| `HAOLIVE_VULKAN=ON` / `HAOLIVE_CUDA=ON` | 需要对应 SDK/toolkit 的可选后端，默认关闭 |
| `HAOLIVE_BUILD_SHARED=ON`               | 构建可选的 C ABI 共享库，Server 不依赖它  |
| `HAOLIVE_BUILD_TESTS=ON/OFF`            | 开启或关闭测试目标                        |

静态库为 `libhaolive.a` / `haolive.lib`。通过 CMake `add_subdirectory(...)` 与 `target_link_libraries(your_app PRIVATE haolive)` 接入。C++ engine 必须比 session 存活更久；C ABI session 保留 engine 所有权。单个 engine 的推理串行执行，各会话的语句状态独立。

## 开发与文档

| 主题                             | 入口                                                       |
| -------------------------------- | ---------------------------------------------------------- |
| 协议、VAD 与宿主进程生命周期     | [协议参考（英文）](docs/protocol.md)                       |
| 模型适配器与引擎所有权           | [架构说明](docs/architecture.md)                           |
| 构建、版本管理与 GitHub Releases | [构建与发布](docs/releasing.md)                            |
| 基准测试命令与指标定义           | [基准测试说明](docs/benchmark.md)                          |
| 模型及运行时验收、已知限制       | [验收记录](docs/validation.md)                             |
| 基准测试结果                     | [汇总](docs/benchmark-results.md)、[详细报告](测试报告.md) |
| Python 流式客户端                | [examples/stream.py](examples/stream.py)                   |

Python 用于脚本和测试，运行发布版 Server 无需安装 Python。真实模型测试需由调用方提供 GGUF 权重与语音文件，构建成功不能代替识别质量验证。

## 许可与开源致谢

HaoLive.cpp 使用 [MIT 许可证](LICENSE)。第三方源码、适配公式、来源和完整许可文本见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，也可运行 `haolive-server --licenses` 查看。

感谢 [CrispASR](https://github.com/CrispStrobe/CrispASR) 项目及其贡献者的开源语音工作。本项目模型适配器参考了 CrispASR 的实现，运行时不依赖其可执行程序。同时感谢 ggml、IXWebSocket、cpp-httplib 和 nlohmann/json。
