# Third-party notices

HaoLive.cpp is MIT licensed. Model weights have their own licenses and are not
included in the source distribution. No model files are embedded in the binary.

## ggml

Vendored from https://github.com/CrispStrobe/ggml at
`2f5a80d258c46e6ac8eee95f1328c0f58376d7ee`, a fork of ggml-org/ggml.
Archive SHA-256:
`8e06f263fffea53b169c7839b3b4456451e0b0a3d24d2732a63daa0d5a42810e`.
License: MIT, retained in `third_party/ggml/LICENSE` and individual source files.
This pinned fork provides the GGUF quantization and Zipformer operations used
by the supplied models. Only configured backends are compiled.

## cpp-httplib

https://github.com/yhirose/cpp-httplib, tag `v0.18.7`, MIT.
`third_party/httplib.h` SHA-256:
`4770f8ea3d3fcd27b0b713c783d9b53bce3baddbd5adad13714011df96142526`.
Copyright (c) 2017 Yuji Hirose. Full license in `third_party/licenses/httplib.txt`.

## nlohmann/json

https://github.com/nlohmann/json, tag `v3.11.3`, MIT.
`third_party/json.hpp` SHA-256:
`9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6`.
Copyright (c) 2013-2023 Niels Lohmann. Full license in `third_party/licenses/json.txt`.

## CrispASR reference

https://github.com/CrispStrobe/CrispASR, local reference revision
`27b2fbe1089f11370222fbf5857b996470f37cb7`.
X-ASR/Zipformer2, Qwen3-ASR, Moonshine Streaming and Nemotron FastConformer /
RNN-T and FireRed VAD / DFSMN tensor equations, feature conventions and layouts
informed this implementation. The graph
equations are adaptations and should retain upstream attribution. Its server,
CLI, model registry, audio codecs, alignment, VAD orchestration, TTS, translation and auxiliary
model runtimes are not included or linked. HaoLive implements its own GGUF
loader, FFT/features, graph lifetime, model adapters, session policy and APIs.

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Embedded Moonshine tokenizer

`src/models/moonshine_tokens.inc` embeds the 251254-byte compact tokenizer
supplied with the requested Moonshine Streaming Medium checkpoint. It is
compiled into the executable; no tokenizer file is read at runtime.
Original model: https://huggingface.co/UsefulSensors/moonshine-streaming-medium
Upstream license: https://github.com/moonshine-ai/moonshine/blob/main/LICENSE
Streaming models are MIT licensed. Copyright (c) 2025 Useful Sensors, Inc.
(dba Moonshine AI). Complete upstream terms retained in
`third_party/licenses/moonshine.txt` and accessible through `--licenses`.
Compact data SHA-256:
`0e90e02b765a10f0fa35b7d67877df29dd22a1fd4890899c9b1b203a19bc8999`.
Model weights remain external and are not included in the binary.

## IXWebSocket

https://github.com/machinezone/IXWebSocket, v11.4.6, commit
`2efe037c9cc96fd536774f17bdb5215161ee5087`, MIT.
Full license: `third_party/ixwebsocket/LICENSE`. Local patches and static
loopback-only build configuration are recorded in `third_party/ixwebsocket/HAOLIVE.md`.

## NVIDIA CUDA 12.6 redistributables (Windows CUDA package only)

The original CUDA license is retained at `third_party/licenses/CUDA-12.6.txt` and packaged as `CUDA-LICENSE.txt`. Source: NVIDIA CUDA 12.6.3 [redistribution manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.6.3.json), `cuda_cudart` 12.6.77 Windows archive (SHA-256 `7a313bc0c93b1a50bb03aa9783a199ae70c3b66e2d8084da65e8254a8577b925`), original archive member `cuda_cudart-windows-x86_64-12.6.77-archive/LICENSE`. CUDA DLLs remain subject to NVIDIA's terms.
