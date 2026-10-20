# Architecture

```
HTTP / C ABI / C++ callers
             |
       Engine + Session
             |
      GGUF architecture dispatch
       /       |       |       \
     Xasr    Qwen  Moonshine  Nemotron
       \       |       |       /
  Device + Weights + Graph + Features
             |
            ggml
```

The public protocol is versioned independently of GGUF architecture names.
Clients use the same session create/accept/finalize lifecycle for every model.
An engine owns weights and device resources. A session owns audio, hypothesis,
feature offsets, encoder caches and lifecycle status. Engine inference is
serialized. Sessions from different callers never share utterance state.

`src/runtime.*` implements device selection, checked GGUF uploads and RAII
graph ownership. Graph metadata stays alive until the scheduler is destroyed.
Weights are loaded once in 4 MiB blocks without a second full model copy.
Backend selection uses the exact ggml registry, including the Metal/MTL alias;
an unavailable requested device is an error. A CPU backend remains available
for individual unsupported GPU operations.

`src/audio.cpp` implements PCM/WAV parsing and mixed-radix FFT. The X-ASR
frontend uses Kaldi/Povey/preemphasis/HTK mel with mirrored edges. Only newly
stable frames are computed on append. Confucius uses the GGUF's Hann window
and mel filters, centered reflect STFT and Whisper log normalization. It
removes padded CNN frames before the audio transformer.

X-ASR builds one fixed encoder graph, preserves host copies of all layer
caches per session, and applies a stateless decoder/joiner. Flush adds the
model's tail padding. Confucius re-encodes bounded cumulative audio on each
scheduled step and uses a model-local KV cache within each generation call.
Recent hypothesis tokens are rolled back before prefix conditioning. KV
storage is not shared between generation calls. F32 cache writes use SET_ROWS
so attention has an explicit dependency on the write.

The Confucius prompt seeds `<asr_text>` directly. This avoids the checkpoint's
early-EOS language-header path without an auxiliary language detector or a
hardcoded input language. The session exposes revised full-text partials. The default step is 160 ms.
Partial generation yields after at most 32 new tokens (or a smaller configured
token limit); finalization still uses the configured full budget and reports
exhaustion. Chunk pipe delimiters never enter the next assistant prefix.
Incomplete UTF-8 suffixes are withheld from display without dropping hypothesis
tokens, so a budget boundary cannot produce invalid JSON.
`Session::accept` also accepts an optional full-hypothesis callback. Qwen
invokes it as each text token becomes decodable. HTTP callers opt into SSE with
`stream=true`: `partial` frames arrive during generation, followed by exactly
one `result` frame; inference failure emits an `error` frame and removes the
session. Existing JSON responses and the C ABI remain available. Other model
adapters currently return their regular chunk results without token callbacks.
This policy is intentionally smaller than upstream's adaptive R2T2 schedule.

The native `/v1/realtime` WebSocket listener uses a dynamically assigned
loopback port and first validates the launch bearer token and session ID.
Each connection separates the receiving loop from an inference worker with a
bounded PCM backlog. Audio received during inference is coalesced for the next
step; the reader never waits on the shared inference mutex. Finals remain
ordered after all audio. The Rust client also separates its writer and reader,
so a submission acknowledgement is independent from recognition progress.
There are at most eight socket connections, each capped at 30 seconds of input.

The compatibility HTTP layer has 4 workers and at most 32 queued connections. Inference and
session mutations are serialized, with at most 8 active sessions, a 1 MiB
request cap, a 30-second utterance cap and lazy 120-second idle cleanup.
The caller controls endpointing. No codec frameworks, resampler,
model downloader, timestamp aligner or translation runtime is linked.

To add a model:

1. Add a private `Model`/`Session` implementation under `src/models`.
2. Register its GGUF architecture in `haolive::load` (not filename heuristics).
3. Keep all model-specific state behind `Session::accept`.
4. Use the same input validation, normalized PCM and full-text result contract.
5. Add unit checks and real-model HTTP/C ABI acceptance checks.
6. Verify requested backends on real hardware before claiming support.

No changes to HTTP response fields or the C ABI are needed for a new model
unless it introduces a genuinely new public capability.

Moonshine uses 80-sample waveform normalization/compression, causal strided
convolutions, local-window encoder attention and a greedy RoPE decoder with
precomputed cross attention and explicit self-KV write dependencies. Cumulative
utterance replay is scheduled by `step_ms`; encoder state is not incrementally
reused. The requested Medium tokenizer is compiled into the executable from
`src/models/moonshine_tokens.inc`; GGUF vocabulary metadata takes precedence.
There is no runtime companion file or SentencePiece library dependency.

Nemotron uses NeMo preemphasis/log-mel with zero-centered 512-point STFT and
retains the terminal frame. A causal three-stage subsampler feeds native
FastConformer chunks. Each session owns per-layer post-FFN attention context,
post-GLU convolution history, two LSTM hidden/cell pairs and the RNN-T token
hypothesis. Convolution history retains the suffix of old + new frames, even
when chunks are shorter than the kernel history. One chunk is held until the
STFT right edge becomes stable; final drains it. A reusable encoder graph is
keyed by past/new frame counts, and all state/constant inputs are uploaded on
every run. Automatic language prompting uses the model's own `auto` table
entry, followed by prompt MLP, encoder/predictor projections and greedy RNN-T.
No LID model or reference runtime is loaded.

The server statically links all third-party code; Metal kernels, Moonshine
vocabulary and license text (`--licenses`) are embedded. Only OS frameworks
and a caller-selected GGUF are needed at runtime. The optional shared C ABI
library is an independent developer artifact, never a server dependency.

Independent VAD is an additive API (`Vad::detect`, `hl_vad_detect`, HTTP
`POST /v1/vad`). It is not an ASR `Engine` or `Session`, and does not change
ASR endpointing/input. The server accepts `--vad-model` with or without
`--model`; unavailable capabilities return a structured 503 error.

`src/vad.cpp` loads and shape-checks the requested FireRed DFSMN GGUF into
immutable F32 CPU arrays, releases the loader buffers, and performs batched
matrix multiplication plus channel-wise finite-memory filtering. On macOS
matrix multiplication uses the OS Accelerate framework; other platforms have
a scalar fallback. No GPU backend is initialized by a VAD-only load. Feature
extraction uses snip edges, int16 input scale, DC removal, preemphasis, Povey
window and linear-Hz triangular filters between HTK mel-spaced centers; it
cannot reuse X-ASR's different mel filterbank. CMVN precedes the eight DFSMN
memory stages. Lookback and lookahead weights are reordered once at load for
contiguous channel loops. Prediction uses sigmoid frame probabilities.

Every request/window has independent scratch, with no persistent audio history
or graph. Concurrent calls are safe. Detection requires a configurable run of
consecutive above-threshold 10 ms frames; probability maxima alone do not
make a speech decision. Shorter-than-25 ms windows report insufficient audio.
Callers may submit overlapping VAD windows while continuing nonoverlapping ASR
appends. Frame probabilities are model outputs, not evidence of live-human
provenance. CPU-only VAD and Metal ASR can be enabled in the same process.
