# FireRed VAD acceptance and latency

Verified on 2026-10-19, macOS arm64 / Apple M4 Max. The independent VAD uses CPU; the existing ASR engines still support CPU and Metal.

## Functional acceptance

- CTest passed: 1738 unit assertions, embedded-license CLI and 15 benchmark tests. The optional shared-library build also passed the C header test.
- Real-model C ABI tests passed 16 checks using English and Chinese speech, empty/short/silent audio, invalid parameters, concurrent independent windows and repeat determinism.
- Python ctypes loaded the development shared library and called the new VAD C ABI successfully.
- VAD-only HTTP run passed 101 request checks. Combined Metal Moonshine ASR + CPU VAD passed 40 request checks.
- Both HTTP runs copied the executable into a temporary directory and used model paths without any companion resources. They covered auth, invalid thresholds/durations (including numeric overflow), malformed/unsupported audio, 25 ms and 30 s boundaries, oversized inputs, PCM/WAV parity and concurrent requests.
- English/Chinese 1 s and 5 s windows were classified as speech. Silence, fixed-seed white noise and a 440 Hz sine fixture were classified as non-speech. This is a small fixture suite, not a general false-positive/false-negative evaluation.
- The first 0.5 s of the Chinese sample returned false: only 60 ms of frames crossed the default threshold. The same sample at 1 s returned true. Short-window sensitivity is a known limitation; callers can supply 1–2 s rolling windows or tune the duration/threshold.
- All four ASR models passed isolated single-binary CPU/Metal inference again (`build/standalone-four-models-vad.json`). X-ASR Chinese HTTP regression passed 60 request checks, including VAD-disabled 503 behavior.

## Warm HTTP latency

Threshold 0.5, minimum consecutive speech 100 ms; one warmup and 10 measured calls for each row. Timings include loopback HTTP, audio decode and JSON. No GPU is initialized in VAD-only mode.

| Fixture | Window | Speech | Mean | p50 | Maximum |
| --- | --- | --- | --- | --- | --- |
| jfk.wav | 500 ms | true | 3.07 ms | 3.07 ms | 3.25 ms |
| jfk.wav | 1000 ms | true | 5.40 ms | 5.35 ms | 6.18 ms |
| jfk.wav | 5000 ms | true | 22.27 ms | 22.23 ms | 23.00 ms |
| paraformer_zh.wav | 500 ms | false | 2.68 ms | 2.69 ms | 2.82 ms |
| paraformer_zh.wav | 1000 ms | true | 5.10 ms | 5.15 ms | 5.70 ms |
| paraformer_zh.wav | 5000 ms | true | 22.76 ms | 22.74 ms | 23.08 ms |

These are warm, small-sample host measurements, not cold-start latency or a guarantee for other hardware.

## Artifact identity

- Single executable: `build/single-file/haolive-server`, 4843120 bytes.
- Server SHA-256: `31eb122192abdcf3bd6b55380e4169dc6581fb38d01c2d9514b11c5f963c5960`.
- FireRed GGUF SHA-256: `72d37db1a5d2a9db386d1452b3f54d13f97bea79710105fc65acd050ea7db600`.
- The archive `build/HaoLive-macos-arm64-single-file.tar.gz` contains exactly one file: `haolive-server`.
- The binary links only macOS system frameworks/libraries. FireRed weights remain an external caller-selected GGUF. No new third-party runtime dependency was added.
- Raw reports: `build/vad-http-final.json`, `build/vad-with-asr-final.json`.

## Reproduction

```sh
python3 tests/vad_integration.py --server build/single-file/haolive-server \
  --model models/firered-vad.gguf --audio speech-en.wav --audio speech-zh.wav \
  --iterations 10 --output build/vad-new-report.json
```
