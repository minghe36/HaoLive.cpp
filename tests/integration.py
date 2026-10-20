#!/usr/bin/env python3
"""Real-model HTTP acceptance checks. Uses Python standard library only."""
import argparse
import io
import json
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "metal"], required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--expected", default="")
    parser.add_argument("--seconds", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with wave.open(str(args.audio)) as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            raise ValueError("test audio must be 16 kHz mono PCM16 WAV")
        pcm = wav.readframes(args.seconds * 16000)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    checks = 0

    def call(path, data=None, method=None, status=200, content="audio/pcm", auth=True):
        nonlocal checks
        headers = {"Content-Type": content}
        if auth:
            headers["Authorization"] = "Bearer integration-test"
        request = urllib.request.Request(base + path, data=data, method=method, headers=headers)
        try:
            response = opener.open(request, timeout=120)
        except urllib.error.HTTPError as error:
            response = error
        body = response.read()
        if response.code != status:
            raise AssertionError(f"{path}: {response.code} expected {status}: {body!r}")
        checks += 1
        return json.loads(body) if body else None

    with tempfile.TemporaryDirectory(prefix="haolive-test-") as tmp:
        log_path = Path(tmp) / "server.log"
        with log_path.open("w") as log:
            process = subprocess.Popen([
                str(args.server.resolve()), "--model", str(args.model.resolve()),
                "--backend", args.backend, "--port", str(port), "--token", "integration-test",
            ], stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 60
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("server exited during model loading")
                    try:
                        health = call("/health")
                        break
                    except (OSError, urllib.error.URLError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("server startup")
                        time.sleep(0.1)
                assert health["backend"] == args.backend
                call("/health", auth=False, status=401)
                models = call("/v1/models")
                assert models["data"][0]["sample_rate"] == 16000
                assert health["vad"] is None and models["vad"] is None
                call("/v1/vad", b"", status=503)
                call("/v1/transcribe", b"x", status=400)
                call("/v1/transcribe", b"x", status=415, content="audio/mp3")
                call("/v1/transcribe", bytes(1024 * 1024 + 1), status=413)
                empty = call("/v1/transcribe", b"")
                assert empty["text"] == "" and empty["type"] == "final"
                ids = [call("/v1/sessions", b"", status=201)["session_id"] for _ in range(8)]
                call("/v1/sessions", b"", status=429)
                for sid in ids:
                    call("/v1/sessions/" + sid, method="DELETE", status=204)
                call("/v1/sessions/" + ids[0], method="DELETE", status=404)
                start = time.monotonic()
                offline = call("/v1/transcribe", pcm)
                offline_seconds = time.monotonic() - start
                assert offline["text"], "real speech must produce text"
                repeat = call("/v1/transcribe", pcm)
                assert repeat["text"] == offline["text"], "previous utterance leaked into the next inference"
                assert offline["type"] == "final" and offline["audio_ms"] == len(pcm) // 32
                if args.expected:
                    assert args.expected.casefold() in offline["text"].casefold(), offline
                # Two interleaved sessions catch encoder/cache leakage across callers.
                sid = call("/v1/sessions", b"", status=201)["session_id"]
                other = call("/v1/sessions", b"", status=201)["session_id"]
                call(f"/v1/sessions/{sid}/audio?final=invalid", b"", status=400)
                call(f"/v1/sessions/{sid}/audio", b"x", status=400)
                partials = []
                start = time.monotonic()
                for offset in range(0, len(pcm), 16000):  # 500 ms chunks
                    result = call(f"/v1/sessions/{sid}/audio", pcm[offset:offset + 16000])
                    assert result["type"] == "partial" and result["sequence"] == len(partials) + 1
                    partials.append(result["text"])
                    empty_other = call(f"/v1/sessions/{other}/audio", b"")
                    assert empty_other["text"] == ""
                final = call(f"/v1/sessions/{sid}/audio?final=true", b"")
                streaming_seconds = time.monotonic() - start
                assert final["type"] == "final" and final["text"]
                if args.expected:
                    assert args.expected.casefold() in final["text"].casefold(), final
                if health["model"] in ("xasr", "moonshine_streaming", "nemotron"):
                    assert final["text"] == offline["text"], (offline, final)
                if health["model"] == "qwen3asr":
                    assert all("|" not in text for text in partials + [final["text"]]), "R2T2 delimiter leaked into text"
                assert any(partials), "stream must emit text before final"
                if health["model"] in ("moonshine_streaming", "nemotron"):
                    # Irregular appends cross FFT/subsampling/chunk boundaries.
                    # Both nonempty final bodies and an empty flush are supported.
                    sid2 = call("/v1/sessions", b"", status=201)["session_id"]
                    offset = 0
                    for samples in (1, 159, 641, 10001, 17, 15555):
                        part = pcm[offset:offset + samples * 2]
                        call(f"/v1/sessions/{sid2}/audio", part)
                        offset += len(part)
                    irregular = call(f"/v1/sessions/{sid2}/audio?final=true", pcm[offset:])
                    assert irregular["text"] == offline["text"], "chunk schedule changed final output"
                    tiny = call("/v1/transcribe", b"\0\0")
                    assert tiny["type"] == "final" and tiny["audio_ms"] == 0
                call(f"/v1/sessions/{sid}/audio?final=true", b"", status=404)
                call(f"/v1/sessions/{other}/audio?final=true", b"")
                # Audio limits must reject before consuming bytes and keep session usable.
                sid = call("/v1/sessions", b"", status=201)["session_id"]
                call(f"/v1/sessions/{sid}/audio", bytes(30 * 32000))
                call(f"/v1/sessions/{sid}/audio", b"\0\0", status=413)
                call(f"/v1/sessions/{sid}", method="DELETE", status=204)
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as wav:
                    wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000); wav.writeframes(pcm)
                wav_result = call("/v1/transcribe", buffer.getvalue(), content="audio/wav")
                assert wav_result["text"] == offline["text"], "PCM/WAV result mismatch"
                report = {
                    "backend": args.backend, "architecture": health["model"], "checks": checks,
                    "audio_ms": len(pcm) // 32, "offline_text": offline["text"],
                    "streaming_text": final["text"], "partials": partials,
                    "offline_seconds": round(offline_seconds, 3),
                    "streaming_seconds": round(streaming_seconds, 3),
                }
                print(json.dumps(report, ensure_ascii=False, indent=2))
                if args.output:
                    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            except BaseException:
                log.flush()
                print(log_path.read_text()[-12000:])
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()


if __name__ == "__main__":
    main()
