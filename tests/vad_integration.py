#!/usr/bin/env python3
"""FireRed VAD HTTP acceptance and small warm latency benchmark (stdlib only)."""
import argparse
import hashlib
import io
import json
import math
import platform
import random
import shutil
import socket
import statistics
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, action="append", required=True)
    parser.add_argument("--asr-model", type=Path)
    parser.add_argument("--backend", choices=["cpu", "metal"], default="cpu")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 100:
        parser.error("iterations must be in [1,100]")
    clips = []
    for path in args.audio:
        with wave.open(str(path)) as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
            clips.append((path.name, wav.readframes(80000)))
    rng = random.Random(42)
    noise = struct.pack("<16000h", *(rng.randint(-2000, 2000) for _ in range(16000)))
    sine = struct.pack("<16000h", *(int(12000 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(16000)))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    checks = 0

    def call(path="/v1/vad", data=b"", status=200, content="audio/pcm", auth=True):
        nonlocal checks
        headers = {"Content-Type": content}
        if auth:
            headers["Authorization"] = "Bearer vad-test"
        request = urllib.request.Request(base + path, data, headers=headers)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            response = opener.open(request, timeout=120)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            body = response.read()
            assert response.code == status, (path, response.code, status, body)
        checks += 1
        return json.loads(body)

    # Isolate the binary and weight path: no shader, tokenizer or shared libs.
    with tempfile.TemporaryDirectory(prefix="haolive-vad-") as directory:
        root = Path(directory)
        executable = root / "haolive-server"
        shutil.copy2(args.server, executable)
        model = root / "vad.gguf"
        model.symlink_to(args.model.resolve())
        command = [str(executable), "--vad-model", str(model), "--port", str(port), "--token", "vad-test"]
        if args.asr_model:
            asr = root / "asr.gguf"
            asr.symlink_to(args.asr_model.resolve())
            command += ["--model", str(asr), "--backend", args.backend]
        log_path = root / "server.log"
        with log_path.open("w") as log:
            start = time.perf_counter()
            process = subprocess.Popen(command, cwd=root, env={"PATH": "/usr/bin:/bin"}, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 60
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(log_path.read_text())
                    try:
                        health = call("/health", None)
                        break
                    except (OSError, urllib.error.URLError):
                        if time.monotonic() > deadline:
                            raise TimeoutError("VAD startup")
                        time.sleep(.02)
                startup_ms = (time.perf_counter() - start) * 1000
                assert health["vad"] == {"model": "firered-vad", "backend": "cpu"}
                models = call("/v1/models", None)
                assert models["vad"]["id"] == "firered-vad"
                if not args.asr_model:
                    assert health["model"] is None and models["data"] == []
                    call("/v1/transcribe", b"", status=503)
                    call("/v1/sessions", b"", status=503)
                call(auth=False, status=401)
                for body, sufficient in [(b"", False), (bytes(798), False), (bytes(800), True), (bytes(32000), True)]:
                    result = call(data=body)
                    assert not result["is_speech"] and result["sufficient_audio"] == sufficient
                call(data=b"x", status=400)
                call(data=b"x", status=415, content="audio/mp3")
                call(data=b"x", status=400, content="audio/wav")
                call(data=bytes(960002), status=413)
                call(data=bytes(1024 * 1024 + 1), status=413)
                assert not call(data=bytes(960000))["is_speech"], "30-second boundary"
                for query in ["threshold=nan", "threshold=inf", "threshold=0", "threshold=1", "threshold=.5x", "threshold=1e99",
                              "min_speech_ms=0", "min_speech_ms=30001", "min_speech_ms=999999999999999999999"]:
                    call("/v1/vad?" + query, status=400)
                nonvoice = {name: call(data=pcm) for name, pcm in [("silence", bytes(32000)), ("white_noise", noise), ("440Hz", sine)]}
                assert all(not r["is_speech"] for r in nonvoice.values()), nonvoice
                rows = []
                for name, pcm in clips:
                    original = call(data=pcm)
                    assert original["is_speech"] and original["frames"] > 0, (name, original)
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        speech = pool.submit(call, data=pcm)
                        silent = pool.submit(call, data=bytes(32000))
                        assert speech.result() == original and not silent.result()["is_speech"]
                    buffer = io.BytesIO()
                    with wave.open(buffer, "wb") as wav:
                        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000); wav.writeframes(pcm)
                    assert call(data=buffer.getvalue(), content="audio/wav") == original
                    for seconds in [.5, 1, 5]:
                        part = pcm[:int(seconds * 32000)]
                        reference = call(data=part)  # warmup
                        latencies = []
                        for _ in range(args.iterations):
                            start = time.perf_counter()
                            assert call(data=part) == reference
                            latencies.append((time.perf_counter() - start) * 1000)
                        rows.append({"audio": name, "audio_ms": len(part) // 32, "result": reference,
                                     "mean_ms": statistics.mean(latencies), "p50_ms": statistics.median(latencies), "max_ms": max(latencies)})
                    if args.asr_model:
                        transcript = call("/v1/transcribe", pcm)
                        assert transcript["text"]
                        assert call(data=pcm) == original, "ASR changed VAD state"
                report = {"checks": checks, "asr_enabled": bool(args.asr_model), "vad_backend": "cpu", "startup_ready_ms": startup_ms,
                          "server_sha256": hashlib.sha256(args.server.read_bytes()).hexdigest(),
                          "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(), "platform": platform.platform(),
                          "iterations": args.iterations, "nonvoice": nonvoice, "measurements": rows}
                rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
                print(rendered, end="")
                if args.output:
                    args.output.write_text(rendered)
            except BaseException:
                log.flush()
                print(log_path.read_text()[-8000:])
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()


if __name__ == "__main__":
    main()
