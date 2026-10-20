#!/usr/bin/env python3
"""Verify a macOS server copied alone, without build libraries or sidecars."""
import argparse
import hashlib
import json
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, type=Path)
    parser.add_argument("--model", required=True, action="append", type=Path)
    parser.add_argument("--backend", action="append", choices=["cpu", "metal"])
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dependencies = subprocess.check_output(["/usr/bin/otool", "-L", str(args.server)], text=True).splitlines()[1:]
    for dependency in dependencies:
        assert dependency.strip().startswith(("/usr/lib/", "/System/Library/")), dependency
    with wave.open(str(args.audio)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        pcm = wav.readframes(5 * 16000)
    assert pcm, "empty speech fixture"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    results = []
    with tempfile.TemporaryDirectory(prefix="haolive-single-file-") as directory:
        root = Path(directory)
        executable = root / "haolive-server"
        shutil.copyfile(args.server, executable)
        executable.chmod(0o755)
        assert [p.name for p in root.iterdir()] == ["haolive-server"]
        licenses = subprocess.check_output([str(executable), "--licenses"], cwd=root, text=True)
        assert "Useful Sensors" in licenses and "Permission is hereby granted" in licenses
        for model in args.model:
            # The path presented to the server has no tokenizer/shader beside it.
            # A symlink avoids duplicating several GB of weights for this test.
            weight = root / "model.gguf"
            weight.symlink_to(model.resolve())
            try:
                for backend in args.backend or ["cpu", "metal"]:
                    with socket.socket() as sock:
                        sock.bind(("127.0.0.1", 0))
                        port = sock.getsockname()[1]
                    base = f"http://127.0.0.1:{port}"
                    process = subprocess.Popen([str(executable), "--model", str(weight), "--backend", backend,
                                                "--port", str(port)], cwd=root, env={"PATH": "/usr/bin:/bin"},
                                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    try:
                        deadline = time.monotonic() + 60
                        while True:
                            if process.poll() is not None:
                                raise RuntimeError(process.communicate()[0].decode(errors="replace"))
                            try:
                                with opener.open(base + "/health", timeout=2) as response:
                                    health = json.load(response)
                                break
                            except (OSError, urllib.error.URLError):
                                if time.monotonic() > deadline:
                                    raise TimeoutError("isolated server startup")
                                time.sleep(.1)
                        request = urllib.request.Request(base + "/v1/transcribe", pcm,
                                                         headers={"Content-Type": "audio/pcm"})
                        with opener.open(request, timeout=120) as response:
                            result = json.load(response)
                        assert result["text"] and result["type"] == "final", result
                        assert health["backend"] == backend
                        results.append({"model": model.name, "backend": backend, "text": result["text"]})
                    finally:
                        process.terminate()
                        try:
                            process.communicate(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.communicate()
            finally:
                weight.unlink()
    report = {"server_sha256": hashlib.sha256(args.server.read_bytes()).hexdigest(),
              "dependencies": [d.strip() for d in dependencies], "results": results}
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
