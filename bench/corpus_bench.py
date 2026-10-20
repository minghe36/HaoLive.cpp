#!/usr/bin/env python3
"""Full-corpus benchmark: native macOS decoding, bounded HTTP windows, wait4 RSS."""
import argparse
import datetime
import hashlib
import json
import os
import platform
import selectors
import shutil
import signal
import subprocess
import sys
import time
import wave
from pathlib import Path

from bench import BenchError, Client, file_identity, run_offline


def windows(pcm, seconds=20):
    size = int(seconds * 16000) * 2
    if size < 2 or size > 30 * 32000 or len(pcm) % 2:
        raise ValueError("invalid PCM window size or unaligned PCM")
    for offset in range(0, len(pcm), size):
        yield offset / 32000, pcm[offset:offset + size]


def rss_bytes(usage, system=None):
    # Darwin wait4 reports bytes; Linux reports KiB. This is the owned child,
    # not the cumulative maximum of all children from getrusage(CHILDREN).
    return int(usage.ru_maxrss * (1 if (system or platform.system()) == "Darwin" else 1024))


def reap(process, grace=10):
    try:
        os.kill(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace
    while True:
        pid, status, usage = os.wait4(process.pid, os.WNOHANG)
        if pid:
            process.returncode = os.waitstatus_to_exitcode(status)
            return usage
        if time.monotonic() >= deadline:
            try:
                os.kill(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
            return usage
        time.sleep(.01)


def start(args, model, backend, log):
    command = [str(args.server.resolve()), "--model", str(model.resolve()), "--backend", backend,
               "--port", "0", "--host", "127.0.0.1", "--threads", str(args.threads),
               "--max-tokens", str(args.max_tokens), "--step-ms", "800"]
    began = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(args.startup_timeout):
                raise BenchError("server readiness timeout")
        raw = process.stdout.readline(4097)
        ready = json.loads(raw)
        if ready.get("event") != "ready" or ready.get("api_version") != 1:
            raise BenchError(f"invalid readiness: {raw!r}")
        client = Client(ready["port"], args.timeout)
        health = client.call("/health", method="GET")
        if health.get("backend") != backend or health.get("api_version") != 1:
            client.close()
            raise BenchError(f"backend mismatch: {health}")
        return process, client, {"command": command, "startup_s": time.perf_counter() - began,
                                 "health": health, "port": ready["port"]}
    except BaseException:
        reap(process)
        process.stdout.close()
        raise


def transcribe_file(args, model, backend, audio, pcm, output):
    row = {"model": model.name, "backend": backend, "audio": audio["relative"],
           "audio_s": len(pcm) / 32000, "iterations": args.iterations, "segments": [],
           "warmup_s": [], "status": "pending"}
    process = client = None
    began = time.perf_counter()
    case_dir = output / backend / model.stem / Path(audio["relative"]).with_suffix("")
    case_dir.mkdir(parents=True)
    row["artifact_dir"] = str(case_dir.relative_to(output))
    try:
        with (case_dir / "server.log").open("w") as log:
            process, client, config = start(args, model, backend, log)
            row.update(config)
            warmup_pcm = next(windows(pcm, args.segment_seconds))[1]
            row["warmup_errors"] = []
            for _ in range(args.warmup):
                warmup_start = time.perf_counter()
                try:
                    run_offline(client, warmup_pcm, allow_empty=True)
                except (Exception,) as error:
                    row["warmup_errors"].append(str(error))
                    client.close()
                    client = Client(config["port"], args.timeout)
                row["warmup_s"].append(time.perf_counter() - warmup_start)
            iterations = []
            for iteration in range(args.iterations):
                inference_start = time.perf_counter()
                segments = []
                for index, (offset, body) in enumerate(windows(pcm, args.segment_seconds)):
                    request_start = time.perf_counter()
                    segment = {"index": index, "start_s": offset, "end_s": offset + len(body) / 32000,
                               "samples": len(body) // 2}
                    try:
                        result = run_offline(client, body, allow_empty=True)
                        segment.update({"wall_s": result["wall_s"], "text": result["text"]})
                    except (Exception,) as error:
                        segment.update({"wall_s": time.perf_counter()-request_start, "text":"", "error":str(error)})
                        client.close()
                        client = Client(config["port"], args.timeout)
                    segments.append(segment)
                    with (case_dir / "segments.jsonl").open("a") as file:
                        file.write(json.dumps({"iteration": iteration + 1, **segment}, ensure_ascii=False) + "\n")
                wall = time.perf_counter() - inference_start
                # Keep all segments, including empty hypotheses and the final short tail.
                text = "\n".join(s["text"].strip() for s in segments)
                iterations.append({"wall_s": wall, "request_s": sum(s["wall_s"] for s in segments),
                                   "text": text, "segments": segments})
            row["runs"] = iterations
            row["segments"] = iterations[0]["segments"]
            row["text"] = iterations[0]["text"]
            row["inference_mean_s"] = sum(r["wall_s"] for r in iterations) / len(iterations)
            row["rtf"] = row["inference_mean_s"] / row["audio_s"]
            row["speed"] = 1 / row["rtf"]
            row["deterministic_text"] = len({r["text"] for r in iterations}) == 1
            row["empty_segments"] = sum(not s["text"].strip() for s in row["segments"])
            row["failed_segments"] = sum("error" in s for s in row["segments"])
            row["status"] = "completed_with_errors" if row["failed_segments"] else "ok"
            (case_dir / "transcript.txt").write_text(row["text"] + "\n")
    except (Exception, KeyboardInterrupt) as error:
        row["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        row["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, KeyboardInterrupt):
            raise
    finally:
        if client:
            client.close()
        if process:
            usage = reap(process)
            process.stdout.close()
            row.update({"peak_rss_bytes": rss_bytes(usage), "process_user_s": usage.ru_utime,
                        "process_system_s": usage.ru_stime, "exit_code": process.returncode})
        row["process_elapsed_s"] = time.perf_counter() - began
        (case_dir / "metrics.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
    return row


def prepare(args):
    converted = args.output / "prepared"
    converted.mkdir(parents=True)
    paths = sorted(p for p in args.input.rglob("*") if p.is_file() and p.suffix.lower() in {".mp3", ".wav", ".m4a", ".flac", ".aiff"})
    if not paths:
        raise ValueError("no supported audio files")
    result = []
    for path in paths:
        relative = path.relative_to(args.input)
        target = converted / relative.with_suffix(".wav")
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [args.afconvert, "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(path), str(target)]
        began = time.perf_counter()
        subprocess.run(command, check=True, capture_output=True)
        conversion_s = time.perf_counter() - began
        with wave.open(str(target)) as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
                raise ValueError(f"unexpected converted format: {target}")
            pcm = wav.readframes(wav.getnframes())
        if not pcm:
            raise ValueError(f"empty audio: {path}")
        identity = {**file_identity(path, True), "relative": str(relative), "conversion_command": command,
                    "conversion_s": conversion_s, "prepared": str(target.resolve()),
                    "samples": len(pcm) // 2, "audio_s": len(pcm) / 32000,
                    "pcm_sha256": hashlib.sha256(pcm).hexdigest()}
        print(f"Prepared {relative}: {identity['audio_s']:.3f}s", flush=True)
        result.append((identity, pcm))
    return result


def write(output, report):
    temporary = output / "results.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output / "results.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--backend", choices=["cpu", "metal"], action="append", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--segment-seconds", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--startup-timeout", type=float, default=90)
    parser.add_argument("--only-errors", type=Path, help="retry only failed configurations from an earlier run")
    parser.add_argument("--afconvert", default=shutil.which("afconvert"))
    args = parser.parse_args(argv)
    if not args.afconvert or not args.input.is_dir() or not args.server.is_file() or any(not p.is_file() for p in args.model):
        parser.error("valid server/models/input directory and macOS afconvert are required")
    if not 1 <= args.segment_seconds <= 30 or not 1 <= args.iterations <= 10 or not 0 <= args.warmup <= 10 or not 1 <= args.threads <= 256 or not 1 <= args.max_tokens <= 2048 or not 0 < args.timeout <= 600 or not 0 < args.startup_timeout <= 300:
        parser.error("invalid benchmark limits")
    if args.output.exists():
        parser.error("output must be a new directory")
    args.output.mkdir(parents=True)
    report = {"schema_version": 1, "started_at": datetime.datetime.now().astimezone().isoformat(),
              "host": {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
                       "sysctl": subprocess.check_output(["sysctl", "-n", "hw.model", "hw.memsize", "hw.ncpu", "machdep.cpu.brand_string"], text=True).strip().splitlines()},
              "server": file_identity(args.server, True), "models": [file_identity(p, True) for p in args.model],
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "model"},
              "audio": [], "results": []}
    selected = None
    if args.only_errors:
        prior=json.loads(args.only_errors.read_text())
        selected={(r["backend"],r["model"],r["audio"]) for r in prior["results"] if r["status"] != "ok"}
    try:
        audio_data = prepare(args)
        report["audio"] = [identity for identity, _ in audio_data]
        write(args.output, report)
        total = len(selected) if selected is not None else len(args.model) * len(args.backend) * len(audio_data)
        for backend in dict.fromkeys(args.backend):
            for model in dict.fromkeys(args.model):
                for identity, pcm in audio_data:
                    if selected is not None and (backend,model.name,identity["relative"]) not in selected:
                        continue
                    print(f"[{len(report['results']) + 1}/{total}] {backend} {model.name} {identity['relative']}", flush=True)
                    row = transcribe_file(args, model, backend, identity, pcm, args.output)
                    report["results"].append(row)
                    write(args.output, report)
                    print(f"  {row['status']} time={row.get('inference_mean_s', 0):.3f}s RTF={row.get('rtf', 0):.4f} RSS={row.get('peak_rss_bytes', 0) / 2**20:.1f}MiB {row.get('error', '')}", flush=True)
    finally:
        report["finished_at"] = datetime.datetime.now().astimezone().isoformat()
        write(args.output, report)
    return int(any(row["status"] != "ok" for row in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
