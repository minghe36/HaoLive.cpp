#!/usr/bin/env python3
"""Reproducible HTTP ASR benchmarks; Python standard library only."""
import argparse
import contextlib
import csv
import datetime
import hashlib
import http.client
import json
import math
import platform
import socket
import statistics
import subprocess
import sys
import time
import wave
from pathlib import Path


class BenchError(RuntimeError):
    pass


def percentile(values, fraction):
    """Linear interpolation between sorted observations, including endpoints."""
    if not 0 <= fraction <= 1:
        raise ValueError("percentile must be in [0, 1]")
    ordered = sorted(values)
    if not ordered:
        return None
    index = (len(ordered) - 1) * fraction
    low = math.floor(index)
    return ordered[low] + (ordered[math.ceil(index)] - ordered[low]) * (index - low)


def read_audio(path, seconds=None):
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
            raise ValueError(f"{path}: expected 16 kHz mono PCM16 WAV")
        frames = audio.getnframes() if seconds is None else min(audio.getnframes(), int(seconds * 16000))
        if not 1 <= frames <= 30 * 16000:
            raise ValueError(f"{path}: audio must be > 0 and <= 30 seconds; use --seconds to select a shorter clip")
        pcm = audio.readframes(frames)
        if len(pcm) != frames * 2:
            raise ValueError(f"{path}: truncated WAV audio")
    return pcm


class Client:
    def __init__(self, port, timeout):
        # Direct loopback connection: environment proxy settings are irrelevant.
        self.connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)

    def call(self, path, body=b"", method="POST"):
        self.connection.request(method, path, body, {"Content-Type": "audio/pcm"})
        response = self.connection.getresponse()
        data = response.read()
        if response.status >= 400:
            raise BenchError(f"HTTP {response.status} {path}: {data.decode('utf-8', errors='replace')}")
        return json.loads(data) if data else {}

    def close(self):
        self.connection.close()


def validate_result(result, final, audio_bytes, allow_empty):
    expected = "final" if final else "partial"
    if (not isinstance(result, dict) or not isinstance(result.get("text"), str) or result.get("type") != expected
            or result.get("audio_ms") != audio_bytes // 32):
        raise BenchError(f"unexpected ASR response: {result!r}")
    if final and not allow_empty and not result["text"].strip():
        raise BenchError("empty final transcript; use --allow-empty only for intentional silence benchmarks")


def run_offline(client, pcm, allow_empty=False, clock=time.perf_counter):
    start = clock()
    result = client.call("/v1/transcribe", pcm)
    elapsed = clock() - start
    validate_result(result, True, len(pcm), allow_empty)
    duration = len(pcm) / 32000
    return {"wall_s": elapsed, "rtf": elapsed / duration, "request_work_rtf": elapsed / duration,
            "text": result["text"], "request_latencies_s": [elapsed]}


def run_stream(client, pcm, chunk_ms, paced, allow_empty=False,
               clock=time.perf_counter, sleep=time.sleep):
    create_start = clock()
    sid = client.call("/v1/sessions")["session_id"]
    create_s = clock() - create_start
    route = f"/v1/sessions/{sid}/audio"
    finished = False
    chunks, latencies = [], []
    first_text_s = first_partial_s = first_text_audio_s = None
    capture_start = clock()
    duration = len(pcm) / 32000
    try:
        chunk_bytes = chunk_ms * 32
        for offset in range(0, len(pcm), chunk_bytes):
            body = pcm[offset:offset + chunk_bytes]
            end = offset + len(body)
            audio_end_s = end / 32000
            if paced:
                sleep(max(0, capture_start + audio_end_s - clock()))
            sent = clock()
            result = client.call(route, body)
            received = clock()
            validate_result(result, False, end, True)
            latency = received - sent
            latencies.append(latency)
            if result["text"].strip() and first_text_s is None:
                first_text_s = first_partial_s = received - capture_start
                first_text_audio_s = audio_end_s
            chunks.append({"audio_end_s": audio_end_s, "send_s": sent - capture_start,
                           "receive_s": received - capture_start, "latency_s": latency,
                           "schedule_lag_s": max(0, sent - capture_start - audio_end_s) if paced else None,
                           "text": result["text"]})
        sent = clock()
        result = client.call(route + "?final=true")
        received = clock()
        finished = True  # Successful HTTP finalization releases the server session.
        validate_result(result, True, len(pcm), allow_empty)
        final_request_s = received - sent
        latencies.append(final_request_s)
        if result["text"].strip() and first_text_s is None:
            first_text_s, first_text_audio_s = received - capture_start, duration
        wall_s = received - capture_start
        return {"wall_s": wall_s, "rtf": wall_s / duration,
                "request_work_rtf": sum(latencies) / duration, "session_create_s": create_s,
                "first_text_s": first_text_s, "first_partial_s": first_partial_s,
                "first_text_audio_s": first_text_audio_s, "final_request_s": final_request_s,
                "capture_to_final_s": max(0, wall_s - duration) if paced else None,
                "max_schedule_lag_s": max(c["schedule_lag_s"] for c in chunks) if paced else None,
                "request_latencies_s": latencies, "chunks": chunks, "text": result["text"]}
    finally:
        if not finished:
            try:
                client.call(f"/v1/sessions/{sid}", method="DELETE")
            except (OSError, http.client.HTTPException, BenchError):
                pass


def summarize(rows):
    """Warmups and failed attempts never contribute to measured summaries."""
    measured = [row for row in rows if not row["warmup"] and "error" not in row]
    if not measured:
        return {"runs": 0}
    latencies = [x for row in measured for x in row["request_latencies_s"]]
    result = {"runs": len(measured), "wall_mean_s": statistics.mean(r["wall_s"] for r in measured),
              "wall_p50_s": percentile([r["wall_s"] for r in measured], .5),
              "wall_p95_s": percentile([r["wall_s"] for r in measured], .95),
              "rtf_mean": statistics.mean(r["rtf"] for r in measured),
              "request_work_rtf_mean": statistics.mean(r["request_work_rtf"] for r in measured),
              "request_p50_s": percentile(latencies, .5), "request_p95_s": percentile(latencies, .95),
              "request_max_s": max(latencies),
              "deterministic_text": len({r["text"] for r in measured}) == 1,
              "empty_transcripts": sum(not r["text"].strip() for r in measured)}
    for field in ["first_text_s", "first_partial_s", "first_text_audio_s", "final_request_s", "capture_to_final_s"]:
        samples = [r[field] for r in measured if r.get(field) is not None]
        result[field[:-2] + "_p50_s"] = percentile(samples, .5)
        result[field[:-2] + "_p95_s"] = percentile(samples, .95)
    result["realtime_speed_mean"] = 1 / result["rtf_mean"] if result["rtf_mean"] > 0 else None
    return result


@contextlib.contextmanager
def running_server(args, model, backend, log_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [str(args.server.resolve()), "--model", str(model.resolve()), "--backend", backend,
               "--host", "127.0.0.1", "--port", str(port), "--threads", str(args.threads),
               "--step-ms", str(args.step_ms), "--max-tokens", str(args.max_tokens)]
    client = Client(port, args.timeout)
    client.connection.timeout = min(.5, args.timeout)
    with log_path.open("w", encoding="utf-8") as log:
        started = time.perf_counter()
        process = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            while True:
                if process.poll() is not None:
                    raise BenchError(f"server exited ({process.returncode}); see {log_path}")
                try:
                    health = client.call("/health", method="GET")
                    break
                except (OSError, http.client.HTTPException):
                    client.close()
                    if time.perf_counter() - started >= args.startup_timeout:
                        raise BenchError(f"server startup timeout; see {log_path}")
                    time.sleep(.02)
            startup_s = time.perf_counter() - started
            if health.get("backend") != backend or health.get("api_version") != 1:
                raise BenchError(f"unexpected server identity: {health}")
            client.connection.timeout = args.timeout
            if client.connection.sock:
                client.connection.sock.settimeout(args.timeout)
            yield client, {"startup_ready_s": startup_s, "health": health, "log": str(log_path)}
        finally:
            client.close()
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def file_identity(path, hash_content=False):
    info = path.stat()
    identity = {"path": str(path.resolve()), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
    if hash_content:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        identity["sha256"] = digest.hexdigest()
    return identity


def write_reports(output, report):
    temporary = output / "results.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "results.json")
    fields = ["model", "backend", "audio", "mode", "stream_mode", "audio_s", "startup_ready_s",
              "runs", "wall_mean_s", "wall_p50_s", "wall_p95_s", "rtf_mean", "realtime_speed_mean",
              "request_work_rtf_mean", "first_text_p50_s", "first_text_audio_p50_s",
              "first_partial_p50_s", "final_request_p50_s", "capture_to_final_p50_s",
              "request_p50_s", "request_p95_s", "request_max_s", "deterministic_text", "empty_transcripts"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["summaries"])


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, action="append", required=True, help="repeat for a model matrix")
    parser.add_argument("--backend", choices=["cpu", "metal", "vulkan", "cuda"], action="append")
    parser.add_argument("--audio", type=Path, action="append", required=True, help="repeat for multiple WAV files")
    parser.add_argument("--output", type=Path, required=True, help="new output directory (must not exist)")
    parser.add_argument("--mode", choices=["offline", "stream", "both"], default="both")
    parser.add_argument("--stream-mode", choices=["paced", "fast"], default="paced")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seconds", type=float, help="use the first N seconds, at most 30")
    parser.add_argument("--chunk-ms", type=int, default=160)
    parser.add_argument("--step-ms", type=int, default=800)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--startup-timeout", type=float, default=60)
    parser.add_argument("--hash-models", action="store_true", help="hash weights before timing; warms file cache")
    parser.add_argument("--allow-empty", action="store_true", help="allow empty transcripts for silence tests")
    args = parser.parse_args(argv)
    for valid, message in [
        (1 <= args.iterations <= 1000, "iterations must be 1..1000"),
        (0 <= args.warmup <= 100, "warmup must be 0..100"),
        (1 <= args.chunk_ms <= 1000, "chunk-ms must be 1..1000"),
        (160 <= args.step_ms <= 10000, "step-ms must be 160..10000"),
        (1 <= args.threads <= 256, "threads must be 1..256"),
        (1 <= args.max_tokens <= 2048, "max-tokens must be 1..2048"),
        (math.isfinite(args.timeout) and 0 < args.timeout <= 600, "timeout must be >0 and <=600"),
        (math.isfinite(args.startup_timeout) and 0 < args.startup_timeout <= 300, "startup-timeout must be >0 and <=300"),
        (args.seconds is None or math.isfinite(args.seconds) and 0 < args.seconds <= 30, "seconds must be >0 and <=30"),
    ]:
        if not valid:
            parser.error(message)
    args.backend = list(dict.fromkeys(args.backend or ["cpu"]))
    args.model = list(dict.fromkeys(p.resolve() for p in args.model))
    args.audio = list(dict.fromkeys(p.resolve() for p in args.audio))
    for path in [args.server] + args.model + args.audio:
        if not path.is_file():
            parser.error(f"file not found: {path}")
    if args.output.exists():
        parser.error(f"output directory already exists: {args.output}; choose a new directory")
    return args


def main(argv=None):
    args = arguments(argv)
    try:
        audio_data = [(path, read_audio(path, args.seconds)) for path in args.audio]
    except (OSError, ValueError, EOFError, wave.Error) as error:
        print(f"ERROR: invalid benchmark audio: {error}", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True)
    report = {"schema_version": 1, "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "host": {"platform": platform.platform(), "machine": platform.machine(),
                       "processor": platform.processor(), "python": platform.python_version()},
              "server": file_identity(args.server, True),
              "models": [file_identity(p, args.hash_models) for p in args.model],
              "settings": {k: v for k, v in vars(args).items() if k not in ["server", "model", "audio", "output"]},
              "audio": [{"path": str(p), "audio_s": len(pcm) / 32000,
                         "pcm_sha256": hashlib.sha256(pcm).hexdigest()} for p, pcm in audio_data],
              "configurations": [], "results": [], "summaries": [], "errors": []}
    modes = ["offline", "stream"] if args.mode == "both" else [args.mode]
    try:
        for model_index, model in enumerate(args.model):
            for backend in args.backend:
                print(f"Loading {model.name} / {backend}", flush=True)
                log_path = args.output / f"server-{model_index}-{backend}.log"
                try:
                    with running_server(args, model, backend, log_path) as (client, configuration):
                        configuration.update({"model": str(model), "backend": backend})
                        report["configurations"].append(configuration)
                        for audio_path, pcm in audio_data:
                            for mode in modes:
                                case = {"model": str(model), "backend": backend, "audio": str(audio_path),
                                        "audio_s": len(pcm) / 32000, "mode": mode,
                                        "stream_mode": args.stream_mode if mode == "stream" else None,
                                        "startup_ready_s": configuration["startup_ready_s"]}
                                rows = []
                                for index in range(args.warmup + args.iterations):
                                    row = {**case, "warmup": index < args.warmup,
                                           "iteration": index + 1 if index < args.warmup else index - args.warmup + 1}
                                    try:
                                        metrics = (run_offline(client, pcm, args.allow_empty) if mode == "offline"
                                                   else run_stream(client, pcm, args.chunk_ms, args.stream_mode == "paced", args.allow_empty))
                                        row.update(metrics)
                                    except (OSError, ValueError, http.client.HTTPException, BenchError) as error:
                                        row["error"] = str(error)
                                        raise BenchError(f"{audio_path.name} {mode} benchmark failed: {error}") from error
                                    except KeyboardInterrupt:
                                        row["error"] = "interrupted"
                                        raise
                                    finally:
                                        rows.append(row)
                                        report["results"].append(row)
                                summary = {**case, **summarize(rows)}
                                report["summaries"].append(summary)
                                print(f"  {audio_path.name} {mode}: RTF={summary['rtf_mean']:.3f}, "
                                      f"request-work RTF={summary['request_work_rtf_mean']:.3f}, "
                                      f"request p95={summary['request_p95_s'] * 1000:.1f} ms", flush=True)
                                write_reports(args.output, report)
                except (OSError, ValueError, http.client.HTTPException, BenchError) as error:
                    report["errors"].append({"model": str(model), "backend": backend, "error": str(error)})
                    print(f"ERROR: {error}", file=sys.stderr, flush=True)
                finally:
                    write_reports(args.output, report)
    except KeyboardInterrupt:
        report["errors"].append({"error": "interrupted"})
        return 130
    finally:
        report["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_reports(args.output, report)
    print(f"Results: {args.output / 'results.json'}\nSummary: {args.output / 'summary.csv'}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
