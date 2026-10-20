#!/usr/bin/env python3
"""Paced multi-clip loopback latency measurement; speech onset is an RMS estimate."""
import argparse
import array
import hashlib
import json
import math
import platform
from pathlib import Path
import statistics
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from socket_streaming import WebSocket
from bench import read_audio, running_server


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    at = (len(ordered) - 1) * fraction
    low = int(at)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (at - low)


def summary(rows):
    valid = [row for row in rows if "error" not in row]
    first = [row["first_text_after_speech_ms"] for row in valid if row["first_text_after_speech_ms"] is not None]
    def distribution(values):
        return {"median": statistics.median(values) if values else None, "p95": percentile(values, .95)}
    return {
        "runs": len(rows), "failed_runs": len(rows) - len(valid),
        "no_partial_runs": sum(row["first_text_after_speech_ms"] is None for row in valid),
        "first_text_ms": distribution(first),
        **{name: distribution([event["timing"][key] for row in valid for event in row["results"]])
           for name, key in [("queue_ms", "queueMs"), ("lock_ms", "lockWaitMs"), ("setup_ms", "sessionSetupMs"), ("inference_ms", "inferenceMs")]},
        "updated_inference_ms": distribution([event["timing"]["inferenceMs"] for row in valid for event in row["results"] if event.get("updated")]),
        "final_flush_ms": distribution([row["final_after_audio_ms"] for row in valid if "final_after_audio_ms" in row]),
        "latest_audio_lag_ms": distribution([max(0, event["at_ms"] - event["audio_ms"]) for row in valid for event in row["updates"]]),
        "send_lag_ms": distribution([value for row in valid for value in row["send_lag_ms"]]),
    }


def measure(port, pcm, packet_ms):
    ws = WebSocket(port, "latency-session")
    assert ws.receive()["event"] == "ready"
    start = time.perf_counter()
    errors, send_lag, updates, results = [], [], [], []
    samples = array.array("h", pcm)
    onset = next((i / 16 for i in range(0, len(samples), 320)
                  if math.sqrt(sum((x / 32768) ** 2 for x in samples[i:i + 320]) / len(samples[i:i + 320])) >= .012), None)

    def capture():
        try:
            for offset in range(0, len(pcm), packet_ms * 32):
                chunk = pcm[offset:offset + packet_ms * 32]
                due = start + (offset + len(chunk)) / 32000
                time.sleep(max(0, due - time.perf_counter()))
                ws.send(chunk)
                send_lag.append(max(0, (time.perf_counter() - due) * 1000))
            ws.send(b'{"event":"final"}', 1)
        except Exception as error:
            errors.append(str(error))

    worker = threading.Thread(target=capture)
    worker.start()
    try:
        while True:
            event = ws.receive()
            event["at_ms"] = (time.perf_counter() - start) * 1000
            assert event["event"] != "error", event
            timing = event.get("timing")
            assert timing and all(math.isfinite(timing[k]) and timing[k] >= 0 for k in
                                  ["queueMs", "lockWaitMs", "sessionSetupMs", "inferenceMs"]), event
            if event["type"] == "final":
                final = event
                break
            if event["event"] == "result":
                results.append(event)
            if event["text"].strip() and (not updates or event["text"] != updates[-1]["text"]):
                updates.append(event)
        worker.join(timeout=15)
        assert not errors and not worker.is_alive(), errors
        assert final["audio_ms"] == len(pcm) // 32
        return {"speech_onset_estimate_ms": onset,
                "first_text_after_speech_ms": updates[0]["at_ms"] - onset if updates and onset is not None else None,
                "final_after_audio_ms": final["at_ms"] - len(pcm) / 32,
                "send_lag_ms": send_lag, "updates": updates, "results": results, "final_text": final["text"]}
    finally:
        worker.join(timeout=15)
        ws.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--audio", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packet-ms", type=int, default=160, choices=[20, 40, 80, 160])
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--backend", choices=["cpu", "metal"], default="cpu")
    args = parser.parse_args()
    if args.repeats < 1 or not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("repeats and seconds must be positive")
    args.threads, args.step_ms, args.max_tokens = 4, 160, 512
    args.timeout, args.startup_timeout = 60, 60
    report = {"host": {"system": platform.platform(), "machine": platform.machine()},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "server_sha256": hashlib.sha256(args.server.read_bytes()).hexdigest(),
              "packet_ms": args.packet_ms, "backend": args.backend, "step_ms": 160,
              "threads": 4, "speech_onset_method": "20 ms RMS >= .012; not forced alignment",
              "scope": "paced PCM to loopback server; OS capture and GUI paint not measured",
              "models": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="haolive-latency-") as temp:
        for model in args.model:
            rows = []
            with running_server(args, model, args.backend, Path(temp) / (model.stem + ".log")) as (client, info):
                assert client.call("/v1/models", method="GET")["data"][0]["step_ms"] == 160
                # Warm model/session kernels separately from measured sessions.
                warm = read_audio(args.audio[0], min(2, args.seconds))
                sid = client.call("/v1/sessions")["session_id"]
                client.call(f"/v1/sessions/{sid}/audio?final=true", warm)
                for repeat in range(args.repeats):
                    for audio in args.audio:
                        pcm = read_audio(audio, args.seconds)
                        row = {"clip": audio.name, "repeat": repeat, "audio_ms": len(pcm) // 32,
                               "pcm_sha256": hashlib.sha256(pcm).hexdigest()}
                        try:
                            row.update(measure(info["health"]["websocket_port"], pcm, args.packet_ms))
                        except Exception as error:
                            row["error"] = str(error)
                        rows.append(row)
                        report["models"][model.name] = {"summary": summary(rows), "runs": rows}
                        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
                        print(json.dumps({"model": model.name, "clip": audio.name, "repeat": repeat,
                              "first_text_ms": row.get("first_text_after_speech_ms"), "error": row.get("error")}), flush=True)
            print(json.dumps({"model": model.name, "summary": summary(rows)}), flush=True)


if __name__ == "__main__":
    main()
