#!/usr/bin/env python3
"""Real-model SSE regression; caller supplies the engine, model and speech WAV."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from bench import read_audio, running_server


def streamed(client, path, pcm):
    started = time.perf_counter()
    client.connection.request("POST", path, pcm, {"Content-Type": "audio/pcm"})
    response = client.connection.getresponse()
    assert response.status == 200
    assert response.getheader("Content-Type").startswith("text/event-stream")
    events, frame = [], []
    while line := response.readline():
        if line == b"\n":
            event = next(x[7:] for x in frame if x.startswith(b"event: "))
            data = next(x[6:] for x in frame if x.startswith(b"data: "))
            events.append((event.decode(), json.loads(data), time.perf_counter() - started))
            frame = []
        else:
            frame.append(line.rstrip(b"\n"))
    assert not frame, "truncated SSE frame"
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "metal"], required=True)
    args = parser.parse_args()
    args.threads, args.step_ms, args.max_tokens = 4, 320, 512
    args.timeout, args.startup_timeout = 60, 60
    pcm = read_audio(args.audio, 12)
    with tempfile.TemporaryDirectory(prefix="haolive-sse-") as tmp:
        with running_server(args, args.model, args.backend, Path(tmp) / "server.log") as (client, _):
            sid = client.call("/v1/sessions")["session_id"]
            route = f"/v1/sessions/{sid}/audio?stream=true"
            first = streamed(client, route, pcm[:3 * 32000])
            assert first[-1][0] == "result"
            partials = [value for event, value, _ in first if event == "partial"]
            assert len({value["text"] for value in partials}) >= 2, first
            assert all(value["session_id"] == sid and value["sequence"] == 1 for value in partials)
            assert all("|" not in value["text"] and value["audio_ms"] == 3000 for value in partials)
            assert first[0][2] < first[-1][2], "partial must arrive before request completion"
            second = streamed(client, route, pcm[3 * 32000:])
            assert second[-1][0] == "result" and second[-1][1]["sequence"] == 2
            assert len([event for event, _, _ in second if event == "partial"]) <= 32
            assert all(value["audio_ms"] == len(pcm) // 32 for event, value, _ in second if event == "partial")
            final = streamed(client, route + "&final=true", b"")
            assert final[-1][0] == "result" and final[-1][1]["type"] == "final"
            assert len(final[-1][1]["text"]) > len(first[-1][1]["text"]), "later speech must remain visible"
            missing = streamed(client, route, b"")
            assert missing[-1][0] == "error" and "code" in missing[-1][1]["error"]
            print(json.dumps({"backend": args.backend, "partial_events": len(partials),
                "first_token_ms": round(first[0][2] * 1000, 1),
                "request_complete_ms": round(first[-1][2] * 1000, 1),
                "first_tokens": [value["text"] for value in partials[:6]],
                "final": final[-1][1]["text"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
