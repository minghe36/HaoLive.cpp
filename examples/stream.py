#!/usr/bin/env python3
"""Send a PCM16/16kHz/mono WAV in chunks over one HTTP connection."""
import argparse
import http.client
import json
import wave
from urllib.parse import urlparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--url", default="http://127.0.0.1:9797")
    parser.add_argument("--token", default="")
    parser.add_argument("--chunk-ms", type=int, default=160)
    args = parser.parse_args()
    url = urlparse(args.url)
    if url.scheme != "http" or not url.hostname or args.chunk_ms < 1:
        parser.error("use an http URL and positive chunk size")
    connection = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=120)
    headers = {"Content-Type": "audio/pcm"}
    if args.token:
        headers["Authorization"] = "Bearer " + args.token

    def call(path, body=b"", method="POST"):
        connection.request(method, path, body, headers)
        response = connection.getresponse()
        data = response.read()
        if response.status >= 400:
            raise RuntimeError(data.decode())
        return json.loads(data) if data else {}

    session_id = None
    try:
        with wave.open(args.audio) as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
                raise ValueError("WAV must contain 16 kHz mono PCM16")
            session_id = call("/v1/sessions")["session_id"]
            route = f"/v1/sessions/{session_id}/audio"
            while chunk := audio.readframes(args.chunk_ms * 16):
                result = call(route, chunk)
                if result["updated"]:
                    print(json.dumps(result, ensure_ascii=False), flush=True)
            print(json.dumps(call(route + "?final=true"), ensure_ascii=False), flush=True)
            session_id = None
    finally:
        if session_id:
            try:
                call(f"/v1/sessions/{session_id}", method="DELETE")
            except (OSError, RuntimeError):
                pass
        connection.close()


if __name__ == "__main__":
    main()
