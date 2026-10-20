#!/usr/bin/env python3
"""Paced, full-duplex PCM acceptance using only Python's standard library."""
import argparse
import array
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import struct
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from bench import read_audio, running_server


class WebSocket:
    def __init__(self, port, session, token=""):
        self.socket = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.lock = threading.Lock()
        key = base64.b64encode(os.urandom(16)).decode()
        self.socket.sendall((f"GET /v1/realtime HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
            f"Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\nAuthorization: Bearer {token}\r\nX-Session-Id: {session}\r\n\r\n").encode())
        headers = b""
        while not headers.endswith(b"\r\n\r\n"):
            headers += self.exact(1)
            assert len(headers) <= 8192
        assert headers.startswith(b"HTTP/1.1 101"), headers
        expected = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest())
        assert expected.lower() in headers.lower()

    def exact(self, size):
        data = b""
        while len(data) < size:
            part = self.socket.recv(size - len(data))
            if not part:
                raise EOFError("WebSocket closed")
            data += part
        return data

    def send(self, payload, opcode=2, final=True):
        mask = os.urandom(4)
        length = len(payload)
        header = bytes([(0x80 if final else 0) | opcode])
        if length < 126:
            header += bytes([0x80 | length])
        elif length <= 65535:
            header += b"\xfe" + struct.pack("!H", length)
        else:
            header += b"\xff" + struct.pack("!Q", length)
        masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        with self.lock:
            self.socket.sendall(header + mask + masked)

    def receive(self):
        while True:
            first, second = self.exact(2)
            assert first & 0x80 and not second & 0x80
            length = second & 127
            if length == 126:
                length = struct.unpack("!H", self.exact(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self.exact(8))[0]
            assert length <= 65536
            payload = self.exact(length)
            opcode = first & 15
            if opcode == 9:
                self.send(payload, 10)
                continue
            if opcode == 8:
                raise EOFError("WebSocket close frame " + str(struct.unpack("!H", payload[:2])[0] if len(payload) >= 2 else 0))
            assert opcode == 1
            return json.loads(payload)

    def close(self):
        self.socket.close()


def run(args):
    pcm = read_audio(args.audio, args.seconds)
    duration = len(pcm) / 32000
    with tempfile.TemporaryDirectory(prefix="haolive-duplex-") as tmp:
        with running_server(args, args.model, args.backend, Path(tmp) / "server.log") as (client, info):
            port = info["health"]["websocket_port"]
            assert port > 0
            ws = WebSocket(port, "paced-session")
            assert ws.receive()["event"] == "ready"
            started = time.perf_counter()
            send_lag, errors, updates = [], [], []

            def capture():
                try:
                    for offset in range(0, len(pcm), 5120):
                        chunk = pcm[offset:offset + 5120]
                        due = started + (offset + len(chunk)) / 32000
                        time.sleep(max(0, due - time.perf_counter()))
                        ws.send(chunk)
                        send_lag.append(max(0, time.perf_counter() - due))
                    ws.send(b'{"event":"final"}', 1)
                except Exception as error:
                    errors.append(str(error))

            capture_thread = threading.Thread(target=capture)
            capture_thread.start()
            try:
                while True:
                    value = ws.receive()
                    received = time.perf_counter() - started
                    assert value["event"] != "error", value
                    assert value["session_id"] == "paced-session"
                    if value["text"] and (not updates or updates[-1]["text"] != value["text"]):
                        updates.append({"at_s": received, **value})
                    if value["type"] == "final":
                        final = value
                        break
            finally:
                capture_thread.join(timeout=15)
                ws.close()
            assert not errors and not capture_thread.is_alive(), errors
            assert final["audio_ms"] == len(pcm) // 32
            assert len(updates) >= 5 and updates[0]["at_s"] < duration
            assert final["text"].strip()
            samples = array.array("h", pcm)
            speech_onset = next((i / 16000 for i in range(0, len(samples), 320)
                if math.sqrt(sum((x / 32768)**2 for x in samples[i:i+320]) / 320) >= .012), 0)
            print(json.dumps({"backend": args.backend, "audio_s": duration,
                "speech_onset_s": speech_onset, "first_text_s": updates[0]["at_s"],
                "first_text_after_speech_ms": (updates[0]["at_s"] - speech_onset) * 1000,
                "max_pcm_send_lag_ms": max(send_lag) * 1000, "text_updates": len(updates),
                "first_updates": [{"at_s": round(x["at_s"], 3), "text": x["text"]} for x in updates[:8]],
                "final": final["text"]}, ensure_ascii=False))
            bad = WebSocket(port, "oversize-session")
            assert bad.receive()["event"] == "ready"
            bad.send(bytes(960002))
            assert bad.receive()["event"] == "error"
            bad.close()
            huge = WebSocket(port, "frame-limit-session")
            assert huge.receive()["event"] == "ready"
            huge.socket.sendall(b"\x82\xff" + struct.pack("!Q", 2 * 1024 * 1024) + os.urandom(4))
            try:
                huge.receive()
                raise AssertionError("oversized announced frame must close before reading its body")
            except EOFError as error:
                assert "1009" in str(error), error
            finally:
                huge.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "metal"], required=True)
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--step-ms", type=int, default=160)
    args = parser.parse_args()
    args.threads, args.max_tokens = 4, 512
    args.timeout, args.startup_timeout = 60, 60
    run(args)


if __name__ == "__main__":
    main()
