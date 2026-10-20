"""CLI defaults and the example client's matching connection target."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave


class ServerDefaultsTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("HAOLIVE_TEST_SERVER"), "requires built server")
    def test_help_documents_default_port(self):
        output = subprocess.check_output(
            [os.environ["HAOLIVE_TEST_SERVER"], "--help"], encoding="utf-8", timeout=30
        )
        self.assertIn("[--port 9797]", output)
        number = (Path(__file__).resolve().parents[1] / "VERSION").read_text().strip()
        self.assertIn("HaoLive.cpp " + number, output)

    @unittest.skipUnless(os.environ.get("HAOLIVE_TEST_SERVER"), "requires built server")
    def test_compiled_backends_include_cpu_and_are_known(self):
        compiled = json.loads(subprocess.check_output(
            [os.environ["HAOLIVE_TEST_SERVER"], "--list-compiled-backends"],
            encoding="utf-8", timeout=30))
        self.assertIn("cpu", compiled)
        self.assertEqual(len(compiled), len(set(compiled)))
        self.assertLessEqual(set(compiled), {"cpu", "metal", "vulkan", "cuda"})

    def test_stream_example_connects_to_default_or_explicit_url(self):
        spec = importlib.util.spec_from_file_location(
            "stream_example", Path(__file__).resolve().parents[1] / "examples" / "stream.py"
        )
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)
        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / "empty.wav"
            with wave.open(str(audio), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            for options, host, port in [([], "127.0.0.1", 9797),
                                        (["--url", "http://localhost:12345"], "localhost", 12345)]:
                with self.subTest(options=options), \
                     patch.object(sys, "argv", ["stream.py", str(audio), *options]), \
                     patch.object(example.http.client, "HTTPConnection") as connection, \
                     patch("builtins.print"):
                    response = connection.return_value.getresponse.return_value
                    response.status = 200
                    response.read.side_effect = [json.dumps({"session_id": "1"}).encode(), b"{}"]
                    example.main()
                    connection.assert_called_once_with(host, port, timeout=120)
                    connection.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
