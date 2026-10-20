import csv
import importlib.util
import io
import json
import tempfile
import unittest
import wave
from contextlib import redirect_stderr
from pathlib import Path

spec = importlib.util.spec_from_file_location("haolive_bench", Path(__file__).resolve().parents[1] / "bench" / "bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeClient:
    def __init__(self, clock, latency=.04, partial=True, corrupt=False):
        self.clock, self.latency, self.partial, self.corrupt = clock, latency, partial, corrupt
        self.audio_bytes = 0
        self.calls = []

    def call(self, path, body=b"", method="POST"):
        self.calls.append((path, body, method))
        if method == "DELETE":
            return {}
        if path == "/v1/sessions":
            self.clock.sleep(.01)
            return {"session_id": "42"}
        if path == "/v1/transcribe":
            self.clock.sleep(.1)
            return {"type": "final", "text": "hello", "audio_ms": len(body) // 32}
        final = path.endswith("?final=true")
        self.clock.sleep(.05 if final else self.latency)
        self.audio_bytes += len(body)
        text = "hello" if final or self.partial and self.audio_bytes >= 24000 else ""
        return {"type": "wrong" if self.corrupt else "final" if final else "partial", "text": text,
                "audio_ms": self.audio_bytes // 32}


class TimingTests(unittest.TestCase):
    def run_stream(self, **kwargs):
        clock = Clock()
        client = FakeClient(clock, **{k: v for k, v in kwargs.items() if k != "paced"})
        result = bench.run_stream(client, bytes(32000), 250, kwargs.get("paced", True), clock=clock, sleep=clock.sleep)
        return result, client

    def test_offline_rtf_uses_audio_duration(self):
        clock = Clock()
        result = bench.run_offline(FakeClient(clock), bytes(64000), clock=clock)
        self.assertAlmostEqual(result["wall_s"], .1)
        self.assertAlmostEqual(result["rtf"], .05)

    def test_paced_capture_clock_and_endpointing(self):
        result, client = self.run_stream()
        self.assertAlmostEqual(result["session_create_s"], .01)
        self.assertAlmostEqual(result["first_partial_s"], .79)
        self.assertAlmostEqual(result["first_text_audio_s"], .75)
        self.assertAlmostEqual(result["capture_to_final_s"], .09)
        self.assertAlmostEqual(result["wall_s"], 1.09)
        self.assertAlmostEqual(result["request_work_rtf"], .21)
        self.assertAlmostEqual(result["max_schedule_lag_s"], 0)
        self.assertEqual(client.audio_bytes, 32000)
        self.assertFalse(any(method == "DELETE" for _, _, method in client.calls))

    def test_slow_inference_records_backlog_without_dropping_audio(self):
        result, client = self.run_stream(latency=.4)
        self.assertAlmostEqual(result["max_schedule_lag_s"], .45)
        self.assertAlmostEqual(result["capture_to_final_s"], .9)
        self.assertEqual(client.audio_bytes, 32000)
        self.assertEqual(len(result["chunks"]), 4)

    def test_fast_mode_excludes_simulated_capture_wait(self):
        result, _ = self.run_stream(paced=False)
        self.assertAlmostEqual(result["wall_s"], .21)
        self.assertAlmostEqual(result["first_text_s"], .12)
        self.assertIsNone(result["capture_to_final_s"])
        self.assertIsNone(result["max_schedule_lag_s"])

    def test_text_only_at_final_is_not_reported_as_partial(self):
        result, _ = self.run_stream(partial=False)
        self.assertIsNone(result["first_partial_s"])
        self.assertAlmostEqual(result["first_text_s"], 1.09)
        self.assertEqual(result["first_text_audio_s"], 1)

    def test_failure_cancels_session(self):
        clock = Clock()
        client = FakeClient(clock, corrupt=True)
        with self.assertRaises(bench.BenchError):
            bench.run_stream(client, bytes(32000), 250, False, clock=clock, sleep=clock.sleep)
        self.assertEqual(client.calls[-1], ("/v1/sessions/42", b"", "DELETE"))

    def test_final_audio_tail_is_never_padded_or_lost(self):
        clock = Clock()
        client = FakeClient(clock)
        result = bench.run_stream(client, bytes(32002), 250, False, clock=clock, sleep=clock.sleep)
        self.assertEqual(client.audio_bytes, 32002)
        self.assertEqual(len(result["chunks"]), 5)


class StatisticsTests(unittest.TestCase):
    def test_percentile_edges_and_interpolation(self):
        self.assertIsNone(bench.percentile([], .5))
        self.assertEqual(bench.percentile([4], .95), 4)
        self.assertEqual(bench.percentile([9, 1, 5], .5), 5)
        self.assertAlmostEqual(bench.percentile([1, 3], .95), 2.9)
        self.assertEqual(bench.percentile([3, 1], 0), 1)
        self.assertEqual(bench.percentile([3, 1], 1), 3)
        with self.assertRaises(ValueError):
            bench.percentile([1], 1.1)

    def test_warmup_and_errors_excluded_from_statistics(self):
        row = {"warmup": False, "wall_s": 1, "rtf": .2, "request_work_rtf": .1,
               "request_latencies_s": [.1, .3], "text": "你好"}
        summary = bench.summarize([row, {**row, "wall_s": 3}, {**row, "warmup": True, "wall_s": 999},
                                   {"warmup": False, "error": "failed"}])
        self.assertEqual(summary["runs"], 2)
        self.assertEqual(summary["wall_mean_s"], 2)
        self.assertAlmostEqual(summary["request_p95_s"], .3)
        self.assertTrue(summary["deterministic_text"])
        self.assertEqual(summary["realtime_speed_mean"], 5)
        self.assertIsNone(summary["first_partial_p50_s"])

    def test_no_successful_samples(self):
        self.assertEqual(bench.summarize([{"warmup": True}]), {"runs": 0})

    def test_empty_or_invalid_transcript_rejected(self):
        row = {"type": "final", "audio_ms": 1000, "text": ""}
        with self.assertRaises(bench.BenchError):
            bench.validate_result(row, True, 32000, False)
        bench.validate_result(row, True, 32000, True)
        with self.assertRaises(bench.BenchError):
            bench.validate_result({**row, "audio_ms": 0}, True, 32000, True)


class InputOutputTests(unittest.TestCase):
    def test_wav_clip_identity_and_format_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "speech.wav"
            pcm = bytes(64000)
            with wave.open(str(path), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(pcm)
            self.assertEqual(bench.read_audio(path, .5), pcm[:16000])
            self.assertEqual(bench.read_audio(path), pcm)
            with wave.open(str(path), "wb") as wav:
                wav.setparams((2, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(pcm)
            with self.assertRaises(ValueError):
                bench.read_audio(path)

    def test_json_and_csv_preserve_unicode_and_csv_quoting(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            row = {"model": "有,逗号.gguf", "backend": "cpu", "runs": 2, "first_text_p50_s": None}
            report = {"summaries": [row], "text": "中文"}
            bench.write_reports(output, report)
            self.assertEqual(json.loads((output / "results.json").read_text()), report)
            with (output / "summary.csv").open(encoding="utf-8-sig", newline="") as file:
                result = list(csv.DictReader(file))[0]
            self.assertEqual(result["model"], row["model"])
            self.assertEqual(result["first_text_p50_s"], "")
            self.assertFalse((output / "results.json.tmp").exists())

    def test_argument_validation_and_output_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input"
            path.write_bytes(b"test")
            base = ["--server", str(path), "--model", str(path), "--audio", str(path), "--output", str(Path(tmp) / "new")]
            args = bench.arguments(base + ["--backend", "cpu", "--backend", "cpu"])
            self.assertEqual(args.backend, ["cpu"])
            for extra in [["--iterations", "0"], ["--chunk-ms", "0"], ["--seconds", "nan"],
                          ["--timeout", "nan"], ["--output", tmp]]:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    bench.arguments(base + extra)

    def test_invalid_wav_fails_before_starting_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.wav"
            path.write_bytes(b"invalid wave bytes")
            output = Path(tmp) / "report"
            args = ["--server", str(path), "--model", str(path), "--audio", str(path), "--output", str(output)]
            with redirect_stderr(io.StringIO()):
                self.assertEqual(bench.main(args), 2)
            self.assertFalse(output.exists())



# Duplex timing aggregation must retain failures and missing partials.
sys_path = str(Path(__file__).resolve().parents[1] / "bench")
import sys
sys.path.insert(0, sys_path)
import latency

class DuplexLatencyTests(unittest.TestCase):
    def test_percentile_and_missing_samples(self):
        self.assertIsNone(latency.percentile([], .95))
        self.assertEqual(latency.percentile([10], .95), 10)
        self.assertEqual(latency.percentile([0, 100], .95), 95)
        empty = latency.summary([{"error": "timeout"}])
        self.assertEqual(empty["failed_runs"], 1)
        self.assertIsNone(empty["first_text_ms"]["median"])

    def test_aggregation_counts_no_partial_and_uses_result_timings(self):
        row = {"first_text_after_speech_ms": None, "send_lag_ms": [1, 3],
               "results": [{"updated": True, "timing": {"queueMs": 10, "lockWaitMs": 2,
                            "sessionSetupMs": 3, "inferenceMs": 20}}],
               "updates": [{"at_ms": 1000, "audio_ms": 800}]}
        result = latency.summary([row, {**row, "first_text_after_speech_ms": 600}, {"error": "failed"}])
        self.assertEqual(result["runs"], 3)
        self.assertEqual(result["failed_runs"], 1)
        self.assertEqual(result["no_partial_runs"], 1)
        self.assertEqual(result["first_text_ms"]["median"], 600)
        self.assertEqual(result["latest_audio_lag_ms"]["median"], 200)
        self.assertEqual(result["inference_ms"]["median"], 20)
        self.assertEqual(result["updated_inference_ms"]["median"], 20)

if __name__ == "__main__":
    unittest.main()
