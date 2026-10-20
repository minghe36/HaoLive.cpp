import importlib.util
import io
import tempfile
from unittest.mock import patch
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import corpus_bench as corpus

class CorpusTests(unittest.TestCase):
    def test_long_audio_preserves_every_sample_and_short_tail(self):
        pcm=bytes(32000*61+2)
        parts=list(corpus.windows(pcm,20))
        self.assertEqual([offset for offset,_ in parts],[0,20,40,60])
        self.assertEqual(b"".join(body for _,body in parts),pcm)
        self.assertEqual(len(parts[-1][1]),32002)
        with self.assertRaises(ValueError): list(corpus.windows(b"x",20))
        with self.assertRaises(ValueError): list(corpus.windows(bytes(2),31))
    def test_failed_warmup_and_window_do_not_skip_remaining_audio(self):
        args=SimpleNamespace(segment_seconds=20,warmup=1,iterations=1,timeout=10)
        child=SimpleNamespace(stdout=io.BytesIO(),returncode=-15)
        client=SimpleNamespace(close=lambda:None)
        usage=SimpleNamespace(ru_maxrss=1234,ru_utime=1,ru_stime=.5)
        calls=[corpus.BenchError('warmup'),corpus.BenchError('window0'),{'wall_s':.1,'text':'second'},{'wall_s':.1,'text':'tail'}]
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus,'start',return_value=(child,client,{'port':123,'startup_s':.1})), patch.object(corpus,'Client',return_value=client), patch.object(corpus,'reap',return_value=usage), patch.object(corpus,'run_offline',side_effect=calls):
            row=corpus.transcribe_file(args,Path('model.gguf'),'cpu',{'relative':'en/audio.mp3'},bytes(41*32000),Path(tmp))
            self.assertEqual(row['status'],'completed_with_errors')
            self.assertEqual(row['failed_segments'],1)
            self.assertEqual(sum(s['samples'] for s in row['segments']),41*16000)
            self.assertEqual(len(row['segments']),3)
            self.assertEqual(row['segments'][-1]['text'],'tail')
            self.assertEqual(len(row['warmup_errors']),1)

    def test_owned_child_rss_units_are_platform_specific(self):
        usage=SimpleNamespace(ru_maxrss=1024)
        self.assertEqual(corpus.rss_bytes(usage,"Darwin"),1024)
        self.assertEqual(corpus.rss_bytes(usage,"Linux"),1024*1024)
    def test_reaps_both_running_and_already_exited_child_without_losing_usage(self):
        for code in ["import time;time.sleep(60)","pass"]:
            process=subprocess.Popen([sys.executable,"-c",code])
            if code=="pass":
                import time
                time.sleep(.1)
            usage=corpus.reap(process)
            self.assertIsNotNone(process.returncode)
            self.assertGreater(corpus.rss_bytes(usage),0)

if __name__=="__main__": unittest.main()
