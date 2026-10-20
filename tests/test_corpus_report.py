import itertools
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"bench"))
import corpus_report as report

def simple_distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1): new.append(min(new[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=new
    return row[-1]

class ReportTests(unittest.TestCase):
    def test_bitvector_matches_dynamic_programming_including_empty_sequences(self):
        strings=[''.join(p) for n in range(5) for p in itertools.product('ab',repeat=n)]
        for a in strings:
            for b in strings: self.assertEqual(report.distance(a,b),simple_distance(a,b),(a,b))
    def test_long_unicode_distance_exceeds_machine_word_size(self):
        reference=list('你好world'*100)
        hypothesis=list('您好world'*100+'!')
        self.assertEqual(report.distance(reference,hypothesis),simple_distance(reference,hypothesis))

    def test_mixed_language_normalization_is_explicit(self):
        self.assertEqual(report.tokens("ＡＩ，你好 2026! Don't.",'wer'),['ai','你','好','2026','dont'])
        self.assertEqual(''.join(report.tokens("ＡＩ，你好 2026! Don't.",'cer')),'ai你好2026dont')
    def test_reference_excludes_both_backends_of_target_and_english_chinese_votes(self):
        def row(model,text,backend='metal'):
            return {'status':'ok','model':model,'backend':backend,'audio':'zh/a.mp3','text':text,'segments':[{'index':0,'start_s':0,'end_s':20,'text':text}]}
        current=row('x-asr.gguf','我说错了','cpu')
        allrows=[current,row('x-asr.gguf','自己不能当真值'),row('confucius.gguf','你好'),row('moonshine.gguf','english')]
        pairs=report.reference_segments({'results':allrows},current)
        self.assertEqual(pairs[0][0],'你好')
        self.assertEqual(pairs[0][2]['candidate_models'],['confucius.gguf'])
    def test_coverage_rejects_missing_cases_and_lost_tail(self):
        model={'path':'/models/x-asr.gguf'}
        audio={'relative':'en/a.mp3','samples':32001}
        segments=[{'samples':32000,'start_s':0,'end_s':2}]
        row={'backend':'metal','model':'x-asr.gguf','audio':'en/a.mp3','status':'ok','runs':[{'segments':segments}]}
        result=report.coverage({'settings':{'backend':['metal','cpu']},'models':[model],'audio':[audio],'results':[row]})
        self.assertEqual(result['expected_cases'],2)
        self.assertEqual(len(result['missing_cases']),1)
        self.assertEqual(len(result['sample_errors']),1)

    def test_merge_preserves_original_failure_and_rejects_changed_settings(self):
        def run(status):
            return {'server':{'sha256':'binary'},'models':[{'path':'/models/a.gguf','sha256':'weights'}],
                    'audio':[{'relative':'en/a.mp3','pcm_sha256':'pcm'}],
                    'settings':dict(threads=4,segment_seconds=20,max_tokens=512,iterations=1,warmup=1),
                    'results':[dict(backend='cpu',model='a.gguf',audio='en/a.mp3',artifact_dir='case',status=status)],'finished_at':'now'}
        base=run('error');retry=run('completed_with_errors')
        merged=report.merge_results(base,Path('/base'),[(retry,Path('/retry'))])
        self.assertEqual(merged['results'][0]['first_attempt']['status'],'error')
        self.assertEqual(merged['results'][0]['artifact_dir'],'/retry/case')
        self.assertEqual(base['results'][0]['artifact_dir'],'case')
        retry['settings']['max_tokens']=2048
        with self.assertRaises(ValueError): report.merge_results(base,Path('/base'),[(retry,Path('/retry'))])

    def test_perfect_difference_and_insertions_above_one_hundred_percent(self):
        self.assertEqual(report.distance(['a'],['a','b','c']),2)
        self.assertEqual(report.medoid({'one':'hello','two':'hello','three':'xyz'}),'one')

if __name__=='__main__': unittest.main()
