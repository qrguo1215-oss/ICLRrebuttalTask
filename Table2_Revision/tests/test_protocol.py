import json
from pathlib import Path
import tempfile
import unittest

from spec import CONDITIONS, configurations, jobs, key, reuse, verify, splits, signature, write
from task import pending, validate_result, sha_text, reject_model_overrides
from report import assemble


class ProtocolTests(unittest.TestCase):
    def test_fixed_scope_and_reuse(self):
        self.assertEqual(verify()['missing'],2000)
        self.assertEqual(len(jobs()),4400)
        self.assertEqual(len(reuse()),2400)
        self.assertEqual(len(configurations()),6)
        self.assertEqual(len(set(splits()['replay'])),200)
        self.assertEqual(key('math7_cps','G',1),key('math7_seal','G',1))
        self.assertNotEqual(key('math7_rebalance','G',1),key('math7_seal','G',1))
        with tempfile.TemporaryDirectory() as d:
            remaining=pending(d,list(configurations()))
            self.assertEqual(len(remaining),2000)
            self.assertEqual({r['condition'] for r in remaining.values()}, {'V','token_replay','thought_replay'})
            self.assertFalse(any(r['config_id'] in ('math15_seal','math15_rebalance') for r in remaining.values()))

    def test_cps7_does_not_double_scale(self):
        self.assertEqual(configurations()['math7_cps']['coefficient'],1.0)
        from spec import read, ROOT
        self.assertEqual(read(ROOT/'assets/cps7_provenance.json')['runtime_coefficient'],1.0)

    def test_fail_closed_result(self):
        job=dict(config_id='math7_cps',condition='V',index=1,seed=0)
        row=dict(job,protocol='abc',smoke=False,max_new_tokens=10000,dtype='bfloat16',cache_policy='native',
                 text='x',text_sha256=sha_text('x'),token_ids=[1],generated_tokens_including_eos=1)
        validate_result(row,job,'abc')
        for field,value in [('smoke',True),('max_new_tokens',256),('dtype','float16'),('text','bad'),('cache_policy','clean_replay')]:
            with self.assertRaises(ValueError):validate_result(dict(row,**{field:value}),job,'abc')

    def test_exact_denominator(self):
        expected=jobs()
        grades={k:dict(status='ok',correct=j['index']%2==0,strict_correct=None,grading_origin='test') for k,j in expected.items()}
        records={k:dict(provenance=dict(kind='new_table2_generation')) for k in expected}
        result=assemble(grades,records)
        self.assertEqual(len(result),24)
        self.assertTrue(all(r['n']==200 and r['accuracy_percent']*2==r['correct_count'] for r in result))
        grades.pop(next(iter(grades)))
        with self.assertRaises(ValueError):assemble(grades,records)

    def test_model_override_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'model.safetensors').touch()
            with self.assertRaises(ValueError):reject_model_overrides(d,{})


if __name__=='__main__':
    unittest.main()
