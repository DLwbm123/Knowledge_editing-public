"""Trust-boundary tests for merging separately bound judge outputs; no inference."""
import json
from pathlib import Path
import tempfile
import unittest

from .qwen_remaining import digest, merge, MODEL, PROTOCOL, REVISION


class HybridMergeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.op=self.root/'operator_hybrid'
        (self.op/'gpu_result/chunks').mkdir(parents=True);(self.root/'operator').mkdir()
        originals={f'source_{i}':dict(query_id=str(i),judge={'model':'gpt-6-astra'}) for i in range(5)}
        lock={'model':MODEL,'revision':REVISION};lock['config_sha256']=digest(lock)
        bindings={digest(dict(v,judge=lock)):dict(v,judge=lock) for k,v in list(originals.items())[2:]}
        ids={oid:'source_'+v['query_id'] for oid,v in bindings.items()}
        self.put(self.root/'operator/BINDINGS.json',originals)
        self.put(self.op/'QWEN_LOCK.json',lock);self.put(self.op/'QWEN_BINDINGS.json',bindings)
        self.put(self.op/'SOURCE_IDS.json',ids)
        self.put(self.op/'ASTRA_ACCEPTED.json',[dict(opaque_query_id=f'source_{i}',judge_model='gpt-6-astra',is_correct=True) for i in range(2)])
        self.put(self.op/'gpu_result/STATUS.json',{'state':'COMPLETE','completed_records':3})
        self.verdicts=[dict(opaque_query_id=oid,is_correct=False,judge_model=MODEL,model_revision=REVISION,protocol=PROTOCOL) for oid in bindings]
        self.output=self.op/'gpu_result/chunks/000000.VERDICTS.json';self.put(self.output,self.verdicts)

    def put(self,path,value):path.write_text(json.dumps(value))

    def test_preserves_actual_judge_labels_and_source_identity(self):
        result=merge(self.root)
        self.assertEqual(result['record_counts'],{'gpt-6-astra':2,MODEL:3})
        rows=[json.loads(x) for x in (self.op/'VERDICTS_HYBRID.jsonl').read_text().splitlines()]
        self.assertEqual({r['opaque_query_id'] for r in rows},{f'source_{i}' for i in range(5)})
        self.assertTrue(all(r['judging_opaque_query_id']!=r['opaque_query_id'] for r in rows[2:]))

    def test_rejects_qwen_outputs_mislabelled_as_astra(self):
        self.verdicts[0]['judge_model']='gpt-6-astra';self.put(self.output,self.verdicts)
        with self.assertRaisesRegex(ValueError,'scientific binding'):merge(self.root)
        self.assertFalse((self.op/'VERDICTS_HYBRID.jsonl').exists())

    def test_rejects_duplicate_verdicts_even_with_expected_count(self):
        self.verdicts[1]=self.verdicts[0];self.put(self.output,self.verdicts)
        with self.assertRaisesRegex(ValueError,'coverage'):merge(self.root)


if __name__=='__main__':unittest.main()
