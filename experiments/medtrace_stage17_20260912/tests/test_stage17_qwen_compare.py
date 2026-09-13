"""Blind packet preparation must reject verdict-bearing or incomplete input."""
import json
from pathlib import Path
import tempfile
import unittest
from scripts.medtrace.stage17_qwen_compare import prepare, accepted_prefix, SNAPSHOT


class BlindPacket(unittest.TestCase):
    def test_reject_extra_score_field(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'operator').mkdir(); (root/'judge_only').mkdir()
            (root/'operator/MANIFEST.json').write_text(json.dumps(dict(records=1,batches=[dict(batch_id='one')])))
            (root/'operator/JUDGE_LOCK.json').write_text(json.dumps(dict(prompt='fixed',config_sha256='lock')))
            row = dict(opaque_query_id='id',question='Q',gold_answer='A',raw_base_answer='B')
            path = root/'judge_only/one.input.json'
            path.write_text(json.dumps(dict(records=[dict(row,is_correct=True)])))
            with self.assertRaises(ValueError): prepare(root,root/'out')
            path.write_text(json.dumps(dict(records=[row])))
            prepare(root,root/'out')
            self.assertEqual(json.loads((root/'out/INPUT.json').read_text())['records'],[row])
            out = root/'out'; cfg = json.loads((out/'INPUT.json').read_text())
            (out/'EXECUTION.json').write_text(json.dumps(dict(completed=1,
                config_id=cfg['config_id'],snapshot=SNAPSHOT)))
            saved = dict(opaque_query_id='id',is_correct=False,config_id=cfg['config_id'],
                snapshot=SNAPSHOT,judge_model=cfg['model'],output_tokens=[1],raw_output=json.dumps(dict(
                    batch_id='batch_0001',decisions=[dict(opaque_query_id='id',is_correct=False)])))
            pending = out/'VERDICTS_QWEN.pending.jsonl'
            pending.write_text(json.dumps(saved)+'\n')
            self.assertEqual(accepted_prefix(out,cfg),[saved])
            pending.write_text(json.dumps(saved)+'\n'+json.dumps(saved)+'\n')
            with self.assertRaises(ValueError): accepted_prefix(out,cfg)
            pending.write_text(json.dumps(dict(saved,opaque_query_id='wrong'))+'\n')
            with self.assertRaises(ValueError): accepted_prefix(out,cfg)
            pending.write_text(json.dumps(saved)+'\n{"unfinished":')
            with self.assertRaises(ValueError): accepted_prefix(out,cfg)


if __name__ == '__main__': unittest.main()
