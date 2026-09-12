"""CPU-only structural acceptance tests; synthetic verdicts are NOT model judgments."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.medtrace import astra_judge_bundle as j


class Contract(unittest.TestCase):
    def test_strict_response(self):
        batch = {'batch_id': 'batch_001', 'records': [{'opaque_query_id': 'a'}, {'opaque_query_id': 'b'}]}
        good = {'batch_id': 'batch_001', 'decisions': [
            {'opaque_query_id': 'a', 'is_correct': True}, {'opaque_query_id': 'b', 'is_correct': False}]}
        self.assertEqual(j.validate(batch, j.loads(json.dumps(good))), good['decisions'])
        bad = []
        for value in ['true', 'false', 0, 1, None, [], {}]:
            response = copy.deepcopy(good)
            response['decisions'][0]['is_correct'] = value
            bad.append(response)
        for decisions in [good['decisions'][:1], good['decisions'][::-1], [good['decisions'][0]] * 2, []]:
            bad.append(dict(good, decisions=decisions))
        bad.extend([dict(good, batch_id='batch_002'), dict(good, explanation='extra'), good['decisions']])
        extra = copy.deepcopy(good)
        extra['decisions'][0]['confidence'] = 1
        bad.append(extra)
        unknown = copy.deepcopy(good)
        unknown['decisions'][0]['opaque_query_id'] = 'unknown'
        bad.append(unknown)
        for response in bad:
            with self.subTest(response=response), self.assertRaises(ValueError):
                j.validate(batch, response)
        for text in ['{"x":1,"x":2}', '{"x":NaN}', '```json\n{}\n```', '{} {}', '{"x":']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                j.loads(text)

    def test_prepare_and_merge(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            operator = root / 'operator'
            operator.mkdir()
            rows = [dict(opaque_query_id=f'{i:064x}', adjudication_pass='SOURCE_ANSWER_JUDGE',
                         question='Synthetic question', gold_answer='Synthetic reference', raw_base_answer='Synthetic candidate')
                    for i in range(3)]
            (operator / 'PACKET.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            bindings = {r['opaque_query_id']: dict(question=r['question'], reference=r['gold_answer'],
                        raw_answer=r['raw_base_answer'], judge_type=r['adjudication_pass'], protocol='test-protocol') for r in rows}
            j.write_new(operator / 'JUDGE_LOCK.json', {'config_sha256': 'test-protocol'})
            j.write_new(operator / 'SIDECAR.json', dict(new=3, reused=0, bindings=bindings, protocol_sha256='test-protocol'))
            j.prepare(root, 2)
            manifest = j.read(operator / 'MANIFEST.json')
            self.assertEqual([b['count'] for b in manifest['batches']], [2, 1])
            self.assertEqual(manifest['new_model_calls'], 0)
            execution = j.read(operator / 'EXECUTION_RECORD.template.json')
            execution.update(actual_model=j.MODEL, surface='SYNTHETIC_TEST_ONLY', reasoning_effort='high',
                             completed_at_utc='TEST_ONLY', context_isolation_verified=True, cloud_data_permission_verified=True)
            for entry in manifest['batches']:
                name = entry['batch_id']
                batch = j.read(root / 'judge_only' / (name + '.input.json'))
                self.assertTrue(all(set(r) == j.VISIBLE_FIELDS for r in batch['records']))
                self.assertNotIn('adjudication_pass', (root / 'judge_only' / (name + '.prompt.md')).read_text())
                response = dict(batch_id=name, decisions=[dict(opaque_query_id=r['opaque_query_id'], is_correct=True) for r in batch['records']])
                j.write_new(operator / 'responses' / (name + '.json'), response)
                execution['batches'][name]['evidence_reference'] = 'SYNTHETIC_TEST_ONLY'
            execution_path = operator / 'EXECUTION_RECORD.json'
            j.write_new(execution_path, execution)
            tampered_path = root / 'judge_only' / 'batch_001.input.json'
            original = tampered_path.read_text()
            tampered = j.loads(original)
            tampered['records'][0]['question'] = 'Changed'
            tampered_path.write_text(json.dumps(tampered))
            with self.assertRaises(ValueError):
                j.merge(root, execution_path)
            self.assertFalse((operator / 'VERDICTS_ASTRA.jsonl').exists())
            tampered_path.write_text(original)
            j.merge(root, execution_path)
            output = operator / 'VERDICTS_ASTRA.jsonl'
            saved = output.read_text()
            merged = [j.loads(line) for line in saved.splitlines()]
            self.assertEqual(len(merged), 3)
            self.assertTrue(all(row['judge_model'] == j.MODEL and row['judge_snapshot_sha'] is None for row in merged))
            self.assertTrue(all(row['adjudication_pass'] == 'SOURCE_ANSWER_JUDGE' for row in merged))
            with self.assertRaises(FileExistsError):
                j.merge(root, execution_path)
            self.assertEqual(output.read_text(), saved)


if __name__ == '__main__':
    unittest.main()
