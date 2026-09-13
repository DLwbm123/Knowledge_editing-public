"""One synthetic prefix/failure test; no model calls or private answers."""
from pathlib import Path
import json
import tempfile
import unittest

from scripts.medtrace.astra_judge_bundle import write_new
from scripts.medtrace.stage17_judge import PROTOCOL, digest, recovery_prefix, flags


class Recovery(unittest.TestCase):
    def test_preserve_prefix_and_reject_semantic_or_completed_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'responses').mkdir(); (root/'execution_evidence').mkdir()
            batches = [dict(batch_id=f'batch_{i:03d}', records=[dict(opaque_query_id=str(i))]) for i in (1,2,3)]
            write_new(root/'EXECUTION_RECORD.json',dict(status='FAILED_NO_RETRY',completed_batches=1))
            write_new(root/'responses/batch_001.json',dict(batch_id='batch_001',decisions=[
                dict(opaque_query_id='1',is_correct=False)]))
            evidence = dict(protocol=PROTOCOL,actual_model='gpt-6-astra',reasoning_effort='high',
                cli_version='same',tool_event_types=[],isolation_checks={'boundary':True})
            write_new(root/'execution_evidence/batch_001.json',dict(evidence,
                input_binding=digest(batches[0]),status='FORMAT_VALID',exit_code=0))
            failure = dict(evidence,input_binding=digest(batches[1]),status='FAILED_NO_RETRY',exit_code=1,
                command=['--output-last-message',str(root/'final.json')],errors=[dict(
                    message='Transport error: network error: error decoding response body')])
            write_new(root/'execution_evidence/batch_002.json',failure)
            approval = dict(decision='APPROVED_BY_USER',transport_recovery_attempts=1,failed_batch='batch_002')
            self.assertEqual(recovery_prefix(root,batches,approval,'same'),1)
            for changed in (dict(approval,decision='PENDING'),dict(approval,failed_batch='batch_001')):
                with self.assertRaises(ValueError): recovery_prefix(root,batches,changed,'same')
            with self.assertRaises(ValueError): recovery_prefix(root,batches,approval,'changed')
            failure['errors'] = [dict(message='stream disconnected before completion: idle timeout waiting for SSE')]
            (root/'execution_evidence/batch_002.json').write_text(json.dumps(failure))
            with self.assertRaises(ValueError): recovery_prefix(root,batches,approval,'same')
            idle_approval = dict(approval,allow_sse_idle_timeout=True)
            self.assertEqual(recovery_prefix(root,batches,idle_approval,'same'),1)
            config = flags(root,explicit_proxy=True)
            self.assertIn('features.respect_system_proxy=false',config)
            self.assertIn('model_reasoning_effort="high"',config)
            self.assertTrue(any('stream_idle_timeout_ms=900000' in arg for arg in config))
            write_new(root/'final.json',dict(result='must not discard'))
            with self.assertRaises(ValueError): recovery_prefix(root,batches,idle_approval,'same')


if __name__ == '__main__':
    unittest.main()
