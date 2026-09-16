"""One synthetic prefix/failure test; no model calls or private answers."""
from pathlib import Path
import json
import os
import tempfile
import unittest

from scripts.medtrace.astra_judge_bundle import write_new
from scripts.medtrace.stage17_judge import PROTOCOL, digest, recovery_prefix, recovery_chain, execution_path, flags


class Recovery(unittest.TestCase):
    def test_third_recovery_reuses_chain_and_rejects_invalid_predecessors(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            batches = [dict(batch_id=f'batch_{i:03d}',records=[dict(opaque_query_id=str(i))])
                for i in range(1,5)]
            attempts = [root,root/'recovery_01',root/'recovery_02']
            evidence = dict(protocol=PROTOCOL,actual_model='gpt-6-astra',reasoning_effort='high',
                cli_version='same',tool_event_types=[],isolation_checks={'boundary':True})
            approvals = [dict(decision='APPROVED_BY_USER',transport_recovery_attempts=1,
                failed_batch=batches[i+1]['batch_id']) for i in range(3)]
            for index, attempt in enumerate(attempts):
                (attempt/'responses').mkdir(parents=True); (attempt/'execution_evidence').mkdir()
                record = dict(status='FAILED_NO_RETRY',completed_batches=index+1,
                    accepted_same_queue_batches_reused=index)
                if index:
                    record['predecessor_execution_record'] = os.path.relpath(
                        attempts[index-1]/'EXECUTION_RECORD.json',attempt)
                    write_new(attempt/'AUTHORIZATION.json',approvals[index-1])
                write_new(attempt/'EXECUTION_RECORD.json',record)
                batch = batches[index]
                write_new(attempt/'responses'/(batch['batch_id']+'.json'),dict(
                    batch_id=batch['batch_id'],decisions=[dict(opaque_query_id=str(index+1),is_correct=True)]))
                write_new(attempt/'execution_evidence'/(batch['batch_id']+'.json'),dict(
                    evidence,input_binding=digest(batch),status='FORMAT_VALID',exit_code=0))
                failed = batches[index+1]
                write_new(attempt/'execution_evidence'/(failed['batch_id']+'.json'),dict(
                    evidence,input_binding=digest(failed),status='FAILED_NO_RETRY',exit_code=1,
                    command=['--output-last-message',str(attempt/'final.json')],errors=[dict(
                        message='Transport error: network error: error decoding response body')]))
            self.assertEqual(recovery_chain(root,'recovery_02',batches,approvals[-1],'same'),
                (attempts[-1],root/'recovery_03',attempts))
            for invalid in (None,'recovery_01','../recovery_02','recovery_002','recovery_00'):
                with self.assertRaises(ValueError): recovery_chain(root,invalid,batches,approvals[-1],'same')
            (root/'recovery_03').mkdir()
            self.assertEqual(execution_path(root),root/'recovery_03/EXECUTION_RECORD.json')
            with self.assertRaises(ValueError): recovery_chain(root,'recovery_02',batches,approvals[-1],'same')
            (root/'recovery_03').rmdir()
            write_new(root/'final.json',dict(result='must retain'))
            with self.assertRaises(ValueError): recovery_chain(root,'recovery_02',batches,approvals[-1],'same')
            (root/'final.json').unlink()
            # An explicit user interruption is distinct from a transport timeout.
            failure_path = attempts[-1]/'execution_evidence/batch_004.json'
            interrupted = json.loads(failure_path.read_text())
            interrupted.update(exit_code=-15,errors=[])
            failure_path.write_text(json.dumps(interrupted))
            approved = dict(approvals[-1],allow_user_interruption=True,
                failed_input_binding=digest(batches[-1]))
            approval_path = root/'user_approval.json'; write_new(approval_path,approved)
            receipt = dict(status='TERMINATED_BY_USER',decision='APPROVED_BY_USER',
                batch_id='batch_004',input_binding=digest(batches[-1]),signal=15,exit_code=-15,
                processes_exited=True,final_exists_before_signal=False,final_exists_after_exit=False,
                authorization=str(approval_path))
            receipt_path = attempts[-1]/'USER_INTERRUPTION.json'; write_new(receipt_path,receipt)
            with self.assertRaises(ValueError): recovery_chain(root,'recovery_02',batches,approvals[-1],'same')
            self.assertEqual(recovery_chain(root,'recovery_02',batches,approved,'same'),
                (attempts[-1],root/'recovery_03',attempts))
            for change in ({'processes_exited':False},{'input_binding':'wrong'},
                    {'final_exists_after_exit':True}):
                receipt_path.write_text(json.dumps(dict(receipt,**change)))
                with self.assertRaises(ValueError): recovery_chain(root,'recovery_02',batches,approved,'same')
            receipt_path.write_text(json.dumps(receipt))
            for change in ({'exit_code':0},{'errors':[{'message':'semantic failure'}]},
                    {'tool_event_types':['command_execution']},{'isolation_checks':{'boundary':False}}):
                failure_path.write_text(json.dumps(dict(interrupted,**change)))
                with self.assertRaises(ValueError): recovery_chain(root,'recovery_02',batches,approved,'same')
            failure_path.write_text(json.dumps(interrupted))
            write_new(attempts[-1]/'final.json',dict(result='must retain'))
            with self.assertRaises(ValueError): recovery_chain(root,'recovery_02',batches,approved,'same')
            record['predecessor_execution_record'] = '../EXECUTION_RECORD.json'
            (attempts[-1]/'EXECUTION_RECORD.json').write_text(json.dumps(record))
            with self.assertRaises(ValueError): recovery_chain(root,'recovery_02',batches,approvals[-1],'same')

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
            self.assertIn('features.respect_system_proxy=true',config)
            self.assertIn('model_reasoning_effort="high"',config)
            self.assertTrue(any('stream_idle_timeout_ms=900000' in arg for arg in config))
            child = root/'recovery_01'
            (child/'responses').mkdir(parents=True); (child/'execution_evidence').mkdir()
            write_new(child/'EXECUTION_RECORD.json',dict(status='FAILED_NO_RETRY',
                completed_batches=2,accepted_same_queue_batches_reused=1))
            write_new(child/'responses/batch_002.json',dict(batch_id='batch_002',decisions=[
                dict(opaque_query_id='2',is_correct=True)]))
            write_new(child/'execution_evidence/batch_002.json',dict(evidence,
                input_binding=digest(batches[1]),status='FORMAT_VALID',exit_code=0))
            write_new(child/'execution_evidence/batch_003.json',dict(failure,input_binding=digest(batches[2])))
            second_approval = dict(idle_approval,failed_batch='batch_003')
            self.assertEqual(recovery_prefix(child,batches,second_approval,'same',[root]),2)
            with self.assertRaises(ValueError): recovery_prefix(child,batches,second_approval,'same')
            write_new(root/'final.json',dict(result='must not discard'))
            with self.assertRaises(ValueError): recovery_prefix(root,batches,idle_approval,'same')


if __name__ == '__main__':
    unittest.main()
