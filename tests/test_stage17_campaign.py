import unittest
import json
from pathlib import Path
import tempfile
import torch
from scripts.medtrace.stage17_campaign import prefixes, query_ids, role_map, schedule, cleanup
from scripts.medtrace.stage17_contract import prefix_router


class CampaignTests(unittest.TestCase):
    def test_precision_amendment_is_scoped_and_preserves_frozen_default(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from scripts.medtrace.stage17_campaign import apply_precision_amendment
        runtime=SimpleNamespace(model=torch.nn.Linear(2,2).half(),generation_config={'dtype':'float16'},
            capture_base_guard=Mock(),resolve_module_inventory=Mock())
        apply_precision_amendment(runtime,{},'lora','sequential')
        self.assertEqual(next(runtime.model.parameters()).dtype,torch.float16)
        variant='LORA_SEQUENTIAL_BF16_STABILITY_V1'
        cfg=dict(precision_variant=variant,runtime_lock=dict(precision_variant=variant),
            methods=dict(generation=dict(dtype='bfloat16')))
        with self.assertRaises(ValueError): apply_precision_amendment(runtime,cfg,'lora','single')
        apply_precision_amendment(runtime,cfg,'lora','sequential')
        self.assertEqual(next(runtime.model.parameters()).dtype,torch.bfloat16)
        self.assertTrue(torch.isfinite(runtime.model(torch.ones(1,2,dtype=torch.bfloat16))).all())
        runtime.capture_base_guard.assert_called_once()
        self.assertEqual(runtime.generation_config['dtype'],'bfloat16')

    def test_finite_order_and_no_future_routes(self):
        self.assertEqual(prefixes(146),[1,50,100,146])
        self.assertEqual(prefixes(1),[1])
        self.assertEqual(len(schedule()),8)
        self.assertEqual(schedule()[:2],[('C_NO_H','sequential'),('balancedit','sequential')])
        entries=[dict(logical_edit_id=str(i),key=torch.tensor([float(i)]),radius=0.1,label=[99]) for i in (1,2)]
        bank=prefix_router(entries,1)
        d=bank.route(torch.tensor([2.]))
        self.assertEqual(d.nearest_logical_edit_id,'1')
        self.assertFalse(d.activated)
        self.assertEqual(bank.labels,[()])
        self.assertEqual(prefix_router(entries,2).route(torch.tensor([2.])).logical_edit_id,'2')

    def test_future_targets_only_enter_evaluator_at_their_prefix(self):
        a=dict(image_sha256='image',question='q',reference='a')
        b=dict(a,reference='b')
        tasks=[dict(edit_id='a',native=a,events=[dict(all_probe_query_ids=['x','a'])]),
               dict(edit_id='b',native=b,events=[])]
        ledger=dict(tasks=tasks,main_T0=['a','b'],queries={'x':a})
        self.assertEqual(query_ids(tasks[0]),['a','x'])
        self.assertEqual(role_map(ledger),{'1':{'x':'a'},'2':{'x':'b'}})

    def test_cleanup_refuses_unfinished_consumer(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); phase=root/'private/balancedit_sequential'; phase.mkdir(parents=True)
            (phase/'COMPLETE.json').write_text(json.dumps({'status':'RUNNING'}))
            with self.assertRaises(ValueError): cleanup({'run':d},'balancedit','sequential')
            self.assertFalse((phase/'CLEANUP_PLAN.json').exists())

    def test_complete_report_modes_and_native_trajectory(self):
        from scripts.medtrace.stage17_campaign_closeout import report
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); be=root/'be'; bundle=root/'bundle'
            def put(p,value):
                p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(value))
            native=dict(image_sha256='image',question='q',reference='a',source_group='s')
            ledger=dict(main_T0=['q'],Base_correctness={'q':False},queries={'q':native},
                tasks=[dict(edit_id='q',native=native,events=[dict(task='T0',all_probe_query_ids=['q'])])])
            put(bundle/'source/COHORT_AND_SUPPORT_LEDGER.json',ledger)
            put(bundle/'operator/EXECUTION_RECORD.json',dict(status='COMPLETE_FORMAT_AND_COVERAGE_VALIDATED'))
            put(bundle/'operator/BINDINGS.json',{'v':{}})
            (bundle/'operator/VERDICTS_ASTRA.jsonl').write_text(json.dumps(dict(opaque_query_id='v',is_correct=True))+'\n')
            records=[]
            for method in ('C_NO_H','balancedit','lora','grace','belora'):
                for mode in ('single','sequential'):
                    if method=='balancedit' and mode=='single': continue
                    routes=('R0','RC','FORCED_ON') if method=='C_NO_H' and mode=='single' else ('R0','RC') if method in ('C_NO_H','balancedit') else ('NATIVE',)
                    for panel in ('panel','native'):
                        for route in routes:
                            records.append(dict(method=method,mode=mode,prefix=1,edit='q',panel=panel,
                                query_id='q',route=route,role='original',source='student',opaque_query_id='v',agreement=False))
            put(bundle/'operator/MODE_MAPPING.json',records)
            put(be/'operator/MODE_MAPPING.json',[dict(edit_id='q',query_id='q',modes={
                m:dict(source='student',opaque_query_id='v') for m in ('R0','RC','FORCED_ON')})])
            (be/'operator/VERDICTS_ASTRA.jsonl').write_text(json.dumps(dict(opaque_query_id='v',is_correct=True))+'\n')
            for m,mode in schedule(): put(bundle/'source/campaign/private'/f'{m}_{mode}'/'COMPLETE.json',dict(N=1,seconds=1))
            report(bundle,root/'base',be)
            result=json.loads((bundle/'public/CAMPAIGN_RESULTS.json').read_text())
            self.assertEqual(len(result['panels']),16)
            self.assertTrue(all(p['primary']['micro']==1 for p in result['panels']))
            self.assertTrue(all(t['counts']=={'1_to_1':1} for t in result['insertion_to_final']))
            (bundle/'public').rename(bundle/'before_amendment')
            put(bundle/'source/campaign/private/EXTERNAL_LORA.json',dict(
                acceptance_amendment={'id':'LORA_SINGLE_FP16_SEQUENTIAL_BF16_V1'}))
            report(bundle,root/'base',be)
            result=json.loads((bundle/'public/CAMPAIGN_RESULTS.json').read_text())
            seq=[p for p in result['panels'] if p['method']=='lora' and p['mode']=='sequential']
            self.assertTrue(all(p['precision']=='bfloat16' for p in seq))
            self.assertFalse(result['protocol_amendments'][0]['same_precision_comparison'])
            self.assertIn('not precision matched',(bundle/'public/GPT_PRO_REVIEW.md').read_text())
            (bundle/'public').rename(bundle/'full_campaign')
            report(bundle,root/'base',be,cnoh_only=True)
            result=json.loads((bundle/'public/CAMPAIGN_RESULTS.json').read_text())
            self.assertEqual({p['method'] for p in result['panels']},{'C_NO_H'})
            self.assertEqual(result['paired'],[])
            self.assertEqual(set(result['costs']),{'C_NO_H_sequential'})
            self.assertNotIn('protocol_amendments',result)
            self.assertIn('C_NO_H only',(bundle/'public/GPT_PRO_REVIEW.md').read_text())
            (bundle/'public').rename(bundle/'cnoh_only')
            ready=[p for p in schedule() if p!=('lora','sequential')]
            report(bundle,root/'base',be,phase_subset=ready)
            result=json.loads((bundle/'public/CAMPAIGN_RESULTS.json').read_text())
            self.assertEqual(result['pending_phases'],[['lora','sequential']])
            self.assertFalse(any(p['method']=='lora' and p['mode']=='sequential' for p in result['panels']))
            self.assertNotIn('lora_sequential',result['costs'])
            self.assertFalse(any(p['method']=='lora' for p in result['insertion_to_final']))

    def test_priority_reuse_rejects_incomplete_changed_and_nonboolean_verdicts(self):
        from scripts.medtrace.stage17_campaign_closeout import accepted_priority
        from scripts.medtrace.stage17_prepare import digest, PROTOCOL
        with tempfile.TemporaryDirectory() as d:
            bundle=Path(d); op=bundle/'operator'; op.mkdir()
            def put(name,value): (op/name).write_text(json.dumps(value))
            lock={'config_sha256':'fixed'}; full={'judge':lock,'query_id':'q'}; oid=digest(full)
            put('JUDGE_LOCK.json',lock)
            put('MANIFEST.json',dict(config_sha256='fixed',cnoh_only=True,records=1))
            put('BINDINGS.json',{oid:full})
            put('EXECUTION_RECORD.json',dict(status='RUNNING'))
            verdict=dict(opaque_query_id=oid,is_correct=True,query_id='q',protocol=PROTOCOL,judge_model='gpt-6-astra')
            put('VERDICTS_ASTRA.jsonl',verdict)
            with self.assertRaises(ValueError): accepted_priority(bundle,lock)
            put('EXECUTION_RECORD.json',dict(status='COMPLETE_FORMAT_AND_COVERAGE_VALIDATED'))
            self.assertEqual(accepted_priority(bundle,lock),({oid:full},[verdict]))
            put('BINDINGS.json',{oid:dict(full,query_id='changed')})
            with self.assertRaises(ValueError): accepted_priority(bundle,lock)
            put('BINDINGS.json',{oid:full})
            put('VERDICTS_ASTRA.jsonl',dict(verdict,is_correct=1))
            with self.assertRaises(ValueError): accepted_priority(bundle,lock)
            put('VERDICTS_ASTRA.jsonl',verdict)
            put('EXECUTION_RECORD.json',dict(status='FAILED_NO_RETRY'))
            (op/'recovery_01').mkdir()
            put('recovery_01/EXECUTION_RECORD.json',dict(status='RUNNING'))
            with self.assertRaises(ValueError): accepted_priority(bundle,lock)
            put('recovery_01/EXECUTION_RECORD.json',dict(status='COMPLETE_FORMAT_AND_COVERAGE_VALIDATED'))
            self.assertEqual(accepted_priority(bundle,lock),({oid:full},[verdict]))
            self.assertEqual(json.loads((op/'EXECUTION_RECORD.json').read_text())['status'],'FAILED_NO_RETRY')
            (op/'recovery_02').mkdir()
            put('recovery_02/EXECUTION_RECORD.json',dict(status='FAILED_NO_RETRY'))
            with self.assertRaises(ValueError): accepted_priority(bundle,lock)
            put('recovery_02/EXECUTION_RECORD.json',dict(status='COMPLETE_FORMAT_AND_COVERAGE_VALIDATED'))
            (op/'recovery_03').mkdir()
            put('recovery_03/EXECUTION_RECORD.json',dict(status='FAILED_NO_RETRY'))
            with self.assertRaises(ValueError): accepted_priority(bundle,lock)
            put('recovery_03/EXECUTION_RECORD.json',dict(status='COMPLETE_FORMAT_AND_COVERAGE_VALIDATED'))
            ready=bundle/'ready'; rop=ready/'operator';rop.mkdir(parents=True)
            full2=dict(full,query_id='new');oid2=digest(full2)
            verdict2=dict(verdict,opaque_query_id=oid2,query_id='new')
            for name,value in {
                'JUDGE_LOCK.json':lock,'MANIFEST.json':dict(config_sha256='fixed',records=1,phase_subset=[['grace','single']]),
                'BINDINGS.json':{oid2:full2},'VERDICTS_ASTRA.jsonl':verdict2,
                'EXECUTION_RECORD.json':dict(status='COMPLETE_FORMAT_AND_COVERAGE_VALIDATED'),
                'REUSED_PRIORITY.json':dict(bundle=str(bundle),bindings={oid:full},verdicts=[verdict]),
            }.items(): (rop/name).write_text(json.dumps(value))
            self.assertEqual(accepted_priority(ready,lock),({oid2:full2,oid:full},[verdict2,verdict]))

    def test_phase_subset_is_registered_and_ordered(self):
        from scripts.medtrace.stage17_campaign_closeout import selected_phases
        self.assertEqual(selected_phases(phase_subset=[['grace','single'],['C_NO_H','sequential']]),
                         [('C_NO_H','sequential'),('grace','single')])
        for invalid in ([],[['unknown','single']],[['grace','single'],['grace','single']]):
            with self.assertRaises(ValueError): selected_phases(phase_subset=invalid)


if __name__=='__main__': unittest.main()
