import unittest
import json
from pathlib import Path
import tempfile
import torch
from scripts.medtrace.stage17_campaign import prefixes, query_ids, role_map, schedule, cleanup
from scripts.medtrace.stage17_contract import prefix_router


class CampaignTests(unittest.TestCase):
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


if __name__=='__main__': unittest.main()
