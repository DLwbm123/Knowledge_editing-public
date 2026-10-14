"""Whole146 synthetic scoring/report fixture; accepted labels are test data only."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import contract
import astra_queue as a


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    packet=json.loads(Path(os.environ['M13_CPU_PACKET']).read_text())
    stage=Path(os.environ['M13_CPU_CODE']);sys.path.insert(0,str(stage))
    spec=importlib.util.spec_from_file_location('prepared_m13_report',stage/'report146.py')
    report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
    plan=contract.coverage_reservation(packet['original_tasks'],packet['base_mask'])
    with tempfile.TemporaryDirectory(prefix='e.',dir=os.environ['M13_CPU_TMP']) as tmp:
        root=Path(tmp);report.RUN=root;q=report.q;q.ROOT=root/'private/judge_m13_146_astra_medium'
        db=q.connect();judge=dict(model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT)
        q.write(q.ROOT/'EPOCH_MANIFEST.json',dict(epoch='SYNTHETIC_FIXTURE_ONLY',**judge))
        q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(status='CPU_FIXTURE'))
        for worker in range(4):q.write(q.ROOT/'workers'/str(worker)/'SCORER_DONE.json',dict(status='CPU_FIXTURE'))
        q.write(root/'private/ORIGINAL146_TASKS.json',packet['original_tasks'])
        q.write(root/'private/ORIGINAL_BASE_MASK146.json',packet['base_mask'])
        for path in ['RESOURCE_LEDGER.json','private/INHERITED_COST.json']:
            q.write(root/path,dict(gpu_sessions=[],gpu_seconds_used=0,Judge_attempts=0))
        for i,row in enumerate(plan):
            inp=dict(query_id=row['query']['query_id'],question='fixture Q',reference='fixture A',image_sha256='fixture image')
            binding=dict(inp,image_path='fixture image',prompt_ids=[1],attention_mask=[1],runtime={'precision':'FP32'},generation={'max_new_tokens':1024})
            out=dict(raw_answer='fixture A',raw_token_ids=[['BASE','M1','M3','M13'].index(row['method'])])
            data=dict(method=row['method'],mode=row['mode'],prefix=row['prefix'],eval_owner=row['owner'],active_target=False,
                binding=dict(input=inp,judge_input=binding),R0=out)
            path=root/'private/outputs'/f'{i}.json';q.write(path,data)
            key=q.payload(db,inp,binding,out)
            db.execute("UPDATE payload SET status='FORMAT_VALID',correct=1 WHERE key=?",(key,))
            db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',
                (str(i),row['method'],row['mode'],row['prefix'],row['owner'],'FIXTURE',inp['query_id'],str(path),q.digest(data),key))
        db.commit();report.main()
        result=q.read(root/'public/M13_146_RESULTS.json')
        assert result['gates']['complete'] and result['T2L']['original_mask_probes']==54 and result['T2L']['original_mask_edits']==31
        assert set(result['panels'])=={'M1','M3','M13'} and result['gates']['M3_damage_per_owner']
        assert result['panels']['M13']['bank_1']['status']=='PRIVATE_SMALL_GROUP_AUDIT_ONLY'
        assert result['gates']['T2L_edit_CI_positive'] is False and result['decision']=='146_COMPATIBILITY_NOT_ESTABLISHED'
        key=db.execute("SELECT payload_key FROM consumer WHERE method='M13' AND query_id='edit_1_1' LIMIT 1").fetchone()[0]
        db.execute('UPDATE payload SET correct=0 WHERE key=?',(key,));db.commit();report.main()
        assert q.read(root/'public/M13_146_RESULTS.json')['gates']['FIT_per_owner'] is False
        db.execute("UPDATE payload SET status='MISSING',correct=NULL WHERE key=?",(key,));db.commit();report.main()
        assert q.read(root/'public/M13_146_RESULTS.json')['gates']['complete'] is False
        db.close()
    checks=dict(status='PASS',synthetic_scores_only=True,consumers=24867,three_method_panels=True,
        original_T2L_31_edits_54_probes=True,one_edit_regression_blocks_success=True,
        missing_blocks_complete=True,zero_CI_blocks_success=True,M3_damage_gate_present=True,
        new_GPU_leases=0,new_VLM_calls=0,new_Judge=0,training_ready=False)
    Path(os.environ['M13_CPU_REPORT']).write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))


if __name__=='__main__':main()
