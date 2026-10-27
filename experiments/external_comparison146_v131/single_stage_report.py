"""Report the fully scored single146 stage without reading sequential verdicts.

The all7494 MEMIT completion/report gate remains unchanged. This separate
stage report never selects, tunes, skips or restarts an experimental consumer.
"""
import collections
import json
import os
from pathlib import Path
import sqlite3
import time

from comparison_report import interval, read, selfcheck, summarize


def main():
    root, parent, cm = [Path(os.environ[key]) for key in ('V131_ROOT','PARENT_RUN','CM_RUN')]
    complete = read(root/'MEMIT_MM/SINGLE_COMPLETE.json')
    assert complete['consumers'] == complete['counts']['formal_consumers'] == 2956
    assert complete['counts']['edits'] == 146 and complete['counts']['inserted'] == 0
    packet = read(parent/'private/M13_SOURCE_PACKET.json')
    tasks, mask = packet['original_tasks'], packet['base_mask']
    assert len(tasks) == 146
    wanted = ('M13P_160','ALPHAEDIT_MM','MEMIT_MM')
    scores, bindings, coverage = {}, {}, {}
    with sqlite3.connect('file:'+str(root/'judge_v131/queue.sqlite')+'?mode=ro',uri=True) as db:
        db.execute('BEGIN')
        for method,owner,qid,state,correct,binding in db.execute(
            "SELECT c.method,c.owner,c.query_id,p.state,p.correct,p.binding FROM consumer c "
            "JOIN payload p ON p.key=c.payload_key WHERE c.mode='single' AND c.prefix=1 "
            "AND c.method IN (?,?,?)", wanted):
            assert state in ('FORMAT_VALID','REUSED') and correct in (0,1)
            key = (method,owner,qid)
            assert key not in scores
            scores[key], bindings[key] = correct, json.loads(binding)['binding']
            coverage[method,state] = coverage.get((method,state),0)+1
        attempts = dict(db.execute("SELECT key,value FROM meta WHERE key!='prompt'"))
    assert collections.Counter(key[0] for key in scores) == dict(M13P_160=2956,ALPHAEDIT_MM=2372,MEMIT_MM=2956)
    with sqlite3.connect('file:'+str(cm/'private/judge_cm146_astra_medium/queue.sqlite')+'?mode=ro',uri=True) as db:
        for owner,qid,state,correct,binding in db.execute(
            "SELECT c.edit_order,c.query_id,p.status,p.correct,p.binding FROM consumer c "
            "JOIN payload p ON p.key=c.payload_key WHERE c.method='C' AND c.mode='single' AND c.prefix=1"):
            assert state == 'FORMAT_VALID' and correct in (0,1)
            key = ('C',owner,qid)
            assert key not in scores
            scores[key], bindings[key] = correct, json.loads(binding)
    assert sum(key[0]=='C' for key in scores) == 2956
    for method,owner,qid in scores:
        if method != 'MEMIT_MM':
            continue
        for other in ('C','M13P_160'):
            left, right = bindings[method,owner,qid], bindings[other,owner,qid]
            for field in ('question','reference','image_sha256','prompt_ids'):
                assert left[field] == right[field], ('Common single query identity',field)
    methods = ('C',*wanted)
    panels, per_owner = {}, {}
    supports = dict(T0=146,FIT=584,T1G=578,T2G=557,T1L=2,T2L=54)
    for method in methods:
        panels[method] = {}
        for role in supports:
            if method == 'ALPHAEDIT_MM' and role == 'FIT':
                panels[method][role] = dict(status='NOT_MEASURED_NOT_ZERO')
                continue
            values = {}
            for task in tasks:
                qids = ([f"edit_{task['order']}_{i}" for i in range(1,5)] if role=='FIT' else
                    list(dict.fromkeys(q for event in task['events'] if event['task']==role
                                       for q in event['all_probe_query_ids'])))
                if role not in ('T0','FIT'):
                    qids = [qid for qid in qids if mask[qid] == role.endswith('L')]
                values[task['order']] = [scores[method,task['order'],qid] for qid in qids]
            per_owner[method,role] = values
            panels[method][role] = summarize(values)
            assert panels[method][role]['observations'] == supports[role]
            assert panels[method][role]['missing'] == 0
    assert panels['M13P_160']['T2G']['correct'] == 427
    assert abs(panels['M13P_160']['T2G']['macro']-76.48401826484019)<1e-9
    assert panels['C']['T2G']['correct'] == 456
    contrasts = {}
    for other in ('C','M13P_160','ALPHAEDIT_MM'):
        contrasts['MEMIT_minus_'+other] = {}
        for role in ('T0','T1G','T2G','T2L'):
            left, right = per_owner[other,role], per_owner['MEMIT_MM',role]
            supported = [task for task in tasks if left[task['order']]]
            delta = [(sum(right[t['order']])-sum(left[t['order']]))/len(left[t['order']]) for t in supported]
            contrasts['MEMIT_minus_'+other][role] = dict(macro_pp=100*sum(delta)/len(delta),
                edit_CI_pp=interval(delta), source_CI_pp=interval(delta,[t['native']['source_group'] for t in supported]),
                MEMIT_better=sum(v>0 for v in delta),same=sum(v==0 for v in delta),MEMIT_worse=sum(v<0 for v in delta))
    locality = {method:dict(observations=56,
        correct=sum(panels[method][role]['correct'] for role in ('T1L','T2L')),
        lost=sum(panels[method][role]['observations']-panels[method][role]['correct'] for role in ('T1L','T2L')),
        reference='Frozen original Base-correct primary locality mask; not fresh medium Base85 audit') for method in methods}
    result = dict(status='SINGLE146_STAGE_COMPLETE_SEQUENTIAL_NOT_COMPLETE',snapshot_time=time.time(),
        MEMIT_single_consumers=2956,MEMIT_single_all_accepted=True,
        MEMIT_single_coverage={state:coverage.get(('MEMIT_MM',state),0) for state in ('FORMAT_VALID','REUSED')},
        panels=panels,task_paired_descriptive_contrasts=contrasts,primary_mask_locality=locality,
        MEMIT_single_complete_counts=complete['counts'],global_judge_cost_snapshot_not_single_only=attempts,
        bootstrap=dict(samples=10000,seed=13120261025,source_groups_not_patients=True),selfcheck=selfcheck(),
        sequential_verdicts_queried=0,new_GPU_or_Judge_calls=0,original_all7494_completion_gate_unchanged=True,
        limitations=['Original146 exposed DEV, not independent or clinical confirmation.',
            'C/M13P use medium Judge; MEMIT and historical Alpha use high. Alpha retains historical FP16.',
            'Paired task/probe identity is not a fully controlled causal method comparison; information and compute differ.',
            'MEMIT is the locked explicit multimodal adaptation, not the authors native VLM implementation.',
            'T1L has one edit/two probes; T2L has31 edits/54 probes. Primary mask differs from fresh Base85 audit.',
            'Single results do not establish cumulative editing performance; four sequential interrupted consumers remain UNKNOWN.',
            'No sequential correctness was queried and no execution choice changes because of these results.'])
    path=root/'MEMIT_SINGLE146_STAGE_RESULTS.json'
    assert not path.exists(), 'Do not silently overwrite a published stage snapshot'
    path.write_text(json.dumps(result,indent=2)+'\n')
    (root/'MEMIT_SINGLE146_STAGE_REPORT_COMPLETE.json').write_text(json.dumps(dict(
        time=time.time(),status=result['status'],single_only=True,all7494_complete=False))+'\n')
    print(json.dumps(dict(panels=panels,locality=locality,MEMIT_single_coverage=result['MEMIT_single_coverage']),indent=2))


if __name__=='__main__':
    main()
