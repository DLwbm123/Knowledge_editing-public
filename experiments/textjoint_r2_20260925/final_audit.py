"""Post-run missing bounds and complete-task paired intervals; never changes selection."""
import json,os,random,hashlib
from pathlib import Path
from collections import defaultdict
from statistics import mean
ROOT=Path(os.environ['RUN_ROOT'])
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
scores={p.stem:read(p)['is_correct'] for p in (ROOT/'private/judge/scores').glob('*.json')}
state=read(ROOT/'RUN_STATUS.json')
groups={'DEV24':state['DEV_jobs'],'VERIFY24_SINGLE':state['VERIFY_jobs'],'VERIFY24_SEQUENTIAL':state['sequential_jobs'],'VERIFY24_BASELINE':state['VERIFY_jobs']+state['baseline_jobs']}
output=[];pairs=[];counts=[]
tasks=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks']
inputs={(t['canonical_edit_id'],r['task'],r['query_id']):(r['image_sha256'],r['question']) for t in tasks for r in t['evaluation']}
def average_bounds(rs):
    edits=defaultdict(list)
    for r in rs:edits[r['edit']].append(scores.get(r['judge_key']))
    assert edits
    lo=mean(sum(v is True for v in vs)/len(vs) for vs in edits.values())
    hi=mean(sum(v is not False for v in vs)/len(vs) for vs in edits.values())
    vals=[scores.get(r['judge_key']) for r in rs]
    return [lo,hi],[sum(v is True for v in vals)/len(vals),sum(v is not False for v in vals)/len(vals)]
for name,jobs in groups.items():
    rows=[];exec_inputs=0
    for jid in jobs:
        for path in (ROOT/'jobs'/jid).glob('*/p*/CONSUMERS.json'):
            rr=read(path);rows+=rr
            # Same image+question identity as worker_v3.input_id; count per actual evaluation call.
            exec_inputs+=len({inputs[(r['edit'],r['task'],r['query_id'])] for r in rr})
    buckets=defaultdict(list)
    for r in rows:buckets[(r['arm'],r['mode'],r['prefix'],r['task'])].append(r)
    counts.append(dict(cohort=name,consumers=len(rows),unique_execution_inputs_summed_per_call=exec_inputs,unique_judge_requests=len({r['judge_key'] for r in rows})))
    for (arm,mode,prefix,task),rs in sorted(buckets.items()):
        assert all(scores.get(r['base_judge_key']) is (task in ['T1L','T2L','T2L_PRESSURE']) for r in rs)
        macro,micro=average_bounds(rs)
        output.append(dict(cohort=name,arm=arm,mode=mode,prefix=prefix,task=task,n=len(rs),missing=sum(r['judge_key'] not in scores for r in rs),edit_macro_bounds=macro,micro_bounds=micro))
        if arm=='B0':continue
        baseline=buckets.get(('B0',mode,prefix,task),[])
        identity=lambda r:(r['edit'],r['query_id'])
        aa={identity(r):r for r in baseline};bb={identity(r):r for r in rs}
        assert set(aa)==set(bb)
        per=defaultdict(list);missing=0;down=up=0
        for k,a in aa.items():
            b=bb[k];x=scores.get(a['judge_key']);y=scores.get(b['judge_key'])
            if x is None or y is None:missing+=1;continue
            per[a['edit']].append(int(y)-int(x));down+=x and not y;up+=y and not x
        vals=[mean(v) for v in per.values()];rng=random.Random(20260925)
        boot=sorted(mean(rng.choices(vals,k=len(vals))) for _ in range(2000)) if vals and not missing else []
        base_macro,_=average_bounds(baseline)
        pairs.append(dict(cohort=name,arm=arm,mode=mode,prefix=prefix,task=task,missing=missing,
            delta_bounds=[macro[0]-base_macro[1],macro[1]-base_macro[0]],
            delta_edit_macro=mean(vals) if vals and not missing else None,CI95=[boot[49],boot[1949]] if boot else None,
            observed_right_to_wrong=int(down),observed_wrong_to_right=int(up),full_paired_point_estimate=not missing))
write(ROOT/'public/MISSING_BOUNDS_AND_TASK_PAIRS.json',dict(role='Post-run descriptive audit; selection lock unchanged; bounds are not confidence intervals',bounds=output,pairs=pairs,counts=counts))
assert average_bounds([dict(edit='e',judge_key='absent')])==([0.,1.],[0.,1.])
print('PASS: fixed Base eligibility, paired identities and explicit missing bounds')
