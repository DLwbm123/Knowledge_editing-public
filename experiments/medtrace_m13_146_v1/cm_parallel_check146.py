"""CPU admission for disjoint unstarted bank work and retained single outputs."""
import os
from pathlib import Path
import json
import cm_parallel146 as worker


def check(rows):
    ids=lambda xs:[worker.protocol.contract.consumer_id(v) for v in xs]
    parts=[worker.assigned(rows,p) for p in range(3)]
    bank=[r for r in rows if r['mode'] in ('bank','insertion')]
    flattened=[v for part in parts for v in part]
    assert len(ids(flattened))==len(set(ids(flattened)))
    assert set(ids(flattened))==set(ids(bank))
    assert not set(ids(flattened))&set(ids([r for r in rows if r['mode'] in ('base','single')]))
    return parts


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    r=worker.RUN;worker.protocol.admit(r,check_weights=True)
    rows=worker.c.read(r/'private/CM_NEW_PLAN1.json');parts=check(rows)
    assert sum(map(len,parts))==9076
    assert all(not worker.w.worker.dest(v).exists() for part in parts for v in part)
    try:check(rows+[parts[0][0]]*3)
    except AssertionError:pass
    else:raise AssertionError('Duplicate consumer accepted')
    fixtures=worker.c.read(r/'private/CM_PARALLEL_FIXTURES1.json')
    assert [v['method'] for v in fixtures]==['C','M13']
    for v in fixtures:
        assert v['owner']==1 and v['mode']=='single'
        data=worker.c.read(worker.w.worker.dest(v))
        assert data['R0']['raw_token_ids'] and data['binding']['judge_input']['runtime']['actual_precision']=='MATH_FP32_FROZEN_FP16_PREFILL'
    receipt=dict(status='PASS',parts=list(map(len,parts)),unique_consumers=9076,duplicates_rejected=True,single_disjoint=True,unstarted=True,fixtures=2,new_GPU_work=0)
    worker.c.write(r/'private/CM_PARALLEL_CPU_CHECK1.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
