"""Freeze role-safe CHECK identities before decoding only selected records."""
import collections
import json
import os
from pathlib import Path
import signal
import time

ROOT=Path(os.environ['EXPERIMENT_ROOT']);PARENT=Path(os.environ['PARENT_RUN']);REPLAY=Path(os.environ['REPLAY_PARENT'])


def write(name,value):
    p=ROOT/name;assert not p.exists(),str(p);p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def main():
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU admission timeout')));signal.alarm(1800)
    import module_run as r
    import stream_index
    from PIL import Image
    assert stream_index.selfcheck()['status']=='PASS'
    deny=json.loads((ROOT/'DENY_GROUPS.private.json').read_text())
    assert len(deny['V95'])==24 and len(deny['V111'])==15
    denied=set(deny['V95'])|set(deny['V111']);clock=json.loads((ROOT/'CLOCK.json').read_text())
    assert time.time()<clock['hard_end']-620
    with (REPLAY/'private/REPLAY_CHECK.json').open('rb') as f:
        metadata=stream_index.index(f);keep=stream_index.select(metadata,denied)
        assert all(v['group_key'] not in denied for v in keep)
        write('ROLE_LOCK.private.json',dict(time=time.time(),groups=31,queries=93,excluded_groups=1,
            records=[{k:r[k] for k in ('group_key','query_id','forced_expert_index','source_qid')} for r in keep],
            source_QA_decoded_before_lock=0,unselected_QA_decoded=0,not_independent_confirmation=True))
        rows=stream_index.decode_selected(f,keep)
    groups=collections.defaultdict(list);images={}
    for row in rows:
        assert row['role']=='REPLAY_CHECK' and row['answer_kind'] in ('yes','no','open')
        idx=row['forced_expert_index'];assert 0<=idx<32
        row['audit_role']='STRESS_CHECK';groups[row['source_group']].append(row)
        p=Path(row['image_path']);assert p.is_file()
        if row['source_group'] in images:assert images[row['source_group']]==str(p)
        else:
            with Image.open(p) as im:im.verify()
            images[row['source_group']]=str(p)
    assert len(groups)==len(images)==31 and all(len(v)==3 for v in groups.values())
    assert all({x['answer_kind'] for x in v}=={'yes','no','open'} for v in groups.values())
    assert len({x['forced_expert_index'] for x in rows})==31
    plan=dict(queries=rows,role='EXPOSED_DEV_FORCED_STRESS',excluded_reserved_groups=1,clinical_truth='UNKNOWN',patient_independence='UNKNOWN')
    write('CHECK_INPUTS_FROZEN.json',plan)
    gpu=json.loads((ROOT/'GPU_EXPECTED.json').read_text())
    lock=dict(physical_GPU=gpu['physical_GPU'],expected_GPU_UUID=gpu['expected_GPU_UUID'],minimum_free_MiB=45056,
        input_binding=r.c.digest(plan),source_commit=os.environ['EXPERIMENT_SOURCE_COMMIT'],budgets=dict(forwards=250000,generations=186,backwards=0,updates=0,vision_forwards=1000),clock=clock)
    write('EVAL_LOCK.json',lock)
    write('EVAL_CPU_ADMISSION.json',dict(status='PASS',time=time.time(),groups=31,queries=93,paired_consumers=186,excluded_reserved_groups=1,
        unselected_QA_decoded=0,images_verified=31,source_qa_only_after_role_lock=True,model_calls=0,clinical='UNKNOWN'))
    print(json.dumps(dict(status='PASS',groups=31,queries=93,excluded_reserved_groups=1,unselected_QA_decoded=0)))

if __name__=='__main__':main()
