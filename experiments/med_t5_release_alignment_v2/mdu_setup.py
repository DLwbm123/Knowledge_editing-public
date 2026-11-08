"""Admit eleven added sources; reuse the 148 protected original MDU experts."""
import json
import os
from pathlib import Path
import sys


def read(p):return json.loads(Path(p).read_text())


def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False))


def setup(root,cfg,data):
    root=Path(root);old=Path(cfg['old_t5']);new=root/'MDU';tools=new/'tools'
    original=read(old/'CONFIG.private.json')
    local=dict(original,neutral=cfg['neutral'],source_commit=cfg['source_commit'])
    write(new/'CONFIG.private.json',local);write(new/'CLOCK.json',read(root/'CLOCK.json'))
    previous=read(old/'TASK_INPUTS.private.json');old_owners={t['order'] for t in previous['tasks']}
    new_events=[dict(e,status='BASE_ELIGIBLE') for e in data['events'] if e['owner'] not in old_owners]
    assert len(new_events)==11 and len(data['tasks'])==159
    write(new/'BASE_MASK_FROZEN.json',dict(events=new_events))
    write(new/'ALIGNED_COHORT_MASK.private.json',dict(events=data['events']))
    owners={t['edit_id']:t['order'] for t in data['tasks']}
    base_plan=[dict(p,owner=owners[p['edit_id']]) for p in data['plan'] if p['role'] in ('T0','T5')]
    write(new/'SOURCE_INPUTS.private.json',dict(tasks=data['tasks'],events=data['events'],base_plan=base_plan))
    (new/'base_outputs').symlink_to(old/'base_outputs',target_is_directory=True)
    (new/'BASE_CONFIGURATION.private.json').write_text((old/'BASE_CONFIGURATION.private.json').read_text())
    os.environ.update(read(original['environment']))
    os.environ.update(T5_REMOTE_ROOT=str(new),T5_TOOLS=str(tools),T5_NEUTRAL_ROOT=cfg['neutral'])
    sys.path.insert(0,str(tools))
    import remote_ops
    # Preparation only: no model process starts before the combined input and
    # reuse bindings below are complete and the shared guard launches workers.
    remote_ops.launch=lambda *args,**kwargs:None
    remote_ops.devices=lambda *_:[cfg['gpus']['MDU']]
    prepared=remote_ops.train(new)
    assert prepared['admission']['admitted_cases']==11, 'Do not silently shrink the corrected cohort'
    created=read(new/'TASK_INPUTS.private.json')
    shards={str(t['order']):str(old/previous['owner_shards'][str(t['order'])]) for t in previous['tasks']}
    shards.update(created['owner_shards'])
    assert set(shards)=={str(t['order']) for t in data['tasks']}
    write(new/'TASK_INPUTS.private.json',dict(tasks=data['tasks'],plan=data['plan'],owner_shards=shards,
        events=data['events'],expected_consumers=1908))
    write(new/'CPU_ADMISSION.json',dict(status='PASS',admitted_cases=159,new_experts=11,reused_experts=148,
        total_consumers=1908,updates_max=5280,prior_only_eligibility=True,followup_unknown_retained=11))
    return prepared
