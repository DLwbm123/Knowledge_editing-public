"""Frozen C/M13 evaluation of retained native+FIT160 experts; no training."""
import collections
import copy
import json
from pathlib import Path
import time
import contract
from m13_only_continue import complete, stopped

METHODS=('C','M13')
METHOD_CONSUMERS={'C':7494,'M13':7494,'BASE':2093}
CONSUMERS=17081
EPOCH='MEDTRACE_CM_NATIVE146_EVAL_20261016_V1'
QUEUE='judge_cm146_astra_medium'
GPU_SECONDS=72000
NON_GENERATION_FORWARDS=40000


def plan(parent_plan):
    rows=copy.deepcopy(parent_plan)
    for row in rows:
        if row['method']=='FISHER_MIN':row['method']='C'
    assert collections.Counter(r['method'] for r in rows)==METHOD_CONSUMERS
    assert len({contract.consumer_id(r) for r in rows})==CONSUMERS
    assert sum(r['mode']=='insertion' for r in rows)==292
    assert not any(r['query'].get('audit_role')=='GFIT' for r in rows)
    return rows


def read(path):return json.loads(Path(path).read_text())


def admit(run,check_weights=False):
    gate=read(run/'private/CM_ADMISSION1.json');parent=Path(gate['parent'])
    assert gate['status']=='PASS' and gate['launch_authorized'] and gate['GPU']==1
    assert gate['training_updates']==0 and gate['max_GPU_seconds']==GPU_SECONDS
    assert gate['source_commit']==read(run/'private/CM_SOURCE1.json')['commit']
    scope=read(run/'private/CM_SCOPE1.json')
    assert scope['methods']==list(METHODS) and scope['steps']==160 and scope['new_training']==0
    assert read(run/'private/SCOPE_LOCK.json')==scope
    assert read(run/'private/GPU_SOURCE_VERSION.json')==read(run/'private/CM_SOURCE1.json')
    devices=read(run/'RUN_MANIFEST.json')['physical_GPUs']
    if devices!=[1]:
        parallel=read(run/'private/CM_PARALLEL_ADMISSION1.json')
        assert devices==[1,3,4,5] and parallel['status']=='PASS'
        assert parallel['generation_GPUs']==[3,4,5] and parallel['original_controller_stopped']
    assert gate['new_Judge_cap']==15987 and gate['new_generation_cap']==16375
    assert read(parent/'private/M13_ONLY_EXIT3.json')['exit_code']==0
    assert read(parent/'private/M13_ONLY_COMPLETE3.json')['status']=='TRAINING_COMPLETE'
    assert not Path('/proc/2265112').exists(),'Old C/M13 worker has not retired'
    for item in gate['paused_processes']:stopped(**item)
    assert read(run/'private/EVAL_PLAN146.json')==plan(read(parent/'private/EVAL_PLAN146.json'))
    full=read(run/'private/EVAL_PLAN146.json');remaining=read(run/'private/CM_NEW_PLAN1.json')
    reused=read(run/'private/CM_REUSE1.json')
    full_ids={contract.consumer_id(v) for v in full}
    new_ids={contract.consumer_id(v) for v in remaining}
    reused_ids=set(reused['consumer_ids'])
    assert len(reused_ids)==706 and len(new_ids)==16375 and not new_ids&reused_ids
    assert full_ids==new_ids|reused_ids
    if check_weights:
        for owner in range(1,147):
            for arm in METHODS:complete(parent,arm,owner)
    return gate


def project_budget(run):
    gate=read(run/'private/CM_ADMISSION1.json');parent=Path(gate['parent'])
    ledger=read(parent/'RESOURCE_LEDGER.json');own=read(run/'RESOURCE_LEDGER.json')
    prior=read(run/'private/INHERITED_COST.json');manifest=read(parent/'RUN_MANIFEST.json')
    active=lambda data:sum(time.time()-s['started_epoch'] for s in data['gpu_sessions'] if not s.get('ended_epoch'))
    delta=0 if 'cm_eval146_v1_GPU_merged' in ledger else own['gpu_seconds_used']-prior['gpu_seconds_used']+active(own)
    assert ledger['gpu_seconds_used']+active(ledger)+delta+1488.5244522094727<manifest['GPU_seconds_limit']
    assert time.time()<manifest['deadline_epoch']
