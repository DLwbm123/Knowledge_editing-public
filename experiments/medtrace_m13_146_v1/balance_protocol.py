"""Finite source-only Fisher balancing; all four prior arms are immutable."""
import json
from pathlib import Path
import time

EPOCH='MEDTRACE_FISHER_BALANCE146_20261017_V1'
QUEUE='judge_balance146_astra_medium'
JUDGE_MERGE_KEY='balance146_Judge_merged'
METHODS=('C','M1','M3','M13','M13B')
METHOD_CONSUMERS={m:2956 for m in METHODS}|{'BASE':2093}
CONSUMERS=16873
CAPS=dict(updates=23360,backwards=47020,operator_backwards=23506,local_solves=0,generation_attempts=2956,non_generation_forwards=100000,Judge_attempts=5000)

def read(p):return json.loads(Path(p).read_text())

def project_budget(run):
    a=read(run/'private/BALANCE_ADMISSION1.json');parent=Path(a['parent'])
    own=read(run/'RESOURCE_LEDGER.json');prior=read(run/'private/INHERITED_COST.json');old=read(parent/'RESOURCE_LEDGER.json');m=read(parent/'RUN_MANIFEST.json')
    active=lambda x:sum(time.time()-s['started_epoch'] for s in x['gpu_sessions'] if not s.get('ended_epoch'))
    merged=old.get('balance146_GPU_merged',{}).get('increment',0)
    assert old['gpu_seconds_used']+active(old)+own['gpu_seconds_used']-prior['gpu_seconds_used']+active(own)-merged+1488.5244522094727<m['GPU_seconds_limit']
    assert time.time()<m['deadline_epoch']

def admit(run):
    from m13_only_continue import stopped
    a=read(run/'private/BALANCE_ADMISSION1.json');assert a['status']=='PASS'
    assert a['owners']==list(range(1,147)) and a['new_methods']==['M13B']
    assert read(run/'PLAN_CONFIG.json')['maximum_counts']==CAPS
    assert read(run/'private/GPU_SOURCE_VERSION.json')['commit']!='NOT_LAUNCHED'
    for proc in a['paused_processes']:stopped(**proc)
    assert read(Path(a['parent'])/'private/FISHER_RELEASE_USER_20261016.json')['status']=='RELEASED_BY_USER'
    project_budget(run);return a
