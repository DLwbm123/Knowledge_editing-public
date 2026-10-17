"""Finite source-only Fisher balancing; all four prior arms are immutable."""
import json
import os
from pathlib import Path
import time

CANDIDATE=os.environ.get('FINITE_VARIANT','M13B')
manifest=Path(os.environ.get('RUN_ROOT','.'))/'RUN_MANIFEST.json'
if manifest.exists():
    saved=json.loads(manifest.read_text()).get('candidate','M13B')
    if 'FINITE_VARIANT' in os.environ:assert saved==CANDIDATE
    CANDIDATE=saved
assert CANDIDATE in ('M13B','M13P')
DIAGNOSE_ENDPOINT=CANDIDATE=='M13B'
REFERENCE='M13' if DIAGNOSE_ENDPOINT else 'M13B'
COST_PREFIX='balance146' if DIAGNOSE_ENDPOINT else 'direction146'
EPOCH='MEDTRACE_FISHER_'+('BALANCE' if DIAGNOSE_ENDPOINT else 'DIRECTION')+'146_20261017_V1'
QUEUE='judge_'+COST_PREFIX+'_astra_medium'
BATCH_LEDGER='Astra_'+COST_PREFIX+'_batches'
JUDGE_MERGE_KEY=COST_PREFIX+'_Judge_merged'
PARENT_QUEUE='judge_factorial146_astra_medium' if DIAGNOSE_ENDPOINT else 'judge_balance146_astra_medium'
PARENT_REPORT='FACTORIAL_RESULTS.json' if DIAGNOSE_ENDPOINT else 'BALANCE_RESULTS.json'
METHODS=('C','M1','M3','M13','M13B')+(() if DIAGNOSE_ENDPOINT else ('M13P',))
METHOD_CONSUMERS={m:2956 for m in METHODS}|{'BASE':2093}
CONSUMERS=sum(METHOD_CONSUMERS.values())
CAPS=dict(updates=23360,backwards=47020 if DIAGNOSE_ENDPOINT else 46728,
    operator_backwards=23506 if DIAGNOSE_ENDPOINT else 23360,local_solves=0,
    generation_attempts=2956,non_generation_forwards=100000,Judge_attempts=5000)

def read(p):return json.loads(Path(p).read_text())

def project_budget(run):
    a=read(run/'private/BALANCE_ADMISSION1.json');parent=Path(a['parent'])
    own=read(run/'RESOURCE_LEDGER.json');prior=read(run/'private/INHERITED_COST.json');old=read(parent/'RESOURCE_LEDGER.json');m=read(parent/'RUN_MANIFEST.json')
    active=lambda x:sum(time.time()-s['started_epoch'] for s in x['gpu_sessions'] if not s.get('ended_epoch'))
    merged=old.get(COST_PREFIX+'_GPU_merged',{}).get('increment',0)
    assert old['gpu_seconds_used']+active(old)+own['gpu_seconds_used']-prior['gpu_seconds_used']+active(own)-merged+1488.5244522094727<m['GPU_seconds_limit']
    assert time.time()<m['deadline_epoch']

def admit(run):
    from m13_only_continue import stopped
    a=read(run/'private/BALANCE_ADMISSION1.json');assert a['status']=='PASS'
    assert a['owners']==list(range(1,147)) and a['new_methods']==[CANDIDATE]
    assert read(run/'PLAN_CONFIG.json')['maximum_counts']==CAPS
    assert a['source_commit']==read(run/'private/GPU_SOURCE_VERSION.json')['commit']!='NOT_LAUNCHED'
    for proc in a['paused_processes']:stopped(**proc)
    assert read(Path(a['parent'])/'private/FISHER_RELEASE_USER_20261016.json')['status']=='RELEASED_BY_USER'
    project_budget(run);return a
