"""Conditional original146 bank evaluation; zero new training."""
import time
import os
from pathlib import Path
from curriculum_protocol import read
EPOCH='MEDTRACE_CURRICULUM_CONTINUOUS146_20261017_V1'
QUEUE='judge_continuous146_astra_medium'
BATCH_LEDGER='Astra_continuous146_batches'
COST_PREFIX='curriculum_continuous146'
JUDGE_MERGE_KEY=COST_PREFIX+'_Judge_merged'
_admission=Path(os.environ.get('RUN_ROOT','.'))/'private/CONTINUOUS_ADMISSION.json'
SELECTED=read(_admission).get('selected_method') if _admission.exists() else None
METHOD_CONSUMERS={SELECTED or 'M13P':7494,'BASE':2093}
CONSUMERS=9587
CAPS=dict(updates=0,backwards=0,operator_backwards=0,local_solves=0,generation_attempts=4538,non_generation_forwards=20000,Judge_attempts=4538)
def gate(result):
    g=result['primary_gate'];panel=result['contrasts']['P160_to_M13P']['panels']['M13P']['single_1']['T2G']
    return all(g[k] is True for k in ('complete_no_missing','FIT','T1G','T1L','T2L','damage')) and panel['macro'] is not None and panel['macro']>80.
def project_budget(run):
    a=read(run/'private/CONTINUOUS_ADMISSION.json');parent=Path(a['parent']);own=read(run/'RESOURCE_LEDGER.json');prior=read(run/'private/INHERITED_COST.json');old=read(parent/'RESOURCE_LEDGER.json');m=read(parent/'RUN_MANIFEST.json')
    active=lambda x:sum(time.time()-s['started_epoch'] for s in x['gpu_sessions'] if not s.get('ended_epoch'))
    assert old['gpu_seconds_used']+active(old)+own['gpu_seconds_used']-prior['gpu_seconds_used']+active(own)-old.get(COST_PREFIX+'_GPU_merged',{}).get('increment',0)+1488.5244522094727<m['GPU_seconds_limit']
    assert time.time()<m['deadline_epoch']
def admit(run):
    a=read(run/'private/CONTINUOUS_ADMISSION.json');assert a['status']=='PASS' and a['conditional_gate_passed']
    from continuous_selection import choose
    selection=read(run/'private/CONTINUOUS_SELECTION.json')
    assert selection['all_scoring_complete'] and a['selected_method']==choose(selection['candidates']) in ('C','M1','M13P')
    assert a['source_commit']==read(run/'private/GPU_SOURCE_VERSION.json')['commit']!='NOT_LAUNCHED'
    assert read(run/'PLAN_CONFIG.json')['maximum_counts']==CAPS
    project_budget(run);return a
