"""Three frozen curricula, with no score-dependent scheduling or tuning."""
import json
import time
from pathlib import Path
METHODS=('C','M1','M13P')
PHASES=(('native',140),('A2',80),('W0',320))
EPOCH='MEDTRACE_CURRICULUM146_20261017_V1'
QUEUE='judge_curriculum146_astra_medium'
BATCH_LEDGER='Astra_curriculum146_batches'
COST_PREFIX='curriculum146'
JUDGE_MERGE_KEY=COST_PREFIX+'_Judge_merged'
METHOD_CONSUMERS={m:2956 for m in (*METHODS,'P160','C160','M1160')}|{'BASE':2093}
CONSUMERS=sum(METHOD_CONSUMERS.values())
CAPS=dict(updates=236520,backwards=411724,operator_backwards=78840,local_solves=0,generation_attempts=8868,non_generation_forwards=500000,Judge_attempts=9000)
def read(p):return json.loads(Path(p).read_text())
def project_budget(run):
    a=read(run/'private/CURRICULUM_ADMISSION.json');parent=Path(a['parent'])
    own=read(run/'RESOURCE_LEDGER.json');prior=read(run/'private/INHERITED_COST.json');old=read(parent/'RESOURCE_LEDGER.json');m=read(parent/'RUN_MANIFEST.json')
    active=lambda x:sum(time.time()-s['started_epoch'] for s in x['gpu_sessions'] if not s.get('ended_epoch'))
    assert old['gpu_seconds_used']+active(old)+own['gpu_seconds_used']-prior['gpu_seconds_used']+active(own)-old.get(COST_PREFIX+'_GPU_merged',{}).get('increment',0)+1488.5244522094727<m['GPU_seconds_limit']
    assert time.time()<m['deadline_epoch']
def admit(run):
    a=read(run/'private/CURRICULUM_ADMISSION.json')
    assert a['status']=='PASS' and a['methods']==list(METHODS) and a['owners']==list(range(1,147))
    assert a['source_commit']==read(run/'private/GPU_SOURCE_VERSION.json')['commit']!='NOT_LAUNCHED'
    assert read(run/'PLAN_CONFIG.json')['maximum_counts']==CAPS
    assert a['storage_write_read'] and a['historical_missing_step_norms']
    project_budget(run);return a

def selected(phase,step):
    assert phase in dict(PHASES) and 1<=step<=dict(PHASES)[phase]
    return ((0,1.),) if phase=='native' else ((0,.5),(1+(step-1)%4,.5))

def gate(result):
    terms=result['paired_native_FIT160_contrasts']['single_1'];p='M13P';ref='P160'
    out=dict(complete_no_missing=result['complete_no_missing'])
    for key in ('FIT','T1G','T1L','T2L'):out[key]=terms[key][p+'_preserves_each_'+ref+'_owner']
    damage=result['protection'][p+'_single_1']
    out['damage']=terms[p+'_damage_no_worse_than_'+ref+'_per_owner'] and damage['new_damage']<=2 and damage['fresh_Base_correct']==85
    out['T2G_improves']=terms['T2G'][p+'_minus_'+ref+'_macro_pp'] is not None and terms['T2G'][p+'_minus_'+ref+'_macro_pp']>0
    out['passed']=all(v is True for v in out.values())
    out['reaches_historical_point_with_protection']=out['passed'] and result['panels'][p]['single_1']['T2G']['macro']>=94.34931506849314
    return out
