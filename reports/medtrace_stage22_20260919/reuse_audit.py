"""Read-only CPU compatibility check before E0/E1 reuse; immutable historical banks."""
import sys,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage18_cfact import extra_schedule,check_cache,state_hash

def read(p):return json.loads(p.read_text())

def audit(old,a,cfg):
    stream=read(old/'private/STREAM.json');prior=torch.load(old/'private/BANKS.pt',map_location='cpu',weights_only=False);noh=torch.load(a/'private/BANKS.pt',map_location='cpu',weights_only=False);rows=[]
    for task in stream['tasks']:
        edit=task['canonical_edit_id'];n=task['order'];w0=torch.load(a/'private/W0'/f'e{n:03d}.pt',map_location='cpu',weights_only=True)
        assert w0['task']==digest(task) and w0['origin_layer']==30
        assert torch.equal(prior['routes'][edit]['key'],noh['routes'][edit]['key']) and prior['routes'][edit]['radius']==noh['routes'][edit]['radius']
        for arm,s in [('E0',noh['experts'][edit]),('E1',prior['banks']['A'][edit])]:
            saved=torch.load(s['path'],map_location='cpu',weights_only=True);b=saved['binding'];assert saved['step']==320 and s['layer_id']==30 and b['layer_id']==30
            assert digest(b['task'])==digest(task)==s['task'] and b['W0']==w0['W0'] and b['steps']==320
            assert b['runtime']==cfg['runtime_lock'] and b['generation']==cfg['generation_lock'] and b['code']==s['code']
            fit=list(range(1,5));random.Random(task['seed']).shuffle(fit);assert b['fit_order']==fit and b['extra_order']==extra_schedule(task)
            assert all(v.dtype==torch.float32 for v in saved['expert'].values())
            curve=saved['curve'];assert len(curve)==320 and all(c['terms']['native']['weight']==.5 and c['terms']['fit']['weight']==.5 and c['terms']['U']['weight']==.01 for c in curve)
            assert all(('extra' in c['terms'])==(arm=='E1') for c in curve)
            if arm=='E1':assert all(c['terms']['extra']['weight']==1 for c in curve)
            rows.append(dict(arm=arm,position=n,task_binding=digest(task),seed=task['seed'],W0=w0['W0'],writer_layer=30,producer_code=s['code'],status='CPU_COMPATIBLE',runtime_and_generation_exact=True,all_320_weights_verified=True,fit_H_schedule_exact=True,router_key_radius_exact=True))
    teacher_rows=[]
    for path in sorted((a/'private/teacher').glob('*.pt')):
        d=torch.load(path,map_location='cpu',weights_only=True);b=d['binding'];check_cache(d,b)
        assert b['runtime']==cfg['runtime_lock'] and b['generation']==cfg['generation_lock'] and b['teacher']=='FROZEN_BASE_ALL_EDITING_OFF' and b['direction']=='Base||student' and b['vocabulary']=='FULL' and b['dtype']=='float32' and not b['truncation']
        assert len(b['predictors'])==d['logp'].shape[0] and b['vocab_size']==d['logp'].shape[1]
        teacher_rows.append(dict(binding=digest(b),tokens=len(b['teacher_tokens']),predictors=len(b['predictors']),vocabulary=b['vocab_size'],source_binding=digest(b['source']),code=b['code']))
    assert len(rows)==90
    return dict(status='CPU_COMPATIBLE_PENDING_CURRENT_GPU_MECHANICAL_REPLAY',rows=rows,teacher_rows=teacher_rows,teacher_cache_count=len(teacher_rows),new_GPU_seconds=0,new_judgments=0,historical_weights_unchanged=True,HSIC_removal='New train reseeds by fixed task seed before optimizer/batches; identical schedules and same L30 CP-W0; no historical HSIC re-labeling',device_lane='New UUID must stay explicitly registered; CPU compatibility is not full GPU outcome equivalence')

if __name__=='__main__':
    old=Path('/root/rivermind-data/job-520/run');a=Path('/root/rivermind-data/job-521/run');cfg=read(Path('/root/rivermind-data/job-521b/run/private/SUPERVISOR_FACT_FIXED_L31_DOWN.json'))
    print(json.dumps(audit(old,a,cfg)))
