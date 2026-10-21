"""Three-stage stopping and save-before-reject on a synthetic model."""
import torch
from unittest.mock import patch
from .cpu_check import Runtime,record
from .editor import calibrate
from .direct_fisher16 import estimate
from .diagnose import candidate,FIRST_D
from .chainprecision_cpu_check import run as prior
from .fp32qualify_engine import qualify


def run():
    prior();torch.manual_seed(42);rt=Runtime(torch.float16);r=record('test');name='model.layers.0.mlp.down_proj'
    train=[record('a'),record('b',image=.2,target='3')]
    f=calibrate(rt,train,[name],train_ids=['a','b'],excluded_ids=['test'],binding={'model':'synthetic'})['factors'][name]
    diag,_=estimate(rt,train,f,lambda e:None);base=rt.model.layer.weight.detach().clone();updates=[]
    p=candidate(rt,r,0,f,FIRST_D/16,solve_factor=diag,update_observer=lambda u:updates.append(u.clone()))
    out={};qualify(rt,r,base,updates[0],out,lambda:None,None,roots=[rt.model])
    assert out['status']=='COMPLETE' and out['stages_completed']==[1,2,3] and out['base_losses_restored']
    from . import fp32qualify_engine as engine
    original=engine.nll;counter=[0];saved=[];out={}
    def bad(model,batch):
        counter[0]+=1
        value=original(model,batch)
        return value+.1 if counter[0]>=3 else value
    # Disturb the second fixed-weight reading, before any roundtrip or reentry.
    with patch.object(engine,'nll',side_effect=bad):
        try:qualify(rt,r,base,updates[0],out,lambda:saved.append(len(out.get('events',[]))),None,roots=[rt.model])
        except ValueError:pass
        else:raise AssertionError('must stop')
    assert out['status']=='STOPPED' and not out['stages_completed']
    assert out['events'][-1]['baseline_difference']>1e-5 and saved
    assert out['base_losses_restored'] and out['precision_flags_hooks_restored'] and torch.equal(base,rt.model.layer.weight)
    return dict(status='PASS',prior_checks=True,three_stages=True,persist_before_failure=True,first_failure_stops=True,full_restore=True,no_scientific_grid=True)
