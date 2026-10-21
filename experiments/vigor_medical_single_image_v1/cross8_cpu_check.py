"""Independent metric routing, no-half safe stop and one temporary endpoint probe."""
from unittest.mock import patch
from unittest import TestCase
import torch
from .normmatch4_cpu_check import run as prior_checks
from .cpu_check import Runtime,record
from .editor import calibrate,paired_batches
from .direct_fisher16 import estimate
from .backoff16 import _attempt
from .cross8 import run_arm
from .finite_null import native_losses


def run():
    prior_checks();torch.manual_seed(42)
    rt=Runtime(torch.float16);r=record('test');name='model.layers.0.mlp.down_proj'
    train=[record('a'),record('b',image=.2,target='3')]
    factor=calibrate(rt,train,[name],train_ids=['a','b'],excluded_ids=['test'],binding={'model':'synthetic'})['factors'][name]
    diagonal,_=estimate(rt,train,factor,lambda e:None)
    pair=paired_batches(rt,r);base=rt.model.layer.weight.detach().clone();losses=native_losses(rt,pair)
    for arm in ('00','11','10','01'):
        observed={}
        def main(*args,**kw):observed['main']=kw.get('solve_factor');return {}
        def corr(*args,**kw):observed['correction']=kw.get('solve_factor');assert kw['compensate']=='secant';return dict(status='APPLIED')
        final=dict(gates=dict(null=True,visual=True,trust=True),accepted=True)
        with patch('experiments.vigor_medical_single_image_v1.backoff16.candidate',side_effect=main),patch('experiments.vigor_medical_single_image_v1.backoff16.correction',side_effect=corr),patch('experiments.vigor_medical_single_image_v1.backoff16.gates',side_effect=[dict(gates=dict(null=False),accepted=False),final]):
            _attempt(rt,r,factor,pair,base,losses,losses[1],.1,None,{},lambda:None,
                diagonal if arm[0]=='1' else None,correction_factor=diagonal if arm[1]=='1' else None,separate_correction=True)
        assert observed['main'] is (diagonal if arm[0]=='1' else None)
        assert observed['correction'] is (diagonal if arm[1]=='1' else None)
    calls=[]
    def rejected(*args,**kw):
        calls.append(args[7]);args[9].update(proposal=dict(requested_reduction=args[7],effective_reduction=args[7],clipped=False))
        with torch.no_grad():rt.model.layer.weight.add_(.01)
        return dict(accepted=False)
    state=dict(arm='10',checks=[])
    with patch('experiments.vigor_medical_single_image_v1.cross8._attempt',side_effect=rejected),patch('experiments.vigor_medical_single_image_v1.cross8.candidate') as probe:
        run_arm(rt,r,factor,diagonal,state,lambda:None,None)
        assert len(calls)==1 and not state['prefix_completed'] and state['accepted_steps']==0
        assert state['steps'][0]['rollback_pass'] and probe.call_count==0
    assert torch.equal(base,rt.model.layer.weight)
    def accepted(*args,**kw):
        args[9].update(proposal=dict(requested_reduction=args[7],effective_reduction=args[7],clipped=False))
        return dict(accepted=True)
    state=dict(arm='01',checks=[])
    from .diagnose import candidate
    with patch('experiments.vigor_medical_single_image_v1.cross8._attempt',side_effect=accepted),patch('experiments.vigor_medical_single_image_v1.cross8.candidate',wraps=candidate) as probe:
        run_arm(rt,r,factor,diagonal,state,lambda:None,None)
        assert state['accepted_steps']==8 and probe.call_count==1 and state['probe']['endpoint_restored']
    assert torch.equal(base,rt.model.layer.weight)
    with TestCase().assertRaises(TimeoutError):run_arm(rt,r,factor,diagonal,dict(arm='10',checks=[]),lambda:None,0)
    assert torch.equal(base,rt.model.layer.weight)
    return dict(status='PASS',prior_checks=True,all_four_independent_metric_routes=True,
        rejected_full_stops_and_rolls_back=True,no_half_candidate=True,no_probe_for_incomplete_prefix=True,
        exactly_one_new_metric_probe_for_completed_prefix=True,probe_and_arm_restore=True,deadline_restore=True)
