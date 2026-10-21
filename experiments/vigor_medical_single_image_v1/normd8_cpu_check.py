"""Old metric routing, one-ratio safe stop and a single matched endpoint probe."""
from unittest.mock import patch
from unittest import TestCase
import torch
from .normpath8_cpu_check import run as prior_checks
from .cpu_check import Runtime,record
from .editor import calibrate,paired_batches
from .direct_fisher16 import estimate
from .diagnose import candidate,FIRST_D
from .finite_null import native_losses
from .batch8 import baseline_check
from .normd8 import run_arm,verify_reference
from .normpath8 import run_probe


def run():
    prior_checks();torch.manual_seed(42)
    rt=Runtime(torch.float16);r=record('test');name='model.layers.0.mlp.down_proj'
    train=[record('a'),record('b',image=.2,target='3')]
    f=calibrate(rt,train,[name],train_ids=['a','b'],excluded_ids=['test'],binding={'model':'synthetic'})['factors'][name]
    d,_=estimate(rt,train,f,lambda e:None)
    base=rt.model.layer.weight.detach().clone();pair=paired_batches(rt,r);losses=native_losses(rt,pair)
    target=candidate(rt,r,0,f,FIRST_D/16,solve_factor=d)['quantization']['native_delta_l2']
    for status in ('MATCH_FAILED','REJECTED'):
        calls=[]
        def rejected(*args,**kw):
            assert args[11] is None and kw['correction_factor'] is None and kw['separate_correction']
            assert kw['native_norm_target']==target
            calls.append(args[7]);args[9].update(status=status,proposal=dict(requested_reduction=1,effective_reduction=1))
            with torch.no_grad():rt.model.layer.weight.add_(.01)
            return dict(accepted=False)
        state=dict(arm='D',checks=[])
        with patch('experiments.vigor_medical_single_image_v1.normd8._attempt',side_effect=rejected),patch('experiments.vigor_medical_single_image_v1.normd8.run_probe') as probe:
            run_arm(rt,r,f,d,state,lambda:None,None,[target]*8,target)
            assert len(calls)==1 and state['accepted_steps']==0 and state['steps'][0]['rollback_pass']
            assert state['status']==('STOPPED_MATCH_FAILED' if status=='MATCH_FAILED' else 'STOPPED_REJECTED')
            assert probe.call_count==0
        assert torch.equal(base,rt.model.layer.weight)
    def accepted(*args,**kw):
        args[9]['proposal']=dict(requested_reduction=1,effective_reduction=1)
        return dict(accepted=True)
    state=dict(arm='D',checks=[])
    with patch('experiments.vigor_medical_single_image_v1.normd8._attempt',side_effect=accepted),patch('experiments.vigor_medical_single_image_v1.normd8.run_probe',wraps=run_probe) as probe:
        run_arm(rt,r,f,d,state,lambda:None,None,[target]*8,target)
        assert state['accepted_steps']==8 and probe.call_count==1 and set(state['probes'])=={'matched'}
        assert state['probes']['matched']['endpoint_restored']
    assert torch.equal(base,rt.model.layer.weight)
    baseline=baseline_check(rt,r,f)
    def broken(*args,**kw):
        with torch.no_grad():rt.model.layer.weight.add_(.01)
        raise RuntimeError('synthetic reference failure')
    with TestCase().assertRaises(RuntimeError),patch('experiments.vigor_medical_single_image_v1.normd8._attempt',side_effect=broken):
        verify_reference(rt,r,f,dict(baseline=baseline),{},lambda:None,None)
    assert torch.equal(base,rt.model.layer.weight)
    with TestCase().assertRaises(TimeoutError):run_arm(rt,r,f,d,dict(arm='D',checks=[]),lambda:None,0,[target]*8,target)
    assert torch.equal(base,rt.model.layer.weight) and native_losses(rt,pair)==losses
    return dict(status='PASS',prior_checks=True,old_main_and_correction_routes=True,each_step_receives_frozen_target=True,
        rejected_match_and_gate_roll_back_stop=True,no_retry_or_half=True,incomplete_prefix_has_no_probe=True,
        complete_prefix_has_one_matched_probe=True,probe_restores=True,reference_exception_restores=True,deadline_restores=True)
