"""Native rounding one-ratio calculation and no-retry lifecycle checks."""
from unittest.mock import patch
from unittest import TestCase
import torch
from .cross8_cpu_check import run as prior_checks
from .cpu_check import Runtime,record
from .editor import calibrate,paired_batches
from .direct_fisher16 import estimate
from .diagnose import candidate,FIRST_D
from .backoff16 import _attempt
from .normpath8 import run_arm,run_probe
from .finite_null import native_losses


def run():
    prior_checks();torch.manual_seed(42)
    rt=Runtime(torch.float16);r=record('test');name='model.layers.0.mlp.down_proj'
    train=[record('a'),record('b',image=.2,target='3')]
    f=calibrate(rt,train,[name],train_ids=['a','b'],excluded_ids=['test'],binding={'model':'synthetic'})['factors'][name]
    d,_=estimate(rt,train,f,lambda e:None)
    base=rt.model.layer.weight.detach().clone();pair=paired_batches(rt,r);losses=native_losses(rt,pair)
    p=candidate(rt,r,0,f,FIRST_D/16,solve_factor=d)
    target=p['quantization']['native_delta_l2']
    q=candidate(rt,r,0,f,FIRST_D/16,solve_factor=d,native_norm_target=target)
    m=q['native_norm_matching']
    assert m['applications']==1 and m['ratio']==1 and m['passed'] and m['relative_error']==0
    assert q['native_after']==p['native_after'] and torch.equal(base,rt.model.layer.weight)
    bad=candidate(rt,r,0,f,FIRST_D/16,solve_factor=d,native_norm_target=1e-10)
    assert not bad['native_norm_matching']['passed']
    verdict=dict(gates=dict(null=False,visual=True,trust=True),accepted=False)
    with patch('experiments.vigor_medical_single_image_v1.backoff16.candidate',return_value=bad),patch('experiments.vigor_medical_single_image_v1.backoff16.gates',return_value=verdict),patch('experiments.vigor_medical_single_image_v1.backoff16.correction') as corr:
        t={};v=_attempt(rt,r,f,pair,base,losses,losses[1],FIRST_D/16,None,t,lambda:None,d,
            correction_factor=None,separate_correction=True,native_norm_target=1e-10)
        assert t['status']=='MATCH_FAILED' and not v['accepted'] and corr.call_count==0
    calls=[]
    def rejected(*args,**kw):
        calls.append(kw['native_norm_target']);args[9].update(status='MATCH_FAILED',proposal=dict(requested_reduction=1,effective_reduction=1))
        with torch.no_grad():rt.model.layer.weight.add_(.01)
        return dict(accepted=False)
    state=dict(arm='C',checks=[])
    with patch('experiments.vigor_medical_single_image_v1.normpath8._attempt',side_effect=rejected),patch('experiments.vigor_medical_single_image_v1.normpath8.run_probe') as probe:
        run_arm(rt,r,f,d,state,lambda:None,None,[target]*8,target)
        assert calls==[target] and state['status']=='STOPPED_MATCH_FAILED' and state['steps'][0]['rollback_pass']
        assert state['accepted_steps']==0 and probe.call_count==0
    assert torch.equal(base,rt.model.layer.weight)
    def accepted(*args,**kw):
        args[9]['proposal']=dict(requested_reduction=1,effective_reduction=1)
        return dict(accepted=True)
    state=dict(arm='C',checks=[])
    with patch('experiments.vigor_medical_single_image_v1.normpath8._attempt',side_effect=accepted),patch('experiments.vigor_medical_single_image_v1.normpath8.run_probe',wraps=run_probe) as probe:
        run_arm(rt,r,f,d,state,lambda:None,None,[target]*8,target)
        assert state['accepted_steps']==8 and probe.call_count==1
        assert state['probes']['matched']['endpoint_restored'] and state['probe_norm_match_passed']
    assert torch.equal(base,rt.model.layer.weight)
    state=dict(probes={},checks=[])
    v=run_probe(rt,r,f,d,pair,losses[1],state,'matched',lambda:None,None,target=1e-10)
    assert not v['matching_passed'] and v['endpoint_restored'] and torch.equal(base,rt.model.layer.weight)
    with TestCase().assertRaises(TimeoutError):run_arm(rt,r,f,d,dict(arm='C',checks=[]),lambda:None,0,[target]*8,target)
    assert torch.equal(base,rt.model.layer.weight)
    return dict(status='PASS',prior_checks=True,native_rounding_ratio_checked=True,exact_target_identity=True,
        one_ratio_only=True,failed_match_skips_correction=True,failed_match_rolls_back_and_stops=True,
        incomplete_prefix_has_no_probe=True,complete_C_has_one_matched_probe=True,failed_probe_restores=True,
        no_half=True,deadline_restores=True)
