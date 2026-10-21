"""Exactly one candidate for each existing metric, with safe norm-failure recovery."""
from unittest.mock import patch
from unittest import TestCase
import torch
from .nllprecision_cpu_check import run as prior_checks
from .cpu_check import Runtime,record
from .editor import calibrate,paired_batches
from .direct_fisher16 import estimate
from .diagnose import candidate,FIRST_D
from .finite_null import native_losses
from .probemetric import metric_probe


def run():
    prior_checks();torch.manual_seed(42)
    rt=Runtime(torch.float16);r=record('test');name='model.layers.0.mlp.down_proj'
    train=[record('a'),record('b',image=.2,target='3')]
    f=calibrate(rt,train,[name],train_ids=['a','b'],excluded_ids=['test'],binding={'model':'synthetic'})['factors'][name]
    diagonal,_=estimate(rt,train,f,lambda e:None)
    base=rt.model.layer.weight.detach().clone();pair=paired_batches(rt,r);losses=native_losses(rt,pair)
    for metric in ('F1','F0'):
        solver=diagonal if metric=='F1' else None
        target=candidate(rt,r,0,f,FIRST_D/16,solve_factor=solver)['quantization']['native_delta_l2']
        state=dict(arm='A',baseline=dict(native=losses),endpoint_native_nll=losses,checks=[],probes={})
        with patch('experiments.vigor_medical_single_image_v1.probemetric.candidate',wraps=candidate) as c:
            p=metric_probe(rt,r,f,diagonal,base,state,metric,target,None,lambda:None,None)
            assert c.call_count==1 and c.call_args.kwargs['solve_factor'] is solver
            assert 'rounded_fp32' not in c.call_args.kwargs and p['matching_passed']
            assert p['proposal']['native_norm_matching']['applications']==1 and p['endpoint_restored']
        assert state['probe_base_restored'] and torch.equal(base,rt.model.layer.weight)
    state=dict(arm='D',baseline=dict(native=losses),endpoint_native_nll=losses,checks=[],probes={})
    with patch('experiments.vigor_medical_single_image_v1.probemetric.candidate',wraps=candidate) as c:
        p=metric_probe(rt,r,f,diagonal,base,state,'F0',1e-10,None,lambda:None,None)
        assert c.call_count==1 and not p['matching_passed'] and p['endpoint_restored']
        assert state['status']=='STOPPED_MATCH_FAILED'
    assert torch.equal(base,rt.model.layer.weight) and native_losses(rt,pair)==losses
    with TestCase().assertRaises(TimeoutError):metric_probe(rt,r,f,diagonal,base,state,'F0',1,None,lambda:None,0)
    with TestCase().assertRaises(PermissionError):metric_probe(rt,r,f,diagonal,base,state,'other',1,None,lambda:None,None)
    return dict(status='PASS',prior_checks=True,existing_F0_F1_routes=True,one_candidate_per_metric_endpoint=True,
        one_norm_ratio_no_retry=True,no_extra_precision_branch=True,norm_failure_restores=True,
        probe_and_base_restored=True,no_probe_correction=True,deadline_and_unknown_metric_stop=True)
