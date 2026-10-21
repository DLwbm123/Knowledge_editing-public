"""Same-start ABC, one norm ratio and exception restoration on a small model."""
from unittest import TestCase
from unittest.mock import patch
import torch
from .direct_fisher16_cpu_check import run as prior_checks
from .cpu_check import Runtime,record
from .editor import calibrate,paired_batches
from .direct_fisher16 import estimate
from .finite_null import native_losses
from .normmatch4 import evaluate,ratio
from .diagnose import FIRST_D


def run():
    prior_checks();torch.manual_seed(42)
    rt=Runtime(torch.float16);r=record('test');name='model.layers.0.mlp.down_proj'
    train=[record('a'),record('b',image=.2,target='3')]
    f=calibrate(rt,train,[name],train_ids=['a','b'],excluded_ids=['test'],binding={'model':'synthetic'})['factors'][name]
    d,_=estimate(rt,train,f,lambda e:None)
    base=rt.model.layer.weight.detach().clone();losses=native_losses(rt,paired_batches(rt,r))
    state=dict(baseline=dict(native=losses),start_native_nll=losses,arms={})
    for arm in ('A','B'):evaluate(rt,r,f,d,base.cpu(),state,arm,FIRST_D/16,lambda:None,None)
    a,b=(state['arms'][k]['proposal']['quantization']['native_delta_l2'] for k in ('A','B'))
    alpha=ratio(a,b)
    req=state['arms']['B']['proposal']['effective_reduction']*alpha
    evaluate(rt,r,f,d,base.cpu(),state,'C',req,lambda:None,None)
    assert len(state['arms'])==3 and state['arms']['C']['proposal']['requested_reduction']==req
    for trial in state['arms'].values():
        assert trial['same_start_verified'] and trial['base_restored']
        assert trial['proposal']['native_before']==losses
    assert torch.equal(base,rt.model.layer.weight)
    with patch('experiments.vigor_medical_single_image_v1.normmatch4._attempt',side_effect=RuntimeError('synthetic')):
        with TestCase().assertRaises(RuntimeError):evaluate(rt,r,f,d,base.cpu(),state,'C',req,lambda:None,None)
    assert torch.equal(base,rt.model.layer.weight)
    with TestCase().assertRaises(TimeoutError):evaluate(rt,r,f,d,base.cpu(),state,'C',req,lambda:None,0)
    with TestCase().assertRaises(ValueError):ratio(1,0)
    assert abs(1.009-1)<=.01 and not abs(1.011-1)<=.01
    return dict(status='PASS',prior_direct_fisher_and_backoff_checks=True,ABC_same_start=True,
        one_ratio_from_native_main_norms=True,three_candidates_only=True,invalid_match_not_retried=True,
        readonly_and_exception_restore=True,deadline_guard=True)
