"""Secant algebra, same-start write, invalid bracket rollback, and native gates."""
from unittest.mock import patch
import copy
import math
import torch
from .stress_precision8_cpu_check import run as prior_checks
from .cpu_check import Runtime,record
from .editor import calibrate
from .batch8 import run_sample
from .precision_abc import reconstruct
from .finite_null import correction,secant_multiplier
from .amplitude_compensation import point


def run():
    prior_checks()
    assert secant_multiplier(.5,1.5)==1.5
    assert math.isclose(secant_multiplier(2,.5),2/3)
    assert secant_multiplier(1,1)==1
    assert secant_multiplier(.5,.8) is None
    assert secant_multiplier(float('nan'),1.5) is None
    assert secant_multiplier(-1,1.5) is None
    torch.manual_seed(42);rt=Runtime(torch.float16);r=record('trial');name='model.layers.0.mlp.down_proj'
    f=calibrate(rt,[record('train')],[name],train_ids=['train'],excluded_ids=['trial'],binding={'model':'synthetic'})['factors'][name]
    f=dict(f,a=f['a']*.001);base=rt.model.layer.weight.detach().clone()
    history=dict(sample='S01');run_sample(rt,r,f,history,lambda s:None,compensate=True)
    history=copy.deepcopy(history);target=max(s['step'] for s in history['method']['steps'] if s['correction'])
    history['method']['steps']=history['method']['steps'][:target];history['method']['steps'][-1]['status']='REJECTED_AND_STOPPED'
    state=dict(checks=[],prefix=[]);snapshot=reconstruct(rt,r,f,history,target,state,lambda:None,compensate=True)
    with torch.no_grad():rt.model.layer.weight.copy_(snapshot['weight'])
    c=correction(rt,r,f,snapshot['baseline']['native'][1],compensate='secant')
    sec=c['compensation']['secant']
    if c['status']=='APPLIED':
        assert c['compensation']['multiplier']==secant_multiplier(sec['first_effect'],sec['second_effect'])
        assert rt.model.layer.weight.dtype==torch.float16
    else:assert c['status']=='INVALID_SECANT' and torch.equal(rt.model.layer.weight.cpu(),snapshot['weight'])
    with torch.no_grad():rt.model.layer.weight.copy_(snapshot['weight'])
    with patch('experiments.vigor_medical_single_image_v1.finite_null.secant_multiplier',return_value=None):
        invalid=correction(rt,r,f,snapshot['baseline']['native'][1],compensate='secant')
    assert invalid['status']=='INVALID_SECANT' and torch.equal(rt.model.layer.weight.cpu(),snapshot['weight'])
    with torch.no_grad():rt.model.layer.weight.copy_(base)
    point(rt,r,f,snapshot,history,state,lambda:None,compensated=True,mode='secant')
    assert state['candidate_count']==1 and state['base_restored'] and torch.equal(base,rt.model.layer.weight)
    return dict(status='PASS',prior_checks=True,analytic_interpolation=True,no_extrapolation=True,
        invalid_no_write=True,one_candidate=True,native_gates_and_restoration=True,generation_calls=0,judge_calls=0)
