"""Compensated prefix replay and rounded/native identity through the ABC executor."""
import copy
from unittest.mock import patch
import torch
from .amplitude_stress16_cpu_check import run as prior_checks
from .cpu_check import Runtime,record
from .editor import calibrate
from .batch8 import run_sample
from .finite_null import correction
from .precision_abc import reconstruct,evaluate


def run():
    prior_checks()
    torch.manual_seed(42);rt=Runtime(torch.float16);r=record('trial');name='model.layers.0.mlp.down_proj'
    f=calibrate(rt,[record('train')],[name],train_ids=['train'],excluded_ids=['trial'],binding={'model':'synthetic'})['factors'][name]
    f=dict(f,a=f['a']*.001);base=rt.model.layer.weight.detach().clone()
    history=dict(sample='S01');run_sample(rt,r,f,history,lambda s:None,compensate=True)
    history=copy.deepcopy(history)
    target=max(st['step'] for st in history['method']['steps'] if st['correction'])
    history['method']['steps']=history['method']['steps'][:target]
    history['method']['steps'][-1]['status']='REJECTED_AND_STOPPED'
    state=dict(checks=[],prefix=[])
    with patch('experiments.vigor_medical_single_image_v1.precision_abc.correction',wraps=correction) as calls:
        snapshot=reconstruct(rt,r,f,history,target,state,lambda:None,compensate=True)
        assert all(c.kwargs['compensate'] for c in calls.call_args_list)
    evaluate(rt,r,f,snapshot,history,state,lambda:None,compensate=True)
    assert state['correction']['rounded_weights_equal'] and state['correction']['native_weights_equal_rounded']
    assert state['base_restored'] and torch.equal(base,rt.model.layer.weight)
    assert state['correction']['compensation']['rho']>0
    return dict(status='PASS',previous_checks=True,compensated_prefix=True,compensated_target=True,
        ABC_same_native_weights=True,base_restored=True,generation_calls=0,judge_calls=0)
