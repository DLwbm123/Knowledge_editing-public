"""Fixed 16-step horizon, verified prefix, and scientific rejection bookkeeping."""
from unittest.mock import patch
import copy
import torch
from .amplitude_batch8_cpu_check import run as prior_checks
from .cpu_check import Runtime,record,rejects
from .editor import calibrate
from .batch8 import run_sample


def run():
    prior_checks()
    torch.manual_seed(42);rt=Runtime(torch.float16);r=record('trial');name='model.layers.0.mlp.down_proj'
    f=calibrate(rt,[record('train')],[name],train_ids=['train'],excluded_ids=['trial'],binding={'model':'synthetic'})['factors'][name]
    f=dict(f,a=f['a']*.001);base=rt.model.layer.weight.detach().clone()
    first=dict(sample='S01');run_sample(rt,r,f,first,lambda s:None,compensate=True)
    prefix=first['method']['steps']
    stress=dict(sample='S01');run_sample(rt,r,f,stress,lambda s:None,compensate=True,steps=16,prefix_reference=prefix)
    m=stress['method']
    assert m['requested_steps']==16 and m['attempted_steps']<=16
    assert m['joint_gate']==(m['trajectory_completed'] and m['edit_threshold_met'] and m['null_gate_on_accepted_states'])
    assert all('prefix_checks' in s for s in m['steps'][:len(prefix)])
    assert torch.equal(base,rt.model.layer.weight) and stress['base_restored']
    wrong=copy.deepcopy(prefix);wrong[0]['proposal']['native_after'][0]+=1
    rejects(lambda:run_sample(rt,r,f,dict(sample='S01'),lambda s:None,compensate=True,steps=16,prefix_reference=wrong))
    assert torch.equal(base,rt.model.layer.weight)
    return dict(status='PASS',previous_checks=True,fixed_horizon=16,prefix_checked_before_continuing=True,
        mismatch_stops_and_restores=True,generation_calls=0,judge_calls=0)
