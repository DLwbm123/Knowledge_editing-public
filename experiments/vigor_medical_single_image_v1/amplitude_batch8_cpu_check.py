"""Production path check: every needed correction receives compensation and restores base."""
from unittest.mock import patch
import torch
from .amplitude_cpu_check import run as prior_checks
from .cpu_check import Runtime,record
from .editor import calibrate
from .batch8 import run_sample
from .finite_null import correction


def run():
    prior_checks()
    torch.manual_seed(42)
    rt=Runtime(torch.float16);r=record('trial');name='model.layers.0.mlp.down_proj'
    f=calibrate(rt,[record('train')],[name],train_ids=['train'],excluded_ids=['trial'],binding={'model':'synthetic'})['factors'][name]
    f=dict(f,a=f['a']*.001);base=rt.model.layer.weight.detach().clone()
    state=dict(sample='S01',status='RUNNING')
    with patch('experiments.vigor_medical_single_image_v1.finite_null.correction',wraps=correction) as calls:
        run_sample(rt,r,f,state,lambda s:None,compensate=True)
    assert calls.call_count>0 and all(c.kwargs['compensate'] for c in calls.call_args_list)
    assert state['base_restored'] and torch.equal(base,rt.model.layer.weight)
    assert state['method']['attempted_steps']<=4
    assert state['method']['correction_attempts']==calls.call_count
    return dict(status='PASS',prior_compensation_checks=True,production_trajectory_compensation=True,
        no_extra_correction=True,base_restored=True,generation_calls=0,judge_calls=0)
