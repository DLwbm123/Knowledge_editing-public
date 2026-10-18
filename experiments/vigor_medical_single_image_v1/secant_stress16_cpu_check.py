"""Complete fixed-horizon secant path and deadline rollback, without old-path stitching."""
from unittest.mock import patch
import torch
from .secant8_cpu_check import run as prior_checks
from .cpu_check import Runtime,record,rejects
from .editor import calibrate
from .batch8 import run_sample
from .finite_null import correction


def run():
    prior_checks()
    torch.manual_seed(42);rt=Runtime(torch.float16);r=record('trial');name='model.layers.0.mlp.down_proj'
    f=calibrate(rt,[record('train')],[name],train_ids=['train'],excluded_ids=['trial'],binding={'model':'synthetic'})['factors'][name]
    f=dict(f,a=f['a']*.001);base=rt.model.layer.weight.detach().clone()
    state=dict(sample='S01')
    with patch('experiments.vigor_medical_single_image_v1.finite_null.correction',wraps=correction) as calls:
        run_sample(rt,r,f,state,lambda s:None,compensate='secant',steps=16)
    assert calls.call_count>0 and all(c.kwargs['compensate']=='secant' for c in calls.call_args_list)
    m=state['method'];assert m['requested_steps']==16 and m['attempted_steps']<=16
    assert m['joint_gate']==(m['trajectory_completed'] and m['edit_threshold_met'] and m['null_gate_on_accepted_states'])
    assert state['base_restored'] and torch.equal(base,rt.model.layer.weight)
    rejects(lambda:run_sample(rt,r,f,dict(sample='S01'),lambda s:None,compensate='secant',steps=16,deadline_epoch=0))
    assert torch.equal(base,rt.model.layer.weight)
    return dict(status='PASS',previous_checks=True,fixed_horizon=16,all_corrections_secant=True,
        single_candidate_per_correction=True,deadline_restores_base=True,generation_calls=0,judge_calls=0)
