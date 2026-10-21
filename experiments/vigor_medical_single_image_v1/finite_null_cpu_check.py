"""Exercise native acceptance and one-correction rejection/rollback on a toy model."""
from unittest.mock import patch
import torch
from .cpu_check import Runtime,record
from .editor import calibrate
from .finite_null import trajectory


def run():
    torch.set_num_threads(2);torch.manual_seed(42)
    runtime=Runtime(torch.float16);name='model.layers.0.mlp.down_proj'
    factor=calibrate(runtime,[record('train')],[name],train_ids=['train'],excluded_ids=['trial'],
        binding={'model':'synthetic'})['factors'][name]
    # Synthetic fixture scale keeps the toy trust cap inactive so finite null repair is exercised.
    factor=dict(factor,a=factor['a']*.001)
    base=runtime.model.layer.weight.detach().clone()
    def forbidden(*args,**kwargs):raise AssertionError('No generation')
    runtime.generate=forbidden
    result=trajectory(runtime,record('trial'),factor,lambda rows:None)
    assert result['accepted_steps']==4 and result['correction_attempts']>0
    assert torch.equal(base,runtime.model.layer.weight)
    for s in result['steps']:
        assert s['final_verdict']['accepted']
        if s['correction']:
            c=s['correction'];tol=1e-5*abs(c['residual'])+1e-7
            assert abs(c['fp32_update']['visual_linear_change'])<tol
            assert abs(c['fp32_update']['null_linear_change']+c['residual'])<tol
    for prev,next_row in zip(result['steps'],result['steps'][1:]):
        assert prev['final_verdict']['native_nll']==next_row['start_native_nll']
    with patch('experiments.vigor_medical_single_image_v1.finite_null.correction',
               return_value={'status':'DEGENERATE'}) as correction:
        stopped=trajectory(runtime,record('trial'),factor,lambda rows:None)
    assert correction.call_count==1 and stopped['attempted_steps']==2 and stopped['accepted_steps']==1
    assert stopped['steps'][-1]['rollback_pass'] and stopped['status']=='STOPPED_REJECTED'
    assert not stopped['joint_gate'] and torch.equal(base,runtime.model.layer.weight)
    assert not runtime.model.layer._forward_hooks and not runtime.model.layer._forward_pre_hooks
    return dict(status='PASS',native_acceptance=True,correction_visual_first_order_protection=True,
        one_correction_limit=True,rejection_rolls_back_and_stops=True,base_restoration=True,
        generation_calls=0,judge_calls=0)
