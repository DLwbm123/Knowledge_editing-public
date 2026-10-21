"""Read-only FP32 chain, fixed increment and exception restoration checks."""
from unittest.mock import patch
import torch
from .probemetric_cpu_check import run as prior
from .cpu_check import Runtime,record
from .editor import calibrate
from .direct_fisher16 import estimate
from .diagnose import candidate,FIRST_D
from .chainprecision_grid import read_grid,chain_fp32


def run():
    prior();torch.manual_seed(42)
    rt=Runtime(torch.float16);r=record('test');name='model.layers.0.mlp.down_proj'
    train=[record('a'),record('b',image=.2,target='3')]
    f=calibrate(rt,train,[name],train_ids=['a','b'],excluded_ids=['test'],binding={'model':'synthetic'})['factors'][name]
    diagonal,_=estimate(rt,train,f,lambda e:None)
    base=rt.model.layer.weight.detach().clone();captured=[]
    seed=candidate(rt,r,0,f,FIRST_D/16,solve_factor=diagonal)
    p=candidate(rt,r,0,f,FIRST_D/16,solve_factor=diagonal,native_norm_target=seed['quantization']['native_delta_l2'],update_observer=lambda u:captured.append(u.clone()))
    assert len(captured)==1
    result=read_grid(rt,r,base,captured[0],p,None,lambda:None,modules=[rt.model])
    assert result['base_restored'] and result['endpoint_restored']
    assert result['chain_audit']['non_fp32_outputs']==0
    assert set(result['mixed']['losses'])=={'0','0.5','1'}
    assert result['chain_fp32']['baseline_repeat_error']==0
    assert rt.model.layer.weight.dtype==torch.float16 and torch.equal(rt.model.layer.weight,base)
    flags=[q.requires_grad for q in rt.model.parameters()]
    try:
        with chain_fp32(rt.model,[rt.model]):
            rt.model.layer.weight.requires_grad_(True)
            raise RuntimeError('forced restoration')
    except RuntimeError:pass
    assert flags==[q.requires_grad for q in rt.model.parameters()]
    assert rt.model.frozen.dtype==torch.float16 and not rt.model.layer._forward_hooks
    with patch('experiments.vigor_medical_single_image_v1.chainprecision_grid.grid',side_effect=ValueError('forced grid failure')):
        try:read_grid(rt,r,base,captured[0],p,None,lambda:None,modules=[rt.model])
        except ValueError:pass
        else:raise AssertionError('failure must stop')
    assert torch.equal(base,rt.model.layer.weight) and rt.model.layer.weight.dtype==torch.float16
    return dict(status='PASS',prior_checks=True,observer_once=True,fixed_positions=True,
        chain_fp32_outputs_only=True,dtype_flag_hook_restoration=True,grid_failure_restoration=True,no_new_direction=True)
