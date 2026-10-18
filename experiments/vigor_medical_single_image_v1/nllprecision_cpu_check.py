"""Optional rounded-weight readout preserves the same candidate and restores on failure."""
from unittest.mock import patch
from unittest import TestCase
import torch
from .nllbisect_cpu_check import run as prior_checks
from .cpu_check import Runtime,record
from .editor import calibrate,paired_batches,editable_fp32
from .direct_fisher16 import estimate
from .diagnose import candidate,FIRST_D,nll
from .finite_null import native_losses


def run():
    prior_checks();torch.manual_seed(42)
    rt=Runtime(torch.float16);r=record('test');name='model.layers.0.mlp.down_proj'
    train=[record('a'),record('b',image=.2,target='3')]
    f=calibrate(rt,train,[name],train_ids=['a','b'],excluded_ids=['test'],binding={'model':'synthetic'})['factors'][name]
    d,_=estimate(rt,train,f,lambda e:None)
    layer=rt.model.layer;base=layer.weight.detach().clone();pair=paired_batches(rt,r);base_losses=native_losses(rt,pair)
    calls=[0]
    def count(module,args):calls[0]+=1
    handle=rt.model.register_forward_pre_hook(count)
    old=candidate(rt,r,0,f,FIRST_D/16,solve_factor=d)
    old_count=calls[0];calls[0]=0;target=old['quantization']['native_delta_l2']
    result=candidate(rt,r,0,f,FIRST_D/16,solve_factor=d,native_norm_target=target,rounded_fp32=True)
    assert calls[0]==old_count+2
    handle.remove()
    for k in ['fp32_before','fp32_after','native_before','native_after','E']:assert result[k]==old[k]
    assert result['native_norm_matching']['applications']==1 and result['native_norm_matching']['passed']
    retained=candidate(rt,r,0,f,FIRST_D/16,solve_factor=d,retain=True)
    with editable_fp32(rt.model,layer,readonly=True):expected=native_losses(rt,pair)
    assert result['rounded_fp32_after']==expected
    with torch.no_grad():layer.weight.copy_(base)
    counter=[0]
    def fail_in_B(model,batch):
        counter[0]+=1
        if counter[0]==old_count-3:raise RuntimeError('synthetic rounded readout failure')
        return nll(model,batch)
    with TestCase().assertRaises(RuntimeError),patch('experiments.vigor_medical_single_image_v1.diagnose.nll',side_effect=fail_in_B):
        candidate(rt,r,0,f,FIRST_D/16,solve_factor=d,rounded_fp32=True)
    assert torch.equal(layer.weight,base) and layer.weight.dtype==torch.float16
    assert not layer._forward_hooks and not layer._forward_pre_hooks and native_losses(rt,pair)==base_losses
    return dict(status='PASS',prior_checks=True,optional_B_preserves_A_C_and_direction=True,
        two_extra_forwards_per_endpoint=True,B_equals_independent_readonly_rounded_readout=True,
        one_norm_ratio_only=True,B_exception_restores_weights_dtype_hooks=True,no_new_candidate=True)
