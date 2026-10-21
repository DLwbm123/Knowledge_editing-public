"""Verify sequential native-state continuity, first-step parity and final restore."""
import torch
from .cpu_check import Runtime,record
from .editor import calibrate
from .diagnose import candidate,FIRST_D
from .multistep import trajectory


def run():
    torch.set_num_threads(2);torch.manual_seed(42)
    runtime=Runtime(torch.float16)
    name='model.layers.0.mlp.down_proj'
    calibration=calibrate(runtime,[record('train')],[name],train_ids=['train'],excluded_ids=['trial'],binding={'model':'synthetic'})
    factor=calibration['factors'][name];item=record('trial')
    def forbidden(*args,**kwargs):raise AssertionError('No generation')
    runtime.generate=forbidden
    base=runtime.model.layer.weight.detach().clone()
    single=candidate(runtime,item,0,factor,FIRST_D/16)
    seen=[]
    result=trajectory(runtime,item,factor,lambda rows:seen.append(len(rows)),first_reference=single['native_after'])
    assert seen==[1,2,3,4] and len(result['steps'])==4
    assert torch.equal(base,runtime.model.layer.weight)
    assert result['steps'][0]['native_after']==single['native_after']
    for previous,current in zip(result['steps'],result['steps'][1:]):
        assert current['native_before']==previous['native_after']
    assert not runtime.model.layer._forward_hooks and not runtime.model.layer._forward_pre_hooks
    return dict(status='PASS',sequential_native_steps=4,first_step_parity=True,base_restoration=True,generation_calls=0,judge_calls=0)
