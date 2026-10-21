"""Existing trajectory lifecycle plus one endpoint observer and frozen panel masks."""
import torch
from .backoff16_cpu_check import run as previous
from .editperf8_plan import run as panel
from . import backoff16
from .cpu_check import Runtime,record
from .editor import calibrate


def run():
    previous();panel();torch.manual_seed(42)
    rt=Runtime(torch.float16);r=record('test');name='model.layers.0.mlp.down_proj'
    f=calibrate(rt,[record('a'),record('b')],[name],train_ids=['a','b'],excluded_ids=['test'],binding={'model':'synthetic'})['factors'][name]
    base=rt.model.layer.weight.detach().clone();seen=[]
    result=backoff16.trajectory(rt,r,f,lambda rows:None,edit_threshold=100,steps=1,endpoint_observer=lambda w,r:seen.append(w.clone()))
    assert len(seen)==1 and torch.equal(rt.model.layer.weight,base) and result['attempted_steps']==1
    return dict(status='PASS',old_backoff_lifecycle=True,endpoint_capture_then_base_restore=True,fixed_formal_panel=True,qualified_denominators=True)
