"""Signed mirror centering and deterministic whole-coordinate selection."""
import torch
from .responseOrthantBalance_cpu_check import run as previous
from .native_null_center import center

def run():
    result=previous()
    base=torch.ones(4,dtype=torch.float16);proposal=(base.float()+.01).half()
    gm=torch.tensor([1.,3.,2.,1.]);g0=torch.tensor([.4,.2,-.1,0.]);gn=-torch.ones(4)
    chosen,info=center(base,proposal,gm,g0,-.0005)
    assert info['status']=='FORMULATED' and info['excursion_sign']==1 and info['dropped_elements']==2
    assert torch.equal(chosen[:2],base[:2]) and torch.equal(chosen[2:],proposal[2:])
    q=chosen.float()-base.float();bias=-.0005+float((q.double()*g0.double()).sum())
    assert abs(bias)<=info['max_removed_null_quantum']+1e-10
    assert (gn*q).sum()<0 and (gm*q).sum()>0
    mirror,minus=center(base,proposal,gm,-g0,.0005)
    assert minus['excursion_sign']==-1 and torch.equal(mirror,chosen)
    no,stop=center(base,proposal,gm,torch.ones(4),.02)
    assert no is None and stop['status']=='STOPPED_SIGNED_CENTER_INSUFFICIENT_RECOVERY'
    result.update(signed_center_positive_and_negative_mirrors=True,signed_impossible_stops=True)
    return result
