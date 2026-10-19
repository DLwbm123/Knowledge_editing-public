"""Greedy null deletion stability, feasible centering, and impossible-case stop."""
import torch
from .responseNativeOrthant_cpu_check import run as previous
from .native_null_balance import balance

def run():
    result=previous()
    base=torch.ones(4,dtype=torch.float16);proposal=(base.float()+.01).half()
    gm=torch.tensor([1.,3.,2.,1.]);g0=torch.tensor([-.4,-.2,.1,0.]);gn=-torch.ones(4)
    chosen,info=balance(base,proposal,gm,g0,.0005)
    assert info['status']=='FORMULATED' and info['dropped_elements']==2
    assert torch.equal(chosen[:2],base[:2]) and torch.equal(chosen[2:],proposal[2:])
    q=chosen.float()-base.float()
    bias=.0005+float((q.double()*g0.double()).sum())
    assert abs(bias)<=info['max_removed_null_quantum']+1e-10
    assert float((gn*q).sum())<0 and float((gm*q).sum())>0
    no,stop=balance(base,proposal,gm,-torch.ones(4),-.02)
    assert no is None and stop['status']=='STOPPED_BALANCE_INSUFFICIENT_RECOVERY'
    base3=torch.ones(3,dtype=torch.float16);p3=(base3.float()+.01).half();dq=float(p3[0]-base3[0])
    tie,record=balance(base3,p3,torch.ones(3),torch.tensor([-1.,-1.,0.]),1.5*dq)
    assert record['dropped_elements']==1 and tie[0]==base3[0] and tie[1]==p3[1]
    result.update(native_null_greedy_prefix=True,stable_flat_index_ties=True,impossible_center_stops_without_fallback=True)
    return result
