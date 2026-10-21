"""Original local NLL projection and cost, alongside shared lifecycle checks."""
import torch
from .responseRefresh_cpu_check import run as previous
from .editor import direction

def run():
    v=previous();g=torch.tensor([[1.,2.,0.]]);g0=torch.tensor([[0.,0.,1.]])
    q,geo=direction(g,g0,torch.ones(3),torch.ones(1));assert q is not None
    C=.5;delta=-(2*C/geo['E'])**.5*q
    assert abs(float(delta.square().sum())/2-C)<1e-6 and float((g*delta).sum())<0 and float((g0*delta).sum())==0
    return dict(v,original_NLL_cost_and_projection=True,new_finite_response_fit=False)
