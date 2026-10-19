"""Native subset identity and coordinate descent/ascent contracts on CPU."""
import torch
from .responseQuad_cpu_check import run as previous
from .responseNativeOrthant_ops import native_dual_subset

def run():
    result=previous()
    base=torch.tensor([[1.,-1.,.5],[.25,-.25,2.]],dtype=torch.float16)
    proposal=(base.float()+torch.tensor([[.01,.01,-.01],[.01,-.01,-.01]])).half()
    gn=torch.tensor([[-2.,1.,1.],[-1.,2.,-1.]])
    gm=torch.tensor([[3.,2.,-2.],[1.,2.,-1.]])
    selected,mask=native_dual_subset(base,proposal,gn,gm)
    q=selected.float()-base.float();old=proposal.float()-base.float()
    assert int(mask.sum())==3
    assert torch.equal(selected[mask],proposal[mask]) and torch.equal(selected[~mask],base[~mask])
    assert (gn*q).sum()<0 and (gm*q).sum()>0
    assert bool(torch.all(gn*q<=0)) and bool(torch.all(gm*q>=0))
    metric=torch.tensor([[1.,2.,3.],[4.,5.,6.]])
    assert (metric*q*q).sum()<=(metric*old*old).sum()
    result.update(native_subset_values_identity=True,coordinate_NLL_margin_contract=True,diagonal_F0_subset_bound=True,new_solver=False)
    return result
