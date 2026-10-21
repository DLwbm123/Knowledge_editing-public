"""One local finite-response fit and shared restoration/solver checks."""
import torch
from .response3_cpu_check import run as prior
from .response3_model import calibrate_vectors

def run():
    v=prior();a=torch.ones(3);b=torch.ones(1)
    g=[torch.tensor([[1.,0.,0.]]),torch.tensor([[0.,1.,0.]]),torch.tensor([[0.,0.,1.]])]
    q=[torch.tensor([[.1,.2,0.]])];y=[[.12],[-.14],[-.06]]
    out=calibrate_vectors(g,q,y,a,b,lambda x:None)
    for vec,target in zip(out,y):assert abs(float((vec.double()*q[0].double()).sum())-target[0])<1e-5
    return dict(v,one_local_anchor_fit=True,rank_one_preserves_orthogonal_components=bool(out[0][0,2]==g[0][0,2]))
