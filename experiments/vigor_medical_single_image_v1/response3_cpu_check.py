"""Three-point fit and singular-anchor stop, plus shared lifecycle checks."""
import torch
from .responseS_cpu_check import run as previous
from .response3_model import calibrate_vectors
from .bdecompose_ops import StopRound

def run():
    v=previous();a=torch.ones(4);b=torch.ones(1)
    g=[torch.tensor([[1.,0.,0.,0.]]),torch.tensor([[0.,1.,0.,0.]]),torch.tensor([[0.,0.,0.,1.]])]
    q=[torch.tensor([[.2,0.,0.,0.]]),torch.tensor([[0.,.3,0.,0.]]),torch.tensor([[0.,0.,.4,0.]])]
    y=[[-.1,.1,-.08],[.2,.1,.5],[0.,0.,.001]]
    out=calibrate_vectors(g,q,y,a,b,lambda s:None)
    for i,bar in enumerate(out):
        for j,anchor in enumerate(q):assert abs(float((bar.double()*anchor.double()).sum())-y[i][j])<1e-5
    try:calibrate_vectors(g,[q[0],q[1],q[0]],y,a,b,lambda s:None)
    except StopRound:pass
    else:raise AssertionError('Dependent third point must stop')
    return dict(v,three_anchor_fit=True,dependent_third_anchor_stops=True)
