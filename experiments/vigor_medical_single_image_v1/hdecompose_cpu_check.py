"""Frozen H reconstruction identity and scale-aligned accounting, without a VLM."""
import torch
from .hybridH_ops import build_h
from .hdecompose_ops import reconstruct_h,RATIO
from .bdecompose_ops import decomposition
from .bdecompose_cpu_check import run as guards


def run():
    guards();gn=torch.tensor([[1.,0.,0.]]);gm=torch.tensor([[-1.,1.,0.]]);a=torch.ones(3);b=torch.ones(1)
    delta,old=build_h(gn,gm,gn,gm,1.,2.,.5,a,b)
    rebuilt=reconstruct_h(gn,gm,old);assert torch.equal(delta,rebuilt)
    w=torch.tensor([[.1,.2,.3]],dtype=torch.float16);u=RATIO*rebuilt
    assert torch.equal((w.float()+u).half(),(w.float()+RATIO*delta).half())
    raw=float((gn.double()*delta.double()).sum());p=float((gn.double()*u.double()).sum())
    d=decomposition(2.,2.2,2.3,2.01,2.32,p,.01,pold=raw*RATIO)
    assert abs(d['D'])<1e-7 and abs(d['closure_error'])<1e-12 and abs(d['historical_gap_closure_error'])<1e-12
    return dict(status='PASS',fixed_H_reconstruction_matches_original_tensor=True,frozen_old_ratio_no_rematch=True,matched_prediction_alignment=True,two_closures=True,prior_forward_identity_and_restore_checks=True)
