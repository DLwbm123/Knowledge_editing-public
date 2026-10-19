"""Two-point fit, singular stop, active/free analytic solutions and gain stop."""
import torch
from .responseS_model import calibrate_vectors,solve_geometry,materialize_solution
from .bdecompose_ops import StopRound
from .bdecompose_cpu_check import run as guards


def run():
    guards();a=torch.ones(3);b=torch.ones(1);gn=torch.tensor([[1.,0.,0.]]);gm=torch.tensor([[-1.,1.,0.]]);g0=torch.tensor([[0.,0.,1.]])
    anchors=[torch.tensor([[.2,0.,0.]]),torch.tensor([[0.,.3,0.]])];y=[[-.1,.1],[.2,.1],[0.,0.]]
    out=calibrate_vectors([gn,gm,g0],anchors,y,a,b,lambda v:None)
    for i,g in enumerate(out):
        for j,q in enumerate(anchors):assert abs(float((g.double()*q.double()).sum())-y[i][j])<1e-5
    try:calibrate_vectors([gn,gm,g0],[anchors[0],anchors[0]],y,a,b,lambda v:None)
    except StopRound as e:assert str(e)=='STOPPED_CALIBRATION_ILL_CONDITIONED'
    else:raise AssertionError('Singular anchors must stop, no ridge')
    active=solve_geometry(1.,2.,-1.,.5,.8,.4453125);assert active['status']=='ELIGIBLE' and active['constraint_active']
    delta=materialize_solution(gn,gm,active);assert abs(float(delta.square().sum())/2-.5)<1e-6 and float((gn*delta).sum())<=-.8+1e-6
    free=solve_geometry(1.,2.,-1.,.5,.2,.4453125);assert free['status']=='ELIGIBLE' and not free['constraint_active']
    assert solve_geometry(1.,2.,-1.,.5,2.,.4453125)['status']=='STOPPED_SOLVER_INFEASIBLE'
    stop=solve_geometry(1.,2.,-1.,.5,.8,2.);assert stop['status']=='STOPPED_PREDICTED_MARGIN_NOT_ABOVE_A'
    return dict(status='PASS',two_anchor_response_fit=True,ill_conditioned_stops_no_ridge=True,active_and_free_solutions=True,infeasible_stops=True,predicted_margin_stop_before_candidate=True,prior_weight_identity_and_restore_checks=True)
