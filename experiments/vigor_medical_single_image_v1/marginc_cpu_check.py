"""Known analytic active solution, inactive/infeasible stops and native matching."""
import torch,math
from .marginc_ops import solve,match_delta
from .marginab_cpu_check import run as previous


def run():
    previous();gn=torch.tensor([[1.,0.,0.]]);gm=torch.tensor([[1.,1.,0.]]);a=torch.ones(3);b=torch.ones(1)
    delta,data=solve(gn,gm,gn,gm,1.,2.,.5,.2)
    assert data['constraint_active'] and data['status']=='ACTIVE_SOLUTION'
    assert torch.allclose(delta,torch.tensor([[-.2,math.sqrt(.96),0.]]),atol=1e-6)
    assert abs(float(delta.square().sum())/2-.5)<1e-6 and float((gn*delta).sum())<=-.2+1e-6
    original=torch.zeros_like(delta,dtype=torch.float16);ratio,match=match_delta(original,delta,.5,a,b,gn,.2)
    assert match['ratio_applications']==1 and match['cost_pass']
    no,data=solve(gn,-gn,gn,-gn,1.,1.,.5,.2);assert no is None and data['status']=='STOPPED_INACTIVE_OLD_B'
    no,data=solve(gn,gm,gn,gm,1.,2.,.5,2.);assert no is None and data['status']=='STOPPED_INFEASIBLE'
    no,data=solve(gn,gn,gn,gn,1.,1.,.5,.2);assert no is None and data['status']=='STOPPED_DEGENERATE'
    return dict(status='PASS',analytic_active_boundary=True,inactive_stops_without_candidate=True,infeasible_stops=True,degenerate_stops=True,one_native_ratio=True,previous_restore_and_token_checks=True)
