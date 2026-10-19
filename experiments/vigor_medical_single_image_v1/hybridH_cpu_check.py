"""Fixed energy split and one native match, with strict finite-effect admission."""
import torch
from .hybridH_ops import build_h,match_once,conditions,FLOOR,MAIN_MARGIN,FINAL_MARGIN
from .bdecompose_cpu_check import run as guards


def run():
    guards();gn=torch.tensor([[1.,0.,0.]]);gm=torch.tensor([[-1.,1.,0.]]);g0=torch.tensor([[0.,0.,1.]]);a=torch.ones(3);b=torch.ones(1)
    delta,data=build_h(gn,gm,gn,gm,1.,2.,.5,a,b)
    assert abs(data['NLL_component_cost']-.375)<1e-6 and abs(data['margin_component_cost']-.125)<1e-6 and abs(data['continuous_total_cost']-.5)<1e-6
    assert data['NLL_orthogonal_residual']==0 and data['F0_orthogonal_residual']==0
    assert data['predicted_NLL_change']<0 and data['predicted_margin_gain']>0 and float((g0*delta).sum())==0
    original=torch.zeros_like(delta,dtype=torch.float16);rounded,m=match_once(original,delta,.5,a,b,gn,gm,g0)
    assert m['ratio_applications']==1 and m['relative_error']<=.01 and m['native_linear_NLL_change']<0 and m['native_linear_margin_gain']>0
    no,d=build_h(gn,-gn,gn,-gn,1.,1.,.5,a,b);assert no is None and d['status']=='STOPPED_DEGENERATE_H'
    assert conditions(FLOOR,MAIN_MARGIN+.01,MAIN_MARGIN)['NLL_pass']
    assert not conditions(FLOOR-2e-5,MAIN_MARGIN+.01,MAIN_MARGIN)['NLL_pass']
    assert not conditions(FLOOR,MAIN_MARGIN,MAIN_MARGIN)['margin_pass']
    assert conditions(FLOOR,FINAL_MARGIN+.01,FINAL_MARGIN)['margin_pass']
    return dict(status='PASS',fixed_75_25_component_costs=True,NLL_and_F0_orthogonal=True,degenerate_stops=True,one_native_match=True,actual_NLL_and_A_margin_admission=True,prior_weight_identity_and_restore_checks=True)
