"""Analytic margin maximization with fixed linear-NLL and old-F0 constraints."""
import math
import torch
import torch.nn.functional as F
from .editor import direction,editable_fp32,nll,finite
from .marginab_ops import fisher_cost


class StopRound(RuntimeError):
    """Frozen scientific stop, never authorization for an alternate candidate."""


def solve(gn,gm,qn,qm,En,Em,cost,floor):
    cross=float((gn.double()*qm.double()).sum());scale=math.sqrt(2*cost/Em)
    free_linear=cross*scale
    data=dict(NLL_energy=En,margin_energy=Em,cross=cross,cost=cost,NLL_descent_floor=floor,unconstrained_linear_NLL_change=free_linear,
        minimum_NLL_cost=floor*floor/(2*En),constraint_active=free_linear>-floor)
    if free_linear<=-floor:
        data['status']='STOPPED_INACTIVE_OLD_B';return None,data
    minimum=data['minimum_NLL_cost']
    if minimum>cost:
        data['status']='STOPPED_INFEASIBLE';return None,data
    perp_energy=Em-cross*cross/En;data['perpendicular_margin_energy']=perp_energy
    if not math.isfinite(perp_energy) or perp_energy<=1e-6*Em or cost-minimum<=0:
        data['status']='STOPPED_DEGENERATE';return None,data
    perpendicular=finite(qm-(cross/En)*qn)
    delta=finite(-(floor/En)*qn+math.sqrt(2*(cost-minimum)/perp_energy)*perpendicular)
    data.update(status='ACTIVE_SOLUTION',predicted_NLL_change=float((gn.double()*delta.double()).sum()),predicted_margin_gain=float((gm.double()*delta.double()).sum()))
    if data['predicted_NLL_change']>-floor+1e-5 or data['predicted_margin_gain']<=0:
        data['status']='STOPPED_LINEAR_INFEASIBLE';return None,data
    return delta,data


def match_delta(original,delta,cost,a,b,gn,floor):
    seed=(original.float()+delta).half().float()-original.float();seed_cost=fisher_cost(seed,a,b)
    if not math.isfinite(seed_cost) or seed_cost<=0:raise StopRound('STOPPED_ZERO_NATIVE_SEED')
    ratio=math.sqrt(cost/seed_cost);native=(original.float()+ratio*delta).half().float()-original.float()
    actual=fisher_cost(native,a,b);error=abs(actual/cost-1);linear=float((gn.double()*native.double()).sum())
    return ratio,dict(target_native_F0_cost=cost,seed_native_F0_cost=seed_cost,ratio=ratio,ratio_applications=1,actual_native_F0_cost=actual,relative_error=error,tolerance=.01,cost_pass=math.isfinite(error) and error<=.01,native_linear_NLL_change=linear,linear_NLL_pass=linear<=-floor+1e-5)


def propose(runtime,layer,pair,factor,identity,cost,floor,history,save):
    original=layer.weight.detach().clone();a,b=(factor[k].to(layer.weight.device) for k in ('a','b'));_,target,j,absolute=identity
    with editable_fp32(runtime.model,layer):
        output=runtime.model(**pair[0].forward_kwargs());logits=output.logits.float();v=logits[0,absolute]
        margin=v[target]-v[j];loss=F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),pair[0].labels[:,1:].reshape(-1),ignore_index=-100)
        gm=finite(torch.autograd.grad(margin,layer.weight,retain_graph=True)[0].detach());gn=finite(torch.autograd.grad(loss,layer.weight)[0].detach());loss_value=float(loss.detach());margin_value=float(margin.detach());del output,logits,v,loss,margin
        loss0=nll(runtime.model,pair[1]);g0=finite(torch.autograd.grad(loss0,layer.weight)[0].detach());null_value=float(loss0.detach());del loss0
        qn,geometry_n=direction(gn,g0,a,b);neg_qm,geometry_m=direction(-gm,g0,a,b)
        data=dict(status='GEOMETRY',fp32_NLL=loss_value,fp32_margin=margin_value,fp32_null_NLL=null_value,NLL_geometry=geometry_n,margin_geometry=geometry_m,candidate_formulated=False,candidate_written=False)
        save(data)
        if qn is None or neg_qm is None:raise StopRound('STOPPED_DEGENERATE')
        differences=dict(NLL_energy_relative=abs(geometry_n['E']/history['arms']['A']['proposal']['E']-1),margin_energy_relative=abs(geometry_m['E']/history['arms']['B']['proposal']['geometry']['E']-1),fp32_NLL_absolute=abs(loss_value-history['arms']['A']['proposal']['fp32_before'][0]),fp32_null_absolute=abs(null_value-history['arms']['A']['proposal']['fp32_before'][1]),fp32_margin_absolute=abs(margin_value+history['arms']['B']['proposal']['fixed_competitor_loss']))
        data['historical_gradient_checks']=differences;save(data)
        if any(not math.isfinite(v) or v>1e-5 for v in differences.values()):raise ValueError('Frozen endpoint gradient geometry mismatch')
        delta,analytic=solve(gn,gm,qn,-neg_qm,geometry_n['E'],geometry_m['E'],cost,floor)
        data['analytic']=analytic;data['status']=analytic['status'];save(data)
        if delta is None:raise StopRound(analytic['status'])
        data['candidate_formulated']=True;data['analytic_F0_cost']=fisher_cost(delta,a,b);save(data)
        if abs(data['analytic_F0_cost']/cost-1)>1e-5:raise ValueError('Analytic trust-radius mismatch')
        ratio,matching=match_delta(original,delta,cost,a,b,gn,floor);data['matching']=matching;save(data)
        if not matching['cost_pass']:raise StopRound('STOPPED_COST_MATCH_FAILED')
        if not matching['linear_NLL_pass']:raise StopRound('STOPPED_QUANTIZED_LINEAR_NLL_FAILED')
        with torch.no_grad():layer.weight.copy_(finite(original.float()+ratio*delta))
        data['candidate_written']=True;data['status']='NATIVE_WRITTEN';save(data)
    actual=layer.weight.detach().float()-original.float();data['actual_native_F0_cost']=fisher_cost(actual,a,b);save(data)
    if abs(data['actual_native_F0_cost']/data['matching']['actual_native_F0_cost']-1)>1e-10:raise ValueError('Native commit differs from preview')
    return data
