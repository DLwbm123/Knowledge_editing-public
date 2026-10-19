"""One fixed 75/25 H construction; all finite native outcomes remain gated."""
import math
import torch
import torch.nn.functional as F
from .editor import direction,editable_fp32,nll,finite
from .finite_null import gates,correction,native_losses
from .marginab_ops import fisher_cost,read_margin
from .bdecompose_ops import weight_guard,StopRound

COST=5.368434395105091e-13
FLOOR=.07246601581573486
MAIN_MARGIN=-5.421875
FINAL_MARGIN=-5.4296875
TOL=1e-5


def build_h(gn,gm,qn,qm,En,Em,cost,a,b):
    cross=float((gn.double()*qm.double()).sum());ep=Em-cross*cross/En
    data=dict(NLL_energy=En,margin_energy=Em,cross=cross,perpendicular_energy=ep,energy_fractions=[.75,.25],target_cost=cost)
    if not all(math.isfinite(x) and x>0 for x in (En,Em,cost)) or not math.isfinite(ep) or ep<=1e-6*Em:
        data['status']='STOPPED_DEGENERATE_H';return None,data
    qp=finite(qm-(cross/En)*qn)
    dn=finite(-math.sqrt(2*.75*cost/En)*qn);dm=finite(math.sqrt(2*.25*cost/ep)*qp);delta=finite(dn+dm)
    kn=fisher_cost(dn,a,b);km=fisher_cost(dm,a,b);kt=fisher_cost(delta,a,b)
    orth=float((qn.double()*qp.double()*b.double()[:,None]*a.double()[None,:]).sum())
    data.update(status='FORMULATED',actual_perpendicular_energy=2*fisher_cost(qp,a,b),NLL_component_cost=kn,margin_component_cost=km,continuous_total_cost=kt,
        F0_orthogonal_residual=abs(orth)/math.sqrt(En*ep),NLL_orthogonal_residual=abs(float((gn.double()*qp.double()).sum()))/math.sqrt(En*ep),
        NLL_component_cost_relative_error=abs(kn/(.75*cost)-1),margin_component_cost_relative_error=abs(km/(.25*cost)-1),continuous_cost_relative_error=abs(kt/cost-1),perpendicular_energy_relative_error=abs(2*fisher_cost(qp,a,b)/ep-1),
        predicted_NLL_change=float((gn.double()*delta.double()).sum()),predicted_margin_gain=float((gm.double()*delta.double()).sum()),
        NLL_component_linear_NLL=float((gn.double()*dn.double()).sum()),margin_component_linear_NLL=float((gn.double()*dm.double()).sum()))
    return delta,data


def match_once(original,delta,cost,a,b,gn,gm,g0):
    seed=(original.float()+delta).half().float()-original.float();seed_cost=fisher_cost(seed,a,b)
    if not math.isfinite(seed_cost) or seed_cost<=0:raise StopRound('STOPPED_ZERO_NATIVE_SEED')
    ratio=math.sqrt(cost/seed_cost);rounded=finite((original.float()+ratio*delta).half());q=rounded.float()-original.float();actual=fisher_cost(q,a,b)
    data=dict(seed_native_F0_cost=seed_cost,target_native_F0_cost=cost,ratio=ratio,ratio_applications=1,actual_native_F0_cost=actual,relative_error=abs(actual/cost-1),tolerance=.01,
        native_linear_NLL_change=float((gn.double()*q.double()).sum()),native_linear_margin_gain=float((gm.double()*q.double()).sum()),native_linear_null_change=float((g0.double()*q.double()).sum()),native_delta_l2=float(torch.linalg.vector_norm(q.double())),changed_elements=int(torch.count_nonzero(q)))
    return rounded,data


def conditions(gain,margin,reference):
    return dict(actual_NLL_gain=gain,NLL_floor=FLOOR,NLL_pass=math.isfinite(gain) and gain>=FLOOR-TOL,margin=margin,margin_reference=reference,margin_pass=math.isfinite(margin) and margin>reference+TOL)


def evaluate(rt,record,layer,pair,factor,identity,history,base_null,start_losses,margin0,data,emit,checks,generate):
    original=layer.weight.detach().clone();w=original.float();a,b=(factor[k].to(layer.weight.device) for k in ('a','b'))
    data.update(candidate_formulated=False,candidate_written=False,candidate_evaluated=False,matching_applications=0,new_secant=0,accepted=False,local_flip=None,historical_weight_direct_comparison=None);emit()
    def near(got,expected,label,relative=False,tolerance=TOL):
        diff=abs(got/expected-1) if relative else abs(got-expected)
        checks.append(dict(field=label,**{'relative_difference' if relative else 'absolute_difference':diff},tolerance=tolerance));emit()
        if not math.isfinite(diff) or diff>tolerance:raise StopRound('STOPPED_RECHECK_'+label)
    def bound(value,tolerance,label):
        checks.append(dict(field=label,absolute_difference=value,tolerance=tolerance));emit()
        if not math.isfinite(value) or value>tolerance:raise StopRound('STOPPED_GEOMETRY_'+label)
    oldA=history['arms']['A'];oldB=history['arms']['B']
    near(oldA['proposal']['native_update']['fisher_cost'],COST,'frozen_A_cost',True)
    near(oldA['proposal']['native_gain'],FLOOR,'frozen_A_gain')
    near(oldA['main_margin']['margin'],MAIN_MARGIN,'frozen_A_main_margin');near(oldA['final_margin']['margin'],FINAL_MARGIN,'frozen_A_final_margin')
    near(start_losses[0],1.5107299089431763,'endpoint.native_visual');near(start_losses[1],3.70239520072937,'endpoint.native_null');near(margin0['mean_nll'],start_losses[0],'endpoint.native_repeat')
    try:
        with editable_fp32(rt.model,layer,readonly=True):
            with weight_guard(layer,w,'H.F_gradient',checks,emit):
                output=rt.model(**pair[0].forward_kwargs());logits=output.logits.float();_,target,j,absolute=identity;v=logits[0,absolute]
                margin=v[target]-v[j];loss=F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),pair[0].labels[:,1:].reshape(-1),ignore_index=-100,reduction='mean')
                gm=finite(torch.autograd.grad(margin,layer.weight,retain_graph=True)[0].detach());gn=finite(torch.autograd.grad(loss,layer.weight)[0].detach())
                f0=float(loss.detach());m0=float(margin.detach());del output,logits,v,margin,loss
            with weight_guard(layer,w,'H.F_null_gradient',checks,emit):
                loss0=nll(rt.model,pair[1]);g0=finite(torch.autograd.grad(loss0,layer.weight)[0].detach());null0=float(loss0.detach());del loss0
            qn,geo_n=direction(gn,g0,a,b);neg_qm,geo_m=direction(-gm,g0,a,b)
            data['geometry']=dict(fp32_before=[f0,null0],fp32_margin=m0,NLL_geometry=geo_n,margin_geometry=geo_m);emit()
            near(f0,oldA['proposal']['fp32_before'][0],'H.historical_F_visual');near(null0,oldB['proposal']['fp32_null_nll'],'H.historical_F_null');near(m0,-oldB['proposal']['fixed_competitor_loss'],'H.historical_F_margin')
            if qn is None or neg_qm is None:raise StopRound('STOPPED_DEGENERATE_GRADIENT')
            near(geo_n['E'],oldA['proposal']['E'],'H.historical_NLL_energy',True);near(geo_m['E'],oldB['proposal']['geometry']['E'],'H.historical_margin_energy',True)
            delta,analytic=build_h(gn,gm,qn,-neg_qm,geo_n['E'],geo_m['E'],COST,a,b)
            data['analytic']=analytic;emit()
            near(analytic['cross'],-476130710069.37573,'H.historical_cross',True)
            if delta is None:raise StopRound(analytic['status'])
            data['candidate_formulated']=True;emit()
            for key in ('NLL_component_cost_relative_error','margin_component_cost_relative_error','continuous_cost_relative_error','perpendicular_energy_relative_error','F0_orthogonal_residual','NLL_orthogonal_residual'):bound(analytic[key],TOL,'H.'+key)
            null_residual=abs(float((g0.double()*delta.double()).sum()))/math.sqrt(2*COST*geo_n['k00'])
            analytic['normalized_null_residual']=null_residual;emit();bound(null_residual,1e-4,'H.null_projection_residual')
            if analytic['predicted_NLL_change']>=0 or analytic['predicted_margin_gain']<=0:raise StopRound('STOPPED_ANALYTIC_DIRECTION_SIGN')
            data['matching_applications']=1;emit()
            rounded,matching=match_once(original,delta,COST,a,b,gn,gm,g0);data['matching']=matching;emit()
            if not math.isfinite(matching['relative_error']) or matching['relative_error']>.01:raise StopRound('STOPPED_NATIVE_COST_MATCH_FAILED')
            if matching['native_linear_NLL_change']>=0 or matching['native_linear_margin_gain']<=0:raise StopRound('STOPPED_NATIVE_LINEAR_SIGN_FAILED')
        # Only the one final matched tensor is written to the native model.
        if layer.weight.dtype!=original.dtype or not torch.equal(layer.weight,original):raise StopRound('STOPPED_GEOMETRY_WEIGHT_RESTORE_FAILED')
        with torch.no_grad():layer.weight.copy_(rounded)
        data['candidate_written']=True;emit()
        near(fisher_cost(layer.weight.detach().float()-w,a,b),matching['actual_native_F0_cost'],'H.native_commit_cost',True,tolerance=1e-10)
        with weight_guard(layer,rounded,'H.native_main',checks,emit,expected_calls=3):
            data['main_gate']=gates(rt,pair,factor,original,start_losses,base_null)
            data['main_margin'],_=read_margin(rt,pair[0],identity[0],identity[2])
        data['candidate_evaluated']=True;emit();near(data['main_margin']['mean_nll'],data['main_gate']['native_nll'][0],'H.main_native_repeat')
        data['main_conditions']=conditions(start_losses[0]-data['main_gate']['native_nll'][0],data['main_margin']['margin'],MAIN_MARGIN)
        data['main_margin_gain']=data['main_margin']['margin']-margin0['margin'];data['status']='MAIN_CHECKED';emit()
        if not data['main_conditions']['NLL_pass']:raise StopRound('STOPPED_MAIN_ACTUAL_NLL_FAILED')
        if not data['main_conditions']['margin_pass']:raise StopRound('STOPPED_MAIN_MARGIN_FAILED')
        if not data['main_gate']['gates']['trust']:raise StopRound('STOPPED_MAIN_TRUST_FAILED')
        data['correction']=None
        if not data['main_gate']['gates']['null']:
            data['new_secant']=1;emit();data['correction']=correction(rt,record,factor,base_null,compensate='secant');emit()
        final_weight=layer.weight.detach().clone()
        with weight_guard(layer,final_weight,'H.native_final',checks,emit,expected_calls=3):
            data['final_gate']=gates(rt,pair,factor,original,start_losses,base_null)
            data['final_margin'],_=read_margin(rt,pair[0],identity[0],identity[2])
        emit();near(data['final_margin']['mean_nll'],data['final_gate']['native_nll'][0],'H.final_native_repeat')
        data['final_conditions']=conditions(start_losses[0]-data['final_gate']['native_nll'][0],data['final_margin']['margin'],FINAL_MARGIN)
        data['final_margin_gain']=data['final_margin']['margin']-margin0['margin']
        data['accepted']=data['final_gate']['accepted'] and data['final_conditions']['NLL_pass'] and data['final_conditions']['margin_pass'] and (data['correction'] is None or data['correction']['status']=='APPLIED');emit()
        if not data['accepted']:raise StopRound('STOPPED_FINAL_CONDITIONS_FAILED')
        generate(data);data['status']='COMPLETE_ACCEPTED_H';emit()
    finally:
        with torch.no_grad():layer.weight.data=original.clone()
        data['endpoint_restore_nll']=native_losses(rt,pair);data['endpoint_restored']=layer.weight.dtype==original.dtype and torch.equal(layer.weight,original) and data['endpoint_restore_nll']==start_losses;emit()
        if not data['endpoint_restored']:raise StopRound('STOPPED_H_ENDPOINT_RESTORE_FAILED')
