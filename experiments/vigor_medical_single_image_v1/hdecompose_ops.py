"""Only the historical complete H, with frozen coefficients and matching ratio."""
import math
import torch
import torch.nn.functional as F
from .editor import editable_fp32,direction,finite,nll
from .finite_null import gates
from .marginab_ops import read_margin,fisher_cost
from .bdecompose_ops import StopRound,weight_guard,decomposition

RATIO=.9974240855956152


def reconstruct_h(qn,qm,old):
    """Exact original operation order, using saved scalars, never new fitting."""
    En=old['NLL_energy'];ep=old['perpendicular_energy'];cost=old['target_cost'];cross=old['cross']
    qp=finite(qm-(cross/En)*qn)
    dn=finite(-math.sqrt(2*.75*cost/En)*qn)
    dm=finite(math.sqrt(2*.25*cost/ep)*qp)
    return finite(dn+dm)


def evaluate(rt,layer,pair,factor,identity,history,base_null,start_losses,data,emit,checks):
    original=layer.weight.detach().clone();w=original.float();a,b=(factor[k].to(layer.weight.device) for k in ('a','b'))
    old=history['hybrid'];oa=old['analytic'];om=old['matching'];og=old['geometry']
    def near(got,expected,label,relative=False):
        diff=abs(got/expected-1) if relative else abs(got-expected)
        checks.append(dict(field=label,**{'relative_difference' if relative else 'absolute_difference':diff},tolerance=1e-5));emit()
        if not math.isfinite(diff) or diff>1e-5:raise StopRound('STOPPED_RECHECK_'+label)
    def same(value,label):
        checks.append(dict(field=label,passed=bool(value)));emit()
        if not value:raise StopRound('STOPPED_IDENTITY_'+label)
    def read_F(expected,label):
        with torch.no_grad():layer.weight.copy_(expected)
        with torch.no_grad(),weight_guard(layer,expected,label,checks,emit):value=float(nll(rt.model,pair[0]))
        data.setdefault('F',{})[label]=value;emit();return value
    data.update(new_cost_matching=0,new_secant=0,new_edit_candidates=0,historical_weight_direct_comparison=None,historical_F_rounded_reference=None,historical_weight_direct_comparison_reason='NO_HISTORICAL_WEIGHT_SNAPSHOT',frozen_ratio=RATIO)
    emit();same(om['ratio']==RATIO,'H.exact_frozen_ratio');same(oa['energy_fractions']==[.75,.25] and oa['target_cost']==5.368434395105091e-13,'H.frozen_construction')
    with torch.no_grad(),weight_guard(layer,original,'N_base_repeat',checks,emit):n0repeat=float(nll(rt.model,pair[0]))
    data['N_base']=start_losses[0];data['N_base_repeat']=n0repeat;emit();near(n0repeat,start_losses[0],'N_base_repeat')
    try:
        with editable_fp32(rt.model,layer,readonly=True):
            try:
                with weight_guard(layer,w,'F_base_gradient',checks,emit):
                    output=rt.model(**pair[0].forward_kwargs());logits=output.logits.float();_,target,j,absolute=identity;v=logits[0,absolute]
                    margin=v[target]-v[j];loss=F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),pair[0].labels[:,1:].reshape(-1),ignore_index=-100,reduction='mean')
                    gm=finite(torch.autograd.grad(margin,layer.weight,retain_graph=True)[0].detach());gn=finite(torch.autograd.grad(loss,layer.weight)[0].detach());f0=float(loss.detach());m0=float(margin.detach())
                del output,logits,v,margin,loss
                data['F']={'F_base_gradient':f0};data['fp32_margin']=m0;emit();near(f0,og['fp32_before'][0],'historical_F_base');near(m0,og['fp32_margin'],'historical_F_margin')
                f0repeat=read_F(w,'F_base_repeat');near(f0repeat,f0,'F_base_repeat')
                with weight_guard(layer,w,'F_null_gradient',checks,emit):
                    null_loss=nll(rt.model,pair[1]);g0=finite(torch.autograd.grad(null_loss,layer.weight)[0].detach());null0=float(null_loss.detach())
                del null_loss
                data['F_null_base']=null0;emit();near(null0,og['fp32_before'][1],'historical_F_null')
                qn,geo_n=direction(gn,g0,a,b);neg_qm,geo_m=direction(-gm,g0,a,b)
                data['NLL_geometry']=geo_n;data['margin_geometry']=geo_m;emit()
                if qn is None or neg_qm is None:raise StopRound('STOPPED_DEGENERATE_H_RECONSTRUCTION')
                near(geo_n['E'],oa['NLL_energy'],'historical_NLL_energy',True);near(geo_m['E'],oa['margin_energy'],'historical_margin_energy',True)
                cross=float((gn.double()*(-neg_qm).double()).sum());near(cross,oa['cross'],'historical_cross',True)
                # Saved coefficients and old ratio; no call to any match/solve routine.
                delta=reconstruct_h(qn,-neg_qm,oa);u=finite(RATIO*delta)
                unrounded=finite(w+RATIO*delta);rounded=finite(unrounded.half());q=rounded.float()-w
                independently_emulated_old_write=(original.float()+om['ratio']*reconstruct_h(qn,-neg_qm,oa)).half()
                same(torch.equal(independently_emulated_old_write,rounded),'current_H_write_path_tensor_equal')
                rawp=float((gn.double()*delta.double()).sum());p=float((gn.double()*u.double()).sum());pold=RATIO*oa['predicted_NLL_change'];gq=float((gn.double()*q.double()).sum());qlin=float((gn.double()*(q-u).double()).sum())
                unorm=float(torch.linalg.vector_norm(u.double()));qnorm=float(torch.linalg.vector_norm(q.double()))
                data['increment']=dict(raw_continuous_prediction=rawp,old_raw_continuous_prediction=oa['predicted_NLL_change'],raw_to_matched_scaling_change=pold-oa['predicted_NLL_change'],P_old=pold,P=p,D=p-pold,g_dot_q=gq,Q_lin=qlin,cost_raw=fisher_cost(delta,a,b),cost_u=fisher_cost(u,a,b),cost_q=fisher_cost(q,a,b),native_linear_margin_gain=float((gm.double()*q.double()).sum()),u_l2=unorm,q_l2=qnorm,rounding_l2=float(torch.linalg.vector_norm((q-u).double())),relative_rounding_l2=float(torch.linalg.vector_norm((q-u).double()))/unorm,cosine=float((u.double()*q.double()).sum())/(unorm*qnorm),unrounded_FP32_realization_error_l2=float(torch.linalg.vector_norm((unrounded-w-u).double())))
                emit();near(rawp,oa['predicted_NLL_change'],'raw_prediction_alignment');near(p,pold,'matched_P_alignment');near(gq,om['native_linear_NLL_change'],'historical_native_linear_NLL');near(data['increment']['native_linear_margin_gain'],om['native_linear_margin_gain'],'historical_native_linear_margin')
                near(data['increment']['cost_raw'],oa['continuous_total_cost'],'historical_continuous_cost',True);near(data['increment']['cost_q'],om['actual_native_F0_cost'],'historical_native_F0_cost',True)
                fu=read_F(unrounded,'F_unrounded')
                with torch.no_grad():layer.weight.copy_(w)
                same(torch.equal(layer.weight,w),'W_restored_before_unrounded_repeat')
                furepeat=read_F(unrounded,'F_unrounded_repeat');near(furepeat,fu,'F_unrounded_repeat')
                fq=read_F(rounded.float(),'F_rounded')
            finally:
                with torch.no_grad():layer.weight.copy_(w)
        same(layer.weight.dtype==original.dtype and torch.equal(layer.weight,original),'F_exit_native_W_restored')
        with torch.no_grad():layer.weight.copy_(rounded)
        with weight_guard(layer,rounded,'N_rounded',checks,emit,expected_calls=3):
            gate=gates(rt,pair,factor,original,start_losses,base_null);margin,_=read_margin(rt,pair[0],identity[0],identity[2])
        data['native_main_gate']=gate;data['native_margin']=margin;data['N_rounded']=gate['native_nll'][0];emit()
        for i,(got,expected) in enumerate(zip(gate['native_nll'],old['main_gate']['native_nll'])):near(got,expected,'historical_N_rounded_'+str(i))
        near(margin['margin'],old['main_margin']['margin'],'historical_N_margin');near(margin['mean_nll'],gate['native_nll'][0],'N_rounded_visual_repeat');same(gate['gates']==old['main_gate']['gates'],'historical_main_gates')
        data['decomposition']=decomposition(f0,fu,fq,start_losses[0],gate['native_nll'][0],p,qlin,pold=pold);data['status']='COMPLETE_READONLY';emit()
        near(data['decomposition']['closure_error'],0,'decomposition_closure');near(data['decomposition']['historical_gap_closure_error'],0,'historical_gap_closure')
    finally:
        with torch.no_grad():layer.weight.data=original.clone()
