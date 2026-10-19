"""Read-only fixed-B decomposition, with actual-forward weight guards."""
from contextlib import contextmanager
import math
import torch
import torch.nn.functional as F
from .editor import direction,editable_fp32,nll,finite
from .finite_null import gates,native_losses
from .marginab_ops import fisher_cost

P_OLD=-0.1516953133602808


class StopRound(RuntimeError):
    pass


def decomposition(f0,fu,fq,n0,nq,p,qlin,pold=P_OLD):
    r=fu-f0-p;q=fq-fu;e=(nq-n0)-(fq-f0);d=p-pold
    return dict(P_old=pold,P=p,D=d,R=r,Q=q,Q_lin=qlin,Q_rem=q-qlin,E=e,
                delta_native=nq-n0,delta_F_unrounded=fu-f0,delta_F_rounded=fq-f0,
                closure_error=(nq-n0)-(p+r+q+e),historical_gap_closure_error=(nq-n0-pold)-(d+r+q+e))


@contextmanager
def weight_guard(layer,expected,label,checks,emit,expected_calls=1):
    """Check the tensor actually presented to Linear; no extra model call."""
    rows=[]
    def before(module,args):
        same_dtype=module.weight.dtype==expected.dtype
        equal=same_dtype and torch.equal(module.weight.detach(),expected)
        row=dict(field=label+'.forward_weight',passed=equal,dtype_matches=same_dtype,
                 max_absolute_difference=float((module.weight.detach().float()-expected.float()).abs().max()),
                 input_dtype_is_FP32=args[0].dtype==torch.float32 if expected.dtype==torch.float32 else None)
        rows.append(row);checks.append(row);emit()
        if not equal or (expected.dtype==torch.float32 and args[0].dtype!=torch.float32):raise ValueError('Actual-forward weight/input identity failed')
    h=layer.register_forward_pre_hook(before)
    try:
        yield
        if len(rows)!=expected_calls:raise ValueError('Unexpected editable layer invocation count')
    finally:h.remove()


def evaluate(rt,layer,pair,factor,identity,history,base_null,start_losses,data,emit,checks):
    original=layer.weight.detach().clone();w=original.float()
    a,b=(factor[k].to(layer.weight.device) for k in ('a','b'))
    oldB=history['arms']['B'];oldA=history['arms']['A'];matching=oldB['proposal']['matching']
    def near(actual,expected,label,relative=False):
        diff=abs(actual/expected-1) if relative else abs(actual-expected)
        checks.append(dict(field=label,**{'relative_difference' if relative else 'absolute_difference':diff},tolerance=1e-5));emit()
        if not math.isfinite(diff) or diff>1e-5:raise StopRound('STOPPED_RECHECK_'+label)
    def same(value,label):
        checks.append(dict(field=label,passed=bool(value)));emit()
        if not value:raise StopRound('STOPPED_IDENTITY_'+label)
    def read_F(expected,label):
        with torch.no_grad():layer.weight.copy_(expected)
        with torch.no_grad(),weight_guard(layer,expected,label,checks,emit):value=float(nll(rt.model,pair[0]))
        data.setdefault('F',{})[label]=value;emit();return value
    data.update(historical_weight_direct_comparison=None,historical_weight_direct_comparison_reason='NO_HISTORICAL_WEIGHT_SNAPSHOT',new_cost_matching=0,new_edit_candidates=0)
    emit()
    with torch.no_grad(),weight_guard(layer,original,'N_base_repeat',checks,emit):n0repeat=float(nll(rt.model,pair[0]))
    data['N_base_repeat']=n0repeat;data['N_base']=start_losses[0];emit();near(n0repeat,start_losses[0],'N_base_repeat')
    try:
        with editable_fp32(rt.model,layer,readonly=True):
            try:
                # One common forward, two distinct objectives. The NLL is full mean CE.
                with weight_guard(layer,w,'F_base_gradient',checks,emit):
                    output=rt.model(**pair[0].forward_kwargs());logits=output.logits.float()
                    loss=F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),pair[0].labels[:,1:].reshape(-1),ignore_index=-100,reduction='mean')
                    _,target,j,absolute=identity;v=logits[0,absolute];margin_loss=v[j]-v[target]
                    gm=finite(torch.autograd.grad(margin_loss,layer.weight,retain_graph=True)[0].detach())
                    gn=finite(torch.autograd.grad(loss,layer.weight)[0].detach())
                    f0=float(loss.detach());margin_value=float(margin_loss.detach())
                del output,logits,loss,v,margin_loss
                data['F']={'F_base_gradient':f0};data['fixed_competitor_loss']=margin_value;emit()
                near(f0,oldA['proposal']['fp32_before'][0],'historical_F_base')
                near(margin_value,oldB['proposal']['fixed_competitor_loss'],'historical_margin_loss')
                f0repeat=read_F(w,'F_base_repeat');near(f0repeat,f0,'F_base_repeat')
                with weight_guard(layer,w,'F_null_gradient',checks,emit):
                    null_loss=nll(rt.model,pair[1]);g0=finite(torch.autograd.grad(null_loss,layer.weight)[0].detach());null_value=float(null_loss.detach())
                del null_loss
                data['F_null_base']=null_value;emit();near(null_value,oldB['proposal']['fp32_null_nll'],'historical_F_null_base')
                tangent,geometry=direction(gm,g0,a,b)
                _,ngeometry=direction(gn,g0,a,b)
                data['margin_geometry']=geometry;data['NLL_geometry']=ngeometry;emit()
                if tangent is None:raise StopRound('STOPPED_DEGENERATE_FIXED_DIRECTION')
                near(geometry['E'],oldB['proposal']['geometry']['E'],'margin_energy',True)
                near(ngeometry['E'],oldA['proposal']['E'],'NLL_energy',True)
                # Frozen scalar scale, not a new seed, cost match, or optimization.
                seed=matching['seed_scale'];ratio=matching['ratio'];scale=seed*ratio
                near(seed,math.sqrt(2*matching['target_native_F0_cost']/geometry['E']),'frozen_seed_scale',True)
                u=finite(-scale*tangent)
                unrounded=finite(w-scale*tangent)  # preserve historical arithmetic order
                rounded=finite(unrounded.half())
                q=rounded.float()-w
                # Independent emulation of the old commit path, within current reconstruction only.
                historical_write=(original.float()-(seed*ratio)*tangent).to(original.dtype)
                same(torch.equal(historical_write,rounded),'current_write_path_tensor_equal')
                p=float((gn.double()*u.double()).sum());gq=float((gn.double()*q.double()).sum());qlin=float((gn.double()*(q-u).double()).sum())
                unorm=float(torch.linalg.vector_norm(u.double()));qnorm=float(torch.linalg.vector_norm(q.double()))
                data['increment']=dict(final_scale=scale,seed_scale=seed,frozen_ratio=ratio,cost_u=fisher_cost(u,a,b),cost_q=fisher_cost(q,a,b),P_old=P_OLD,P=p,D=p-P_OLD,g_dot_q=gq,Q_lin=qlin,
                    u_l2=unorm,q_l2=qnorm,rounding_l2=float(torch.linalg.vector_norm((q-u).double())),relative_rounding_l2=float(torch.linalg.vector_norm((q-u).double()))/unorm,
                    cosine=float((u.double()*q.double()).sum())/(unorm*qnorm),unrounded_FP32_realization_error_l2=float(torch.linalg.vector_norm((unrounded-w-u).double())))
                emit();near(data['increment']['cost_q'],oldB['proposal']['actual_native_F0_cost'],'native_F0_cost',True)
                near(p,P_OLD,'P_alignment')
                fu=read_F(unrounded,'F_unrounded')
                # Independent restore and setup; never build the unrounded branch via FP16.
                with torch.no_grad():layer.weight.copy_(w)
                same(torch.equal(layer.weight,w),'W_restored_before_unrounded_repeat')
                furepeat=read_F(unrounded,'F_unrounded_repeat');near(furepeat,fu,'F_unrounded_repeat')
                fq=read_F(rounded.float(),'F_rounded')
                near(fq,oldB['correction']['fp32_before'][0],'historical_F_rounded')
            finally:
                with torch.no_grad():layer.weight.copy_(w)
        same(torch.equal(layer.weight,original) and layer.weight.dtype==original.dtype,'F_exit_native_W_restored')
        with torch.no_grad():layer.weight.copy_(rounded)
        with weight_guard(layer,rounded,'N_rounded',checks,emit,expected_calls=2):
            native_gate=gates(rt,pair,factor,original,start_losses,base_null)
        data['native_main_gate']=native_gate;data['N_rounded']=native_gate['native_nll'][0];emit()
        for i,(got,old) in enumerate(zip(native_gate['native_nll'],oldB['main_gate']['native_nll'])):near(got,old,'historical_N_rounded_'+str(i))
        same(native_gate['gates']==oldB['main_gate']['gates'],'historical_main_gates')
        data['decomposition']=decomposition(f0,fu,fq,start_losses[0],native_gate['native_nll'][0],p,qlin)
        data['status']='COMPLETE_READONLY';emit()
        near(data['decomposition']['closure_error'],0,'decomposition_closure')
        near(data['decomposition']['historical_gap_closure_error'],0,'historical_gap_closure')
    finally:
        with torch.no_grad():layer.weight.data=original.clone()
