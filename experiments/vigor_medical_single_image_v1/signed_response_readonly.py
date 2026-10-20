"""Fixed native-increment response attribution, never a candidate search."""
import math
import torch
from .editor import editable_fp32
from .finite_null import native_losses,gates
from .marginab_ops import read_margin,fisher_cost
from .bdecompose_ops import weight_guard,StopRound


def parts(f0,fu,fq,n0,nq,p):
    r=fu-f0-p;q=fq-fu;e=(nq-n0)-(fq-f0)
    return dict(P=p,R=r,Q=q,E=e,delta_native=nq-n0,delta_F_unrounded=fu-f0,
                delta_F_rounded=fq-f0,closure_error=(nq-n0)-(p+r+q+e))


def diagnose(rt,layer,pair,factor,identity,base,proposal,base_losses,base_margin,
             base_null,predicted,gradient_baseline,old,data,checks,emit):
    d=dict(status='READING',read_only=True,new_edit_candidates=0,accepted=False,
           diagnostic_forward_max=20,diagnostic_forward_calls=0,positions=[0,.5,1],
           historical_native_full_forward_calls=0,chain='editable_layer_FP32_not_full_chain_FP32',
           historical_weight_direct_comparison=None)
    data['fixed_response_diagnostic']=d;emit()
    def fail(ok,label):
        checks.append(dict(field='signed_readonly.'+label,passed=bool(ok)));emit()
        if not ok:raise StopRound('STOPPED_DIAGNOSTIC_'+label)
    def near(x,y,label):
        diff=abs(x-y);checks.append(dict(field='signed_readonly.'+label,absolute_difference=diff,tolerance=1e-5));emit()
        if not math.isfinite(diff) or diff>1e-5:raise StopRound('STOPPED_DIAGNOSTIC_'+label)
    w=base.float();q=proposal.float()-w;mid=w+.5*q;mid_native=mid.half()
    fail(torch.equal(w+q,proposal.float()),'full_increment_FP32_identity')
    fail(torch.equal(mid_native,mid_native.float().half()),'mid_native_grid')
    fail(old['candidate_evaluated'] and not old['accepted'],'old_terminal_failed_proposal')
    fail(old['continuation_start']['native_nll']==list(base_losses),'old_same_start')
    near(old['continuation_start']['margin']['margin'],base_margin['margin'],'old_start_margin')
    a,b=(factor[k].to(base.device) for k in ('a','b'))
    near(fisher_cost(q,a,b)/old['small_native_admission']['actual_native_F0_cost'],1,'old_increment_cost')
    for i,x in enumerate(predicted):near(x,old['small_native_admission']['native_linear_change'][i],'old_increment_dot_'+str(i))
    fail(data['native_null_balance']['retained_elements']==old['native_null_balance']['retained_elements'],'old_mask_count')
    def budget(n):
        d['diagnostic_forward_calls']+=n;emit()
        fail(d['diagnostic_forward_calls']<=20,'forward_cap')
    def read(expected,label):
        with torch.no_grad():layer.weight.copy_(expected)
        budget(3)
        with torch.no_grad(),weight_guard(layer,expected,'signed_readonly.'+label,checks,emit,expected_calls=3):
            losses=native_losses(rt,pair);m,_=read_margin(rt,pair[0],identity[0],identity[2])
        v=dict(nll=losses[0],null=losses[1],margin=m['margin'],fixed_margin=m['fixed_competitor_margin'],competitor_changed=m['strongest_competitor_changed'])
        d[label]=v;emit() # Preserve readings before judging qualification.
        near(m['mean_nll'],losses[0],label+'.repeat')
        fail(all(math.isfinite(v[k]) for k in ['nll','null','margin','fixed_margin']),label+'.finite')
        return v
    def restore_read(expected,label,reference):
        with torch.no_grad():layer.weight.copy_(expected)
        budget(2)
        with torch.no_grad(),weight_guard(layer,expected,'signed_readonly.'+label,checks,emit,expected_calls=2):v=native_losses(rt,pair)
        d[label]=dict(nll=v[0],null=v[1]);emit()
        for i,k in enumerate(['nll','null']):near(v[i],reference[k],label+'.'+k)
    # F reference changes only editable Linear weight/input dtype and casts output back.
    with editable_fp32(rt.model,layer,readonly=True):
        try:
            f0=read(w,'F0')
            for k,x in zip(['nll','null','fixed_margin'],[*gradient_baseline['F_baseline'],gradient_baseline['F_margin']]):near(f0[k],x,'gradient_baseline.'+k)
            fu=read(mid,'F_half_unrounded')
            fq=read(mid_native.float(),'F_half_rounded')
            f1=read(proposal.float(),'F_full')
            restore_read(w,'F0_return',f0)
        finally:
            with torch.no_grad():layer.weight.copy_(w)
    fail(layer.weight.dtype==base.dtype and torch.equal(layer.weight,base),'F_context_native_restore')
    # Native half position is a pre-registered read-only measurement, never selected.
    with torch.no_grad():layer.weight.copy_(mid_native)
    budget(3)
    with torch.no_grad(),weight_guard(layer,mid_native,'signed_readonly.N_half',checks,emit,expected_calls=3):
        verdict=gates(rt,pair,factor,base,base_losses,base_null);m,_=read_margin(rt,pair[0],identity[0],identity[2])
    nh=dict(nll=verdict['native_nll'][0],null=verdict['native_nll'][1],margin=m['margin'],fixed_margin=m['fixed_competitor_margin'],competitor_changed=m['strongest_competitor_changed'])
    d['N_half']=nh;d['N_half_original_gate_diagnostic_only']=verdict;emit();near(m['mean_nll'],nh['nll'],'N_half.repeat')
    n0=dict(nll=base_losses[0],null=base_losses[1],margin=base_margin['margin'],fixed_margin=base_margin['fixed_competitor_margin'])
    n1=dict(nll=old['main_gate']['native_nll'][0],null=old['main_gate']['native_nll'][1],margin=old['main_margin']['margin'],fixed_margin=old['main_margin']['fixed_competitor_margin'])
    d['N0_reused_control']=n0;d['N_full_reused_historical']=n1;emit()
    restore_read(base,'N0_return',n0)
    result={}
    for i,k in enumerate(['nll','fixed_margin','null']):
        v={}
        for t,ff,fr,nn in [(.5,fu,fq,nh),(1.,f1,f1,n1)]:
            z=parts(f0[k],ff[k],fr[k],n0[k],nn[k],t*predicted[i]);v[str(t)]=z
            fail(abs(z['closure_error'])<=1e-10,'closure.'+k+'.'+str(t))
        r1=v['1.0']['R'];v['half_to_full_R_ratio']=v['0.5']['R']/r1 if abs(r1)>1e-12 else None
        result[k]=v
    d['decomposition']=result;d['status']='COMPLETE_READ_ONLY_FIXED_RESPONSE';emit()
