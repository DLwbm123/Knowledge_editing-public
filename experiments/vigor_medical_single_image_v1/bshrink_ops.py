"""Two predeclared B scales; actual native NLL before any null correction."""
import math
import torch
import torch.nn.functional as F
from .editor import editable_fp32,direction,finite,nll
from .finite_null import correction,gates,native_losses
from .marginab_ops import read_margin,fisher_cost
from .bdecompose_ops import weight_guard,P_OLD,StopRound

SCALES=(.5,.25)
FLOOR=.07246601581573486
TOL=1e-5


def conditions(gain,margin,start_margin,alpha):
    return dict(actual_NLL_gain=gain,NLL_floor=alpha*FLOOR,NLL_pass=math.isfinite(gain) and gain>=alpha*FLOOR-TOL,
                margin_gain=margin-start_margin,margin_pass=math.isfinite(margin) and margin>start_margin+TOL)


def action(row,index):
    if not row['main_gate']['gates']['trust']:return 'STOP_TRUST'
    if row['main_conditions']['NLL_pass'] and row['main_conditions']['margin_pass']:return 'PRESELECT'
    return 'NEXT_FROZEN_SCALE' if index==0 else 'STOP_NO_ELIGIBLE_SCALE'


def frozen_direction(rt,layer,pair,factor,identity,history,data,checks,emit):
    original=layer.weight.detach().clone();w=original.float();a,b=(factor[k].to(layer.weight.device) for k in ('a','b'))
    def near(got,old,label,relative=False):
        diff=abs(got/old-1) if relative else abs(got-old)
        checks.append(dict(field=label,**{'relative_difference' if relative else 'absolute_difference':diff},tolerance=TOL));emit()
        if not math.isfinite(diff) or diff>TOL:raise StopRound('STOPPED_RECHECK_'+label)
    oldB=history['arms']['B']['proposal'];oldA=history['arms']['A']['proposal'];matching=oldB['matching']
    exact_scale=matching['seed_scale']==3.193810020752729e-13 and matching['ratio']==0.9975550929003212
    checks.append(dict(field='fixed_B.exact_historical_scale',passed=exact_scale));emit()
    if not exact_scale:raise StopRound('STOPPED_FIXED_SCALE_BINDING')
    near(oldA['native_gain'],FLOOR,'fixed_A_floor_binding')
    with editable_fp32(rt.model,layer,readonly=True):
        with weight_guard(layer,w,'fixed_B.F_gradient',checks,emit):
            output=rt.model(**pair[0].forward_kwargs());logits=output.logits.float();_,target,j,absolute=identity;v=logits[0,absolute]
            margin_loss=v[j]-v[target];loss=F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),pair[0].labels[:,1:].reshape(-1),ignore_index=-100,reduction='mean')
            gm=finite(torch.autograd.grad(margin_loss,layer.weight,retain_graph=True)[0].detach());gn=finite(torch.autograd.grad(loss,layer.weight)[0].detach())
            f0=float(loss.detach());margin_loss_value=float(margin_loss.detach());del output,logits,v,loss,margin_loss
        with weight_guard(layer,w,'fixed_B.F_null_gradient',checks,emit):
            null_loss=nll(rt.model,pair[1]);g0=finite(torch.autograd.grad(null_loss,layer.weight)[0].detach());null_value=float(null_loss.detach());del null_loss
        tangent,geometry=direction(gm,g0,a,b);_,ngeometry=direction(gn,g0,a,b)
        data.update(fp32_before=[f0,null_value],fixed_competitor_loss=margin_loss_value,margin_geometry=geometry,NLL_geometry=ngeometry,historical_weight_direct_comparison=None)
        emit()
        near(f0,oldA['fp32_before'][0],'fixed_B.F_visual');near(null_value,oldB['fp32_null_nll'],'fixed_B.F_null');near(margin_loss_value,oldB['fixed_competitor_loss'],'fixed_B.competitor_loss')
        if tangent is None:raise StopRound('STOPPED_FIXED_DIRECTION_DEGENERATE')
        near(geometry['E'],oldB['geometry']['E'],'fixed_B.margin_energy',True);near(ngeometry['E'],oldA['E'],'fixed_B.NLL_energy',True)
        scale=matching['seed_scale']*matching['ratio'];u=finite(-scale*tangent)
        data.update(seed_scale=matching['seed_scale'],fixed_old_ratio=matching['ratio'],final_scale=scale,unrounded_F0_cost=fisher_cost(u,a,b),linear_NLL=float((gn.double()*u.double()).sum()),P_old=P_OLD)
        emit();near(matching['seed_scale'],math.sqrt(2*matching['target_native_F0_cost']/geometry['E']),'fixed_B.seed_scale',True);near(data['linear_NLL'],P_OLD,'fixed_B.P_alignment')
    if layer.weight.dtype!=original.dtype or not torch.equal(layer.weight,original):raise ValueError('Fixed-direction extraction changed native W')
    return u,gn,a,b


def evaluate(rt,record,layer,pair,factor,identity,history,base_null,start_losses,margin0,state,emit,checks,generate):
    original=layer.weight.detach().clone();w=original.float();state.update(candidates=[],selected_scale=None,preselected_scale=None,new_secant=0,new_cost_matching=0,geometry={});emit()
    def near(got,expected,label):
        diff=abs(got-expected);checks.append(dict(field=label,absolute_difference=diff,tolerance=TOL));emit()
        if not math.isfinite(diff) or diff>TOL:raise StopRound('STOPPED_RECHECK_'+label)
    def restore(row):
        with torch.no_grad():layer.weight.data=original.clone()
        got=native_losses(rt,pair)
        row['endpoint_restore_nll']=got;row['endpoint_restored']=layer.weight.dtype==original.dtype and torch.equal(layer.weight,original) and got==start_losses;emit()
        if not row['endpoint_restored']:raise StopRound('STOPPED_CANDIDATE_RESTORE_FAILED')
    near(start_losses[0],1.5107299089431763,'endpoint.native_visual');near(start_losses[1],3.70239520072937,'endpoint.native_null');near(margin0['mean_nll'],start_losses[0],'endpoint.native_repeat')
    u,gn,a,b=frozen_direction(rt,layer,pair,factor,identity,history,state['geometry'],checks,emit)
    previous=None
    for index,alpha in enumerate(SCALES):
        row=dict(alpha=alpha,status='FORMING',accepted=False,correction=None);state['candidates'].append(row);emit()
        try:
            if not torch.equal(layer.weight,original) or layer.weight.dtype!=original.dtype:raise ValueError('Candidate common-start mismatch')
            rounded=finite((w+alpha*u).half());q=rounded.float()-w
            row['native_increment']=dict(l2=float(torch.linalg.vector_norm(q.double())),F0_cost=fisher_cost(q,a,b),linear_NLL=float((gn.double()*q.double()).sum()),changed_elements=int(torch.count_nonzero(q)));emit()
            if not row['native_increment']['changed_elements']:raise StopRound('STOPPED_QUANTIZED_ZERO')
            if previous is not None and torch.equal(rounded,previous):raise StopRound('STOPPED_DUPLICATE_QUANTIZED_CANDIDATE')
            previous=rounded.clone()
            with torch.no_grad():layer.weight.copy_(rounded)
            with weight_guard(layer,rounded,f'alpha_{alpha}.native_main',checks,emit,expected_calls=3):
                row['main_gate']=gates(rt,pair,factor,original,start_losses,base_null)
                row['main_margin'],_=read_margin(rt,pair[0],identity[0],identity[2])
            emit();near(row['main_margin']['mean_nll'],row['main_gate']['native_nll'][0],f'alpha_{alpha}.main_visual_repeat')
            row['main_conditions']=conditions(start_losses[0]-row['main_gate']['native_nll'][0],row['main_margin']['margin'],margin0['margin'],alpha)
            row['action']=action(row,index);row['status']='MAIN_CHECKED';emit()
            if row['action']=='STOP_TRUST':raise StopRound('STOPPED_MAIN_TRUST_FAILED')
            if row['action']!='PRESELECT':
                row['status']='MAIN_REJECTED';emit()
                continue
            # First eligible candidate locks search before any secant result is observed.
            state['preselected_scale']=alpha;row['status']='PRESELECTED_SEARCH_CLOSED';emit()
            if not row['main_gate']['gates']['null']:
                if state['new_secant']>=1:raise RuntimeError('Secant ceiling')
                state['new_secant']+=1;emit()
                row['correction']=correction(rt,record,factor,base_null,compensate='secant');emit()
            row['final_gate']=gates(rt,pair,factor,original,start_losses,base_null)
            row['final_margin'],_=read_margin(rt,pair[0],identity[0],identity[2]);emit()
            near(row['final_margin']['mean_nll'],row['final_gate']['native_nll'][0],f'alpha_{alpha}.final_visual_repeat')
            row['final_conditions']=conditions(start_losses[0]-row['final_gate']['native_nll'][0],row['final_margin']['margin'],margin0['margin'],alpha)
            correction_ok=row['correction'] is None or row['correction']['status']=='APPLIED'
            row['accepted']=correction_ok and row['final_gate']['accepted'] and row['final_conditions']['NLL_pass'] and row['final_conditions']['margin_pass'];emit()
            if not row['accepted']:
                row['status']='POST_CONDITIONS_REJECTED';emit();raise StopRound('STOPPED_POST_CONDITIONS_FAILED')
            state['selected_scale']=alpha;row['status']='ACCEPTED';emit()
            generate(row)
            state['status']='COMPLETE_FIRST_ELIGIBLE';emit();return
        finally:restore(row)
    state['status']='STOPPED_NO_ELIGIBLE_SCALE';emit();raise StopRound('STOPPED_NO_ELIGIBLE_SCALE')
