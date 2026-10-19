"""A/H endpoints are memory-only anchors; one new S is the only possible write."""
import math
import torch
import torch.nn.functional as F
from .editor import editable_fp32,direction,finite,nll
from .finite_null import gates,correction,native_losses
from .marginab_ops import fisher_cost,read_margin
from .bdecompose_ops import weight_guard,StopRound
from .hdecompose_ops import reconstruct_h,RATIO
from .hybridH_ops import match_once,conditions,COST,FLOOR,MAIN_MARGIN,FINAL_MARGIN
from .responseS_model import calibrate_vectors,solve_geometry,materialize_solution
from .response3_model import calibrate_vectors as calibrate_three


def evaluate(rt,record,layer,pair,factor,identity,history,base_null,start_losses,margin0,data,emit,checks,generate):
    original=layer.weight.detach().clone();w=original.float();a,b=(factor[k].to(layer.weight.device) for k in ('a','b'))
    A=history['AB']['arms']['A'];H=history['H']['hybrid'];data.update(candidate_formulated=False,candidate_written=False,candidate_evaluated=False,matching_applications=0,new_secant=0,historical_endpoint_forward_calls=0,accepted=False,local_flip=None,historical_weight_direct_comparison=None);emit()
    def near(got,expected,label,relative=False,tolerance=1e-5):
        error=abs(got/expected-1) if relative else abs(got-expected)
        checks.append(dict(field=label,**{'relative_difference' if relative else 'absolute_difference':error},tolerance=tolerance));emit()
        if not math.isfinite(error) or error>tolerance:raise StopRound('STOPPED_RECHECK_'+label)
    def same(ok,label):
        checks.append(dict(field=label,passed=bool(ok)));emit()
        if not ok:raise StopRound('STOPPED_BINDING_'+label)
    near(start_losses[0],1.5107299089431763,'S.endpoint_visual');near(start_losses[1],3.70239520072937,'S.endpoint_null');near(margin0['mean_nll'],start_losses[0],'S.endpoint_repeat')
    same(A['proposal']['native_before']==start_losses,'S.A_common_native_start')
    same(history['H']['method']['final_accepted_native_nll']==start_losses,'S.H_common_native_start')
    same(H['matching']['ratio']==RATIO,'S.H_frozen_ratio');same(H['analytic']['energy_fractions']==[.75,.25],'S.H_frozen_fraction')
    near(A['proposal']['native_update']['fisher_cost'],COST,'S.A_cost',True);near(A['proposal']['native_gain'],FLOOR,'S.A_NLL_floor');near(A['main_margin']['margin'],MAIN_MARGIN,'S.A_main_margin');near(A['final_margin']['margin'],FINAL_MARGIN,'S.A_final_margin')
    try:
        with editable_fp32(rt.model,layer,readonly=True):
            with weight_guard(layer,w,'S.base_gradient',checks,emit):
                output=rt.model(**pair[0].forward_kwargs());logits=output.logits.float();_,target,j,absolute=identity;v=logits[0,absolute]
                margin=v[target]-v[j];loss=F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),pair[0].labels[:,1:].reshape(-1),ignore_index=-100,reduction='mean')
                gm=finite(torch.autograd.grad(margin,layer.weight,retain_graph=True)[0].detach());gn=finite(torch.autograd.grad(loss,layer.weight)[0].detach());f0=float(loss.detach());m0=float(margin.detach());del output,logits,v,margin,loss
            with weight_guard(layer,w,'S.base_null_gradient',checks,emit):
                loss0=nll(rt.model,pair[1]);g0=finite(torch.autograd.grad(loss0,layer.weight)[0].detach());n0=float(loss0.detach());del loss0
            qn,geo_n=direction(gn,g0,a,b);neg_qm,geo_m=direction(-gm,g0,a,b)
            data['original_geometry']=dict(fp32_before=[f0,n0],fp32_margin=m0,NLL_geometry=geo_n,margin_geometry=geo_m);emit()
            near(f0,A['proposal']['fp32_before'][0],'S.historical_F_visual');near(n0,A['proposal']['fp32_before'][1],'S.historical_F_null');near(m0,H['geometry']['fp32_margin'],'S.historical_F_margin')
            if qn is None or neg_qm is None:raise StopRound('STOPPED_ANCHOR_DIRECTION_DEGENERATE')
            near(geo_n['E'],A['proposal']['E'],'S.historical_NLL_energy',True);near(geo_m['E'],H['analytic']['margin_energy'],'S.historical_margin_energy',True)
            near(float((gn.double()*(-neg_qm).double()).sum()),H['analytic']['cross'],'S.historical_cross',True)
            # Reconstruct only tensors. Neither anchor is installed or forwarded.
            da=finite(-A['proposal']['effective_reduction']/A['proposal']['E']*qn)
            aw=w.clone();aw.add_(da);roundedA=aw.half();qA=roundedA.float()-w
            dh=reconstruct_h(qn,-neg_qm,H['analytic']);roundedH=(w+RATIO*dh).half();qH=roundedH.float()-w
            anchor_data={}
            for name,q,ref in [('A',qA,A['proposal']['native_update']),('H',qH,dict(fisher_cost=H['matching']['actual_native_F0_cost'],visual_linear_change=H['matching']['native_linear_NLL_change'],null_linear_change=H['matching']['native_linear_null_change']))]:
                cost=fisher_cost(q,a,b);nl=float((gn.double()*q.double()).sum());nu=float((g0.double()*q.double()).sum())
                anchor_data[name]=dict(F0_cost=cost,NLL_directional=nl,null_directional=nu,margin_directional=float((gm.double()*q.double()).sum()),l2=float(torch.linalg.vector_norm(q.double())))
                data['anchor_checks']=anchor_data;emit();near(cost,ref['fisher_cost'],'S.'+name+'_cost',True);near(nl,ref['visual_linear_change'],'S.'+name+'_NLL_directional');near(nu,ref['null_linear_change'],'S.'+name+'_null_directional')
            near(anchor_data['A']['l2'],A['proposal']['quantization']['native_delta_l2'],'S.A_native_norm',True);near(anchor_data['H']['l2'],H['matching']['native_delta_l2'],'S.H_native_norm',True);near(anchor_data['H']['margin_directional'],H['matching']['native_linear_margin_gain'],'S.H_margin_directional')
            responses=[[A['main_gate']['native_nll'][0]-start_losses[0],H['main_gate']['native_nll'][0]-start_losses[0]],[A['main_margin']['margin']-margin0['margin'],H['main_margin']['margin']-margin0['margin']],[A['main_gate']['native_nll'][1]-start_losses[1],H['main_gate']['native_nll'][1]-start_losses[1]]]
            # Reconstruct the terminated Sreserve increment using its frozen ratio,
            # with no installation, new native matching, or historical forward.
            R=history['R']['response']
            same(R['candidate_evaluated'] and history['R']['status']=='STOPPED_MAIN_ACTUAL_NLL_FAILED','R3.historical_R_terminal')
            same(history['R']['method']['final_accepted_native_nll']==start_losses,'R3.common_start')
            def save_two(v):data['historical_two_calibration']=v;emit()
            bn2,bm2,b02=calibrate_vectors([gn,gm,g0],[qA,qH],responses,a,b,save_two)
            qr_n,qr_gn=direction(bn2,b02,a,b);qr_m_neg,qr_gm=direction(-bm2,b02,a,b)
            if qr_n is None or qr_m_neg is None:raise StopRound('STOPPED_R_ANCHOR_DEGENERATE')
            qr_m=-qr_m_neg
            cross2=float((bn2.double()*qr_m.double()).sum())
            analytic2=solve_geometry(qr_gn['E'],qr_gm['E'],cross2,COST,1.05*FLOOR,.4453125)
            deltaR=materialize_solution(qr_n,qr_m,analytic2)
            ratioR=R['matching']['ratio']
            roundedR=finite((w+ratioR*deltaR).half());qR=roundedR.float()-w
            near(ratioR,.9885765253507064,'R3.frozen_ratio')
            near(fisher_cost(qR,a,b),R['matching']['actual_native_F0_cost'],'R3.anchor_cost',True)
            near(float(torch.linalg.vector_norm(qR.double())),R['matching']['native_delta_l2'],'R3.anchor_norm',True)
            for vec,key in [(bn2,'NLL_change'),(bm2,'margin_gain'),(b02,'null_change')]:
                near(float((vec.double()*qR.double()).sum()),R['matching']['native_calibrated_'+key],'R3.anchor_'+key)
            data['third_anchor_reconstruction']=dict(ratio=ratioR,F0_cost=fisher_cost(qR,a,b),l2=float(torch.linalg.vector_norm(qR.double())),forward_calls=0,direct_historical_weight_comparison=None);emit()
            responses[0].append(R['main_gate']['native_nll'][0]-start_losses[0])
            responses[1].append(R['main_margin']['margin']-margin0['margin'])
            responses[2].append(R['main_gate']['native_nll'][1]-start_losses[1])
            del qr_n,qr_m,qr_m_neg,deltaR,bn2,bm2,b02
            def save_cal(v):data['calibration']=v;emit()
            bn,bm,b0=calibrate_three([gn,gm,g0],[qA,qH,qR],responses,a,b,save_cal)
            qln,caln=direction(bn,b0,a,b);neg_qlm,calm=direction(-bm,b0,a,b)
            data['calibrated_geometry']=dict(NLL=caln,margin=calm);emit()
            if qln is None or neg_qlm is None:raise StopRound('STOPPED_CALIBRATED_DIRECTION_DEGENERATE')
            qm=-neg_qlm;cross=float((bn.double()*qm.double()).sum());analytic=solve_geometry(caln['E'],calm['E'],cross,COST,1.05*FLOOR,.4453125)
            data['solver_reserve_factor']=1.05;data['actual_NLL_floor']=FLOOR;data['analytic']=analytic;emit()
            if analytic['status']!='ELIGIBLE':raise StopRound(analytic['status'])
            delta=materialize_solution(qln,qm,analytic);data['candidate_formulated']=False
            data['analytic_recheck']=dict(F0_cost=fisher_cost(delta,a,b),calibrated_NLL=float((bn.double()*delta.double()).sum()),calibrated_margin=float((bm.double()*delta.double()).sum()),normalized_null=abs(float((b0.double()*delta.double()).sum()))/math.sqrt(2*COST*caln['k00']));emit()
            near(data['analytic_recheck']['F0_cost'],COST,'S.analytic_cost',True)
            if data['analytic_recheck']['calibrated_NLL']>-1.05*FLOOR+1e-5 or data['analytic_recheck']['calibrated_margin']<=.4453125+1e-5 or data['analytic_recheck']['normalized_null']>1e-4:raise StopRound('STOPPED_ANALYTIC_CONSTRAINT_RECHECK')
            old3=history['R3']['response']
            same(history['R3']['status']=='COMPLETE' and old3['accepted'],'follow.R3_terminal')
            same(history['R3']['method']['final_accepted_native_nll']==start_losses,'follow.common_start')
            ratio3=old3['matching']['ratio']
            near(ratio3,.9856479649560461,'follow.frozen_ratio')
            rounded=finite((w+ratio3*delta).half());q3=rounded.float()-w
            for vec,key in [(bn,'NLL_change'),(bm,'margin_gain'),(b0,'null_change')]:
                near(float((vec.double()*q3.double()).sum()),old3['matching']['native_calibrated_'+key],'follow.R3_'+key)
            near(fisher_cost(q3,a,b),old3['matching']['actual_native_F0_cost'],'follow.R3_cost',True)
            near(float(torch.linalg.vector_norm(q3.double())),old3['matching']['native_delta_l2'],'follow.R3_norm',True)
            data['historical_R3_matching']=old3['matching'];data['historical_R3_forward_calls']=0;emit()
        same(layer.weight.dtype==original.dtype and torch.equal(layer.weight,original),'S.no_historical_endpoint_write')
        # One mandatory R3 endpoint control before local-gradient response refresh.
        with torch.no_grad():layer.weight.copy_(rounded)
        with weight_guard(layer,rounded,'follow.control_R3',checks,emit,expected_calls=3):
            step_start=native_losses(rt,pair)
            step_margin,_=read_margin(rt,pair[0],identity[0],identity[2])
        data['historical_R3_forward_calls']=3
        data['continuation_start']=dict(native_nll=step_start,margin=step_margin)
        for i,x in enumerate(step_start):near(x,old3['final_gate']['native_nll'][i],'follow.control_loss_'+str(i))
        near(step_margin['margin'],old3['final_margin']['margin'],'follow.control_margin')
        near(step_margin['mean_nll'],step_start[0],'follow.control_repeat')
        same(step_margin['strongest_competitor_changed']==old3['final_margin']['strongest_competitor_changed'],'follow.control_competitor')
        step_original=rounded.detach().clone();step_w=step_original.float()
        main_reference=step_margin['margin']+.4453125
        final_reference=step_margin['margin']+.4375
        failed_rounded=finite((step_original.float()+q3).half())
        q_follow=failed_rounded.float()-step_w
        old_follow=history['Follow']['response']
        same(history['Follow']['status']=='STOPPED_MAIN_ACTUAL_NLL_FAILED','refresh.failed_history_terminal')
        same(old_follow['continuation_start']['native_nll']==step_start,'refresh.failed_history_start')
        near(fisher_cost(q_follow,a,b),old_follow['continuation_preview']['actual_F0_cost'],'refresh.failed_increment_cost',True)
        # The failed endpoint is never installed or forwarded in this round.
        responses1=[[old_follow['main_gate']['native_nll'][0]-step_start[0]],
                    [old_follow['main_margin']['margin']-step_margin['margin']],
                    [old_follow['main_gate']['native_nll'][1]-step_start[1]]]
        data['historical_R3_calibration']=data.pop('calibration')
        data['historical_R3_analytic']=data.pop('analytic')
        data['historical_R3_analytic_recheck']=data.pop('analytic_recheck')
        with editable_fp32(rt.model,layer,readonly=True):
            with weight_guard(layer,step_w,'refresh.local_visual_gradient',checks,emit):
                output=rt.model(**pair[0].forward_kwargs());logits=output.logits.float();_,target,j,absolute=identity;v=logits[0,absolute]
                margin=v[target]-v[j]
                loss=F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),pair[0].labels[:,1:].reshape(-1),ignore_index=-100,reduction='mean')
                gm_local=finite(torch.autograd.grad(margin,layer.weight,retain_graph=True)[0].detach())
                gn_local=finite(torch.autograd.grad(loss,layer.weight)[0].detach())
                local_F_NLL=float(loss.detach());local_F_margin=float(margin.detach());del output,logits,v,margin,loss
            with weight_guard(layer,step_w,'refresh.local_null_gradient',checks,emit):
                loss0=nll(rt.model,pair[1]);g0_local=finite(torch.autograd.grad(loss0,layer.weight)[0].detach());local_F_null=float(loss0.detach());del loss0
            local_dot=[float((g.double()*q_follow.double()).sum()) for g in [gn_local,gm_local,g0_local]]
            old_dot=[float((g.double()*q_follow.double()).sum()) for g in [gn,gm,g0]]
            data['local_gradient_forward_calls']=2
            data['local_gradient_diagnostic']=dict(F_baseline=[local_F_NLL,local_F_null],F_margin=local_F_margin,original_endpoint_dot=old_dot,local_endpoint_dot=local_dot,observed_native_response=[x[0] for x in responses1],local_native_minus_linear=[x[0]-v for x,v in zip(responses1,local_dot)],failed_endpoint_new_forward_calls=0);emit()
            def save_local(v):data['calibration']=v;emit()
            bn_local,bm_local,b0_local=calibrate_three([gn_local,gm_local,g0_local],[q_follow],responses1,a,b,save_local)
            qln_local,geo_local_n=direction(bn_local,b0_local,a,b)
            qm_neg_local,geo_local_m=direction(-bm_local,b0_local,a,b)
            if qln_local is None or qm_neg_local is None:raise StopRound('STOPPED_REFRESH_DIRECTION_DEGENERATE')
            qm_local=-qm_neg_local;cross_local=float((bn_local.double()*qm_local.double()).sum())
            analytic_local=solve_geometry(geo_local_n['E'],geo_local_m['E'],cross_local,COST,1.05*FLOOR,.4453125)
            data['calibrated_geometry']=dict(NLL=geo_local_n,margin=geo_local_m)
            data['analytic']=analytic_local;emit()
            if analytic_local['status']!='ELIGIBLE':raise StopRound(analytic_local['status'])
            delta_local=materialize_solution(qln_local,qm_local,analytic_local)
            data['analytic_recheck']=dict(F0_cost=fisher_cost(delta_local,a,b),calibrated_NLL=float((bn_local.double()*delta_local.double()).sum()),calibrated_margin=float((bm_local.double()*delta_local.double()).sum()),normalized_null=abs(float((b0_local.double()*delta_local.double()).sum()))/math.sqrt(2*COST*geo_local_n['k00']));emit()
            near(data['analytic_recheck']['F0_cost'],COST,'refresh.analytic_cost',True)
            if data['analytic_recheck']['calibrated_NLL']>-1.05*FLOOR+1e-5 or data['analytic_recheck']['calibrated_margin']<=.4453125+1e-5 or data['analytic_recheck']['normalized_null']>1e-4:raise StopRound('STOPPED_REFRESH_ANALYTIC_RECHECK')
            data['candidate_formulated']=True;data['matching_applications']=1;emit()
            rounded,match=match_once(step_original,delta_local,COST,a,b,bn_local,bm_local,b0_local)
            match['prediction_model']='local_R3_one_observed_finite_response_not_true_gradients'
            for key in ('NLL_change','margin_gain','null_change'):match['native_calibrated_'+key]=match.pop('native_linear_'+key)
            data['matching']=match;emit()
            if not math.isfinite(match['relative_error']) or match['relative_error']>.01:raise StopRound('STOPPED_NATIVE_MATCH_FAILED')
            if match['native_calibrated_NLL_change']>-FLOOR+1e-5 or match['native_calibrated_margin_gain']<=.4453125+1e-5 or abs(step_start[1]-base_null+match['native_calibrated_null_change'])>.01:raise StopRound('STOPPED_QUANTIZED_MODEL_CONSTRAINT_FAILED')
            if torch.equal(rounded,step_original) or torch.equal(rounded,failed_rounded):raise StopRound('STOPPED_ZERO_OR_FAILED_DUPLICATE')
        same(torch.equal(layer.weight,step_original) and layer.weight.dtype==step_original.dtype,'refresh.native_control_restored')
        native_cost=match['actual_native_F0_cost']
        data['candidate_formulated']=True
        with torch.no_grad():layer.weight.copy_(rounded)
        data['candidate_written']=True;emit()
        with weight_guard(layer,rounded,'S.native_main',checks,emit,expected_calls=3):
            data['main_gate']=gates(rt,pair,factor,step_original,step_start,base_null);data['main_margin'],_=read_margin(rt,pair[0],identity[0],identity[2])
        data['candidate_evaluated']=True;emit();near(data['main_margin']['mean_nll'],data['main_gate']['native_nll'][0],'S.main_native_repeat')
        near(data['main_gate']['combined_native_fisher_cost'],native_cost,'S.native_commit_cost',True,tolerance=1e-10)
        data['main_conditions']=conditions(step_start[0]-data['main_gate']['native_nll'][0],data['main_margin']['margin'],main_reference);data['main_margin_gain']=data['main_margin']['margin']-step_margin['margin'];emit()
        if not data['main_conditions']['NLL_pass']:raise StopRound('STOPPED_MAIN_ACTUAL_NLL_FAILED')
        if not data['main_conditions']['margin_pass']:raise StopRound('STOPPED_MAIN_MARGIN_FAILED')
        if not data['main_gate']['gates']['trust']:raise StopRound('STOPPED_MAIN_TRUST_FAILED')
        data['correction']=None
        if not data['main_gate']['gates']['null']:
            data['new_secant']=1;emit();data['correction']=correction(rt,record,factor,base_null,compensate='secant');emit()
        data['final_gate']=gates(rt,pair,factor,step_original,step_start,base_null);data['final_margin'],_=read_margin(rt,pair[0],identity[0],identity[2]);emit()
        near(data['final_margin']['mean_nll'],data['final_gate']['native_nll'][0],'S.final_native_repeat')
        data['final_conditions']=conditions(step_start[0]-data['final_gate']['native_nll'][0],data['final_margin']['margin'],final_reference);data['final_margin_gain']=data['final_margin']['margin']-step_margin['margin'];data['accepted']=data['final_gate']['accepted'] and data['final_conditions']['NLL_pass'] and data['final_conditions']['margin_pass'] and (data['correction'] is None or data['correction']['status']=='APPLIED');emit()
        if not data['accepted']:raise StopRound('STOPPED_FINAL_CONDITIONS_FAILED')
        data['cumulative_F0_cost']=fisher_cost(layer.weight.detach().float()-original.float(),a,b);generate(data);data['status']='COMPLETE_ACCEPTED_REFRESH';emit()
    finally:
        with torch.no_grad():layer.weight.data=original.clone()
        data['endpoint_restore_nll']=native_losses(rt,pair);data['endpoint_restored']=layer.weight.dtype==original.dtype and torch.equal(layer.weight,original) and data['endpoint_restore_nll']==start_losses;emit()
        if not data['endpoint_restored']:raise StopRound('STOPPED_S_ENDPOINT_RESTORE_FAILED')
