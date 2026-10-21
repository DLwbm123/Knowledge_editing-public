"""Memory reconstruction and read-only endpoint derivatives; zero new candidates."""
import math
import torch
import torch.nn.functional as F
from .editor import editable_fp32,direction,finite,nll
from .finite_null import gates,correction,native_losses
from .marginab_ops import fisher_cost,read_margin
from .bdecompose_ops import weight_guard,StopRound
from .hdecompose_ops import reconstruct_h,RATIO
from .hybridH_ops import conditions,COST,FLOOR,MAIN_MARGIN,FINAL_MARGIN
from .responseS_model import calibrate_vectors,solve_geometry,materialize_solution
from .response3_model import calibrate_vectors as calibrate_three
from .responseQuad_model import admissibility
from .native_null_balance import balance
from .native_null_center import center


def native_dual_subset(base,proposal,gn,gm):
    if base.dtype!=torch.float16 or proposal.dtype!=torch.float16:raise ValueError('Native half values required')
    q=proposal.float()-base.float()
    mask=(q!=0)&(gn*q<0)&(gm*q>0)
    return torch.where(mask,proposal,base),mask


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
        data['historical_R3_calibrated_geometry']=data.pop('calibrated_geometry')
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
            N=history['N']['response'];RF=history['Refresh']['response']
            for tag,entry in [('N',history['N']),('Refresh',history['Refresh'])]:
                same(entry['status']=='STOPPED_MAIN_ACTUAL_NLL_FAILED','quad.'+tag+'_terminal')
                same(entry['response']['continuation_start']['native_nll']==step_start,'quad.'+tag+'_common_start')
            qn_local,geo_local_n=direction(gn_local,g0_local,a,b)
            if qn_local is None:raise StopRound('STOPPED_QUAD_N_ANCHOR_DEGENERATE')
            delta_n=finite(-math.sqrt(2*COST/geo_local_n['E'])*qn_local)
            roundedN=finite((step_w+N['matching']['ratio']*delta_n).half());qN=roundedN.float()-step_w
            near(N['matching']['ratio'],.9978326557816087,'quad.N_frozen_ratio')
            near(fisher_cost(qN,a,b),N['matching']['actual_native_F0_cost'],'quad.N_cost',True)
            near(float(torch.linalg.vector_norm(qN.double())),N['matching']['native_delta_l2'],'quad.N_norm',True)
            for vec,key in [(gn_local,'NLL_change'),(gm_local,'margin_gain'),(g0_local,'null_change')]:
                near(float((vec.double()*qN.double()).sum()),N['matching']['native_linear_'+key],'quad.N_'+key)
            def save_rf(v):data['historical_Refresh_fit']=v;emit()
            bn_rf,bm_rf,b0_rf=calibrate_three([gn_local,gm_local,g0_local],[q_follow],responses1,a,b,save_rf)
            qrfn,geon=direction(bn_rf,b0_rf,a,b);qrfm_neg,geom=direction(-bm_rf,b0_rf,a,b)
            if qrfn is None or qrfm_neg is None:raise StopRound('STOPPED_QUAD_REFRESH_ANCHOR_DEGENERATE')
            qrfm=-qrfm_neg;cross=float((bn_rf.double()*qrfm.double()).sum())
            ar=solve_geometry(geon['E'],geom['E'],cross,COST,1.05*FLOOR,.4453125)
            delta_rf=materialize_solution(qrfn,qrfm,ar)
            near(RF['matching']['ratio'],.9971336372809206,'quad.Refresh_frozen_ratio')
            roundedRF=finite((step_w+RF['matching']['ratio']*delta_rf).half());qRF=roundedRF.float()-step_w
            near(fisher_cost(qRF,a,b),RF['matching']['actual_native_F0_cost'],'quad.Refresh_cost',True)
            near(float(torch.linalg.vector_norm(qRF.double())),RF['matching']['native_delta_l2'],'quad.Refresh_norm',True)
            for vec,key in [(bn_rf,'NLL_change'),(bm_rf,'margin_gain'),(b0_rf,'null_change')]:
                near(float((vec.double()*qRF.double()).sum()),RF['matching']['native_calibrated_'+key],'quad.Refresh_'+key)
            vectors=[qN,qRF]
            Gram=[[float((x.double()*y.double()*b.double()[:,None]*a.double()[None,:]).sum()) for y in vectors] for x in vectors]
            G=torch.tensor(Gram,dtype=torch.float64);norm=G.diag().sqrt();eigen=torch.linalg.eigvalsh(G/norm[:,None]/norm[None,:]);condition=float(eigen[-1]/eigen[0]) if eigen[0]>0 else float('inf')
            data['quadratic_Gram']=dict(Gram=Gram,normalized_condition=condition if math.isfinite(condition) else None,eigenvalues=eigen.tolist());emit()
            if not math.isfinite(condition) or condition>1e4:raise StopRound('STOPPED_QUADRATIC_SPAN_ILL_CONDITIONED')
            linear=[[float((g.double()*q.double()).sum()) for q in vectors] for g in [gn_local,gm_local,g0_local]]
            axes=[[N['main_gate']['native_nll'][0]-step_start[0],RF['main_gate']['native_nll'][0]-step_start[0]],
                  [N['main_margin']['margin']-step_margin['margin'],RF['main_margin']['margin']-step_margin['margin']],
                  [N['main_gate']['native_nll'][1]-step_start[1],RF['main_gate']['native_nll'][1]-step_start[1]]]
            C=history['Center']['response'];transfer=history['Transfer']
            same(history['Center']['status']=='COMPLETE' and C['accepted'],'path.center_terminal')
            near(C['correction']['compensation']['multiplier'],1.5660931117171595,'path.frozen_center_alpha')
            tangent_c,geometry_c=direction(g0_local,gn_local,a,b)
            if tangent_c is None:raise StopRound('STOPPED_CENTER_RECONSTRUCTION_DEGENERATE')
            near(geometry_c['E'],C['correction']['null_correction_energy'],'path.center_energy',True)
            residual_c=step_start[1]-base_null
            near(residual_c,C['correction']['residual'],'path.center_residual')
            delta_c=finite(-residual_c/geometry_c['E']*tangent_c)
            center_rounded=finite((step_w+delta_c*C['correction']['compensation']['multiplier']).half())
            center_q=center_rounded.float()-step_w
            near(fisher_cost(center_q,a,b),C['correction']['native_update']['fisher_cost'],'path.center_cost',True)
            near(float(torch.linalg.vector_norm(center_q.double())),C['correction']['quantization']['native_delta_l2'],'path.center_norm',True)
            near(float((gn_local.double()*center_q.double()).sum()),C['correction']['native_update']['visual_linear_change'],'path.center_NLL_dot')
            near(float((g0_local.double()*center_q.double()).sum()),C['correction']['native_update']['null_linear_change'],'path.center_null_dot')
            near(fisher_cost(center_rounded.float()-original.float(),a,b),C['final_gate']['combined_native_fisher_cost'],'path.center_cumulative_cost',True)
            data['memory_center_reconstruction']=dict(frozen_alpha=C['correction']['compensation']['multiplier'],new_secant_calls=0,direct_historical_weight_comparison=None);emit()
        same(torch.equal(layer.weight,step_original) and layer.weight.dtype==step_original.dtype,'path.R3_control_restored')
        with torch.no_grad():layer.weight.copy_(center_rounded)
        with weight_guard(layer,center_rounded,'path.center_control',checks,emit,expected_calls=3):
            center_losses=native_losses(rt,pair)
            center_margin,_=read_margin(rt,pair[0],identity[0],identity[2])
        for i,x in enumerate(center_losses):near(x,C['final_gate']['native_nll'][i],'path.center_control_loss_'+str(i))
        near(center_margin['margin'],C['final_margin']['margin'],'path.center_control_margin')
        near(center_margin['mean_nll'],center_losses[0],'path.center_repeat')
        same(center_margin['strongest_competitor_changed']==C['final_margin']['strongest_competitor_changed'],'path.center_competitor')
        data['uncentered_R3_control']=data['continuation_start']
        step_original=center_rounded.clone();step_start=center_losses;step_margin=center_margin
        data['continuation_start']=dict(native_nll=step_start,margin=step_margin)
        data['center_control_forward_calls']=3
        same(transfer['status']=='ELIGIBLE_FP32_REPRESENTATION','path.offline_solver_qualified')
        same(transfer['dataset_solver_invocations']==1 and transfer['source_CPU_stop_preserved'],'path.one_solver_preserved_stop')
        same(transfer['frozen_old_model']==history['Quad']['response']['quadratic_model'],'path.frozen_model_binding')
        near(step_start[1]-base_null,transfer['center_null_bias'],'path.measured_center_bias')
        for i in range(2):
            for j in range(2):near(Gram[i][j],transfer['Gram'][i][j],'path.frozen_Gram',True)
        coefficients=transfer['materialization_FP32_coefficients'];raw_coefficients=transfer['raw_solver_coefficients']
        delta_new=finite(coefficients[0]*qN+coefficients[1]*qRF)
        raw_delta=finite(raw_coefficients[0]*qN+raw_coefficients[1]*qRF)
        same(torch.equal(delta_new,raw_delta),'path.raw_and_FP32_coefficient_delta_equal')
        target_cost=fisher_cost(delta_new,a,b)
        if not math.isfinite(target_cost) or target_cost<=0 or target_cost>COST*(1+1e-5):raise StopRound('STOPPED_PATH_TARGET_COST')
        data['memory_increment_constructed']=True;data['matching_applications']=0;data['offline_transfer']=transfer;emit()
        rounded=finite((step_original.float()+delta_new).half())
        native_delta=rounded.float()-step_original.float();native_cost=fisher_cost(native_delta,a,b)
        model_check=admissibility(transfer['frozen_old_model']['models'],coefficients,step_start[1]-base_null,1e-5,0.,Gram,COST)
        data['native_budget_validation']=dict(continuous_target_cost=target_cost,actual_native_F0_cost=native_cost,budget_C_A=COST,bound_pass=math.isfinite(native_cost) and native_cost>0 and native_cost<=COST,rounding_cost_gap_relative=native_cost/target_cost-1,not_an_equality_match=True,coefficients_unchanged=coefficients,quadratic_recheck=model_check)
        same(history['PreviousPath']['status']=='STOPPED_TRANSFER_MATCHED_PREDICTION_FAILED','native.previous_path_terminal')
        near(native_cost,history['PreviousPath']['response']['matching']['seed_native_F0_cost'],'native.previous_unwritten_seed_cost',True)
        data['native_budget_validation']['historical_seed_weight_direct_comparison']=None;emit()
        if not data['native_budget_validation']['bound_pass']:raise StopRound('STOPPED_NATIVE_BUDGET_BOUND_FAILED')
        if not all(model_check[k] for k in ['in_hull','NLL_pass','margin_pass','null_pass','cost_pass']):raise StopRound('STOPPED_UNSCALED_TRANSFER_PREDICTION_FAILED')
        if torch.equal(rounded,step_original) or torch.equal(rounded,roundedN) or torch.equal(rounded,roundedRF) or torch.equal(rounded,failed_rounded):raise StopRound('STOPPED_NATIVE_PATH_DUPLICATE')
        same(torch.equal(layer.weight,step_original) and layer.weight.dtype==step_original.dtype,'native.center_control_restored')
        old=history['Native']['response']
        same(history['Native']['status']=='COMPLETE' and old['accepted'],'probe.native_predecessor_complete')
        same(old['continuation_start']['native_nll']==step_start,'probe.native_start_binding')
        near(native_cost,old['native_budget_validation']['actual_native_F0_cost'],'probe.native_main_cost',True)
        # Reconstruct the accepted correction in memory from its frozen multiplier.
        # This is not a newly solved correction or a replay of its secant probes.
        q_main=rounded.float()-step_original.float()
        with torch.no_grad():layer.weight.copy_(rounded)
        with editable_fp32(rt.model,layer,readonly=True):
            with weight_guard(layer,rounded.float(),'probe.main_visual_gradient',checks,emit):
                lv=nll(rt.model,pair[0]);g_main_n=finite(torch.autograd.grad(lv,layer.weight)[0].detach());main_f_n=float(lv.detach());del lv
            with weight_guard(layer,rounded.float(),'probe.main_null_gradient',checks,emit):
                ln=nll(rt.model,pair[1]);g_main_0=finite(torch.autograd.grad(ln,layer.weight)[0].detach());main_f_0=float(ln.detach());del ln
            correction_old=old['correction']
            near(main_f_n,correction_old['fp32_before'][0],'probe.main_F_NLL')
            near(main_f_0,correction_old['fp32_before'][1],'probe.main_F_null')
            qc,gc=direction(g_main_0,g_main_n,a,b)
            if qc is None:raise StopRound('STOPPED_PROBE_CORRECTION_DEGENERATE')
            near(gc['E'],correction_old['null_correction_energy'],'probe.correction_energy',True)
            residual=correction_old['residual'];alpha=correction_old['compensation']['multiplier']
            near(alpha,1.4634371306169494,'probe.frozen_native_alpha')
            dc=finite(-residual/gc['E']*qc)
            final_rounded=finite((rounded.float()+alpha*dc).half());q_c=final_rounded.float()-rounded.float()
            near(fisher_cost(q_c,a,b),correction_old['native_update']['fisher_cost'],'probe.correction_cost',True)
            near(float(torch.linalg.vector_norm(q_c.double())),correction_old['quantization']['native_delta_l2'],'probe.correction_norm',True)
            near(float((g_main_n.double()*q_c.double()).sum()),correction_old['native_update']['visual_linear_change'],'probe.correction_NLL_dot')
            near(float((g_main_0.double()*q_c.double()).sum()),correction_old['native_update']['null_linear_change'],'probe.correction_null_dot')
        same(torch.equal(layer.weight,rounded),'probe.main_gradient_weight_restored')
        with torch.no_grad():layer.weight.copy_(final_rounded)
        with weight_guard(layer,final_rounded,'probe.accepted_endpoint_control',checks,emit,expected_calls=3):
            endpoint_losses=native_losses(rt,pair)
            endpoint_margin,_=read_margin(rt,pair[0],identity[0],identity[2])
        for i,x in enumerate(endpoint_losses):near(x,old['final_gate']['native_nll'][i],'probe.accepted_control_loss_'+str(i))
        near(endpoint_margin['margin'],old['final_margin']['margin'],'probe.accepted_control_margin')
        near(endpoint_margin['mean_nll'],endpoint_losses[0],'probe.accepted_control_repeat')
        same(endpoint_margin['strongest_competitor_changed']==old['final_margin']['strongest_competitor_changed'],'probe.accepted_competitor')
        data['memory_native_reconstruction']=dict(frozen_alpha=alpha,new_secant_calls=0,historical_direct_weight_comparison=None,control_forward_calls=3,gradient_forward_calls=2)
        data['accepted_endpoint_control']=dict(native_nll=endpoint_losses,margin=endpoint_margin)
        near(fisher_cost(final_rounded.float()-original.float(),a,b),old['cumulative_F0_cost'],'probe.full_path_cost',True)
        previous=history['Nullstep']['response']
        same(history['Nullstep']['status']=='COMPLETE' and previous['accepted'],'small.previous_complete')
        same(previous['continuation_start']['native_nll']==endpoint_losses,'small.previous_start')
        frozen_scale=history['NullstepScale']['scale']
        near(frozen_scale,.19166364845040093,'small.frozen_previous_scale')
        small_rounded=finite((final_rounded.float()+frozen_scale*q_main).half())
        small_q=small_rounded.float()-final_rounded.float()
        cap=previous['nullstep_eligibility']['native_F0_cost']
        near(cap,2.4043746118789385e-15,'small.frozen_cost_cap',True)
        near(fisher_cost(small_q,a,b),cap,'small.reconstructed_previous_cost',True)
        same(previous['correction'] is None and previous['new_secant']==0,'small.previous_no_correction')
        with torch.no_grad():layer.weight.copy_(small_rounded)
        with weight_guard(layer,small_rounded,'small.previous_endpoint_control',checks,emit,expected_calls=3):
            small_losses=native_losses(rt,pair);small_margin,_=read_margin(rt,pair[0],identity[0],identity[2])
        for i,x in enumerate(small_losses):near(x,previous['final_gate']['native_nll'][i],'small.previous_loss_'+str(i))
        near(small_margin['margin'],previous['final_margin']['margin'],'small.previous_margin')
        near(small_margin['mean_nll'],small_losses[0],'small.previous_repeat')
        same(small_margin['strongest_competitor_changed']==previous['final_margin']['strongest_competitor_changed'],'small.previous_competitor')
        data['previous_Native_start']=data.pop('continuation_start')
        data['continuation_start']=dict(native_nll=small_losses,margin=small_margin)
        data['small_endpoint_control']=dict(control_forward_calls=3,frozen_previous_scale=frozen_scale,historical_weight_direct_comparison=None)
        with editable_fp32(rt.model,layer,readonly=True):
            with weight_guard(layer,small_rounded.float(),'small.local_visual_gradient',checks,emit):
                output=rt.model(**pair[0].forward_kwargs());logits=output.logits.float();_,target,j,absolute=identity;v=logits[0,absolute]
                lm=v[target]-v[j]
                lv=F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),pair[0].labels[:,1:].reshape(-1),ignore_index=-100,reduction='mean')
                gm_s=finite(torch.autograd.grad(lm,layer.weight,retain_graph=True)[0].detach());gn_s=finite(torch.autograd.grad(lv,layer.weight)[0].detach())
                f_n=float(lv.detach());f_m=float(lm.detach());del output,logits,v,lm,lv
            with weight_guard(layer,small_rounded.float(),'small.local_null_gradient',checks,emit):
                ln=nll(rt.model,pair[1]);g0_s=finite(torch.autograd.grad(ln,layer.weight)[0].detach());f_0=float(ln.detach());del ln
            data['new_local_geometry']=dict(F_baseline=[f_n,f_0],F_margin=f_m,gradient_forward_calls=2,chain='editable_layer_FP32_not_full_chain_FP32');emit()
            qs_n,geo_s_n=direction(gn_s,g0_s,a,b);qs_m_neg,geo_s_m=direction(-gm_s,g0_s,a,b)
            if qs_n is None or qs_m_neg is None:raise StopRound('STOPPED_SMALL_LOCAL_DIRECTION_DEGENERATE')
            qs_m=-qs_m_neg
            cross=float((gn_s.double()*qs_m.double()).sum())
            old_small=history['SmallDirection']['response']
            same(history['SmallDirection']['status']=='STOPPED_SMALL_NATIVE_ADMISSION_FAILED','orthant.old_terminal')
            same(not old_small['candidate_written'] and not old_small['candidate_evaluated'],'orthant.old_not_evaluated')
            same(history['QuantReserve']['status']=='STOPPED_SMALL_NATIVE_ADMISSION_FAILED','orthant.reserve_terminal')
            same(old_small['continuation_start']['native_nll']==small_losses,'orthant.same_old_start')
            old_analytic=old_small['small_analytic']
            near(geo_s_n['E'],old_analytic['NLL_energy'],'orthant.bound_NLL_energy',True)
            near(geo_s_m['E'],old_analytic['margin_energy'],'orthant.bound_margin_energy',True)
            near(cross,old_analytic['cross'],'orthant.bound_cross',True)
            delta_old=materialize_solution(qs_n,qs_m,old_analytic)
            old_proposed=finite((small_rounded.float()+delta_old).half())
            old_native=old_proposed.float()-small_rounded.float()
            old_cost=fisher_cost(old_native,a,b)
            old_dot=[float((g.double()*old_native.double()).sum()) for g in [gn_s,gm_s,g0_s]]
            near(old_cost,old_small['small_native_admission']['actual_native_F0_cost'],'orthant.old_native_cost',True)
            for j,x in enumerate(old_dot):near(x,old_small['small_native_admission']['native_linear_change'][j],'orthant.old_native_dot_'+str(j))
            previous_orthant=history['NativeOrthant']['response']
            same(history['NativeOrthant']['status']=='COMPLETE' and previous_orthant['accepted'],'radius.small_orthant_complete')
            same(previous_orthant['continuation_start']['native_nll']==small_losses,'radius.same_start')
            small_subset,_=native_dual_subset(small_rounded,old_proposed,gn_s,gm_s)
            near(fisher_cost(small_subset.float()-small_rounded.float(),a,b),previous_orthant['small_native_admission']['actual_native_F0_cost'],'radius.small_subset_cost',True)
            reference_scale=math.sqrt(COST/cap)
            near(reference_scale,history['RadiusScale']['scale'],'radius.frozen_reference_scale')
            delta_large=finite(reference_scale*delta_old)
            reference_continuous_cost=fisher_cost(delta_large,a,b)
            near(reference_continuous_cost,COST,'radius.continuous_reference_cost',True)
            data['historical_radius']=dict(original_small_cap=cap,historical_reference_cap=COST,scale=reference_scale,new_scale_choices=1,scale_search=False,small_subset_forward_calls=0,reference_continuous_cost=reference_continuous_cost,equality_cost_matching=False)
            old_proposed=finite((small_rounded.float()+delta_large).half())
            old_native=old_proposed.float()-small_rounded.float();old_cost=fisher_cost(old_native,a,b)
            old_dot=[float((g.double()*old_native.double()).sum()) for g in [gn_s,gm_s,g0_s]]
            cap=COST
            raw_dual,raw_keep=native_dual_subset(small_rounded,old_proposed,gn_s,gm_s)
            radius_old=history['OrthantRadius']['response']
            same(history['OrthantRadius']['status']=='STOPPED_ORTHANT_NATIVE_ADMISSION_FAILED','balance.radius_terminal')
            same(not radius_old['candidate_written'] and not radius_old['candidate_evaluated'],'balance.radius_not_evaluated')
            same(radius_old['continuation_start']['native_nll']==small_losses,'balance.same_start')
            raw_q=raw_dual.float()-small_rounded.float();raw_cost=fisher_cost(raw_q,a,b)
            raw_dot=[float((g.double()*raw_q.double()).sum()) for g in [gn_s,gm_s,g0_s]]
            near(raw_cost,radius_old['small_native_admission']['actual_native_F0_cost'],'balance.raw_cost',True)
            for j,x in enumerate(raw_dot):near(x,radius_old['small_native_admission']['native_linear_change'][j],'balance.raw_dot_'+str(j))
            same(int(raw_keep.sum())==radius_old['native_subset']['selected_elements'],'balance.raw_selected_count')
            proposed,balance_info=balance(small_rounded,raw_dual,gm_s,g0_s,small_losses[1]-base_null)
            data['native_null_balance']=balance_info;data['radius_memory_reference']=dict(native_F0_cost=raw_cost,native_linear_change=raw_dot,forward_calls=0);emit()
            if proposed is None:raise StopRound(balance_info['status'])
            data['memory_balance_reconstructed']=True;emit()
            if torch.equal(proposed,small_rounded) or torch.equal(proposed,raw_dual) or torch.equal(proposed,small_subset):raise StopRound('STOPPED_BALANCE_EMPTY_OR_DUPLICATE')
            retained=(proposed!=small_rounded)
            same(torch.equal(proposed[retained],raw_dual[retained]) and torch.equal(proposed[~retained],small_rounded[~retained]),'balance.native_value_subset_identity')
            native_s=proposed.float()-small_rounded.float();native_cost_s=fisher_cost(native_s,a,b)
            predicted=[float((g.double()*native_s.double()).sum()) for g in [gn_s,gm_s,g0_s]]
            same(bool(torch.all(gn_s*native_s<=0)) and bool(torch.all(gm_s*native_s>=0)),'balance.coordinate_sign_contract')
            same(native_cost_s<=raw_cost*(1+1e-10),'balance.diagonal_F0_subset_bound')
            predicted_bias=small_losses[1]-base_null+predicted[2]
            balance_info.update(retained_elements=int(retained.sum()),final_predicted_base_null=predicted_bias,final_native_linear_change=predicted,final_native_F0_cost=native_cost_s,null_tangent_projection_preserved=False)
            same(abs(predicted_bias)<=balance_info['max_removed_null_quantum']+1e-10,'balance.discrete_centering_quantum_bound')
            checks_small=dict(actual_native_F0_cost=native_cost_s,cap=cap,not_equality_matching=True,native_linear_change=predicted,predicted_original_base_null=predicted_bias,cost_pass=math.isfinite(native_cost_s) and 0<native_cost_s<=cap,NLL_pass=predicted[0]<-1e-5,margin_pass=predicted[1]>1e-5,null_pass=abs(predicted_bias)<=.01)
            data['small_native_admission']=checks_small;emit()
            if not all(checks_small[k] for k in ['cost_pass','NLL_pass','margin_pass','null_pass']):raise StopRound('STOPPED_BALANCE_NATIVE_ADMISSION_FAILED')
        same(torch.equal(layer.weight,small_rounded),'refresh_balance.old_control_restored')
        old_balance=history['Balance']['response']
        same(history['Balance']['status']=='COMPLETE' and old_balance['accepted'],'refresh_balance.predecessor_complete')
        same(old_balance['correction'] is None and old_balance['new_secant']==0,'refresh_balance.previous_no_new_secant')
        near(native_cost_s,old_balance['small_native_admission']['actual_native_F0_cost'],'refresh_balance.previous_cost',True)
        for j,x in enumerate(predicted):near(x,old_balance['small_native_admission']['native_linear_change'][j],'refresh_balance.previous_dot_'+str(j))
        same(balance_info['dropped_elements']==old_balance['native_null_balance']['dropped_elements'],'refresh_balance.previous_drop_count')
        near(fisher_cost(proposed.float()-original.float(),a,b),old_balance['cumulative_F0_cost'],'refresh_balance.previous_cumulative_cost',True)
        refreshed_base=proposed.clone()
        with torch.no_grad():layer.weight.copy_(refreshed_base)
        with weight_guard(layer,refreshed_base,'refresh_balance.new_endpoint_control',checks,emit,expected_calls=3):
            refreshed_losses=native_losses(rt,pair);refreshed_margin,_=read_margin(rt,pair[0],identity[0],identity[2])
        for j,x in enumerate(refreshed_losses):near(x,old_balance['final_gate']['native_nll'][j],'refresh_balance.endpoint_loss_'+str(j))
        near(refreshed_margin['margin'],old_balance['final_margin']['margin'],'refresh_balance.endpoint_margin')
        near(refreshed_margin['mean_nll'],refreshed_losses[0],'refresh_balance.endpoint_repeat')
        same(refreshed_margin['strongest_competitor_changed']==old_balance['final_margin']['strongest_competitor_changed'],'refresh_balance.endpoint_competitor')
        data['memory_balance_reconstruction']=dict(frozen_scale=reference_scale,control_forward_calls=3,new_historical_secant_calls=0,historical_weight_direct_comparison=None)
        data['previous_Nullstep_start']=data.pop('continuation_start')
        data['historical_balance_admission']=data.pop('small_native_admission')
        data['historical_native_null_balance']=data.pop('native_null_balance')
        data['continuation_start']=dict(native_nll=refreshed_losses,margin=refreshed_margin)
        small_rounded=refreshed_base;small_losses=refreshed_losses;small_margin=refreshed_margin
        with editable_fp32(rt.model,layer,readonly=True):
            with weight_guard(layer,small_rounded.float(),'refresh_balance.local_visual_gradient',checks,emit):
                output=rt.model(**pair[0].forward_kwargs());logits=output.logits.float();_,target,j,absolute=identity;v=logits[0,absolute]
                lm=v[target]-v[j]
                lv=F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),pair[0].labels[:,1:].reshape(-1),ignore_index=-100,reduction='mean')
                gm_f=finite(torch.autograd.grad(lm,layer.weight,retain_graph=True)[0].detach());gn_f=finite(torch.autograd.grad(lv,layer.weight)[0].detach())
                f_n=float(lv.detach());f_m=float(lm.detach());del output,logits,v,lm,lv
            with weight_guard(layer,small_rounded.float(),'refresh_balance.local_null_gradient',checks,emit):
                ln=nll(rt.model,pair[1]);g0_f=finite(torch.autograd.grad(ln,layer.weight)[0].detach());f_0=float(ln.detach());del ln
            data['refreshed_local_gradient']=dict(F_baseline=[f_n,f_0],F_margin=f_m,forward_calls=2,chain='editable_layer_FP32_not_full_chain_FP32');emit()
            qf_n,geo_f_n=direction(gn_f,g0_f,a,b);qf_m_neg,geo_f_m=direction(-gm_f,g0_f,a,b)
            if qf_n is None or qf_m_neg is None:raise StopRound('STOPPED_REFRESH_BALANCE_DIRECTION_DEGENERATE')
            qf_m=-qf_m_neg;cross_f=float((gn_f.double()*qf_m.double()).sum())
            seed_cap=2.4043746118789385e-15
            refresh_old=history['BalancedRefresh']['response']
            same(history['BalancedRefresh']['status']=='STOPPED_BALANCE_NO_NEGATIVE_EXCURSION','signed.previous_one_sided_stop')
            same(not refresh_old['candidate_formulated'] and not refresh_old['candidate_written'] and not refresh_old['candidate_evaluated'],'signed.previous_no_final_candidate')
            same(refresh_old['continuation_start']['native_nll']==small_losses,'signed.same_frozen_start')
            analytic_f=refresh_old['refreshed_seed_analytic']
            near(geo_f_n['E'],analytic_f['NLL_energy'],'signed.frozen_NLL_energy',True)
            near(geo_f_m['E'],analytic_f['margin_energy'],'signed.frozen_margin_energy',True)
            near(cross_f,analytic_f['cross'],'signed.frozen_cross',True)
            data['new_refreshed_seed_solves']=0;emit()
            data['refreshed_seed_analytic']=analytic_f;data['refreshed_geometry']=dict(NLL=geo_f_n,margin=geo_f_m,cross=cross_f);emit()
            if analytic_f['status']!='ELIGIBLE':raise StopRound('STOPPED_REFRESH_BALANCE_ANALYTIC_'+analytic_f['status'])
            delta_f=finite(reference_scale*materialize_solution(qf_n,qf_m,analytic_f))
            near(fisher_cost(delta_f,a,b),COST,'refresh_balance.continuous_reference_cost',True)
            raw_f=finite((small_rounded.float()+delta_f).half())
            dual_f,keep_f=native_dual_subset(small_rounded,raw_f,gn_f,gm_f)
            qdual_f=dual_f.float()-small_rounded.float();dual_cost=fisher_cost(qdual_f,a,b)
            dual_dot=[float((g.double()*qdual_f.double()).sum()) for g in [gn_f,gm_f,g0_f]]
            data['refreshed_native_dual']=dict(retained_elements=int(keep_f.sum()),native_F0_cost=dual_cost,native_linear_change=dual_dot,frozen_scale=reference_scale)
            emit()
            if not math.isfinite(dual_cost) or dual_cost<=0 or dual_cost>COST:raise StopRound('STOPPED_REFRESH_BALANCE_DUAL_COST')
            near(dual_cost,refresh_old['refreshed_native_dual']['native_F0_cost'],'signed.frozen_dual_cost',True)
            same(int(keep_f.sum())==refresh_old['refreshed_native_dual']['retained_elements'],'signed.frozen_dual_count')
            for j,x in enumerate(dual_dot):near(x,refresh_old['refreshed_native_dual']['native_linear_change'][j],'signed.frozen_dual_dot_'+str(j))
            current_bias=small_losses[1]-base_null+dual_dot[2]
            near(current_bias,refresh_old['native_null_balance']['initial_predicted_bias'],'signed.frozen_positive_excursion')
            same(current_bias>0,'signed.only_frozen_positive_side')
            proposed,new_balance_info=center(small_rounded,dual_f,gm_f,g0_f,small_losses[1]-base_null)
            data['native_null_balance']=new_balance_info;emit()
            if proposed is None:raise StopRound(new_balance_info['status'])
            data['candidate_formulated']=True;emit()
            if torch.equal(proposed,small_rounded) or torch.equal(proposed,dual_f):raise StopRound('STOPPED_REFRESH_BALANCE_EMPTY_OR_DUPLICATE')
            retained=(proposed!=small_rounded)
            same(torch.equal(proposed[retained],dual_f[retained]) and torch.equal(proposed[~retained],small_rounded[~retained]),'refresh_balance.native_values_identity')
            native_s=proposed.float()-small_rounded.float();native_cost_s=fisher_cost(native_s,a,b)
            predicted=[float((g.double()*native_s.double()).sum()) for g in [gn_f,gm_f,g0_f]]
            same(bool(torch.all(gn_f*native_s<=0)) and bool(torch.all(gm_f*native_s>=0)),'refresh_balance.coordinate_sign_contract')
            same(native_cost_s<=dual_cost*(1+1e-10),'refresh_balance.diagonal_F0_subset_bound')
            predicted_bias=small_losses[1]-base_null+predicted[2]
            new_balance_info.update(retained_elements=int(retained.sum()),final_predicted_base_null=predicted_bias,final_native_linear_change=predicted,final_native_F0_cost=native_cost_s,null_tangent_projection_preserved=False)
            same(abs(predicted_bias)<=new_balance_info['max_removed_null_quantum']+1e-10,'refresh_balance.centering_quantum_bound')
            cap=COST
            checks_small=dict(actual_native_F0_cost=native_cost_s,cap=cap,not_equality_matching=True,native_linear_change=predicted,predicted_original_base_null=predicted_bias,cost_pass=math.isfinite(native_cost_s) and 0<native_cost_s<=cap,NLL_pass=predicted[0]<-1e-5,margin_pass=predicted[1]>1e-5,null_pass=abs(predicted_bias)<=.01)
            data['small_native_admission']=checks_small;emit()
            if not all(checks_small[k] for k in ['cost_pass','NLL_pass','margin_pass','null_pass']):raise StopRound('STOPPED_REFRESH_BALANCE_NATIVE_ADMISSION_FAILED')
        same(torch.equal(layer.weight,small_rounded) and layer.weight.dtype==original.dtype,'small.gradient_weight_restored')
        step_original=small_rounded;step_start=small_losses;step_margin=small_margin
        def progress(losses,margin):
            gain=step_start[0]-losses[0];mg=margin-step_margin['margin']
            return dict(actual_NLL_gain=gain,NLL_floor=1e-5,NLL_pass=gain>1e-5,margin=margin,margin_reference=step_margin['margin']+1e-5,margin_gain=mg,margin_pass=mg>1e-5,criterion='new_monotone_path_not_old_fixed_per_step_A_comparison')
        with torch.no_grad():layer.weight.copy_(proposed)
        data['candidate_written']=True;emit()
        with weight_guard(layer,proposed,'small.native_main',checks,emit,expected_calls=3):
            data['main_gate']=gates(rt,pair,factor,step_original,step_start,base_null);data['main_margin'],_=read_margin(rt,pair[0],identity[0],identity[2])
        data['candidate_evaluated']=True;emit()
        near(data['main_margin']['mean_nll'],data['main_gate']['native_nll'][0],'small.main_repeat')
        near(data['main_gate']['combined_native_fisher_cost'],native_cost_s,'small.committed_cost',True,tolerance=1e-10)
        data['main_conditions']=progress(data['main_gate']['native_nll'],data['main_margin']['margin']);data['main_margin_gain']=data['main_conditions']['margin_gain'];emit()
        if not data['main_conditions']['NLL_pass']:raise StopRound('STOPPED_MAIN_ACTUAL_NLL_FAILED')
        if not data['main_conditions']['margin_pass']:raise StopRound('STOPPED_MAIN_MARGIN_FAILED')
        if not data['main_gate']['gates']['trust']:raise StopRound('STOPPED_MAIN_TRUST_FAILED')
        data['correction']=None
        if not data['main_gate']['gates']['null']:
            data['new_secant']=1;emit();data['correction']=correction(rt,record,factor,base_null,compensate='secant');emit()
        data['final_gate']=gates(rt,pair,factor,step_original,step_start,base_null);data['final_margin'],_=read_margin(rt,pair[0],identity[0],identity[2]);emit()
        near(data['final_margin']['mean_nll'],data['final_gate']['native_nll'][0],'small.final_repeat')
        data['final_conditions']=progress(data['final_gate']['native_nll'],data['final_margin']['margin']);data['final_margin_gain']=data['final_conditions']['margin_gain'];data['overall_conditions']=conditions(start_losses[0]-data['final_gate']['native_nll'][0],data['final_margin']['margin'],FINAL_MARGIN)
        data['accepted']=data['overall_conditions']['NLL_pass'] and data['overall_conditions']['margin_pass'] and data['final_gate']['accepted'] and data['final_conditions']['NLL_pass'] and data['final_conditions']['margin_pass'] and (data['correction'] is None or data['correction']['status']=='APPLIED');emit()
        if not data['accepted']:raise StopRound('STOPPED_FINAL_CONDITIONS_FAILED')
        data['cumulative_F0_cost']=fisher_cost(layer.weight.detach().float()-original.float(),a,b);generate(data);data['status']='COMPLETE_ACCEPTED_SIGNED_NATIVE_BALANCE';emit()
    finally:
        with torch.no_grad():layer.weight.data=original.clone()
        data['endpoint_restore_nll']=native_losses(rt,pair);data['endpoint_restored']=layer.weight.dtype==original.dtype and torch.equal(layer.weight,original) and data['endpoint_restore_nll']==start_losses;emit()
        if not data['endpoint_restored']:raise StopRound('STOPPED_PROBE_ENDPOINT_RESTORE_FAILED')
