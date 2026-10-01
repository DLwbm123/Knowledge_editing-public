"""Bounded original-weight mechanical qualification, never a scientific edit run."""
from __future__ import annotations
import gc,json,os,resource,signal,subprocess,sys,time,traceback
from dataclasses import replace
from pathlib import Path
import torch
from .gate import require_external_approval
from .contracts import digest
from .editor import MatrixRuntime,mean_answer_logprob
from .numerics import FrozenGGN,cg,solve_step,protection_kl,protection_gradient
from .native_io import load,prepare,WEIGHT

FILES=['NATIVE_BINDINGS','NATIVE_TEMPLATE_AUDIT','NATIVE_DERIVATIVE_AUDIT','NATIVE_PRECISION_AUDIT',
       'NATIVE_GGN_AUDIT','NATIVE_QP_AUDIT','NATIVE_WRITE_ROLLBACK_AUDIT','NATIVE_CLEAN_RELOAD_AUDIT',
       'NATIVE_RESOURCE_PROFILE','PILOT_COST_PROJECTION']


def run(config,approval,trusted):
    require_external_approval('NATIVE_SMOKE',config['bindings'],approval,config,trusted_authorization=trusted)
    if config['protocol'].get('editor_function')!='FP32_FULL_FUNCTIONAL':
        raise PermissionError('explicit FP32 functional precision contract required')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    root=Path(config['run_root']);root.mkdir(parents=True,exist_ok=True)
    records={name:dict(status='NOT_RUN',result_kind='MECHANICAL_NATIVE_VALIDATION') for name in FILES}
    profile=[];ggn_calls=0;rt=None;started=time.monotonic()
    def flush():
        for name,value in records.items():(root/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    def timeout(*_):raise TimeoutError('authorized native wall budget exhausted')
    torch.manual_seed(20260930)
    torch.use_deterministic_algorithms(True)
    signal.signal(signal.SIGALRM,timeout);signal.alarm(config['native_wall_seconds_cap']-30)
    def check():
        if time.monotonic()-started>=config['native_wall_seconds_cap']:raise TimeoutError('native wall budget exhausted')
    def bench(name,fn):
        check();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated();calls=ggn_calls;t=time.monotonic()
        try:return fn()
        finally:
            torch.cuda.synchronize()
            profile.append(dict(operation=name,wall_seconds=time.monotonic()-t,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                activation_memory_increment_bytes=max(0,torch.cuda.max_memory_allocated()-before),
                host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                ggn_calls=ggn_calls-calls))
    def operator(curvature,point):
        def apply(v):
            nonlocal ggn_calls
            check()
            if ggn_calls>=config['maximum_GGN_calls']:raise TimeoutError('authorized exact GGN call cap exhausted')
            ggn_calls+=1
            return curvature(v.reshape(point.shape)).reshape(v.shape)+config['damping']*v
        return apply
    def grad(fn,point):
        variable=point.detach().requires_grad_(True)
        return torch.autograd.grad(fn(variable),variable)[0].detach()
    def finite(t):return bool(torch.isfinite(t).all())
    try:
        flush()
        model,tokenizer,processor=bench('clean_model_load',lambda:load(config))
        selected=dict(model.named_parameters(remove_duplicate=False)).get(WEIGHT)
        if selected is None or list(selected.shape)!=[4096,14336] or selected.dtype!=torch.bfloat16:
            raise RuntimeError('exact native matrix path/shape/BF16 binding mismatch')
        rt=MatrixRuntime(model,WEIGHT)
        aliases=[k for k,v in model.state_dict().items() if v.untyped_storage().data_ptr()==selected.untyped_storage().data_ptr()]
        if aliases!=[WEIGHT]:raise RuntimeError('native shared storage mismatch')
        hooks=rt.hooks();snapshot=rt.base_state
        records['NATIVE_BINDINGS'].update(status='RUNTIME_BOUND',static_bindings=config['static_bindings'],
            runtime=dict(actual_parameter_count=sum(p.numel() for p in model.parameters()),state_keys=list(model.state_dict()),
                resolved_path=WEIGHT,shape=list(selected.shape),weight_dtype=str(selected.dtype),storage_aliases=aliases,
                selected_storage_pointer=selected.untyped_storage().data_ptr(),selected_bias=model.get_submodule(WEIGHT.rsplit('.',1)[0]).bias is not None,
                GPU_UUID=config['leased_gpu_uuids'][0],torch=torch.__version__,CUDA=torch.version.cuda,
                transformers=__import__('transformers').__version__,attention_backend=model.config._attn_implementation,
                deployment_dtype=str(selected.dtype),compute_dtype=str(next(model.parameters()).dtype),
                generation_config=model.generation_config.to_dict(),tokenizer_class=type(tokenizer).__name__,processor_class=type(processor).__name__))
        flush()
        inputs,labels,generation,template=bench('template_image_expansion',lambda:prepare(model,tokenizer,processor,config['smoke_row']))
        (root/'TEMPLATE.private.json').write_text(json.dumps(template,indent=2)+'\n')
        normal=lambda:rt.normal_logits(inputs).detach().float()
        with torch.no_grad():
            z=bench('Base_forward',normal);repeat=bench('deterministic_repeat_forward',normal)
            fz=bench('functional_call_parity',lambda:rt.logits(rt.base,inputs).detach().float())
        repeat_difference=float((z-repeat).abs().max());functional_difference=float((z-fz).abs().max())
        target=labels[:,1:];mask=(target!=-100)&(target!=tokenizer.eos_token_id)
        predictors=mask.nonzero()[:,1]
        effective=torch.cat((mask,torch.zeros(mask.shape[0],1,dtype=torch.bool,device=mask.device)),1)
        if not mask.any() or not bool(inputs['attention_mask'][effective].all()):raise RuntimeError('invalid expanded answer/padding alignment')
        score=lambda logits:mean_answer_logprob(logits.float(),labels,eos_id=tokenizer.eos_token_id).mean()
        independent=z[0,predictors].log_softmax(-1).gather(-1,target[0,predictors,None]).mean()
        alignment_error=abs(float(score(z)-independent))
        keys=bench('W_KEY_physical_input',lambda:rt.capture_input(inputs))
        if keys.shape[-1]!=14336 or keys.shape[:2]!=labels.shape or rt.hooks()!=hooks:raise RuntimeError('physical down_proj key geometry mismatch')
        records['NATIVE_TEMPLATE_AUDIT'].update(status='PASS' if repeat_difference==0 and functional_difference<=1e-3 and alignment_error<=1e-6 else 'FAIL',
            input_sequence_length=template['input_length'],expanded_sequence_length=template['expanded_length'],
            predictor_positions=predictors.cpu().tolist(),padding_valid=True,eos_policy=template['eos_policy'],vocabulary_size=z.shape[-1],
            logits_dtype=str(model(**inputs).logits.dtype),selected_layer_input_dtype=str(keys.dtype),selected_weight_dtype=str(selected.dtype),
            repeat_max_difference=repeat_difference,functional_max_difference=functional_difference,answer_alignment_error=alignment_error,
            W_KEY_geometry=dict(shape=list(keys.shape),dimension=14336,module_calls=1,hook_removed=rt.hooks()==hooks,aligned_predictor_positions=True),
            template_digest=digest(template),answer_score_mechanical_only=float(score(z)))
        if records['NATIVE_TEMPLATE_AUDIT']['status']!='PASS':raise RuntimeError('native template/forward parity qualification failed')
        del repeat,fz,keys;flush()
        raw=rt.bind_inputs(inputs)
        bfpoint=rt.base.clone()
        bf_logits=lambda w:raw(w)[:,predictors,:].float()
        bf_score=lambda w:score(raw(w))
        answer_grad=bench('answer_score_backward',lambda:grad(bf_score,bfpoint))
        if not finite(answer_grad) or float(answer_grad.norm())==0:raise RuntimeError('native answer gradient invalid')
        anchor=bf_logits(bfpoint).softmax(-1).detach();weights=torch.full(anchor.shape[:-1],1/len(predictors),device=anchor.device,dtype=torch.float32)
        bf_loss=lambda w:protection_kl(bf_logits(w),anchor,weights)
        base_grad=bench('protection_KL_backward_Base',lambda:grad(bf_loss,bfpoint))
        index=int(answer_grad.abs().reshape(-1).argmax());epsilon=max(float(bfpoint.reshape(-1)[index].abs())/16,1e-3)
        perturb=bfpoint.clone();perturb.reshape(-1)[index]+=epsilon
        perturb_grad=bench('protection_KL_backward_perturbed',lambda:grad(bf_loss,perturb))
        no_others=all(p.grad is None for p in model.parameters())
        derivative_ok=finite(base_grad) and finite(perturb_grad) and float(base_grad.float().norm())<=1e-5*max(1,float(answer_grad.float().norm())) and float(perturb_grad.float().norm())>0 and no_others
        records['NATIVE_DERIVATIVE_AUDIT'].update(status='PASS' if derivative_ok else 'FAIL',answer_gradient_dtype=str(answer_grad.dtype),
            answer_gradient_norm=float(answer_grad.float().norm()),answer_gradient_finite=finite(answer_grad),answer_gradient_shape=list(answer_grad.shape),
            preservation_Base_gradient_norm=float(base_grad.float().norm()),preservation_perturbed_gradient_norm=float(perturb_grad.float().norm()),
            perturb_delta_norm=float((perturb.float()-bfpoint.float()).norm()),fixed_detached_anchor=True,no_other_parameter_gradients=no_others)
        flush()
        if not derivative_ok:raise RuntimeError('fixed-anchor native derivative qualification failed')
        # The separately approved FP32 function is not the rounded BF16 map.
        deployment_anchor=anchor
        raw32=rt.bind_inputs(inputs,arithmetic_dtype=torch.float32)
        point=bfpoint.float()
        logits=lambda w:raw32(w)[:,predictors,:].float()
        answer=lambda w:score(raw32(w))
        with torch.no_grad():fp32_z=raw32(point).float()
        cross_difference=float((fp32_z-z).abs().max())
        anchor=logits(point).softmax(-1).detach()
        loss=lambda w:protection_kl(logits(w),anchor,weights)
        a=bench('solver_coordinate_answer_backward',lambda:grad(answer,point))
        autograd_g=bench('solver_coordinate_protection_backward',lambda:grad(loss,point))
        # At this exact teacher point, J^T(p-q)=0 analytically. Check equality
        # and bound autograd normalization roundoff before using that identity.
        if not torch.equal(logits(point).softmax(-1),anchor) or not finite(autograd_g) or float(autograd_g.norm())>1e-5*max(1,float(a.norm())):
            raise RuntimeError('stationary fixed-anchor identity failed')
        g=bench('analytic_categorical_KL_pullback',lambda:protection_gradient(logits,point,anchor,weights))
        if torch.count_nonzero(g):raise RuntimeError('stationary categorical KL pullback must vanish')
        curvature=bench('solver_FrozenGGN_anchor',lambda:FrozenGGN(logits,point,weights))
        op=operator(curvature,point)
        u=torch.zeros_like(point);v=torch.zeros_like(point);u.reshape(-1)[index]=1;v.reshape(-1)[(index+14336)%point.numel()]=1
        # Independent JVP/VJP support and timing, without Fisher approximation.
        _,jv=bench('native_JVP',lambda:torch.func.jvp(logits,(point,),(u,)))
        def vjp():
            _,pullback=torch.func.vjp(logits,point)
            return pullback(torch.ones_like(anchor)/anchor.numel())[0].detach()
        vjp_result=bench('native_VJP',vjp)
        fu=bench('exact_GGN_matvec',lambda:op(u)-config['damping']*u)
        repeated=bench('repeated_exact_GGN_matvec',lambda:op(u)-config['damping']*u)
        fv=bench('symmetry_exact_GGN_matvec',lambda:op(v)-config['damping']*v)
        uv=float((u.float()*fv.float()).sum());vu=float((v.float()*fu.float()).sum());quad=float((u.float()*fu.float()).sum())
        symmetric=abs(uv-vu)<=1e-5*max(1,abs(uv),abs(vu));psd=quad>=-1e-6
        records['NATIVE_GGN_AUDIT'].update(status='PASS' if finite(fu) and finite(fv) and symmetric and psd and torch.equal(fu,repeated) else 'FAIL',
            JVP_supported=True,VJP_supported=True,JVP_tangent_dtype=str(u.dtype),JVP_result_dtype=str(jv.dtype),VJP_result_dtype=str(vjp_result.dtype),
            GGN_output_dtype=str(fu.dtype),GGN_norm=float(fu.float().norm()),u_Fv=uv,v_Fu=vu,symmetry_absolute_error=abs(uv-vu),
            PSD_quadratic=quad,repeated_max_difference=float((fu.float()-repeated.float()).abs().max()),approximation_used=False)
        solved=bench('Q_inverse_CG',lambda:cg(op,a,rtol=1e-4,max_iter=config['maximum_CG_iterations']))
        plus=point.clone();minus=point.clone();plus.reshape(-1)[index]+=epsilon;minus.reshape(-1)[index]-=epsilon
        with torch.no_grad():fd=float((answer(plus)-answer(minus))/(2*epsilon))
        analytic=float(a.reshape(-1)[index]);fd_error=abs(fd-analytic)/max(abs(analytic),1e-8)
        records['NATIVE_PRECISION_AUDIT'].update(status='PASS' if solved.status=='CONVERGED' and fd_error<=.2 else 'FAIL',
            deployment_dtype='torch.bfloat16',solver_coordinate_dtype=str(point.dtype),logit_protection_arithmetic_dtype='torch.float32',
            FP32_editor_state_used=True,editor_function='FP32_FULL_FUNCTIONAL',
            curvature_semantics='exact JVP/VJP GGN of FP32 functional model, not rounded BF16 map',
            cross_precision_max_difference=cross_difference,cross_precision_parity_passed=cross_difference<=1e-3,
            cross_precision_parity_is_admission_gate=False,approved_precision_contract=config['approved_protocol_digest'],
            preservation_gradient_semantics='analytic J^T(p-q)=0 at identical fixed FP32 teacher',
            autograd_preservation_normalization_roundoff_norm=float(autograd_g.norm()),
            constraint_gradient_dtype=str(a.dtype),preservation_gradient_dtype=str(g.dtype),CG_inner_product_dtype=str((a*a).sum().dtype),
            solver_CG_status=solved.status,solver_CG_residual=solved.relative_residual,CG_rtol=1e-4,CG_iterations=solved.iterations,
            recursive_relative_residuals=solved.recursive_relative_residuals,
            finite_difference=dict(epsilon=epsilon,analytic=analytic,central_difference=fd,relative_error=fd_error,
                plus_represented_delta_norm=float((plus.to(torch.bfloat16).float()-bfpoint.float()).norm()),
                minus_represented_delta_norm=float((minus.to(torch.bfloat16).float()-bfpoint.float()).norm()),frozen_relative_tolerance=.2),
            gradient_zero_fraction=float((a==0).float().mean()))
        flush()
        if records['NATIVE_GGN_AUDIT']['status']!='PASS' or records['NATIVE_PRECISION_AUDIT']['status']!='PASS':raise RuntimeError('native GGN/precision qualification failed')
        constraint=a.reshape(1,-1);rhs=torch.tensor([1e-4],device=point.device,dtype=point.dtype)
        solved_g=bench('Q_inverse_stationary_preservation_CG',lambda:cg(op,g,rtol=1e-4,max_iter=config['maximum_CG_iterations']))
        qp1=bench('one_constraint_QP',lambda:solve_step(op,constraint,rhs,g,nu=10,cg_rtol=1e-4,cg_max_iter=config['maximum_CG_iterations'],dual_tol=1e-8,max_active=2,inverse_solutions=[solved_g,solved]))
        qp2=bench('two_constraint_QP',lambda:solve_step(op,torch.cat((constraint,.5*constraint)),torch.cat((rhs,.5*rhs)),g,nu=10,cg_rtol=1e-4,cg_max_iter=config['maximum_CG_iterations'],dual_tol=1e-8,max_active=2,inverse_solutions=[solved_g,solved,replace(solved,value=.5*solved.value)]))
        records['NATIVE_QP_AUDIT'].update(status='PASS' if qp1.status==qp2.status=='CONVERGED' else 'FAIL',
            one_constraint=dict(status=qp1.status,kkt=qp1.kkt,slack=qp1.slack.tolist()),two_constraints=dict(status=qp2.status,kkt=qp2.kkt,slack=qp2.slack.tolist()),
            includes_preservation_linear_term=True,explicit_slack=True,max_active=2,max_edit_steps=1,
            inverse_reuse='same frozen Q, fresh actual residual check per RHS',
            reused_inverse_residuals=[[s.relative_residual for s in q.cg] for q in [qp1,qp2]],
            second_constraint='mechanical redundant row A2=0.5*A1; no independent medical evidence claim')
        flush()
        if records['NATIVE_QP_AUDIT']['status']!='PASS':raise RuntimeError('frozen native QP convergence failed')
        direction=qp2.direction;direction=direction*min(1,config['trust_radius']/max(float(direction.norm()),1e-30))
        def transaction():
            rt.write(point+direction);rt.audit(snapshot,hooks)
            with torch.no_grad():candidate_z=normal();candidate_loss=float(protection_kl(candidate_z[:,predictors,:],deployment_anchor,weights))
            candidate_score=float(score(candidate_z));base_score=float(score(z))
            rounding=dict(rt.last_rounding)
            rt.reset_single();restored=torch.equal(rt.weight,rt.base)
            with torch.no_grad():restoration_error=float((normal()-z).abs().max())
            return candidate_loss,rounding,restored,restoration_error,candidate_score,base_score
        candidate_loss,rounding,restored,restoration_error,candidate_score,base_score=bench('write_normal_forward_protection_rollback',transaction)
        transaction_ok=restored and restoration_error<=1e-3 and candidate_loss<=1e-3 and rounding['actual_delta_norm']>0
        records['NATIVE_WRITE_ROLLBACK_AUDIT'].update(status='PASS' if transaction_ok else 'FAIL',only_selected_changed=True,
            protection_KL=candidate_loss,protection_anchor_dtype='BF16 deployment logits promoted to FP32 for KL only',
            base_answer_score=base_score,candidate_answer_score=candidate_score,
            actual_deployed_score_change=candidate_score-base_score,content_score_is_mechanical_only=True,
            protection_budget=1e-3,trust_radius=config['trust_radius'],rounding=rounding,
            bitwise_W_restoration=restored,normal_forward_restoration_max_difference=restoration_error)
        flush()
        if not transaction_ok:raise RuntimeError('bounded deployed mechanical write/rollback qualification failed')
        rt.write(point+direction);rt.audit(snapshot,hooks)
        with torch.no_grad():edited_z=normal().cpu();tokens=bench('ordinary_unjudged_free_generation',lambda:model.generate(**generation)).cpu()
        edited_path=root/'edited_matrix.private.pt';torch.save(rt.weight.detach().cpu(),edited_path)
        rt.reset_single();rt.audit(snapshot,hooks)
        del rt,model,selected,snapshot,raw,raw32,bf_logits,bf_score,bf_loss,loss,logits,answer,curvature,op,inputs,generation
        rt=None;gc.collect();torch.cuda.empty_cache()
        reload_config=dict(config,edited_matrix=str(edited_path),reload_result=str(root/'clean_reload.private.pt'))
        (root/'RELOAD_CONFIG.private.json').write_text(json.dumps(reload_config))
        env=dict(os.environ,NATIVE_RELOAD_CONFIG=str(root/'RELOAD_CONFIG.private.json'))
        code="import os,json\nfrom experiments.directw_evidence_v1.native_io import clean_reload\nclean_reload(json.load(open(os.environ['NATIVE_RELOAD_CONFIG'])))\n"
        remaining=max(1,int(config['native_wall_seconds_cap']-(time.monotonic()-started)))
        result=bench('editor_free_clean_reload',lambda:subprocess.run([sys.executable,'-'],input=code,text=True,env=env,timeout=remaining,capture_output=True))
        if result.returncode:raise RuntimeError('editor-free clean reload failed: '+result.stderr[-1200:])
        reloaded=torch.load(root/'clean_reload.private.pt',weights_only=True)
        logit_error=float((edited_z-reloaded['logits']).abs().max());token_equal=torch.equal(tokens,reloaded['tokens'])
        records['NATIVE_CLEAN_RELOAD_AUDIT'].update(status='PASS' if logit_error<=1e-3 and token_equal and reloaded['hooks']==0 and not reloaded['prohibited_parameter_names'] and not reloaded['editor_imported'] else 'FAIL',
            editor_free_process=True,editor_imported=reloaded['editor_imported'],hooks=reloaded['hooks'],prohibited_parameters=reloaded['prohibited_parameter_names'],
            normal_logit_max_difference=logit_error,unjudged_generation_token_parity=token_equal,Base_restored_in_parent_before_reload=True,
            child_wall_seconds=reloaded['wall_seconds'],child_peak_gpu_bytes=reloaded['peak_gpu_bytes'],child_peak_host_rss_bytes=reloaded['host_peak_rss_bytes'])
    except BaseException as error:
        failure=dict(exception=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
        (root/'FAILURE.private.json').write_text(json.dumps(failure,indent=2))
        records['NATIVE_RESOURCE_PROFILE']['failure']=dict(exception=type(error).__name__,reason=str(error))
    finally:
        if rt is not None:
            try:
                rt.reset_single()
                records['NATIVE_WRITE_ROLLBACK_AUDIT']['final_Base_W_bitwise_restored']=torch.equal(rt.weight,rt.base)
            except BaseException as error:records['NATIVE_WRITE_ROLLBACK_AUDIT']['restore_exception']=str(error)
        signal.alarm(0)
        elapsed=time.monotonic()-started
        records['NATIVE_RESOURCE_PROFILE'].update(status='MEASURED_PARTIAL' if profile else 'NOT_MEASURED',microbenchmarks=profile,
            elapsed_seconds=elapsed,gpu_hours_used=elapsed/3600,gpu_hours_cap=config['authorized_gpu_hours'],exact_GGN_calls=ggn_calls,
            exact_GGN_call_cap=config['maximum_GGN_calls'],solver_vector_bytes=4096*14336*4,
            snapshot_bytes=4096*14336*2,full_logit_bytes=int(z.numel()*z.element_size()) if 'z' in locals() else None,
            candidate_matrix_bytes=4096*14336*4,full_Backbone_CPU_snapshot_bytes=config['static_bindings']['checkpoint_tensor_elements']*2,
            new_artifact_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
            paid_judge_calls=0,real_pilot_started=False,sequential_started=False)
        measured=[p['wall_seconds'] for p in profile if p['operation']=='exact_GGN_matvec']
        if measured:
            seconds=measured[0];step=948*seconds
            records['PILOT_COST_PROJECTION'].update(status='MEASURED_GGN_PARTIAL_PROJECTION',seconds_per_GGN=seconds,worst_case_GGN_calls_per_QP_step=948,
                worst_case_QP_GGN_seconds=step,one_edit_20steps_GGN_hours=20*step/3600,eight_edit_20steps_GGN_hours=8*20*step/3600,
                twelve_edit_20steps_GGN_hours=12*20*step/3600,non_GGN_overheads='SEE_MICROBENCHMARKS; projection is lower bound without all overheads',
                scalability_status='EXACT_GGN_SCALABILITY_BLOCKER' if 20*step>config['native_wall_seconds_cap'] else 'REQUIRES_EXTERNAL_COST_REVIEW',pilot_allowed=False)
        else:records['PILOT_COST_PROJECTION'].update(status='NOT_MEASURED',reason='No completed exact native GGN matvec; no static cost substituted',pilot_allowed=False)
        required=['NATIVE_TEMPLATE_AUDIT','NATIVE_DERIVATIVE_AUDIT','NATIVE_PRECISION_AUDIT','NATIVE_GGN_AUDIT','NATIVE_QP_AUDIT','NATIVE_WRITE_ROLLBACK_AUDIT','NATIVE_CLEAN_RELOAD_AUDIT']
        passed=all(records[k]['status']=='PASS' for k in required) and bool(measured)
        records['NATIVE_RESOURCE_PROFILE']['native_smoke_status']='PASS' if passed else 'FAIL'
        failure_type=records['NATIVE_RESOURCE_PROFILE'].get('failure',{}).get('exception')
        resource_blocked=failure_type in {'OutOfMemoryError','TimeoutError'}
        if resource_blocked:records['NATIVE_RESOURCE_PROFILE']['native_smoke_status']='BLOCKED_RESOURCE'
        records['NATIVE_RESOURCE_PROFILE']['current_state']='WAITING_FOR_PILOT_REVIEW' if passed else ('BLOCKED_RESOURCE' if resource_blocked else 'BLOCKED_NATIVE')
        flush()
    return records
