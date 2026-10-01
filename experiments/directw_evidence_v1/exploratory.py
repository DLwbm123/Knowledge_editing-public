"""User-authorized exploratory train-support run; no scientific FIT claim."""
from __future__ import annotations
import gc,json,os,resource,signal,subprocess,sys,time,traceback
from dataclasses import asdict
from pathlib import Path
import torch
from .contracts import digest
from .gate import require_external_approval
from .editor import MatrixRuntime,Constraint,ProtectionGroup,EditConfig,edit_one,mean_answer_logprob
from .numerics import FrozenGGN
from .native_io import load,prepare,WEIGHT


def select_cases(rows, count=8):
    """Source-only deterministic round robin; permanent smoke exclusions."""
    groups={}
    for row in rows:
        if row['role']=='CANDIDATE_ONLY' and row['source_split']=='train' and row.get('original_role')!='SMOKE_MECHANICAL':
            groups.setdefault(row['source_group'],[]).append(row)
    queues=[sorted(group,key=lambda r:digest([r['image_hash'],r['question_hash'],r['id']])) for _,group in sorted(groups.items())]
    ordered=[q[i] for i in range(max(map(len,queues),default=0)) for q in queues if i<len(q)]
    if len(ordered)<count:raise ValueError('insufficient original non-smoke training candidates')
    pairs=[]
    for row in ordered[:count]:
        reference=next((x for x in ordered if x['image_hash']!=row['image_hash']),None)
        if reference is None:raise ValueError('distinct-image exploratory reference unavailable')
        pairs.append((row,reference))
    return pairs


def combine(prepared):
    """Right-pad native expanded inputs; physical and functional paths share them."""
    length=max(p[0]['inputs_embeds'].shape[1] for p in prepared)
    first=prepared[0][0]['inputs_embeds']
    embeds=first.new_zeros(len(prepared),length,first.shape[-1])
    mask=torch.zeros(len(prepared),length,device=first.device,dtype=torch.bool)
    labels=torch.full((len(prepared),length),-100,device=first.device,dtype=torch.long)
    for i,(inputs,y,_,_) in enumerate(prepared):
        n=inputs['inputs_embeds'].shape[1];embeds[i,:n]=inputs['inputs_embeds'][0]
        mask[i,:n]=inputs['attention_mask'][0];labels[i,:n]=y[0]
    return dict(inputs_embeds=embeds,attention_mask=mask,position_ids=None,use_cache=False,return_dict=True),labels


def resumed_cases(config):
    """Carry forward only a verified terminal prefix of independent cases."""
    if not config.get('resume_from'):
        return []
    parent=Path(config['resume_from'])
    if parent.resolve()==Path(config['run_root']).resolve():
        raise ValueError('resume requires a new run directory')
    previous=json.loads((parent/'CONFIG.private.json').read_text())['config']
    status=json.loads((parent/'STATUS.json').read_text())
    keys=('model_binding','editable_weight_path','data_manifest_digest','protocol','candidate_rows',
          'exploratory_inputs','target_logprob_gain','max_edit_steps','maximum_CG_iterations')
    if any(previous[k]!=config[k] for k in keys):
        raise ValueError('resume scientific inputs or protocol changed')
    if status['state']!='STOPPED_ON_ERROR' or status.get('final_Base_restored') is not True:
        raise ValueError('resume requires stopped, restored parent')
    cases=status['cases']
    if len(cases)!=status['completed'] or not 0<=len(cases)<config['exploratory_inputs']:
        raise ValueError('invalid completed prefix')
    for i,c in enumerate(cases):
        if str(c['index'])!=str(i) or not c.get('Base_restored') or c['status'] not in (
            'ACCEPTED','BACKTRACK_REJECTED','NOT_SATISFIED','SOLVER_NOT_CONVERGED'):
            raise ValueError('nonterminal or noncontiguous inherited case')
        if c['status']=='ACCEPTED' and c.get('changed_original_W') and not c.get('clean_reload'):
            raise ValueError('inherited accepted edit lacks clean reload')
    consumed=status['previous_native_seconds']+status['elapsed_seconds']
    if config['prior_GGN_calls']!=status['cumulative_GGN_calls'] or abs(config['prior_native_seconds']-consumed)>1e-6:
        raise ValueError('resume cumulative ledger mismatch')
    return cases


def run(config,approval,trusted):
    require_external_approval('EXPLORATORY_PILOT',config['bindings'],approval,config,trusted_authorization=trusted)
    inherited=resumed_cases(config);resume_count=len(inherited)
    root=Path(config['run_root']);started=time.monotonic();total_ggn=0;runtime=None
    status=dict(state='LOADING',result_kind='EXPLORATORY_TRAINING_ONLY',completed=len(inherited),intended=config['exploratory_inputs'],
                previous_native_seconds=config['prior_native_seconds'],previous_GGN_calls=config['prior_GGN_calls'],new_GGN_calls=0,judge_calls=0,
                original_scientific_data_status='BLOCKED_DATA',cases=inherited,native_regression='NOT_RUN',
                inherited_from=Path(config['resume_from']).name if config.get('resume_from') else None)
    def save():
        status.update(elapsed_seconds=time.monotonic()-started,new_GGN_calls=total_ggn,
                      cumulative_GGN_calls=config['prior_GGN_calls']+total_ggn,
                      cumulative_native_seconds=config['prior_native_seconds']+time.monotonic()-started,
                      host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        temp=root/'STATUS.tmp';temp.write_text(json.dumps(status,indent=2,allow_nan=False)+'\n');temp.replace(root/'STATUS.json')
    def timeout(*_):raise TimeoutError('exploratory process safety time ceiling reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(config['native_wall_seconds_cap'])
    torch.manual_seed(20261001);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def case(row,reference,index,mechanical=False):
        nonlocal total_ggn
        runtime.reset_single();before=time.monotonic();torch.cuda.reset_peak_memory_stats()
        prepared=[prepare(model,tokenizer,processor,r) for r in (row,reference)]
        inputs,labels=combine(prepared)
        with torch.no_grad():
            physical=runtime.normal_logits(inputs).float()
            same=runtime.logits(runtime.base,inputs).float()
        parity=float((physical-same).abs().max())
        if parity>1e-3:raise RuntimeError('same-precision native functional parity failed')
        smooth=runtime.bind_inputs(inputs,arithmetic_dtype=torch.float32)
        normal=runtime.bind_normal_inputs(inputs)
        score=lambda z:mean_answer_logprob(z.float(),labels,eos_id=tokenizer.eos_token_id)[0]
        point=runtime.base.float()
        with torch.no_grad():fp=smooth(point).float()
        base_score=float(score(physical));fp_score=float(score(fp))
        control=None
        if not mechanical and config.get('historical_control_scores') is not None:
            control=config['historical_control_scores'][int(index)]
            if abs(base_score-control['base_score'])>1e-6 or abs(fp_score-control['FP32_base_score'])>1e-6:
                raise RuntimeError('historical control initial score parity failed')
        # Exploratory reference scope is UNKNOWN, not claimed as legal BG/NEAR FIT.
        valid=(labels[1,1:]!=-100)&(labels[1,1:]!=tokenizer.eos_token_id)
        positions=valid.nonzero().flatten()
        f_logits=lambda w:smooth(w)[1:2,positions,:].float()
        n_logits=lambda:normal()[1:2,positions,:].float()
        bind=dict(model=config['model_binding'],weight_version='clean_Base',input=digest(reference['id']),
                  prefix=digest(prepared[1][3]),mask=digest(positions.cpu().tolist()),backend='eager',
                  config=config['approved_protocol_digest'],teacher_version='clean_Base')
        group=ProtectionGroup('reference',f_logits,fp[1:2,positions,:].softmax(-1).detach(),
            torch.ones(1,len(positions),device=point.device,dtype=torch.bool),('original_train',),1.,.001,
            dict(bind,dtype='torch.float32'),normal_logits=n_logits,
            deployment_anchor=physical[1:2,positions,:].softmax(-1).detach(),deployment_binding=dict(bind,dtype=str(runtime.weight.dtype)))
        diagnostic=None
        if mechanical:
            F=FrozenGGN(f_logits,point,group.weights)
            u=torch.zeros_like(point);v=torch.zeros_like(point);u[0,0]=1.;v[1,1]=1.
            total_ggn+=1;fu=F(u);total_ggn+=1;fv=F(v);total_ggn+=1;again=F(u)
            symmetry=abs(float((u*fv).sum()-(v*fu).sum()));psd=float((u*fu).sum())
            diagnostic=dict(symmetry_error=symmetry,PSD_probe=psd,repeat_exact=torch.equal(fu,again),finite=bool(torch.isfinite(fu).all() and torch.isfinite(fv).all()))
            if symmetry>1e-5 or psd< -1e-6 or not diagnostic['repeat_exact'] or not diagnostic['finite']:
                raise RuntimeError('repaired native exact-GGN regression failed')
            del F,u,v,fu,fv,again
        gain=1e-4 if mechanical else config['target_logprob_gain']
        threshold=min(max(base_score,fp_score)+gain,-1e-4)
        constraint=Constraint('target','EXPLORATORY_TRAIN',lambda w:score(smooth(w)),threshold,
                              normal_score=lambda:score(normal()))
        settings=EditConfig(tau=100.,nu=10.,trust_radius=.1,base_drift_limit=.3,edit_drift_limit=.3,
            max_steps=1 if mechanical else config['max_edit_steps'],max_active=1,cg_max_iter=config['maximum_CG_iterations'],
            cg_rtol=1e-4,tolerance=1e-6,arithmetic_dtype=torch.float32,
            constraint_value_mode=config.get('constraint_value_mode','functional'),
            record_trial_diagnostics=config.get('record_trial_diagnostics',False),
            max_ggn_calls=config['maximum_GGN_calls']-total_ggn)
        result=edit_one(runtime,[constraint],[group],settings,'W_FUNCTIONAL_QP',attempt_log=root/f'{index}.attempts.private.jsonl')
        total_ggn+=result.ggn_calls
        runtime.audit(runtime.base_state,runtime.base_hooks)
        with torch.no_grad():final_score=float(score(normal()))
        changed=not torch.equal(runtime.weight,runtime.base)
        if result.status!='ACCEPTED' and changed:raise RuntimeError('rejected edit did not restore original W')
        # Failures of implementation or solvers stop, rather than silently granting qualification.
        if result.status in ('EXCEPTION_ROLLED_BACK','ROLLBACK_FAILED') or (mechanical and result.status=='SOLVER_NOT_CONVERGED'):
            (root/f'{index}.failure.private.json').write_text(json.dumps(asdict(result),indent=2,default=str))
            raise RuntimeError('native editor failure: '+result.status+' '+str(result.error))
        receipt=dict(index=index,mechanical=mechanical,status=result.status,accepted_steps=result.accepted_steps,
            ggn_calls=result.ggn_calls,rollback=result.rollback,base_score=base_score,FP32_base_score=fp_score,
            threshold=threshold,final_deployed_score=final_score,changed_original_W=changed,
            same_precision_parity=parity,cross_precision_max_difference=float((fp-physical).abs().max()),
            constraint_value_mode=settings.constraint_value_mode,historical_control_parity=control is not None,
            native_deployment_dtype=str(runtime.weight.dtype),
            exact_GGN=diagnostic,attempts=result.attempts,
            elapsed_seconds=time.monotonic()-before,peak_gpu_bytes=torch.cuda.max_memory_allocated())
        (root/f'{index}.result.json').write_text(json.dumps(receipt,indent=2,allow_nan=False))
        if mechanical and not result.attempts:raise RuntimeError('native regression did not exercise solver/write path')
        if changed and not mechanical:
            # Last consumer is clean native reload; delete this generated matrix after parity.
            path=root/f'{index}.matrix.private.pt';torch.save(runtime.weight.detach().cpu(),path)
            with torch.no_grad():
                expected=model(**prepared[0][0]).logits.float().cpu()
                tokens=model.generate(**prepared[0][2]).cpu()
            torch.save(dict(logits=expected,tokens=tokens),root/f'{index}.expected.private.pt')
            receipt['export']=str(path)
        runtime.reset_single();receipt['Base_restored']=torch.equal(runtime.weight,runtime.base)
        return receipt
    try:
        save();model,tokenizer,processor=load(config);runtime=MatrixRuntime(model,WEIGHT)
        status['state']='NATIVE_REGRESSION';save()
        smoke=config['smoke_rows']
        native=case(smoke[0],smoke[1],'native',True);gc.collect();torch.cuda.empty_cache()
        status['native_regression']='PASS';status['native_details']=native
        status['state']='EXPERIMENT_RUNNING';save()
        for i,(row,reference) in enumerate(select_cases(config['candidate_rows'],config['exploratory_inputs'])):
            if i<resume_count:continue
            if time.monotonic()-started>=config['native_wall_seconds_cap']:raise TimeoutError('time ceiling')
            status['active_case']=i;save()
            result=case(row,reference,str(i));gc.collect();torch.cuda.empty_cache()
            if result.get('export'):
                reload_cfg=dict(config,smoke_row=row,edited_matrix=result['export'],reload_result=str(root/f'{i}.reload.private.pt'))
                cfgpath=root/f'{i}.reload_config.private.json';cfgpath.write_text(json.dumps(reload_cfg))
                code="import os,json\nfrom experiments.directw_evidence_v1.native_io import clean_reload\nclean_reload(json.load(open(os.environ['NATIVE_RELOAD_CONFIG'])))\n"
                child=subprocess.run([sys.executable,'-'],input=code,text=True,capture_output=True,
                    env=dict(os.environ,NATIVE_RELOAD_CONFIG=str(cfgpath)),timeout=max(1,int(config['native_wall_seconds_cap']-(time.monotonic()-started))))
                if child.returncode:raise RuntimeError('clean reload failed: '+child.stderr[-1000:])
                expected=torch.load(root/f'{i}.expected.private.pt',weights_only=True);actual=torch.load(root/f'{i}.reload.private.pt',weights_only=True)
                difference=float((expected['logits']-actual['logits']).abs().max())
                if difference>1e-3 or not torch.equal(expected['tokens'],actual['tokens']) or actual['hooks'] or actual['editor_imported'] or actual['prohibited_parameter_names']:
                    raise RuntimeError('clean export/reload parity failed')
                result['clean_reload']=dict(logit_max_difference=difference,generation_equal=True,editor_imported=False)
                path=Path(result.pop('export'));size=path.stat().st_size;path.unlink()
                result['checkpoint_lifecycle']=dict(deleted_after_final_consumer=True,bytes=size)
            status['cases'].append(result);status['completed']=len(status['cases']);save()
        status['state']='EXPERIMENT_COMPLETE';status['accepted']=sum(x['status']=='ACCEPTED' for x in status['cases'])
    except BaseException as error:
        status['state']='STOPPED_ON_ERROR';status['error']=type(error).__name__+': '+str(error)
        (root/'FAILURE.private.json').write_text(json.dumps(dict(error=status['error'],traceback=traceback.format_exc()),indent=2))
    finally:
        if runtime is not None:
            try:runtime.reset_single();status['final_Base_restored']=True
            except BaseException as error:status['final_Base_restored']=False;status['restore_error']=str(error)
        signal.alarm(0);save()
    return status
