"""Bounded31-step cumulative feasibility; first transitions reuse the frozen V7 audit."""
import os,json,time,copy,math
from pathlib import Path
import torch
from prefix_cache import PrefixCache
from qp import project,selfcheck as qp_check
from constraints import build,selfcheck as constraint_check
from path_math import direct,capped,distance,selfcheck as path_check
from engine import proposal,construct,match,violation,same,first_feasible,selfcheck as engine_check

def run(runtime,r,out,partition,counts):
    shared=out.parent;parent=Path(os.environ['PARENT_RUN']);prior=Path(os.environ['V7_ROOT'])/str(partition);old8=Path(os.environ['V8_ROOT'])/str(partition);read=lambda p:json.loads(Path(p).read_text())
    def write(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,default=str)+'\n');tmp.replace(path)
    def save(path,value):
        path=Path(path);tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)
    def budget():
        import shutil
        assert time.time()<float(os.environ['DEADLINE_EPOCH']) and not (shared/'STOP').exists()
        assert counts['forwards']<=3500 and counts['backwards']<=1200 and counts['candidate_checks']<=40 and counts['updates']<=19 and counts['logical_transitions']<=20 and counts['generations']==0
        assert shutil.disk_usage(shared).free>8*1024**3
    r.c.budget=budget;write(out/'CPU_ADMISSION.json',dict(QP=qp_check(),constraints=constraint_check(),engine=engine_check(),path=path_check()))
    task=read(shared/'DATA_ADMISSION.json')['tasks'][partition];pool=read(shared/'SUPPORT_FROZEN.json');assert len(pool)==9
    template=read(parent/'private/M13_SOURCE_PACKET.json')['original_tasks'][4];bindings=read(parent/'private/EVAL_BINDINGS.json')
    rows=[task['source']]
    for c in pool:
        for form in ('{}.', 'The answer is {}.', 'It is {}.'):
            for text in (c['row']['reference'],c['wrong_choice']):rows.append(dict(c['row'],query_id='S'+str(len(rows)-1),reference=form.format(text.rstrip('.'))))
    runtime.model.half();items=r.s.prepare(runtime,bindings,template,rows,True);r.old.fp32(runtime);batches=[]
    for index,(row,value,raw,expanded,binding) in enumerate(items):
        if index:
            kwargs,labels,tokens=value;start=raw['input_ids'][0].tolist();full=runtime._tokenize_prompt(raw['prompt']+' '+row['reference'])[0].tolist();assert full[:len(start)]==start
            content=tuple(full[len(start):]);assert tuple(tokens[:len(content)])==content and tuple(tokens[len(content):])==(28705,2)
            positions=torch.where(labels[0]!=-100)[0];labels=labels.clone();labels[0,positions[len(content):]]=-100;value=(dict(kwargs,labels=labels),labels,content)
        else:
            base_binding=read(read(shared/'BASE_REUSE.json')[row['query_id']])['binding'];assert all(base_binding[k]==binding[k] for k in ('image_sha256','question','reference','prompt_ids','generation'))
        old_binding=read(prior/'bindings'/(row['query_id']+'.json'));new_binding=dict(binding,preprocessing_seed=raw['preprocessing_seed']);assert all(new_binding[k]==old_binding[k] for k in old_binding if k!='image_path'),'V7 input binding drift';batches.append(r.old.batch_on(runtime,value));write(out/'bindings'/(row['query_id']+'.json'),dict(binding,preprocessing_seed=raw['preprocessing_seed']))
    initial=r.p.tr.TT4(template['seed'],8,8).state_dict();base=r.mm.canonicalize(initial)[0];expert=r.p.expert(initial,template['seed'],runtime.device);expert.load_state_dict(base)
    before=r.old.snapshot(expert);keys=list(before);params=dict(expert.named_parameters());assert set(keys)==set(params)
    hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert)
    old=read(prior/'AUDIT_COMPLETE.json');assert read(prior/'ENGINEERING_ADMISSION.json')['support_initial_difference']<=1e-6;assert old['candidate_qualified'] and read(prior/'EXIT.json')['code']==0
    qp_inputs=torch.load(prior/'QP_INPUTS.pt',map_location='cpu',weights_only=True)
    source_full=read(prior/'candidates/SOURCE_1.0.json');initial_source=source_full['private_metrics']['source_loss']+source_full['summary']['source_gain'];anchor=read(shared/'REFERENCE_POOL.json')['nll']
    initial_before={k:v.clone() for k,v in before.items()};scalar0=dict(step=0,second_moments=[0.,0.]);raw,scalar1,d0=proposal(r,before,qp_inputs['source_gradient'],scalar0,keys)
    assert torch.equal(d0,qp_inputs['source_direction']),'Exact V7 raw proposal reconstruction failed'
    dstar,qp0=project(d0,qp_inputs['matrix'],qp_inputs['bounds']);assert qp0['status']=='PASS';torch.testing.assert_close(dstar,qp_inputs['projected_direction'],rtol=0,atol=1e-10)
    zero,_=project(d0,torch.empty((0,d0.numel())),torch.empty(0));assert same(construct(r,before,raw,d0,zero,1.,keys),construct(r,before,raw,d0,d0,1.,keys))
    eligible=[]
    for scale in [2.**-j for j in range(8)]:
        row=read(prior/('candidates/PROJECT_'+str(scale)+'.json'))['summary'];eligible.append(row)
    chosen=first_feasible(eligible);assert chosen is not None
    scale=chosen['scale'];project_state=construct(r,before,raw,d0,dstar,scale,keys);target=r.fm.map_norm(before,project_state)
    full_source=construct(r,before,raw,d0,d0,1.,keys);full_norm=r.fm.map_norm(before,full_source)
    if target>full_norm:project_state=match(r,before,project_state,full_norm);target=r.fm.map_norm(before,project_state)
    assert target==chosen['actual_operator_norm'],'V7 projected actual norm reconstruction drift'
    norm_state=match(r,before,full_source,target);old_norm=read(prior/('candidates/NORM_SOURCE_'+str(scale)+'.json'));assert r.fm.map_norm(before,norm_state)==old_norm['summary']['actual_operator_norm']
    initial_project_metrics=read(prior/('candidates/PROJECT_'+str(scale)+'.json'))['private_metrics'];initial_norm_metrics=old_norm['private_metrics']
    project_curve=[dict(step=1,reused_V7=True,source_loss_before=initial_source,source_loss_after=initial_project_metrics['source_loss'],source_gain=initial_source-initial_project_metrics['source_loss'],actual_norm=target,max_support_regression=violation(anchor,initial_project_metrics['support_nll']),scale=scale,QP=qp0)]
    counts['reused_transitions']=1;counts['logical_transitions']=1
    write(out/'REUSE_ADMISSION.json',dict(status='PASS',old_candidate_scale=scale,input_bindings_matched=55,mask_proof='Frozen V7 masking algorithm reconstructed from identical question/reference/prompt ids/tokenizer/runtime/precision/preprocessing; EOS exclusion assertions identical',raw_proposal_exact=True,initial_optimizer_rebuilt=True,zero_constraint_state_exact=True,actual_project_and_control_norm_exact=True,new_language_forwards=counts['forwards'],new_NNback=counts['backwards']))
    assert counts['forwards']==counts['backwards']==0,'No fresh V7-successful language calls allowed'
    old_curve=read(old8/'PROJECT_COMPLETE.json')['curve'];assert len(old_curve)>=20
    project_scalar=scalar1;parity=[]
    def values(state,mode=True,gradient=False,native=False):
        expert.load_state_dict(state);numbers=[];gradients=[]
        with torch.set_grad_enabled(mode):
            for batch in batches:
                budget();hook.set_teacher_routing(batch.labels);loss=(cache.original_loss(batch) if native else runtime.compute_loss(batch));numbers.append(float(loss.detach()))
                if gradient:
                    gs=torch.autograd.grad(loss,tuple(params[k] for k in keys),allow_unused=True);counts['backwards']+=1
                    gradients.append(torch.cat([(v if v is not None else torch.zeros_like(params[k])).detach().cpu().double().reshape(-1) for k,v in zip(keys,gs)]))
        assert all(math.isfinite(x) for x in numbers)
        return dict(source_loss=numbers[0],support_nll=numbers[1:]),gradients
    def legacy_candidate(before,raw,d0,delta,scale):
        candidate=construct(r,before,raw,d0,delta,scale,keys);norm=r.fm.map_norm(before,candidate);full=construct(r,before,raw,d0,d0,1.,keys);full_norm=r.fm.map_norm(before,full)
        if norm>full_norm:candidate=match(r,before,candidate,full_norm)
        return candidate,full_norm
    def frozen_metrics(step):
        if step==1:return initial_project_metrics
        return read(old8/('candidates/step'+str(step)+'_'+str(old_curve[step-1]['scale'])+'.json'))['private_metrics']
    hook.attach()
    try:
        with PrefixCache(runtime,batches) as cache:
            for clock in range(2,21):
                before={k:v.clone() for k,v in project_state.items()};current,grads=values(before,True,True)
                row=dict(step=clock,max_gradient_vs_previous_accept_NLL=distance(current,frozen_metrics(clock-1)),source_before_difference=abs(current['source_loss']-old_curve[clock-1]['source_loss_before']))
                assert row['source_before_difference']<=1e-6,'Reconstruction source gradient input drift'
                raw,updated,d0=proposal(r,before,grads[0],project_scalar,keys);matrix,bounds=build(torch.stack(grads[1:]),torch.tensor(current['support_nll'],dtype=torch.float64),torch.tensor(anchor,dtype=torch.float64));delta,qp=project(d0,matrix,bounds);assert qp['status']=='PASS'
                project_state,_=legacy_candidate(before,raw,d0,delta,old_curve[clock-1]['scale']);norm=r.fm.map_norm(before,project_state);row['norm_exact']=norm==old_curve[clock-1]['actual_norm'];assert row['norm_exact'],'Old accepted update reconstruction norm drift'
                project_scalar=updated;counts['updates']+=1;counts['logical_transitions']+=1;parity.append(row)
                save(out/'active_RECON.pt',dict(state={k:v.cpu() for k,v in project_state.items()},scalar=project_scalar,parity=parity,counts=copy.deepcopy(counts)))
                write(out/'PROGRESS.json',dict(phase='RECONSTRUCT',accepted_old_step=clock,counts=counts));print(task['key'],'RECONSTRUCT',clock,flush=True)
            before={k:v.clone() for k,v in project_state.items()};current,grads=values(before,True,True);write(out/'RECONSTRUCTION_COMPLETE.json',dict(status='PASS',steps=20,new_updates=19,reused=1,parity=parity,endpoint_gradient_vs_old_no_grad=distance(current,frozen_metrics(20)),counts=counts))
            raw,updated,d0=proposal(r,before,grads[0],project_scalar,keys);matrix,bounds=build(torch.stack(grads[1:]),torch.tensor(current['support_nll'],dtype=torch.float64),torch.tensor(anchor,dtype=torch.float64));delta,qp=project(d0,matrix,bounds);assert qp['status']=='PASS'
            bundle=dict(state={k:v.cpu() for k,v in before.items()},scalar=project_scalar,source_direction=d0,projected_direction=delta,matrix=matrix,bounds=bounds,gradients=torch.stack(grads))
            save(out/'AUDIT_INPUTS.pt',bundle);saved=torch.load(out/'AUDIT_INPUTS.pt',map_location='cpu',weights_only=True);assert all(torch.equal(saved['state'][k],before[k].cpu()) for k in keys)
            base_ng,_=values(before,False);repeat_ng,_=values(before,False);native_ng,_=values(before,False,native=True)
            zero_legacy=construct(r,before,raw,d0,delta,0.,keys);zero_fixed=direct(before,delta,0.,keys);assert same(zero_fixed,before)
            zero_metrics,_=values(zero_legacy,False);counts['candidate_checks']+=1
            baseline=dict(grad_vs_no_grad=distance(current,base_ng),no_grad_repeat=distance(base_ng,repeat_ng),cached_vs_native=distance(base_ng,native_ng),legacy_zero_tensor_exact=same(zero_legacy,before),legacy_zero_operator_norm=r.fm.map_norm(before,zero_legacy),legacy_zero_NLL_drift=distance(base_ng,zero_metrics),fixed_zero_tensor_exact=True)
            write(out/'BASELINE_AUDIT.json',dict(status='COMPLETE',baseline=baseline,private_metrics=dict(grad=current,no_grad=base_ng,repeat=repeat_ng,native=native_ng,legacy_zero=zero_metrics),counts=counts))
            full_source=construct(r,before,raw,d0,d0,1.,keys);full_norm=r.fm.map_norm(before,full_source);results=[];state_cache=[(before,True,current),(before,False,base_ng),(zero_legacy,False,zero_metrics)]
            for scale in [2.**-j for j in range(8)]:
                for kind in ('LEGACY','DIRECT'):
                    candidate,actual_scale=legacy_candidate(before,raw,d0,delta,scale) if kind=='LEGACY' else capped(before,delta,scale,keys,full_norm,r.fm.map_norm)
                    if kind=='LEGACY':actual_scale=scale
                    actual_norm=r.fm.map_norm(before,candidate)
                    parameter_delta=torch.cat([(candidate[k]-before[k]).detach().cpu().double().reshape(-1) for k in keys]);linear=matrix@parameter_delta-bounds
                    for mode in (False,True):
                        reused=next((v for state,flag,v in state_cache if flag==mode and same(state,candidate)),None)
                        if reused is None:metric,_=values(candidate,mode);state_cache.append(({k:v.clone() for k,v in candidate.items()},mode,metric))
                        else:metric=reused
                        counts['candidate_checks']+=1;gain=current['source_loss']-metric['source_loss'];reg=violation(anchor,metric['support_nll']);row=dict(path=kind,grad_enabled=mode,scale=scale,actual_parameter_scale=actual_scale,source_gain=gain,max_support_regression=reg,actual_operator_norm=actual_norm,linear_max_violation=float(linear.max()),fixed_constraint=reg<=1e-4,qualified=reg<=1e-4 and gain>=1e-6 and actual_norm>0 and actual_norm<=full_norm*1.002+1e-10,reused_exact_state=reused is not None)
                        results.append(row);write(out/'candidates'/(kind+'_'+str(scale)+'_'+str(mode)+'.json'),dict(summary=row,private_metrics=metric))
                write(out/'PROGRESS.json',dict(phase='FACTORIAL',scale=scale,checks=len(results),counts=counts));print(task['key'],'FACTORIAL',scale,flush=True)
            cache_pass=baseline['no_grad_repeat']<=1e-6 and baseline['cached_vs_native']<=1e-6
            fixed_qualified=[x['scale'] for x in results if x['path']=='DIRECT' and x['grad_enabled'] and x['qualified']]
            write(out/'AUDIT_COMPLETE.json',dict(status='COMPLETE',task=task['key'],baseline=baseline,QP=qp,candidates=results,save_load_exact=True,mechanical_repeat_cache_pass=cache_pass,fixed_candidate_qualified=cache_pass and bool(fixed_qualified),fixed_qualified_scales=fixed_qualified,counts=counts,root_cause_requires_factorial_analysis=True,formal_new_training=0,semantic_effectiveness_tested=False))
    finally:hook.detach()
