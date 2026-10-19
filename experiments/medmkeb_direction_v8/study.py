"""Bounded31-step cumulative feasibility; first transitions reuse the frozen V7 audit."""
import os,json,time,copy,math
from pathlib import Path
import torch
from prefix_cache import PrefixCache
from qp import project,selfcheck as qp_check
from constraints import build,selfcheck as constraint_check
from engine import proposal,construct,match,violation,same,first_feasible,selfcheck as engine_check

def run(runtime,r,out,partition,counts):
    shared=out.parent;parent=Path(os.environ['PARENT_RUN']);prior=Path(os.environ['V7_ROOT'])/str(partition);read=lambda p:json.loads(Path(p).read_text())
    def write(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,default=str)+'\n');tmp.replace(path)
    def save(path,value):
        path=Path(path);tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)
    def budget():
        import shutil
        assert time.time()<float(os.environ['DEADLINE_EPOCH']) and not (shared/'STOP').exists()
        assert not any((shared/str(i)/'FAILURE.json').exists() for i in (0,1))
        assert counts['forwards']<=18000 and counts['backwards']<=1800 and counts['candidate_checks']<=300 and counts['updates']<=60 and counts['logical_transitions']<=62 and counts['generations']==0
        assert shutil.disk_usage(shared).free>8*1024**3
    r.c.budget=budget;write(out/'CPU_ADMISSION.json',dict(QP=qp_check(),constraints=constraint_check(),engine=engine_check()))
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
    old=read(prior/'AUDIT_COMPLETE.json');assert old['candidate_qualified'] and read(prior/'EXIT.json')['code']==0
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
    counts['reused_transitions']=2;counts['logical_transitions']=2
    write(out/'REUSE_ADMISSION.json',dict(status='PASS',old_candidate_scale=scale,input_bindings_matched=55,mask_proof='Frozen V7 masking algorithm reconstructed from identical question/reference/prompt ids/tokenizer/runtime/precision/preprocessing; EOS exclusion assertions identical',raw_proposal_exact=True,initial_optimizer_rebuilt=True,zero_constraint_state_exact=True,actual_project_and_control_norm_exact=True,new_language_forwards=counts['forwards'],new_NNback=counts['backwards']))
    assert counts['forwards']==counts['backwards']==0,'No fresh V7-successful language calls allowed'
    project_scalar=scalar1;status='COMPLETE';attempted=1;metrics_current=initial_project_metrics
    def loss_values(gradient=False):
        values=[];grads=[]
        for batch in batches:
            budget();hook.set_teacher_routing(batch.labels)
            if gradient:
                loss=runtime.compute_loss(batch);g=torch.autograd.grad(loss,tuple(params[k] for k in keys),allow_unused=True);counts['backwards']+=1
                grads.append(torch.cat([(v if v is not None else torch.zeros_like(params[k])).detach().cpu().double().reshape(-1) for k,v in zip(keys,g)]));values.append(float(loss.detach()))
            else:
                with torch.no_grad():values.append(float(runtime.compute_loss(batch)))
        assert all(math.isfinite(v) for v in values)
        return dict(source_loss=values[0],support_nll=values[1:]),grads
    def checkpoint(arm,state,scalar,curve):
        path=out/('active_'+arm+'.pt');save(path,dict(state={k:v.detach().cpu() for k,v in state.items()},scalar=scalar,curve=curve,counts=copy.deepcopy(counts)))
    hook.attach()
    try:
        with PrefixCache(runtime,batches):
            checkpoint('PROJECT',project_state,project_scalar,project_curve)
            for clock in range(2,32):
                budget();attempted=clock;expert.load_state_dict(project_state);before=r.old.snapshot(expert);current,grads=loss_values(True)
                assert violation(anchor,current['support_nll'])<=1e-4,'Current accepted state lost cumulative feasibility'
                raw,updated,d0=proposal(r,before,grads[0],project_scalar,keys);matrix,bounds=build(torch.stack(grads[1:]),torch.tensor(current['support_nll'],dtype=torch.float64),torch.tensor(anchor,dtype=torch.float64));dstar,qp=project(d0,matrix,bounds)
                write(out/'LAST_QP.json',dict(step=clock,QP=qp,counts=counts));assert qp['status']=='PASS','Fixed cumulative QP did not converge'
                zero,_=project(d0,torch.empty((0,d0.numel())),torch.empty(0));full_source=construct(r,before,raw,d0,d0,1.,keys);assert same(full_source,construct(r,before,raw,d0,zero,1.,keys));full_norm=r.fm.map_norm(before,full_source)
                accepted=None
                for scale in [2.**-j for j in range(8)]:
                    candidate=construct(r,before,raw,d0,dstar,scale,keys);norm=r.fm.map_norm(before,candidate)
                    if norm>full_norm:candidate=match(r,before,candidate,full_norm);norm=r.fm.map_norm(before,candidate)
                    assert norm<=full_norm*1.002+1e-10
                    expert.load_state_dict(candidate);value,_=loss_values();counts['candidate_checks']+=1;row=dict(step=clock,scale=scale,source_gain=current['source_loss']-value['source_loss'],actual_operator_norm=norm,max_support_regression=violation(anchor,value['support_nll']))
                    write(out/'candidates'/('step'+str(clock)+'_'+str(scale)+'.json'),dict(summary=row,private_metrics=value,QP=qp))
                    if first_feasible([row]):accepted=(candidate,value,row);break
                if accepted is None:
                    status='NO_FEASIBLE_LEARNING_STEP';expert.load_state_dict(before);write(out/'STOPPING_POINT.json',dict(status=status,step=clock,accepted_steps=len(project_curve),remaining_steps=31-len(project_curve),counts=counts));break
                project_state,metrics_current,row=accepted;project_scalar=updated;counts['updates']+=1;counts['logical_transitions']+=1
                project_curve.append(dict(step=clock,reused_V7=False,source_loss_before=current['source_loss'],source_loss_after=metrics_current['source_loss'],source_gain=row['source_gain'],actual_norm=row['actual_operator_norm'],max_support_regression=row['max_support_regression'],scale=row['scale'],QP=qp));checkpoint('PROJECT',project_state,project_scalar,project_curve)
                write(out/'PROGRESS.json',dict(task=task['key'],arm='PROJECT',step=clock,source_loss=metrics_current['source_loss'],counts=counts));print(task['key'],'PROJECT',clock,flush=True)
            save(out/'final_PROJECT.pt',dict(state={k:v.cpu() for k,v in project_state.items()}));saved=torch.load(out/'final_PROJECT.pt',map_location='cpu',weights_only=True)['state'];assert all(torch.equal(v.cpu(),saved[k]) for k,v in project_state.items())
            write(out/'PROJECT_COMPLETE.json',dict(status=status,task=task['key'],accepted_steps=len(project_curve),attempted_step=attempted,remaining_steps=31-len(project_curve),curve=project_curve,final_metrics=metrics_current,save_load_exact=True,counts=counts))
            norm_scalar=scalar1;norm_curve=[dict(step=1,reused_V7=True,source_loss_before=initial_source,source_loss_after=initial_norm_metrics['source_loss'],actual_norm=r.fm.map_norm(initial_before,norm_state),target_norm=project_curve[0]['actual_norm'],max_support_regression=violation(anchor,initial_norm_metrics['support_nll']))];norm_metrics=initial_norm_metrics;checkpoint('NORM_SOURCE',norm_state,norm_scalar,norm_curve)
            for clock in range(2,len(project_curve)+1):
                budget();expert.load_state_dict(norm_state);before=r.old.snapshot(expert);batch=batches[0];hook.set_teacher_routing(batch.labels);loss=runtime.compute_loss(batch);g=torch.autograd.grad(loss,tuple(params[k] for k in keys),allow_unused=True);counts['backwards']+=1;gradient=torch.cat([(v if v is not None else torch.zeros_like(params[k])).detach().cpu().double().reshape(-1) for k,v in zip(keys,g)])
                raw,norm_scalar,d0=proposal(r,before,gradient,norm_scalar,keys);candidate=construct(r,before,raw,d0,d0,1.,keys);target=project_curve[clock-1]['actual_norm'];norm_state=match(r,before,candidate,target);actual=r.fm.map_norm(before,norm_state);assert abs(actual-target)<=1e-10+.002*target
                expert.load_state_dict(norm_state);norm_metrics,_=loss_values();counts['candidate_checks']+=1;counts['updates']+=1;counts['logical_transitions']+=1;norm_curve.append(dict(step=clock,reused_V7=False,source_loss_before=float(loss.detach()),source_loss_after=norm_metrics['source_loss'],actual_norm=actual,target_norm=target,max_support_regression=violation(anchor,norm_metrics['support_nll'])));checkpoint('NORM_SOURCE',norm_state,norm_scalar,norm_curve);write(out/'PROGRESS.json',dict(task=task['key'],arm='NORM_SOURCE',step=clock,source_loss=norm_metrics['source_loss'],counts=counts))
            save(out/'final_NORM_SOURCE.pt',dict(state={k:v.cpu() for k,v in norm_state.items()}));saved=torch.load(out/'final_NORM_SOURCE.pt',map_location='cpu',weights_only=True)['state'];assert all(torch.equal(v.cpu(),saved[k]) for k,v in norm_state.items())
            write(out/'NORM_COMPLETE.json',dict(status='COMPLETE_MATCHED_PREFIX',steps=len(norm_curve),remaining_steps=31-len(norm_curve),curve=norm_curve,final_metrics=norm_metrics,save_load_exact=True,counts=counts))
            reference=read(shared/'REFERENCE_TRAJECTORY.json')[task['key']];reference_gain=reference['initial_source_loss']-reference['source_loss_after31'];gain=initial_source-metrics_current['source_loss'];qualified=status=='COMPLETE' and len(project_curve)==31 and all(x['max_support_regression']<=1e-4 for x in project_curve) and gain>=.1*reference_gain and gain>1e-3
            write(out/'TRAJECTORY_COMPLETE.json',dict(status=status,task=task['key'],accepted_steps=len(project_curve),remaining_steps=31-len(project_curve),source_gain=gain,reference_NONE31_gain=reference_gain,trajectory_qualified=qualified,matched_control_gain=initial_source-norm_metrics['source_loss'],counts=counts,semantic_effectiveness_tested=False))
    finally:hook.detach()
