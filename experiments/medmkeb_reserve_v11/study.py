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
        assert counts['forwards']<=15500 and counts['backwards']<=1800 and counts['candidate_checks']<=248 and counts['updates']<=31 and counts['logical_transitions']<=31 and counts['generations']==0
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
    inputs=torch.load(prior/'QP_INPUTS.pt',map_location='cpu',weights_only=True);assert read(prior/'ENGINEERING_ADMISSION.json')['support_initial_difference']<=1e-6 and read(prior/'EXIT.json')['code']==0
    reference=read(prior/'candidates/SOURCE_1.0.json');initial_source=reference['private_metrics']['source_loss']+reference['summary']['source_gain'];anchor=read(shared/'REFERENCE_POOL.json')['nll'];initial_metrics=dict(source_loss=initial_source,support_nll=anchor)
    scalar=dict(step=0,second_moments=[0.,0.]);state=before;current=initial_metrics;curve=[];status='COMPLETE';counts['reused_gradient_sequences']=55
    def values(gradient=False):
        data=[];grads=[]
        for batch in batches:
            budget();hook.set_teacher_routing(batch.labels)
            with torch.set_grad_enabled(gradient):
                loss=runtime.compute_loss(batch);data.append(float(loss.detach()))
                if gradient:
                    gs=torch.autograd.grad(loss,tuple(params[k] for k in keys),allow_unused=True);counts['backwards']+=1
                    grads.append(torch.cat([(v if v is not None else torch.zeros_like(params[k])).detach().cpu().double().reshape(-1) for k,v in zip(keys,gs)]))
        assert all(math.isfinite(x) for x in data)
        return dict(source_loss=data[0],support_nll=data[1:]),grads
    def first_cache(before,raw,d0):
        cache=[({k:v.cpu().clone() for k,v in before.items()},initial_metrics)];full=construct(r,before,raw,d0,d0,1.,keys);limit=r.fm.map_norm(before,full)
        for scale in [2.**-j for j in range(8)]:
            source=construct(r,before,raw,d0,d0,scale,keys);protected=construct(r,before,raw,d0,inputs['projected_direction'],scale,keys);norm=r.fm.map_norm(before,protected)
            if norm>limit:protected=match(r,before,protected,limit);norm=r.fm.map_norm(before,protected)
            control=match(r,before,full,norm)
            for kind,value in [('SOURCE',source),('PROJECT',protected),('NORM_SOURCE',control)]:
                cache.append(({k:v.cpu().clone() for k,v in value.items()},read(prior/('candidates/'+kind+'_'+str(scale)+'.json'))['private_metrics']))
        return cache
    hook.attach()
    try:
        with PrefixCache(runtime,batches):
            for clock in range(1,32):
                budget();expert.load_state_dict(state);before=r.old.snapshot(expert)
                if clock==1:source_gradient=inputs['source_gradient'];matrix=inputs['matrix'];bounds=torch.full((54,),5e-5,dtype=torch.float64)
                else:
                    current,grads=values(True);assert violation(anchor,current['support_nll'])<=1e-4
                    source_gradient=grads[0];matrix,bounds=build(torch.stack(grads[1:]),torch.tensor(current['support_nll'],dtype=torch.float64),torch.tensor(anchor,dtype=torch.float64),epsilon=5e-5)
                raw,updated,d0=proposal(r,before,source_gradient,scalar,keys)
                if clock==1:assert torch.equal(d0,inputs['source_direction'])
                delta,qp=project(d0,matrix,bounds);assert qp['status']=='PASS','Fixed reserve QP did not converge'
                zero,_=project(d0,torch.empty((0,d0.numel())),torch.empty(0));full=construct(r,before,raw,d0,d0,1.,keys);assert same(full,construct(r,before,raw,d0,zero,1.,keys));assert same(before,construct(r,before,raw,d0,delta,0.,keys));limit=r.fm.map_norm(before,full)
                save(out/'LAST_LINEAR.pt',dict(state={k:v.cpu() for k,v in before.items()},source_gradient=source_gradient,matrix=matrix,bounds=bounds,projected_direction=delta,scalar=scalar,current_metrics=current,step=clock,counts=copy.deepcopy(counts)))
                if clock==1:write(out/'ENGINEERING_ADMISSION.json',dict(status='PASS',bindings=55,raw_initial_proposal_exact=True,old_Base_reference_difference=read(prior/'ENGINEERING_ADMISSION.json')['support_initial_difference'],reused_gradient_sequences=55,zero_scale_state_exact=True,zero_constraint_source_exact=True,inner=5e-5,outer=1e-4,QP=qp,new_language_calls=counts['forwards']))
                cache=first_cache(before,raw,d0) if clock==1 else [({k:v.cpu().clone() for k,v in before.items()},current)];accepted=None
                for scale in [2.**-j for j in range(8)]:
                    candidate=construct(r,before,raw,d0,delta,scale,keys);norm=r.fm.map_norm(before,candidate)
                    if norm>limit:candidate=match(r,before,candidate,limit);norm=r.fm.map_norm(before,candidate)
                    cpu={k:v.cpu() for k,v in candidate.items()};metric=next((v for old,v in cache if same(old,cpu)),None);reused=metric is not None
                    if not reused:expert.load_state_dict(candidate);metric,_=values();cache.append(({k:v.clone() for k,v in cpu.items()},metric))
                    counts['candidate_checks']+=1;row=dict(step=clock,scale=scale,source_gain=current['source_loss']-metric['source_loss'],actual_operator_norm=norm,max_support_regression=violation(anchor,metric['support_nll']),reused_exact_metrics=reused)
                    write(out/'candidates'/('step'+str(clock)+'_'+str(scale)+'.json'),dict(summary=row,private_metrics=metric,QP=qp));assert norm<=limit*1.002+1e-10
                    if first_feasible([row]):accepted=(candidate,metric,row);break
                if accepted is None:status='NO_FEASIBLE_LEARNING_STEP';expert.load_state_dict(before);write(out/'STOPPING_POINT.json',dict(step=clock,status=status,accepted_steps=len(curve),remaining_NA=31-len(curve),counts=counts));break
                state,metric,row=accepted;scalar=updated;curve.append(dict(step=clock,source_loss_before=current['source_loss'],source_loss_after=metric['source_loss'],**{k:v for k,v in row.items() if k!='step'},QP=qp));current=metric;counts['updates']+=1;counts['logical_transitions']+=1
                save(out/'active.pt',dict(state={k:v.cpu() for k,v in state.items()},scalar=scalar,curve=curve,counts=copy.deepcopy(counts)));write(out/'PROGRESS.json',dict(step=clock,source_loss=current['source_loss'],max_regression=row['max_support_regression'],counts=counts));print(task['key'],'RESERVE',clock,flush=True)
            if not (out/'active.pt').exists():save(out/'active.pt',dict(state={k:v.cpu() for k,v in state.items()},scalar=scalar,curve=curve,counts=counts))
            save(out/'final.pt',dict(state={k:v.cpu() for k,v in state.items()}));loaded=torch.load(out/'final.pt',map_location='cpu',weights_only=True)['state'];assert same(loaded,{k:v.cpu() for k,v in state.items()})
            ref=read(shared/'REFERENCE_TRAJECTORY.json')[task['key']];gain=initial_source-current['source_loss'];refgain=ref['initial_source_loss']-ref['source_loss_after31'];qualified=len(curve)==31 and all(x['max_support_regression']<=1e-4 for x in curve) and gain>=.1*refgain and gain>1e-3
            write(out/'STUDY_COMPLETE.json',dict(status=status,task=task['key'],accepted_steps=len(curve),remaining_NA=31-len(curve),curve=curve,final_metrics=current,source_gain=gain,reference_NONE31_gain=refgain,trajectory_qualified=qualified,save_load_exact=True,inner=5e-5,outer=1e-4,counts=counts,semantic_effectiveness_tested=False))
    finally:hook.detach()
