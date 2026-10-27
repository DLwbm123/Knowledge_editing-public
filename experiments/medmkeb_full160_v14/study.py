"""Fixed160 SOURCE continuation and cumulative RESERVE; no checkpoint selection."""
import os,json,time,copy,math,shutil
from pathlib import Path
import torch
from cache_v14 import PrefixCache
from path_v14 import proposal,construct,match,same,violation,first_feasible
from qp_v14 import project,selfcheck as qp_check
from constraints_v14 import build,selfcheck as constraint_check
from answer_v14 import check,selfcheck as answer_check
from admission_v14 import replay_metrics,source_resume,selfcheck as admission_check

def run(runtime,r,out,partition,counts):
    shared=out.parent;parent=Path(os.environ['PARENT_RUN']);v7=Path(os.environ['V7_ROOT'])/str(partition);v11=Path(os.environ['V11_ROOT'])/str(partition);v13=Path(os.environ['V13_ROOT'])/str(partition)
    read=lambda p:json.loads(Path(p).read_text())
    def write(p,v):
        p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,default=str)+'\n');tmp.replace(p)
    def save(p,v):
        p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');torch.save(v,tmp);tmp.replace(p)
    def budget():
        assert time.time()<float(os.environ['DEADLINE_EPOCH']) and not (shared/'STOP').exists()
        assert counts['forwards']<=135000 and counts['backwards']<=9000 and counts['updates']<=289 and counts['candidate_checks']<=1200 and counts['generations']<=64
        assert shutil.disk_usage(shared).free>8*1024**3
    r.c.budget=budget;write(out/'CPU_ADMISSION.json',dict(QP=qp_check(),constraints=constraint_check(),answer=answer_check(),replay=admission_check()))
    task=read(shared/'DATA_ADMISSION.json')['tasks'][partition];pool=read(shared/'SUPPORT_FROZEN.json');assert len(pool)==9
    template=read(parent/'private/M13_SOURCE_PACKET.json')['original_tasks'][4];bindings=read(parent/'private/EVAL_BINDINGS.json');rows=[task['source']]
    for c in pool:
        for form in ('{}.', 'The answer is {}.', 'It is {}.'):
            for text in (c['row']['reference'],c['wrong_choice']):rows.append(dict(c['row'],query_id='S'+str(len(rows)-1),reference=form.format(text.rstrip('.'))))
    runtime.model.half();items=r.s.prepare(runtime,bindings,template,rows,True);r.old.fp32(runtime);batches=[]
    for index,(row,value,raw,expanded,binding) in enumerate(items):
        if index:
            kwargs,labels,tokens=value;start=raw['input_ids'][0].tolist();full=runtime._tokenize_prompt(raw['prompt']+' '+row['reference'])[0].tolist();assert full[:len(start)]==start
            content=tuple(full[len(start):]);assert tuple(tokens[:len(content)])==content and tuple(tokens[len(content):])==(28705,2)
            positions=torch.where(labels[0]!=-100)[0];labels=labels.clone();labels[0,positions[len(content):]]=-100;value=(dict(kwargs,labels=labels),labels,content)
        old=read(v7/'bindings'/(row['query_id']+'.json'));new=dict(binding,preprocessing_seed=raw['preprocessing_seed']);assert all(new[k]==old[k] for k in old if k!='image_path'),'V7 input binding drift'
        batches.append(r.old.batch_on(runtime,value));write(out/'bindings'/(row['query_id']+'.json'),new)
    initial=r.p.tr.TT4(template['seed'],8,8).state_dict();base=r.mm.canonicalize(initial)[0];expert=r.p.expert(initial,template['seed'],runtime.device);expert.load_state_dict(base)
    hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert);keys=list(base);params=dict(expert.named_parameters());assert set(keys)==set(params)
    inputs=torch.load(v7/'QP_INPUTS.pt',map_location='cpu',weights_only=True);ref=read(v7/'candidates/SOURCE_1.0.json');initial_source=ref['private_metrics']['source_loss']+ref['summary']['source_gain'];anchor=read(shared/'REFERENCE_POOL.json')['nll'];assert len(anchor)==54
    assert read(v7/'EXIT.json')['code']==read(v11/'EXIT.json')['code']==read(v13/'EXIT.json')['code']==0
    counts['reused_gradient_sequences']=55
    def values(gradient=False,source_only=False):
        data=[];grads=[]
        for batch in (batches[:1] if source_only else batches):
            budget();hook.set_teacher_routing(batch.labels)
            with torch.set_grad_enabled(gradient):
                loss=runtime.compute_loss(batch);data.append(float(loss.detach()))
                if gradient:
                    gs=torch.autograd.grad(loss,tuple(params[k] for k in keys),allow_unused=True);counts['backwards']+=1
                    grads.append(torch.cat([(v if v is not None else torch.zeros_like(params[k])).detach().cpu().double().reshape(-1) for k,v in zip(keys,gs)]))
        assert all(math.isfinite(x) for x in data)
        return dict(source_loss=data[0],support_nll=data[1:]),grads
    def snapshot():return {k:v.cpu() for k,v in r.old.snapshot(expert).items()}

    # This continuation keeps the original scalar moments, not a fresh optimizer.
    arm=out/'SOURCE';arm.mkdir(exist_ok=True);saved=torch.load(v13/'active.pt',map_location='cpu',weights_only=True);old_source=read(v13/'TRAIN_COMPLETE.json');expected_source=source_resume(saved,old_source);scalar=copy.deepcopy(saved['scalar']);expert.load_state_dict(saved['state']);assert same(snapshot(),saved['state']);curve=copy.deepcopy(old_source['curve'])
    hook.attach()
    try:
        with PrefixCache(runtime,batches[:1]):
            for step in range(32,161):
                before=r.old.snapshot(expert);metric,grad=values(True,True)
                if step==32:
                    assert abs(metric['source_loss']-expected_source)<=1e-6
                    write(out/'SOURCE_RESUME_ADMISSION.json',dict(status='PASS',step31=31,optimizer_moments_preserved=True,state_load_exact=True,source_NLL_difference=abs(metric['source_loss']-expected_source),new_backward_calls=counts['backwards']))
                raw,scalar,direction=proposal(r,before,grad[0],scalar,keys);zero,_=project(direction,torch.empty((0,direction.numel())),torch.empty(0));candidate=construct(r,before,raw,direction,direction,1.,keys);assert same(candidate,construct(r,before,raw,direction,zero,1.,keys));norm=r.fm.map_norm(before,candidate);assert norm>0
                expert.load_state_dict(candidate);counts['updates']+=1;counts['logical_transitions']+=1;counts['candidate_checks']+=1;curve.append(dict(step=step,source_loss_before=metric['source_loss'],actual_operator_norm=norm))
                save(arm/'active.pt',dict(state=snapshot(),scalar=scalar,curve=curve,counts=copy.deepcopy(counts)));write(out/'PROGRESS.json',dict(arm='SOURCE',step=step,counts=counts))
            metric,_=values(False,True)
            for j,row in enumerate(curve):row['source_loss_after']=curve[j+1]['source_loss_before'] if j+1<len(curve) else metric['source_loss']
    finally:hook.detach()
    save(arm/'final.pt',dict(state=snapshot()));assert same(torch.load(arm/'final.pt',map_location='cpu',weights_only=True)['state'],snapshot())
    write(arm/'TRAIN_COMPLETE.json',dict(status='COMPLETE',accepted_steps=160,remaining_NA=0,curve=curve,final_metrics=metric,source_gain=initial_source-metric['source_loss'],save_load_exact=True,old_updates_reused=31,new_updates=129))

    arm=out/'RESERVE';arm.mkdir(exist_ok=True);expert.load_state_dict(base);scalar=dict(step=0,second_moments=[0.,0.]);current=dict(source_loss=initial_source,support_nll=anchor);curve=[];status='COMPLETE';old_curve=read(v11/'STUDY_COMPLETE.json')['curve'];assert len(old_curve)==31;pending=None;replay_differences=[]
    hook.attach()
    try:
        with PrefixCache(runtime,batches):
            for step in range(1,161):
                budget();before=r.old.snapshot(expert)
                if step==1:source_gradient=inputs['source_gradient'];matrix=inputs['matrix'];bounds=torch.full((54,),5e-5,dtype=torch.float64)
                else:
                    current,grads=values(True)
                    if pending is not None:
                        old_metric=read(v11/'candidates'/('step'+str(pending['step'])+'_'+str(pending['scale'])+'.json'))['private_metrics'];difference=replay_metrics(current,old_metric);replay_differences.append(difference)
                        assert violation(anchor,current['support_nll'])<=1e-4 and pending['source_loss_before']-current['source_loss']>=1e-6
                        pending.update(source_loss_after=current['source_loss'],source_gain=pending['source_loss_before']-current['source_loss'],max_support_regression=violation(anchor,current['support_nll']),post_metrics_measured_in_next_gradient=True)
                        save(arm/'active.pt',dict(state=snapshot(),scalar=scalar,curve=curve,counts=copy.deepcopy(counts)))
                        pending=None
                    if step==32:
                        write(out/'RECONSTRUCTION_COMPLETE.json',dict(status='PASS',updates=31,all55_metrics_checked=True,max_NLL_difference=max(replay_differences),historical_candidate_metrics_reused=False,old_candidate_search_repeated=False,exact_old_tensor_reference_available=False,validation='actual_next_registered_gradient_pass'))
                    assert violation(anchor,current['support_nll'])<=1e-4
                    source_gradient=grads[0];matrix,bounds=build(torch.stack(grads[1:]),torch.tensor(current['support_nll'],dtype=torch.float64),torch.tensor(anchor,dtype=torch.float64),epsilon=5e-5)
                raw,updated,d0=proposal(r,before,source_gradient,scalar,keys)
                if step==1:assert torch.equal(d0,inputs['source_direction'])
                delta,qp=project(d0,matrix,bounds)
                save(arm/'active.pt',dict(state=snapshot(),scalar=scalar,curve=curve,linear_step=step,source_gradient=source_gradient,matrix=matrix,bounds=bounds,projected_direction=delta,current_metrics=current,counts=copy.deepcopy(counts)))
                assert qp['status']=='PASS','Fixed reserve QP did not converge'
                full=construct(r,before,raw,d0,d0,1.,keys);limit=r.fm.map_norm(before,full);zero,_=project(d0,torch.empty((0,d0.numel())),torch.empty(0));assert same(full,construct(r,before,raw,d0,zero,1.,keys)) and same(before,construct(r,before,raw,d0,delta,0.,keys))
                accepted=None
                scales=[old_curve[step-1]['scale']] if step<=31 else [2.**-j for j in range(8)]
                for scale in scales:
                    candidate=construct(r,before,raw,d0,delta,scale,keys);norm=r.fm.map_norm(before,candidate)
                    if norm>limit:candidate=match(r,before,candidate,limit);norm=r.fm.map_norm(before,candidate)
                    assert norm<=limit*1.002+1e-10;counts['candidate_checks']+=1
                    if step<=31:
                        old=old_curve[step-1];assert abs(current['source_loss']-old['source_loss_before'])<=1e-6 and abs(norm-old['actual_operator_norm'])<=1e-12,('RECONSTRUCTION_CONSTRUCTION_DRIFT',step)
                        assert qp['sweeps']==old['QP']['sweeps']
                        row=dict(step=step,scale=scale,source_loss_before=current['source_loss'],actual_operator_norm=norm,prefix_reconstruction=True,QP=qp);pending=row;accepted=(candidate,None,row);break
                    expert.load_state_dict(candidate);metric,_=values();row=dict(step=step,scale=scale,source_loss_before=current['source_loss'],source_loss_after=metric['source_loss'],source_gain=current['source_loss']-metric['source_loss'],actual_operator_norm=norm,max_support_regression=violation(anchor,metric['support_nll']),prefix_reconstruction=False,QP=qp)
                    write(arm/'candidates'/('step'+str(step)+'_'+str(scale)+'.json'),dict(summary=row,private_metrics=metric))
                    if first_feasible([row]):accepted=(candidate,metric,row);break
                if accepted is None:
                    status='NO_FEASIBLE_LEARNING_STEP';expert.load_state_dict(before);write(out/'STOPPING_POINT.json',dict(step=step,accepted_steps=len(curve),remaining_NA=160-len(curve),counts=counts));break
                state,metric,row=accepted;expert.load_state_dict(state);scalar=updated;curve.append(row)
                if metric is not None:current=metric
                counts['updates']+=1;counts['logical_transitions']+=1
                if step<=31:counts['reconstruction_updates']=counts.get('reconstruction_updates',0)+1
                save(arm/'active.pt',dict(state=snapshot(),scalar=scalar,curve=curve,counts=copy.deepcopy(counts)));write(out/'PROGRESS.json',dict(arm='RESERVE',step=step,prefix_validation_pending=step<=31,counts=counts));print(task['key'],'RESERVE',step,flush=True)
            assert pending is None and len(replay_differences)==31
    finally:hook.detach()
    save(arm/'final.pt',dict(state=snapshot()));assert same(torch.load(arm/'final.pt',map_location='cpu',weights_only=True)['state'],snapshot())
    assert all(x['max_support_regression']<=1e-4 for x in curve)
    write(arm/'TRAIN_COMPLETE.json',dict(status=status,accepted_steps=len(curve),remaining_NA=160-len(curve),curve=curve,final_metrics=current,source_gain=initial_source-current['source_loss'],save_load_exact=True,all_actual_outer_constraints_passed=True,inner=5e-5,outer=1e-4,reconstruction_updates=31))
    write(shared/('FROZEN_'+str(partition)+'.json'),dict(task=task['key'],arms=['SOURCE','RESERVE'],counts=counts))
    while not all((shared/('FROZEN_'+str(i)+'.json')).exists() for i in (0,1)):
        budget();assert not any((shared/str(i)/'FAILURE.json').exists() for i in (0,1));time.sleep(2)
    rows=read(shared/'EVAL_ROWS.json')+[task['rephrase']];assert len(rows)==len({x['query_id'] for x in rows})==32
    runtime.model.half();evaluation=r.s.prepare(runtime,bindings,template,rows,False);r.old.fp32(runtime)
    for row,_,raw,expanded,binding in evaluation:
        old=read(read(shared/'BASE_REUSE.json')[row['query_id']])['binding'];assert all(binding[k]==old[k] for k in ('image_sha256','question','reference','prompt_ids','generation'));write(out/'eval_bindings'/(row['query_id']+'.json'),dict(binding,preprocessing_seed=raw['preprocessing_seed']))
    write(out/'ENGINEERING_ADMISSION.json',dict(status='PASS',training_bindings=55,evaluation_bindings=32,SOURCE_resume=True,RESERVE_prefix_actual_validation=True,new_Base_answers=0))
    for name in ('SOURCE','RESERVE'):
        arm=out/name;state=torch.load(arm/'final.pt',map_location='cpu',weights_only=True)['state'];expert.load_state_dict(state);assert same(snapshot(),state);hook.attach()
        try:
            for index,(row,_,raw,expanded,binding) in enumerate(evaluation):
                budget();dest=arm/'outputs'/(row['query_id']+'.json');attempt=arm/'attempts'/dest.name;assert not dest.exists() and not attempt.exists();write(attempt,dict(attempt=1,time=time.time()));counts['generations']+=1
                answer,trace=r.s.fit.short.generate(runtime,hook,raw,expanded);write(dest,dict(binding=binding,raw_answer=answer.decoded_text,raw_token_ids=list(answer.raw_token_ids),generation_trace=trace,EOS=answer.raw_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(answer.raw_token_ids)>=1024));write(out/'PROGRESS.json',dict(arm=name,phase='GENERATION',completed=index+1,expected=32,counts=counts))
            assert check(arm/'outputs',[x['query_id'] for x in rows])==32
        finally:hook.detach()
    budget();write(out/'STUDY_COMPLETE.json',dict(status='COMPLETE',task=task['key'],arms=['SOURCE','RESERVE'],answers=64,counts=counts,semantic_review_complete=False))
