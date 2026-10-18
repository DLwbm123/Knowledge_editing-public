"""Six frozen arms: Fisher off/on crossed with none/content-CE/semantic retention."""
import os,json,time,copy
from pathlib import Path
import torch
import trajectory,training_math
import owner5_semantic_loss as semantic
from prefix_cache import PrefixCache
from m3bench_repro.editors.llava_runtime import seed_everything

def selfcheck():
    semantic.selfcheck()
    x=torch.arange(1.,7.,dtype=torch.float64,requires_grad=True)
    (sum(x[i] for i in (0,2,4))/3).backward()
    assert torch.equal(x.grad,torch.tensor([1/3,0,1/3,0,1/3,0],dtype=x.dtype))
    assert [f'{f}_{m}' for f in ('OFF','ON') for m in ('NONE','CE','SEM')]==['OFF_NONE','OFF_CE','OFF_SEM','ON_NONE','ON_CE','ON_SEM']
    return dict(status='PASS',semantic_hinge_gradient=True,content_ce_gradient=True,frozen_arms=6)

def step(r,before,edit,retention,cache,coefficient,scalar,target,fisher):
    params={k:v.detach().clone().requires_grad_(True) for k,v in before.items()}
    extra={k:torch.zeros_like(v) for k,v in params.items()};penalty=None;balance=direction=None
    if fisher:
        penalty=r.mm.operator_penalty(params,cache)
        extra=dict(zip(params,torch.autograd.grad(coefficient*penalty,tuple(params.values()))))
        from balanced_fisher import cap_gradient,remove_opposing
        extra,balance=cap_gradient(edit,extra,r.mm.GROUPS)
        extra,direction=remove_opposing(edit,extra,r.mm.GROUPS)
    for k,v in params.items():v.grad=edit[k].detach()+extra[k].detach()+retention[k].detach()
    norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.)
    assert torch.isfinite(norm) and all(torch.isfinite(v.grad).all() for v in params.values())
    proposed,updated=r.mm.canonical_rms_step(before,{k:v.grad.detach() for k,v in params.items()},copy.deepcopy(scalar))
    if r.fm.map_norm(before,proposed)<=1e-14:selected={k:v.clone() for k,v in before.items()};matching=dict(status='ZERO_DIRECTION')
    else:
        proposed,matching=r.match_proposal(before,proposed,target);selected=r.mm.canonicalize(proposed)[0]
        assert abs(r.fm.map_norm(before,selected)-target)<=1e-10+.002*target
    return selected,updated,dict(gradient_norm=float(norm),matching=matching,balance=balance,anti_parallel=direction,penalty_before=float(penalty.detach()) if penalty is not None else None)

def run(runtime,r,out,partition,counts):
    shared=out.parent;parent=Path(os.environ['PARENT_RUN']);reference=Path(os.environ['REFERENCE_RUN']);original_budget=r.c.budget
    def read(path):return json.loads(Path(path).read_text())
    def write(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,default=str)+'\n');temp.replace(path)
    def save(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)
    def budget():
        original_budget()
        assert not (shared/'STOP').exists()
        assert not any((shared/str(i)/'FAILURE.json').exists() for i in (0,1)),'Other partition failed; preserve state'
    r.c.budget=budget
    assert read(shared/'IMPLEMENTATION_ADMISSION.json')['status']=='PASS'
    write(out/'STUDY_SELF_CHECK.json',selfcheck())
    task=read(parent/'private/M13_SOURCE_PACKET.json')['original_tasks'][4]
    source=trajectory.source_rows(read(parent/'private/M13_SOURCE_PACKET.json'),5)
    bindings=read(parent/'private/EVAL_BINDINGS.json');construct=read(shared/'CONSTRUCT.json');assert len(construct)==2
    assert construct[0]['query_id']=='positive_construct' and construct[1]['query_id']=='negative_construct'
    def prepare(rows,training):
        runtime.model.half();items=r.s.prepare(runtime,bindings,task,rows,training);r.old.fp32(runtime)
        for row,value,raw,expanded,binding in items:
            write(out/('train_bindings' if training else 'eval_bindings')/(row['query_id']+'.json'),dict(binding,preprocessing_seed=raw['preprocessing_seed']))
        return items
    items=prepare(source+construct,True);batches=[r.old.batch_on(runtime,v[1]) for v in items]
    candidates=prepare([dict(construct[1],query_id='semantic_'+str(i),reference=text) for i,text in enumerate(semantic.ANSWERS)],True)
    semantic_batches=[]
    for row,value,raw,_,binding in candidates:
        kwargs,labels,tokens=value;prefix=raw['input_ids'][0].tolist();full=runtime._tokenize_prompt(raw['prompt']+' '+row['reference'])[0].tolist()
        assert full[:len(prefix)]==prefix
        content=tuple(full[len(prefix):]);assert content and tuple(tokens[:len(content)])==content and tuple(tokens[len(content):])==(28705,2)
        positions=torch.where(labels[0]!=-100)[0];labels=labels.clone();labels[0,positions[len(content):]]=-100
        semantic_batches.append(r.old.batch_on(runtime,(dict(kwargs,labels=labels),labels,content)))
        write(out/'semantic_bindings'/(row['query_id']+'.json'),dict(binding,content_token_ids=list(content),labels=labels.tolist(),prediction_positions=(positions[:len(content)]-1).tolist()))
    initial=r.p.tr.TT4(task['seed'],8,8).state_dict();base=r.mm.canonicalize(initial)[0]
    expected=r.p.load_state(parent/'private/candidates/C/5.pt')['states']['BASE'];assert all(torch.equal(initial[k],expected[k]) for k in initial)
    expert=r.p.expert(initial,task['seed'],runtime.device);expert.load_state_dict(base)
    assert not runtime.get_module(r.c.LAYER)._forward_hooks
    hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert);hook.attach()
    targets=[v['operator_step_norm'] for v in read(parent/'private/results/C/5.json')['curve']];assert len(targets)==160
    coefficient=read(parent/'private/calibration146/5.json')['coefficient']
    cache={k:v.to(runtime.device) if isinstance(v,torch.Tensor) else v for k,v in r.p.load_state(parent/'private/OPERATOR_FISHER.pt').items()}
    fisher=partition==1;prefix='ON' if fisher else 'OFF';curve_files=[]
    def score():
        values=[]
        with torch.no_grad():
            for batch in semantic_batches:hook.set_teacher_routing(batch.labels);values.append(float(runtime.compute_loss(batch)))
        return semantic.summary(values)
    try:
        with PrefixCache(runtime,batches+semantic_batches):
            s0=score()['score'];write(shared/('PROXY_'+str(partition)+'.json'),dict(score=s0,positive=s0>0,content_tokens=[list(b.target_token_ids) for b in semantic_batches]))
            while not all((shared/('PROXY_'+str(i)+'.json')).exists() for i in (0,1)):budget();time.sleep(2)
            proxies=[read(shared/('PROXY_'+str(i)+'.json')) for i in (0,1)]
            assert abs(proxies[0]['score']-proxies[1]['score'])<=1e-6 and proxies[0]['content_tokens']==proxies[1]['content_tokens'],'Proxy partition identity drift'
            if not all(v['positive'] for v in proxies):
                write(out/'STUDY_COMPLETE.json',dict(status='PROXY_ADMISSION_FAILED',counts=counts));return
            for mode in ('NONE','CE','SEM'):
                arm=prefix+'_'+mode
                with (out/('ONCE_'+arm)).open('x'):pass
                seed_everything(task['seed']);expert.load_state_dict(base);scalar=dict(step=0,second_moments=[0.,0.]);curve=[]
                for clock in range(1,161):
                    budget();before=r.old.snapshot(expert);old_scalar=copy.deepcopy(scalar);expert.zero_grad(set_to_none=True);losses=[]
                    for index,weight in ((0,.25),(1+(clock-1)%4,.25),(5,.5)):
                        hook.set_teacher_routing(batches[index].labels);loss=runtime.compute_loss(batches[index]);assert torch.isfinite(loss)
                        (weight*loss).backward();counts['backwards']+=1;losses.append(float(loss.detach()))
                    edit={k:v.grad.detach().clone() for k,v in expert.named_parameters()};contrast=score();hinge=max(0.,s0-contrast['score'])
                    expert.zero_grad(set_to_none=True);active=mode=='CE' or mode=='SEM' and hinge>0
                    if active:
                        for i in ((0,2,4) if mode=='CE' else range(6)):
                            batch=semantic_batches[i];hook.set_teacher_routing(batch.labels);loss=runtime.compute_loss(batch)
                            ((1/3 if mode=='CE' else semantic.GRADIENT_WEIGHTS[i])*loss).backward();counts['backwards']+=1
                    retention={k:v.grad.detach().clone() if v.grad is not None else torch.zeros_like(v) for k,v in expert.named_parameters()}
                    selected,scalar,audit=step(r,before,edit,retention,cache,coefficient,scalar,targets[clock-1],fisher);counts['operator_backwards']+=int(fisher)
                    if clock==1 and mode=='NONE':
                        old,_,_=training_math.step(r,before,edit,cache,coefficient if fisher else 0.,old_scalar,targets[0],'M13P' if fisher else 'M1');counts['operator_backwards']+=int(fisher)
                        assert all(torch.equal(old[k],selected[k]) for k in old),'Zero-retention implementation parity failed'
                    expert.load_state_dict(selected);counts['updates']+=1
                    curve.append(dict(step=clock,losses=losses,score=contrast,hinge=hinge,retention_active=active,proposal=audit,operator_step_norm=r.fm.map_norm(before,selected)))
                    save(out/('active_'+arm+'.pt'),dict(state={k:v.detach().cpu() for k,v in selected.items()},scalar=scalar,step=clock,counts=copy.deepcopy(counts),curve=curve))
                    if clock%20==0:write(out/'STUDY_PROGRESS.json',dict(arm=arm,step=clock,counts=counts));print(arm,clock,flush=True)
                final=r.old.snapshot(expert);path=out/'finals'/(arm+'.pt');save(path,dict(state={k:v.cpu() for k,v in final.items()}))
                saved=r.p.load_state(path)['state'];assert all(torch.equal(saved[k],v.cpu()) for k,v in final.items())
                if arm=='ON_NONE':
                    old=r.p.load_state(reference/'CONTROL.pt')['state'];assert all(torch.equal(saved[k],old[k]) for k in saved),'ON_NONE endpoint parity failed'
                    previous=read(reference/'CONTROL_TRAIN.json')['curve'];assert all(a['losses']==b['losses'] for a,b in zip(curve,previous))
                    write(out/'ON_NONE_PARITY.json',dict(endpoint_tensors_exact=True,editing_losses_exact=True))
                record=dict(arm=arm,status='FROZEN',steps=160,curve=curve,save_load_exact=True,final_semantic=score(),source_only=True)
                write(out/(arm+'_TRAIN.json'),record);curve_files.append(arm)
        write(shared/('FROZEN_'+str(partition)+'.json'),dict(arms=curve_files,counts=counts,check_used_for_training=False))
        hook.detach()
        while not all((shared/('FROZEN_'+str(i)+'.json')).exists() for i in (0,1)):budget();time.sleep(2)
        assert (shared/'1/ON_NONE_PARITY.json').exists()
        rows=read(shared/'EVAL_ROWS.json');assert 1<=len(rows)<=36 and len({v['query_id'] for v in rows})==len(rows)
        evaluation=prepare(rows,False)
        # Existing Base consumers are reused only after exact current input/precision identity checks.
        reuse=read(shared/'BASE_REUSE.json')
        for row,_,raw,expanded,binding in evaluation:
            previous=read(reuse[row['query_id']]);prior=previous['binding']
            assert all(binding[k]==prior[k] for k in ('question','reference','image_sha256','prompt_ids','generation')),'Base reuse input drift'
            assert prior['runtime']['actual_precision']=='MATH_FP32_FROZEN_FP16_PREFILL'
        write(out/'BASE_REUSE_ADMISSION.json',dict(matched=len(rows),precision='MATH_FP32_FROZEN_FP16_PREFILL'))
        for arm in curve_files:
            expert.load_state_dict(r.p.load_state(out/'finals'/(arm+'.pt'))['state']);hook.attach()
            try:
                for row,_,raw,expanded,binding in evaluation:
                    budget();dest=out/'outputs'/arm/(row['query_id']+'.json');assert not dest.exists()
                    write(out/'attempts'/arm/(row['query_id']+'.json'),dict(attempt=1,time=time.time()));counts['generations']+=1
                    answer,trace=r.s.fit.short.generate(runtime,hook,raw,expanded)
                    write(dest,dict(binding=binding,raw_answer=answer.decoded_text,raw_token_ids=list(answer.raw_token_ids),generation_trace=trace,EOS=answer.raw_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(answer.raw_token_ids)>=1024))
            finally:hook.detach()
        assert counts['updates']==480
        write(out/'STUDY_COMPLETE.json',dict(status='COMPLETE',arms=curve_files,evaluation_per_arm=len(rows),counts=counts,check_used_for_selection=False))
    finally:
        hook.detach();r.c.budget=original_budget
