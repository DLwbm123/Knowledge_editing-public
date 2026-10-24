"""All qualified independent edits, seven frozen conditions, complete consumers."""
import os,json,time,copy,shutil
from pathlib import Path
import torch
from cache_v18 import PrefixCache
from path_v18 import flatten,proposal,construct,match,same
from qp_v18 import project
from math_v18 import constraints,violation,content_logits,kl
import be_v18
from m3bench_repro.editors.llava_runtime import seed_everything
ARMS=('SOURCE','CE','KL','CE_PROJ','RESERVE','BALANCEDIT','BALANCEDIT_CE')

def run(runtime,r,out,gpu,counts):
    root=out.parent;P=Path(os.environ['PARENT_RUN']);read=lambda p:json.loads(Path(p).read_text())
    def write(path,v):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,default=str)+'\n');tmp.replace(path)
    def save(path,v):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(v,tmp);tmp.replace(path)
    last_assets=[0.]
    def budget():
        if time.time()-last_assets[0]>60:
            assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<16*1024**3
            last_assets[0]=time.time()
        assert time.time()<float(os.environ['DEADLINE_EPOCH']) and not (root/'STOP').exists()
        assert counts['updates']<=900 and counts['backwards']<=50000 and counts['forwards']<=3000000 and counts['generations']<=175
        assert shutil.disk_usage(root).free>8*1024**3
    r.c.budget=budget;config=read(root/'DATA_ADMISSION.json');assert config['status']=='PASS' and len(config['tasks'])==2
    assert read(root/'FORMAL_CPU_ADMISSION.json')['status']=='PASS'
    pool=read(root/'SUPPORT_FROZEN.json');assert len(pool)==8
    template=read(P/'private/M13_SOURCE_PACKET.json')['original_tasks'][4];bindings=read(P/'private/EVAL_BINDINGS.json');reuse=read(root/'BASE_REUSE.json')
    def prepare(rows,training):
        budget();runtime.model.half();items=r.s.prepare(runtime,bindings,template,rows,training);r.old.fp32(runtime)
        for row,value,raw,expanded,binding in items:write(out/('training_bindings' if training else 'eval_bindings')/(row['query_id']+'.json'),dict(binding,preprocessing_seed=raw['preprocessing_seed']))
        return items
    rows=[]
    for c in pool:
        for form in ('{}.', 'The answer is {}.', 'It is {}.'):
            for text in [c['row']['reference'],c['wrong_choice']]:rows.append(dict(c['row'],query_id='S'+str(len(rows)),reference=form.format(text.rstrip('.'))))
    items=prepare(rows,True);batches=[];signatures=[]
    for row,value,raw,_,binding in items:
        kwargs,labels,tokens=value;prefix=raw['input_ids'][0].tolist();full=runtime._tokenize_prompt(raw['prompt']+' '+row['reference'])[0].tolist();assert full[:len(prefix)]==prefix;content=tuple(full[len(prefix):]);assert content and tuple(tokens[:len(content)])==content and tuple(tokens[len(content):])==(28705,2)
        labels=labels.clone();positions=torch.where(labels[0]!=-100)[0];labels[0,positions[len(content):]]=-100;batches.append(r.old.batch_on(runtime,(dict(kwargs,labels=labels),labels,content)))
        signatures.append({k:binding[k] for k in ('image_sha256','question','reference','prompt_ids','generation')})
    assert len(batches)==48
    initial=r.p.tr.TT4(template['seed'],8,8).state_dict();base=r.mm.canonicalize(initial)[0];expert=r.p.expert(initial,template['seed'],runtime.device);expert.load_state_dict(base);keys=list(base);params=dict(expert.named_parameters());assert set(keys)==set(params)
    hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert)
    def grad(loss):
        gs=torch.autograd.grad(loss,tuple(params[k] for k in keys),allow_unused=True);counts['backwards']+=1
        return torch.cat([(g if g is not None else torch.zeros_like(params[k])).detach().cpu().double().reshape(-1) for k,g in zip(keys,gs)])
    def nll(batch,gradient=False):
        budget();hook.set_teacher_routing(batch.labels)
        with torch.set_grad_enabled(gradient):loss=runtime.compute_loss(batch);assert torch.isfinite(loss);value=float(loss.detach());g=grad(loss) if gradient else None
        return value,g
    teacher_path=root/'TEACHER_CACHE.pt'
    if gpu==0:
        with (root/'TEACHER_ONCE').open('x'):pass
        anchor=[];teacher={};expert.load_state_dict(base);hook.attach()
        try:
            with PrefixCache(runtime,batches),torch.no_grad():
                for i,batch in enumerate(batches):
                    budget();hook.set_teacher_routing(batch.labels)
                    if i%2==0:
                        output=runtime.model(**batch.forward_kwargs());logits,targets=content_logits(output,batch);loss=torch.nn.functional.cross_entropy(logits,targets);assert abs(float(loss)-float(output.loss))<=1e-6;teacher[i]=logits.log_softmax(-1).cpu();anchor.append(float(loss));del output,logits,targets
                    else:anchor.append(nll(batch)[0])
        finally:hook.detach()
        save(teacher_path,dict(log_probabilities=teacher,anchor=anchor,signatures=signatures));write(root/'TEACHER_READY.json',dict(status='PASS',train_questions=8,expressions=24,teacher_forward=24,negative_anchor_forward=24,eval_teacher_access=False))
    while not (root/'TEACHER_READY.json').exists():budget();time.sleep(2)
    cache=torch.load(teacher_path,map_location='cpu',weights_only=True);assert cache['signatures']==signatures;anchor=cache['anchor'];assert len(anchor)==48
    for task in config['tasks'][gpu::2]:
        key=task['key'];folder=out/key;source=prepare([task['source']],True);source_batch=r.old.batch_on(runtime,source[0][1]);old=read(reuse[task['source']['query_id']])['binding'];assert all(old[k]==source[0][4][k] for k in ('image_sha256','question','reference','prompt_ids','generation'))
        for arm in ARMS[:5]:
            dest=folder/arm;dest.mkdir(parents=True,exist_ok=True)
            with (dest/'ONCE').open('x'):pass
            seed_everything(template['seed']);expert.load_state_dict(base);scalar=dict(step=0,second_moments=[0.,0.]);curve=[];status='COMPLETE';initial_source=None
            hook.attach()
            try:
                with PrefixCache(runtime,[source_batch]+batches):
                    for step in range(1,161):
                        before=r.old.snapshot(expert);source_loss,source_grad=nll(source_batch,True)
                        if initial_source is None:initial_source=source_loss
                        values=[];grads=[];total=source_grad.clone()
                        for i,batch in enumerate(batches):
                            needs_grad=arm in ('CE_PROJ','RESERVE') or arm=='CE' and i%2==0
                            if arm=='KL' and i%2==0:
                                budget();hook.set_teacher_routing(batch.labels);output=runtime.model(**batch.forward_kwargs());logits,targets=content_logits(output,batch);values.append(float(torch.nn.functional.cross_entropy(logits,targets).detach()));loss=kl(logits,cache['log_probabilities'][i]);total+=grad(loss)/24;del output,logits,targets
                            else:
                                value,g=nll(batch,needs_grad);values.append(value)
                                if arm in ('CE_PROJ','RESERVE'):grads.append(g)
                                elif arm=='CE' and i%2==0:total+=g/24
                        if step==1:
                            difference=max(abs(a-b) for a,b in zip(anchor,values));assert difference<=1e-6,'Initial shared support numerical binding drift'
                            write(dest/'ENGINEERING_ADMISSION.json',dict(status='PASS',shared_support_NLL_difference=difference,zero_constraint_same_path=True,content_expressions=48))
                        raw,updated,direction=proposal(r,before,total,scalar,keys);empty=torch.empty((0,direction.numel()));zero,zero_audit=project(direction,empty,torch.empty(0));full=construct(r,before,raw,direction,direction,1.,keys);assert same(full,construct(r,before,raw,direction,zero,1.,keys))
                        limit=r.fm.map_norm(before,full);accepted=None;qp=None
                        if arm in ('CE_PROJ','RESERVE'):
                            matrix,bounds=constraints(torch.stack(grads),torch.tensor(values,dtype=torch.float64),torch.tensor(anchor,dtype=torch.float64),arm=='RESERVE');delta,qp=project(direction,matrix,bounds)
                            save(dest/'active.pt',dict(state=before,scalar=scalar,curve=curve,linear_step=step,matrix=matrix,bounds=bounds,direction=direction,delta=delta,counts=copy.deepcopy(counts)))
                            assert qp['status']=='PASS','Fixed QP engineering nonconvergence'
                            for scale in [2.**-i for i in range(8)]:
                                candidate=construct(r,before,raw,direction,delta,scale,keys);norm=r.fm.map_norm(before,candidate)
                                if norm>limit:candidate=match(r,before,candidate,limit);norm=r.fm.map_norm(before,candidate)
                                if norm>limit:
                                    low=0.;high=1.;best={k:v.clone() for k,v in before.items()}
                                    for _ in range(32):
                                        mid=(low+high)/2;trial_state=r.mm.canonicalize({k:before[k]+mid*(candidate[k]-before[k]) for k in keys})[0]
                                        if r.fm.map_norm(before,trial_state)<=limit:low=mid;best=trial_state
                                        else:high=mid
                                    candidate=best;norm=r.fm.map_norm(before,candidate)
                                assert norm<=limit;expert.load_state_dict(candidate);after=nll(source_batch)[0];trial=[nll(b)[0] for b in batches];counts['candidate_checks']+=1
                                row=dict(step=step,scale=scale,source_loss_before=source_loss,source_loss_after=after,source_gain=source_loss-after,actual_norm=norm,max_support_regression=violation(anchor,trial,arm=='RESERVE'),QP=qp)
                                write(dest/'candidates'/('%03d_%s.json'%(step,scale)),dict(summary=row,support_nll=trial))
                                if row['max_support_regression']<=1e-4 and row['source_gain']>=1e-6 and norm>0:accepted=(candidate,row);break
                            if accepted is None:expert.load_state_dict(before);status='NO_FEASIBLE_LEARNING_STEP';break
                        else:
                            assert limit>0;accepted=(full,dict(step=step,source_loss_before=source_loss,support_nll_before=values,actual_norm=limit));counts['candidate_checks']+=1
                        expert.load_state_dict(accepted[0]);scalar=updated;curve.append(accepted[1]);counts['updates']+=1
                        save(dest/'active.pt',dict(state=r.old.snapshot(expert),scalar=scalar,curve=curve,counts=copy.deepcopy(counts)))
                        if step%10==0:write(out/'PROGRESS.json',dict(task=key,arm=arm,step=step,counts=counts,time=time.time()));print(key,arm,step,flush=True)
                    final_source=nll(source_batch)[0];final_nll=[nll(b)[0] for b in batches]
            finally:hook.detach()
            state=r.old.snapshot(expert);save(dest/'final.pt',dict(state=state));assert same(state,torch.load(dest/'final.pt',map_location=runtime.device,weights_only=True)['state'])
            write(dest/'TRAIN_COMPLETE.json',dict(status=status,accepted_steps=len(curve),remaining_NA=160-len(curve),curve=curve,final_source_loss=final_source,source_gain=initial_source-final_source,final_support_nll=final_nll,save_load_exact=True))
        for variant in (0,1):be_v18.train(runtime,r,template,task,source_batch,batches,variant,folder,counts,budget,save,write)
        write(folder/'FROZEN.json',dict(arms=ARMS,all_terminals_frozen=True,counts=counts))
        evaluation=prepare(read(root/'EVAL_ROWS.json')+[task[k] for k in ('source','cross','rephrase')],False);assert len(evaluation)==25
        for row,_,raw,expanded,binding in evaluation:
            prior=read(reuse[row['query_id']])['binding'];assert all(prior[k]==binding[k] for k in ('image_sha256','question','reference','prompt_ids','generation'))
        def emit(dest,active_hook,raw,expanded,binding):
            budget();assert not dest.exists();attempt=dest.parent.parent.parent/'attempts'/dest.parent.name/dest.name;write(attempt,dict(attempt=1,time=time.time()));counts['generations']+=1;answer,trace=r.s.fit.short.generate(runtime,active_hook,raw,expanded);tokens=list(answer.raw_token_ids);write(dest,dict(binding=binding,raw_answer=answer.decoded_text,raw_token_ids=tokens,generation_trace=trace,EOS=bool(tokens) and tokens[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(tokens)>=1024))
        for arm in ARMS[:5]:
            expert.load_state_dict(torch.load(folder/arm/'final.pt',map_location=runtime.device,weights_only=True)['state']);hook.attach()
            try:
                for row,_,raw,expanded,binding in evaluation:emit(folder/'outputs'/arm/(row['query_id']+'.json'),hook,raw,expanded,binding)
            finally:hook.detach()
            write(folder/arm/'ROUTING_SCOPE.json',dict(primary='FORCED_SINGLE_EXPERT',natural_routing='NOT_EVALUATED_NOT_AN_END_TO_END_ROUTED_CLAIM',generation_trace_preserved=True))
        for variant in (0,1):be_v18.evaluate(runtime,r,template,task,variant,folder,evaluation,counts,budget,emit,write)
        outputs=list((folder/'outputs').glob('*/*.json'));assert len(outputs)==175
        write(folder/'MODEL_CONSUMERS_COMPLETE.json',dict(status='PASS',answers=175,arms=ARMS,save_load=True,route_scope_recorded=True,counts=counts))
    write(out/'STUDY_COMPLETE.json',dict(status='COMPLETE',counts=counts,seconds=time.time()-float(os.environ['FORMAL_START'])))
