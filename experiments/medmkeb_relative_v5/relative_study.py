"""Finite independent factual-source edits with a shared Base-qualified retention construct."""
import os,json,time,copy
from pathlib import Path
import torch
import training_math
import owner5_semantic_loss as semantic
from prefix_cache import PrefixCache
from m3bench_repro.editors.llava_runtime import seed_everything
from separated_step import step,selfcheck
from fresh import fisher as build_fisher,source_step
import equal_baseline

def run(runtime,r,out,partition,counts):
    shared=out.parent;parent=Path(os.environ['PARENT_RUN']);original_budget=r.c.budget
    def read(path):return json.loads(Path(path).read_text())
    def write(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,default=str)+'\n');tmp.replace(path)
    def save(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)
    def budget():
        original_budget()
        import shutil
        assert shutil.disk_usage(out).free>8*1024**3
        assert time.time()<float(os.environ['DEADLINE_EPOCH'])
        assert counts['updates']<=1060 and counts['backwards']<=10000 and counts['operator_backwards']<=1100 and counts['generations']<=256 and counts['forwards']<=400060
        assert not (shared/'STOP').exists()
        assert not any((shared/str(i)/'FAILURE.json').exists() for i in (0,1)),'Other partition failed; preserve state'
    r.c.budget=budget
    assert read(shared/'IMPLEMENTATION_ADMISSION.json')['status']=='PASS'
    write(out/'STUDY_SELF_CHECK.json',selfcheck())
    config=read(shared/'DATA_ADMISSION.json');assert config['protocol']=='V5_RELATIVE_SIGN_INVARIANT';tasks=config['tasks'];assert len(tasks)==2
    assert read(shared/'PREFILL_CHECK_COMPLETE.json')['status']=='PASS'
    template=read(parent/'private/M13_SOURCE_PACKET.json')['original_tasks'][4]
    bindings=read(parent/'private/EVAL_BINDINGS.json')
    def prepare(rows,training):
        budget();runtime.model.half();items=r.s.prepare(runtime,bindings,template,rows,training);r.old.fp32(runtime)
        for row,value,raw,expanded,binding in items:
            write(out/('train_bindings' if training else 'eval_bindings')/(row['query_id']+'.json'),dict(binding,preprocessing_seed=raw['preprocessing_seed']))
        return items
    def emit(dest,active_hook,raw,expanded,binding):
        budget();assert not dest.exists()
        write(dest.parent.parent.parent/'attempts'/dest.parent.name/dest.name,dict(attempt=1,time=time.time()))
        counts['generations']+=1
        answer,trace=r.s.fit.short.generate(runtime,active_hook,raw,expanded)
        write(dest,dict(binding=binding,raw_answer=answer.decoded_text,raw_token_ids=list(answer.raw_token_ids),generation_trace=trace,EOS=answer.raw_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(answer.raw_token_ids)>=1024))
    initial=r.p.tr.TT4(template['seed'],8,8).state_dict();base=r.mm.canonicalize(initial)[0]
    expected=r.p.load_state(parent/'private/candidates/C/5.pt')['states']['BASE'];assert all(torch.equal(initial[k],expected[k]) for k in initial)
    expert=r.p.expert(initial,template['seed'],runtime.device);expert.load_state_dict(base)
    assert not runtime.get_module(r.c.LAYER)._forward_hooks
    hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert)
    # New training-side Fisher only. No inherited task data/curves/calibration.
    if partition==0:
        with (shared/'FISHER_ONCE').open('x'):pass
        basis=prepare(config['fisher_rows'],True);basis_batches=[r.old.batch_on(runtime,v[1]) for v in basis]
        expert.load_state_dict(base);hook.attach()
        try:new_cache=build_fisher(runtime,r,expert,hook,basis_batches,counts,budget)
        finally:hook.detach()
        save(shared/'NEW_FISHER.pt',new_cache)
        write(shared/'FISHER_COMPLETE.json',dict(status='PASS',questions=8,probes=256,zero_TT_pullback=True,train_only=True,counts=counts))
        del basis,basis_batches,new_cache
    while not (shared/'FISHER_COMPLETE.json').exists():budget();time.sleep(2)
    cache={k:v.to(runtime.device) if isinstance(v,torch.Tensor) else v for k,v in r.p.load_state(shared/'NEW_FISHER.pt').items()}
    targets=coefficient=None
    fisher=partition==1;prefix='ON' if fisher else 'OFF';all_arms=[]
    # All49 Base/rephrase outputs are immutable V4 consumers reused by input identity.
    try:
        for task in tasks:
            targets=coefficient=None
            key=task['key'];folder=out/key;source=prepare([task['source']],True);source_batch=r.old.batch_on(runtime,source[0][1])
            reused=read(read(shared/'BASE_REUSE.json')[task['source']['query_id']])['binding']
            for identity in ('question','reference','image_sha256','prompt_ids','generation'):assert source[0][4][identity]==reused[identity],'Edit Base input identity drift'
            assert reused['runtime']['actual_precision']=='MATH_FP32_FROZEN_FP16_PREFILL'
            answers=[]
            for form in ('{}.', 'The answer is {}.', 'It is {}.'):
                answers.extend([form.format(task['retention_correct'].rstrip('.')),form.format(task['retention_wrong'].rstrip('.'))])
            candidates=prepare([dict(task['retention'],query_id=key+'_sem_'+str(i),reference=text) for i,text in enumerate(answers)],True)
            sem_batches=[]
            for row,value,raw,_,binding in candidates:
                kwargs,labels,tokens=value;start=raw['input_ids'][0].tolist();full=runtime._tokenize_prompt(raw['prompt']+' '+row['reference'])[0].tolist()
                assert full[:len(start)]==start
                content=tuple(full[len(start):]);assert content and tuple(tokens[:len(content)])==content and tuple(tokens[len(content):])==(28705,2)
                positions=torch.where(labels[0]!=-100)[0];labels=labels.clone();labels[0,positions[len(content):]]=-100
                sem_batches.append(r.old.batch_on(runtime,(dict(kwargs,labels=labels),labels,content)))
                write(folder/'semantic_bindings'/(row['query_id']+'.json'),dict(binding,content_token_ids=list(content)))
            def score():
                values=[]
                with torch.no_grad():
                    for batch in sem_batches:hook.set_teacher_routing(batch.labels);values.append(float(runtime.compute_loss(batch)))
                return semantic.summary(values)
            expert.load_state_dict(base);hook.attach()
            try:
                with PrefixCache(runtime,[source_batch]+sem_batches):
                    s0=score()['score'];proxy=dict(score=s0,positive=s0>0,content_tokens=[list(b.target_token_ids) for b in sem_batches]);write(shared/('PROXY_'+key+'_'+str(partition)+'.json'),proxy)
                    while not all((shared/('PROXY_'+key+'_'+str(i)+'.json')).exists() for i in (0,1)):budget();time.sleep(2)
                    proxies=[read(shared/('PROXY_'+key+'_'+str(i)+'.json')) for i in (0,1)]
                    assert abs(proxies[0]['score']-proxies[1]['score'])<=1e-6 and proxies[0]['content_tokens']==proxies[1]['content_tokens'],'Proxy input identity drift'
                    if not all(__import__('math').isfinite(v['score']) for v in proxies):
                        write(out/'STUDY_COMPLETE.json',dict(status='NONFINITE_PROXY_ADMISSION_FAILED',counts=counts));return
                    for mode in ('NONE','CE','SEM'):
                        reference_arm=partition==0 and mode=='NONE'
                        if not reference_arm:
                            while not (shared/('REFERENCE_'+key+'.json')).exists():budget();time.sleep(2)
                            reference=read(shared/('REFERENCE_'+key+'.json'));targets=reference['targets'];coefficient=reference['coefficient'];assert len(targets)==160
                        reference_targets=[]
                        arm=prefix+'_'+mode
                        folder.mkdir(exist_ok=True)
                        with (folder/('ONCE_'+arm)).open('x'):pass
                        seed_everything(template['seed']);expert.load_state_dict(base);scalar=dict(step=0,second_moments=[0.,0.]);curve=[]
                        for clock in range(1,161):
                            budget();before=r.old.snapshot(expert);old_scalar=copy.deepcopy(scalar);expert.zero_grad(set_to_none=True)
                            hook.set_teacher_routing(source_batch.labels);loss=runtime.compute_loss(source_batch);assert torch.isfinite(loss);loss.backward();counts['backwards']+=1
                            edit={k:v.grad.detach().clone() for k,v in expert.named_parameters()};contrast=score();hinge=max(0.,s0-contrast['score'])
                            expert.zero_grad(set_to_none=True);active=mode=='CE' or mode=='SEM' and hinge>0
                            if active:
                                for i in ((0,2,4) if mode=='CE' else range(6)):
                                    batch=sem_batches[i];hook.set_teacher_routing(batch.labels);ret=runtime.compute_loss(batch)
                                    ((1/3 if mode=='CE' else semantic.GRADIENT_WEIGHTS[i])*ret).backward();counts['backwards']+=1
                            retention={k:v.grad.detach().clone() if v.grad is not None else torch.zeros_like(v) for k,v in expert.named_parameters()}
                            if reference_arm:
                                selected,scalar,target=source_step(r,before,edit,scalar);reference_targets.append(target)
                                audit=dict(status='CURRENT_TASK_SOURCE_ONLY_REFERENCE',operator_step_norm=target)
                                if clock==1:
                                    expert.load_state_dict(selected);hook.set_teacher_routing(source_batch.labels)
                                    with torch.no_grad():after=float(runtime.compute_loss(source_batch))
                                    gain=float(loss.detach())-after;penalty=float(r.mm.operator_penalty(selected,cache))
                                    coefficient=training_math.coefficient(gain,penalty)
                                    write(shared/('CALIBRATION_'+key+'.json'),dict(source_gain=gain,penalty=penalty,coefficient=coefficient,new_training_only_Fisher=True))
                            else:
                                selected,scalar,audit=step(r,before,edit,retention,cache,coefficient,scalar,targets[clock-1],fisher);counts['operator_backwards']+=int(fisher)
                            if clock==1 and mode=='NONE':
                                old,_,_=training_math.step(r,before,edit,cache,coefficient if fisher else 0.,old_scalar,(target if reference_arm else targets[0]),'M13P' if fisher else 'M1');counts['operator_backwards']+=int(fisher)
                                for k in old:torch.testing.assert_close(old[k],selected[k],rtol=1e-5,atol=1e-7)
                            expert.load_state_dict(selected);counts['updates']+=1
                            curve.append(dict(step=clock,source_loss=float(loss.detach()),score=contrast,hinge=hinge,retention_active=active,proposal=audit,operator_step_norm=r.fm.map_norm(before,selected)))
                            save(folder/('active_'+arm+'.pt'),dict(state={k:v.detach().cpu() for k,v in selected.items()},scalar=scalar,step=clock,counts=copy.deepcopy(counts),curve=curve))
                            if clock%20==0:write(out/'STUDY_PROGRESS.json',dict(task=key,arm=arm,step=clock,counts=counts));print(key,arm,clock,flush=True)
                        final=r.old.snapshot(expert);path=folder/'finals'/(arm+'.pt');save(path,dict(state={k:v.cpu() for k,v in final.items()}))
                        saved=r.p.load_state(path)['state'];assert all(torch.equal(saved[k],v.cpu()) for k,v in final.items())
                        write(folder/(arm+'_TRAIN.json'),dict(arm=arm,status='FROZEN',steps=160,curve=curve,save_load_exact=True,final_semantic=score(),source_only=True,first_none_step_parity=True))
                        all_arms.append(dict(task=key,arm=arm))
                        if reference_arm:
                            write(shared/('REFERENCE_'+key+'.json'),dict(status='PASS',targets=reference_targets,coefficient=coefficient,source_only=True,steps=160,formal_arm='OFF_NONE'))
            finally:hook.detach()
            baseline_arm=equal_baseline.train(runtime,r,template,task,source_batch,sem_batches,partition,folder,counts,budget,save,write)
            all_arms.append(dict(task=key,arm=baseline_arm))
        write(shared/('FROZEN_'+str(partition)+'.json'),dict(arms=all_arms,counts=counts,check_used_for_training=False))
        while not all((shared/('FROZEN_'+str(i)+'.json')).exists() for i in (0,1)):budget();time.sleep(2)
        for task in tasks:
            rows=read(shared/'EVAL_ROWS.json')+[task['rephrase']];assert len(rows)<=261 and len({v['query_id'] for v in rows})==len(rows)
            evaluation=prepare(rows,False);reuse=read(shared/'BASE_REUSE.json')
            assert task['rephrase']['query_id'] in reuse
            for row,_,raw,expanded,binding in evaluation:
                prior=read(reuse[row['query_id']])['binding'];assert all(binding[k]==prior[k] for k in ('question','reference','image_sha256','prompt_ids','generation')),'Base reuse identity drift'
                assert prior['runtime']['actual_precision']=='MATH_FP32_FROZEN_FP16_PREFILL'
            write(out/task['key']/'BASE_REUSE_ADMISSION.json',dict(matched=len(rows)))
            for arm in (prefix+'_NONE',prefix+'_CE',prefix+'_SEM'):
                expert.load_state_dict(r.p.load_state(out/task['key']/'finals'/(arm+'.pt'))['state']);hook.attach()
                try:
                    for row,_,raw,expanded,binding in evaluation:
                        budget();dest=out/task['key']/'outputs'/arm/(row['query_id']+'.json');assert not dest.exists()
                        write(out/task['key']/'attempts'/arm/(row['query_id']+'.json'),dict(attempt=1,time=time.time()));counts['generations']+=1
                        answer,trace=r.s.fit.short.generate(runtime,hook,raw,expanded)
                        write(dest,dict(binding=binding,raw_answer=answer.decoded_text,raw_token_ids=list(answer.raw_token_ids),generation_trace=trace,EOS=answer.raw_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(answer.raw_token_ids)>=1024))
                finally:hook.detach()
            equal_baseline.evaluate(runtime,r,template,task,partition,out/task['key'],evaluation,counts,budget,emit,write)
        assert counts['updates']==530*len(tasks)
        write(out/'STUDY_COMPLETE.json',dict(status='COMPLETE',arms=all_arms,evaluation_per_arm=len(rows),counts=counts,check_used_for_selection=False))
    finally:hook.detach();r.c.budget=original_budget
