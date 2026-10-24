"""Frozen multi-support mechanism development; no evaluation-driven decisions."""
import os,json,time,copy,math
from pathlib import Path
import torch
from prefix_cache import PrefixCache
from fresh import source_step
from m3bench_repro.editors.llava_runtime import seed_everything

def selfcheck():
    before=[0.,0.];after=[1.,-0.5]
    assert sum(after)>sum(before) and after[1]<before[1]
    scales=[2.**-k for k in range(8)]+[0.]
    assert len(scales)==9 and scales[-1]==0. and all(0<=x<=1 for x in scales)
    x=torch.tensor([1.,2.,3.,2.],requires_grad=True)
    initial=torch.tensor([1.,0.]);margins=x[1::2]-x[::2]
    torch.relu(initial-margins).mean().backward()
    assert torch.equal(x.grad,torch.tensor([0.,0.,.5,-.5]))
    return dict(status='PASS',mean_can_mask_pair=True,pair_gradient=True,finite_backtrack=True)

def run(runtime,r,out,partition,counts):
    shared=out.parent;parent=Path(os.environ['PARENT_RUN'])
    read=lambda p:json.loads(Path(p).read_text())
    def write(p,v):
        p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,default=str)+'\n');tmp.replace(p)
    def save(p,v):
        p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');torch.save(v,tmp);tmp.replace(p)
    def budget():
        import shutil
        assert time.time()<float(os.environ['DEADLINE_EPOCH']) and not (shared/'STOP').exists()
        assert not any((shared/str(i)/'FAILURE.json').exists() for i in (0,1))
        assert counts['updates']<=(960 if partition==0 else 640) and counts['backwards']<=20000 and counts['forwards']<=220000 and counts['generations']<=(192 if partition==0 else 128)
        assert shutil.disk_usage(shared).free>8*1024**3
    r.c.budget=budget;write(out/'CPU_ADMISSION.json',selfcheck())
    config=read(shared/'DATA_ADMISSION.json');pool=read(shared/'SUPPORT_FROZEN.json');assert len(pool)==9
    template=read(parent/'private/M13_SOURCE_PACKET.json')['original_tasks'][4];bindings=read(parent/'private/EVAL_BINDINGS.json')
    def prepare(rows,training):
        budget();runtime.model.half();items=r.s.prepare(runtime,bindings,template,rows,training);r.old.fp32(runtime)
        for row,value,raw,expanded,binding in items:write(out/('train_bindings' if training else 'eval_bindings')/(row['query_id']+'.json'),dict(binding,preprocessing_seed=raw['preprocessing_seed']))
        return items
    initial=r.p.tr.TT4(template['seed'],8,8).state_dict();base=r.mm.canonicalize(initial)[0];expert=r.p.expert(initial,template['seed'],runtime.device);expert.load_state_dict(base)
    hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert)
    rows=[]
    for c in pool:
        for form in ('{}.', 'The answer is {}.', 'It is {}.'):
            for text in (c['row']['reference'],c['wrong_choice']):rows.append(dict(c['row'],query_id='S'+str(len(rows)),reference=form.format(text.rstrip('.'))))
    items=prepare(rows,True);batches=[]
    for row,value,raw,expanded,binding in items:
        kwargs,labels,tokens=value;start=raw['input_ids'][0].tolist();full=runtime._tokenize_prompt(raw['prompt']+' '+row['reference'])[0].tolist();assert full[:len(start)]==start
        content=tuple(full[len(start):]);assert tuple(tokens[:len(content)])==content and tuple(tokens[len(content):])==(28705,2)
        positions=torch.where(labels[0]!=-100)[0];labels=labels.clone();labels[0,positions[len(content):]]=-100
        batches.append(r.old.batch_on(runtime,(dict(kwargs,labels=labels),labels,content)))
        write(out/'semantic_bindings'/(row['query_id']+'.json'),dict(binding,content_token_ids=list(content)))
    assert len(batches)==54
    def score():
        values=[]
        with torch.no_grad():
            for batch in batches:budget();hook.set_teacher_routing(batch.labels);values.append(float(runtime.compute_loss(batch)))
        assert all(math.isfinite(x) for x in values)
        return dict(nll=values,margins=[values[j+1]-values[j] for j in range(0,54,2)])
    expert.load_state_dict(base);hook.attach()
    try:
        with PrefixCache(runtime,batches):s0=score()
    finally:hook.detach()
    write(shared/f'POOL_{partition}.json',s0)
    while not all((shared/f'POOL_{i}.json').exists() for i in (0,1)):budget();time.sleep(2)
    other=read(shared/f'POOL_{1-partition}.json');assert max(abs(a-b) for a,b in zip(s0['nll'],other['nll']))<=1e-6
    own=('NONE','POOL_MEAN','NORM_REPLAY') if partition==0 else ('POOL_PAIR','LOOKAHEAD');arms=[]
    for task in config['tasks']:
        key=task['key'];folder=out/key;source=prepare([task['source']],True);source_batch=r.old.batch_on(runtime,source[0][1]);source_id=task['source']['query_id'];prior=read(read(shared/'BASE_REUSE.json')[source_id])['binding'];assert all(prior[k]==source[0][4][k] for k in ('image_sha256','question','reference','prompt_ids','generation'))
        for arm in own:
            dest=folder/(arm+'_TRAIN.json');assert not dest.exists();seed_everything(template['seed']);expert.load_state_dict(base);scalar=dict(step=0,second_moments=[0.,0.]);curve=[]
            if arm not in ('NONE','LOOKAHEAD'):
                while not (shared/f'REFERENCE_{key}.json').exists():budget();time.sleep(2)
                targets=read(shared/f'REFERENCE_{key}.json')['targets']
            if arm=='NORM_REPLAY':
                while not (shared/f'LOOKAHEAD_{key}.json').exists():budget();time.sleep(2)
                replay=read(shared/f'LOOKAHEAD_{key}.json')['norms']
            hook.attach()
            try:
                with PrefixCache(runtime,[source_batch]+batches):
                    for clock in range(160):
                        budget();before=r.old.snapshot(expert);expert.zero_grad(set_to_none=True);hook.set_teacher_routing(source_batch.labels);loss=runtime.compute_loss(source_batch);assert torch.isfinite(loss);loss.backward();counts['backwards']+=1
                        edit={k:v.grad.detach().clone() for k,v in expert.named_parameters()};current=score();expert.zero_grad(set_to_none=True)
                        deficits=[a-b for a,b in zip(s0['margins'],current['margins'])];active=[]
                        if arm=='POOL_MEAN' and sum(deficits)>0:active=list(range(27))
                        elif arm=='POOL_PAIR':active=[j for j,d in enumerate(deficits) if d>0]
                        for j in active:
                            for index,sign in ((2*j,1.),(2*j+1,-1.)):
                                budget();batch=batches[index];hook.set_teacher_routing(batch.labels);ret=runtime.compute_loss(batch);(sign*ret/27).backward();counts['backwards']+=1
                        gradient={k:edit[k]+(v.grad.detach() if v.grad is not None else 0.) for k,v in expert.named_parameters()}
                        selected,updated,raw_norm=source_step(r,before,gradient,scalar)
                        alpha=1.;checks=0;violations=None
                        if arm in ('POOL_MEAN','POOL_PAIR','NORM_REPLAY'):
                            target=targets[clock] if arm!='NORM_REPLAY' else replay[clock]
                            if target==0:selected={k:v.clone() for k,v in before.items()}
                            else:selected=r.mm.canonicalize(r.match_proposal(before,selected,target)[0])[0]
                        if arm=='LOOKAHEAD':
                            proposal={k:v.clone() for k,v in selected.items()}
                            for alpha in [2.**-k for k in range(8)]+[0.]:
                                budget();selected=r.mm.canonicalize({k:before[k]+alpha*(proposal[k]-before[k]) for k in before})[0] if alpha else {k:v.clone() for k,v in before.items()};expert.load_state_dict(selected);trial=score();checks+=1
                                violations=max(max(a-b for a,b in zip(s0['margins'],trial['margins'])),max(trial['nll'][j]-s0['nll'][j] for j in range(0,54,2)))
                                if violations<=1e-4:break
                            else:raise AssertionError('Even zero displacement violated frozen constraints')
                        scalar=updated;expert.load_state_dict(selected);counts['updates']+=1;actual=r.fm.map_norm(before,selected)
                        curve.append(dict(step=clock+1,source_loss=float(loss.detach()),support=current,active_pairs=len(active),actual_norm=actual,raw_norm=raw_norm,alpha=alpha,proposal_checks=checks,post_violation=violations))
                        save(folder/f'active_{arm}.pt',dict(state={k:v.cpu() for k,v in selected.items()},scalar=scalar,curve=curve,counts=copy.deepcopy(counts)))
                        if (clock+1)%20==0:write(out/'PROGRESS.json',dict(task=key,arm=arm,step=clock+1,counts=counts));print(key,arm,clock+1,flush=True)
                    final=r.old.snapshot(expert);path=folder/'finals'/(arm+'.pt');save(path,dict(state={k:v.cpu() for k,v in final.items()}));assert all(torch.equal(v.cpu(),r.p.load_state(path)['state'][k]) for k,v in final.items());write(dest,dict(status='FROZEN',curve=curve,final_support=score(),save_load_exact=True))
                    if arm=='NONE':write(shared/f'REFERENCE_{key}.json',dict(targets=[x['actual_norm'] for x in curve]))
                    if arm=='LOOKAHEAD':write(shared/f'LOOKAHEAD_{key}.json',dict(norms=[x['actual_norm'] for x in curve]))
            finally:hook.detach()
            arms.append(dict(task=key,arm=arm))
    write(shared/f'FROZEN_{partition}.json',dict(arms=arms,counts=counts))
    while not all((shared/f'FROZEN_{i}.json').exists() for i in (0,1)):budget();time.sleep(2)
    for task in config['tasks']:
        evaluation=prepare(read(shared/'EVAL_ROWS.json')+[task['rephrase']],False);assert len(evaluation)==32
        for row,_,raw,expanded,binding in evaluation:
            prior=read(read(shared/'BASE_REUSE.json')[row['query_id']])['binding'];assert all(binding[k]==prior[k] for k in ('image_sha256','question','reference','prompt_ids','generation'))
        for arm in own:
            expert.load_state_dict(r.p.load_state(out/task['key']/'finals'/(arm+'.pt'))['state']);hook.attach()
            try:
                for row,_,raw,expanded,binding in evaluation:
                    budget();dest=out/task['key']/'outputs'/arm/(row['query_id']+'.json');assert not dest.exists();write(dest.parent.parent.parent/'attempts'/arm/dest.name,dict(attempt=1,time=time.time()));counts['generations']+=1;answer,trace=r.s.fit.short.generate(runtime,hook,raw,expanded);write(dest,dict(binding=binding,raw_answer=answer.decoded_text,raw_token_ids=list(answer.raw_token_ids),generation_trace=trace,EOS=answer.raw_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(answer.raw_token_ids)>=1024))
            finally:hook.detach()
    write(out/'STUDY_COMPLETE.json',dict(status='COMPLETE',arms=arms,counts=counts))
