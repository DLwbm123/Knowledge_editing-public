"""One finite shard: native-only admission, common OFF, then three fixed writers."""
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
from types import SimpleNamespace

ROOT=Path(os.environ['V24_ROOT']);PART=int(os.environ['PARTITION']);OUT=ROOT/str(PART)
P=Path(os.environ['PARENT_RUN']);OLD=Path(os.environ['V24_PRIOR_ROOT']);START=time.time()
counts=dict(forwards=0,backwards=0,updates=0,generations=0,vision_forwards=0,query_keys=0)
last_space=0.
HARD_END=None
GEN_CAP=None
PRIOR_SECONDS=0.


def read(p):return json.loads(Path(p).read_text())


def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)


def save(p,v):
    import torch
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');torch.save(v,tmp);tmp.replace(p)


def budget():
    global last_space
    assert not (ROOT/'STOP').exists(),'Global STOP'
    assert time.time()-START+PRIOR_SECONDS < 4*3600-60
    assert time.time()<HARD_END-300
    assert counts['forwards']<=220000 and counts['backwards']<=1000 and counts['updates']<=960
    assert counts['generations']<=GEN_CAP
    assert time.time()-read(ROOT/'RESOURCE_GUARD_HEALTH.json')['time']<120,'Independent guard stale'
    if time.time()-last_space>30:
        assert shutil.disk_usage(ROOT).free>8*1024**3
        assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())<1500000000
        last_space=time.time()


def main():
    global HARD_END,GEN_CAP,PRIOR_SECONDS
    with (OUT/'ONCE').open('x'):pass
    assert (OUT/'GO.json').exists()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Fixed shard alarm')))
    if (ROOT/'PRIOR_COST.json').exists():
        prior=read(ROOT/'PRIOR_COST.json')[str(PART)]
        assert all(v==0 for v in prior['counts'].values()),'This repair is pre-model only'
        PRIOR_SECONDS=prior['seconds']
    remaining=4*3600-60-PRIOR_SECONDS
    if (ROOT/'GPU_DEADLINES.json').exists():remaining=min(remaining,read(ROOT/'GPU_DEADLINES.json')[str(PART)]-time.time()-60)
    assert remaining>0
    signal.alarm(int(remaining))
    sys.path[:0]=[str(ROOT/'tools'),str(P/'private/tools'),str(P/'private/source'),str(P/'private/official_llava')]
    import torch
    import module_run as r
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from alignment import ARMS,TARGETS,TT,Hook,snapshot,next_step,map_change
    from cache_v18 import PrefixCache
    assert read(ROOT/'CPU_ADMISSION.json')['status']=='PASS'
    HARD_END=read(ROOT/'RESOURCE_LIMITS.json')['hard_end']
    GEN_CAP=36+3*sum(len(t['eval_rows']) for t in read(ROOT/'DATA_ADMISSION.json')['tasks'][PART::5])
    r.c.budget=budget
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.matmul.allow_tf32=False
    budget();runtime=load_real_runtime(SimpleNamespace(cpu_gate=P/'private/cpu_gate'))
    runtime.model.eval();r.old.fp32(runtime)
    assert not any(v.requires_grad for v in runtime.model.parameters())
    assert len(runtime.model.model.layers)==32
    config=read(ROOT/'DATA_ADMISSION.json');tasks=config['tasks'][PART::5]
    template=read(P/'private/M13_SOURCE_PACKET.json')['original_tasks'][4]
    seed=template['seed'];bindings=read(P/'private/EVAL_BINDINGS.json')
    assert runtime.generation_config==next(iter(bindings.values()))['generation']
    def count(*_):
        budget();assert counts['forwards']<220000;counts['forwards']+=1
    fh=runtime.model.register_forward_pre_hook(count)
    vh=runtime.model.get_vision_tower().register_forward_pre_hook(lambda *_:counts.__setitem__('vision_forwards',counts['vision_forwards']+1))

    def prepare(rows,training):
        runtime.model.half()
        try:items=r.s.prepare(runtime,bindings,template,rows,training)
        finally:r.old.fp32(runtime)
        for row,_,raw,expanded,binding in items:
            assert all(v==1 for v in binding['attention_mask'])
            prior=read(next((OLD/'base').glob('*/'+row['query_id']+'.json')))['binding']
            assert all(binding[k]==prior[k] for k in ('image_sha256','question','reference','prompt_ids','attention_mask','generation','runtime'))
        return items

    def make(arm):
        e=TT(seed,up=arm=='C').to(runtime.device)
        e.load_state_dict(r.mm.canonicalize(snapshot(e))[0])
        return e,Hook(runtime.get_module(TARGETS[arm]),e,arm!='A')

    def teacher(hook,batch):
        hook.teacher(batch.labels,batch.forward_kwargs().get('attention_mask'))

    def mechanical(arm,batch):
        expert,hook=make(arm)
        kwargs=batch.forward_kwargs();first=int(torch.where(batch.labels[0]!=-100)[0][0])
        with torch.no_grad():base=runtime.model(**kwargs).logits[0,first-1].detach().cpu().double()
        hook.attach()
        try:
            teacher(hook,batch)
            with torch.no_grad():zero=runtime.model(**kwargs).logits[0,first-1].detach().cpu().double()
            torch.testing.assert_close(base,zero,rtol=0,atol=1e-6)
            with torch.no_grad():expert.G1.fill_(.001)
            teacher(hook,batch);out=runtime.model(**kwargs)
            assert torch.isfinite(out.loss)
            gs=torch.autograd.grad(out.loss,tuple(expert.parameters()));counts['backwards']+=1
            assert all(torch.isfinite(g).all() for g in gs)
            expected=out.logits[0,first-1:first+1].detach().cpu().double();del out,gs
            prefix={k:v for k,v in kwargs.items() if k!='labels'};prefix['use_cache']=True
            for name in ('inputs_embeds','attention_mask','position_ids'):
                if isinstance(prefix.get(name),torch.Tensor):prefix[name]=prefix[name][:,:first]
            with torch.no_grad(),hook.generation_request():
                out=runtime.model(**prefix);actual=out.logits[0,-1].detach().cpu().double()
                decode=dict(prefix,past_key_values=out.past_key_values)
                decode['inputs_embeds']=kwargs['inputs_embeds'][:,first:first+1]
                if isinstance(kwargs.get('attention_mask'),torch.Tensor):decode['attention_mask']=kwargs['attention_mask'][:,:first+1]
                if isinstance(kwargs.get('position_ids'),torch.Tensor):decode['position_ids']=kwargs['position_ids'][:,first:first+1]
                out=runtime.model(**decode);actual_next=out.logits[0,-1].detach().cpu().double()
            torch.testing.assert_close(actual,expected[0],rtol=1e-5,atol=5e-4)
            torch.testing.assert_close(actual_next,expected[1],rtol=1e-5,atol=5e-4)
            trace=list(hook.last_generation_trace)
            assert len(trace)==2 and trace[0]['active_predictor_count']==(1 if arm=='A' else first)
            assert trace[1]['active_predictor_count']==1 and all(t['active_residual_norm']>0 for t in trace)
            return dict(status='PASS',zero_logits_max_abs=float((base-zero).abs().max()),
                        teacher_prefill_max_abs=float((actual-expected[0]).abs().max()),
                        teacher_KV_decode_max_abs=float((actual_next-expected[1]).abs().max()),trace=trace,
                        source_only=True,real_generations=0)
        finally:hook.detach();del expert,hook

    try:
        write(OUT/'MODEL_READY.json',dict(time=time.time(),gpu=os.environ['CUDA_VISIBLE_DEVICES']))
        source=prepare([tasks[0]['source']],True)[0]
        batch=r.old.batch_on(runtime,source[1])
        real={a:mechanical(a,batch) for a in ARMS}
        write(OUT/'REAL_ADMISSION.json',dict(status='PASS',arms=real,counts=counts,time=time.time()))
        while not all((ROOT/str(i)/'REAL_ADMISSION.json').exists() for i in range(5)):
            budget();time.sleep(1)
        assert all(read(ROOT/str(i)/'REAL_ADMISSION.json')['status']=='PASS' for i in range(5))
        write(OUT/'TRAINING_RELEASE.json',dict(time=time.time(),all_five_real_admissions_pass=True))
        del source,batch
        def emit(dest,hook,item):
            row,_,raw,expanded,binding=item
            budget();assert not dest.exists()
            attempt=dest.parent.parent.parent/'attempts'/dest.parent.name/dest.name
            assert not attempt.exists(),'Never retry attempted generation'
            write(attempt,dict(time=time.time(),attempt=1));counts['generations']+=1
            answer,trace=r.s.fit.short.generate(runtime,hook,raw,expanded)
            tokens=list(answer.raw_token_ids)
            write(dest,dict(binding=binding,raw_answer=answer.decoded_text,raw_token_ids=tokens,
                            generation_trace=trace,EOS=bool(tokens) and tokens[-1]==runtime.adapter.tokenizer.eos_token_id,
                            at_cap=len(tokens)>=1024))
        queries=read(ROOT/'QUERY_ROWS.json')
        for i,qid in enumerate(sorted(queries)):
            if i%5==PART:emit(ROOT/'base'/str(PART)/(qid+'.json'),None,prepare([queries[qid]],False)[0])
        write(OUT/'BASE_COMPLETE.json',dict(status='PASS',answers=36,counts=counts))
        for task in tasks:
            folder=OUT/task['key'];item=prepare([task['source']],True)[0]
            batch=r.old.batch_on(runtime,item[1]);initials=[]
            for arm in ARMS:
                dest=folder/arm;dest.mkdir(parents=True,exist_ok=True)
                with (dest/'ONCE').open('x'):pass
                expert,hook=make(arm);initials.append(snapshot(expert))
                if arm=='B':assert all(torch.equal(v,initials[0][k]) for k,v in initials[1].items())
                scalar=dict(step=0,second_moments=[0.,0.]);curve=[]
                hook.attach()
                try:
                    with PrefixCache(runtime,[batch]):
                        for step in range(1,161):
                            budget();before=snapshot(expert);teacher(hook,batch)
                            loss=runtime.compute_loss(batch);assert torch.isfinite(loss)
                            gs=torch.autograd.grad(loss,tuple(expert.parameters()));counts['backwards']+=1
                            assert all(torch.isfinite(g).all() for g in gs)
                            gradient=torch.cat([g.detach().cpu().double().flatten() for g in gs])
                            state,scalar=next_step(r,before,gradient,scalar)
                            norm=map_change(before,state,r.mm);assert norm>0
                            expert.load_state_dict(state);counts['updates']+=1
                            curve.append(dict(step=step,source_loss=float(loss.detach()),operator_change=norm))
                            save(dest/'active.pt',dict(state=snapshot(expert),scalar=scalar,curve=curve,counts=counts))
                            if step%10==0:write(OUT/'PROGRESS.json',dict(task=task['key'],arm=arm,step=step,counts=counts,time=time.time()))
                            del loss,gs,gradient
                        teacher(hook,batch)
                        with torch.no_grad():final_loss=float(runtime.compute_loss(batch))
                finally:hook.detach()
                state=snapshot(expert);save(dest/'final.pt',dict(state=state))
                loaded=torch.load(dest/'final.pt',map_location=runtime.device,weights_only=True)['state']
                assert all(torch.equal(v,loaded[k]) for k,v in state.items())
                write(dest/'TRAIN_COMPLETE.json',dict(status='COMPLETE',accepted_steps=160,remaining_NA=0,
                      curve=curve,final_source_loss=final_loss,save_load_exact=True,parameters=7168,scope='ANSWER_PREDICTORS' if arm=='A' else 'ALL_VALID_TOKENS',target=TARGETS[arm]))
                del expert,hook,loaded,state
            evaluation=prepare(task['eval_rows'],False)
            for arm in ARMS:
                expert,hook=make(arm)
                expert.load_state_dict(torch.load(folder/arm/'final.pt',map_location=runtime.device,weights_only=True)['state'])
                hook.attach()
                try:
                    for value in evaluation:emit(folder/'outputs'/arm/(value[0]['query_id']+'.json'),hook,value)
                finally:hook.detach()
                del expert,hook
            assert len(list((folder/'outputs').glob('*/*.json')))==3*len(evaluation)
            write(folder/'MODEL_CONSUMERS_COMPLETE.json',dict(status='PASS',outputs=3*len(evaluation),save_load=True,mechanical_admitted=True,no_bank_or_sequential_consumer=True))
            files=[folder/a/n for a in ARMS for n in ('active.pt','final.pt')]
            assert all(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(ROOT.resolve()) for p in files)
            deleted=[dict(file=str(p.relative_to(ROOT)),bytes=p.stat().st_size) for p in files]
            write(folder/'RETIREMENT_PREPARED.json',dict(files=deleted,last_consumers='forced outputs, mechanical admission, save-load'))
            for p in files:p.unlink()
            write(folder/'LIFECYCLE.json',dict(status='RETIRED',files=deleted,no_weight_copy=True,rebuild='retraining required'))
            del batch,item,evaluation,initials
        write(OUT/'MODEL_COMPLETE.json',dict(status='PASS',counts=counts,time=time.time()))
    finally:fh.remove();vh.remove()


if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        code=1
        write(OUT/'FAILURE.json',dict(error=traceback.format_exc(),counts=counts,time=time.time(),automatic_retry=False))
        if not (ROOT/'STOP').exists():write(ROOT/'STOP',dict(reason='WORKER_FAILURE',partition=PART,time=time.time()))
        raise
    finally:write(OUT/'EXIT.json',dict(code=code,counts=counts,seconds=time.time()-START+PRIOR_SECONDS))
