"""Fixed six-query Base opportunity qualification; no formal updates."""
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
from types import SimpleNamespace

ROOT=Path(os.environ['V27_ROOT']);PART=int(os.environ['PARTITION']);OUT=ROOT/str(PART)
P=Path(os.environ['PARENT_RUN']);OLD=Path(os.environ['V27_PRIOR_ROOT']);START=time.time()
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
    assert time.time()-START+PRIOR_SECONDS < 2*3600-60
    assert time.time()<HARD_END-300
    assert counts['forwards']<=80000 and counts['backwards']<=660 and counts['updates']<=640
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
        assert prior['counts']['updates']==0 and prior['counts']['query_keys']==0 and prior['counts']['generations']==1 and read(ROOT/'ENGINEERING_REPAIR1.json')['bounded_missing_output_retry']==1,'Only unexecuted training/generation may be repaired'
        counts.update(prior['counts'])
        PRIOR_SECONDS=prior['seconds']
    remaining=2*3600-60-PRIOR_SECONDS
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
    GEN_CAP=7
    r.c.budget=budget
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.matmul.allow_tf32=False
    budget();runtime=load_real_runtime(SimpleNamespace(cpu_gate=P/'private/cpu_gate'))
    runtime.model.eval();r.old.fp32(runtime)
    assert not any(v.requires_grad for v in runtime.model.parameters())
    assert len(runtime.model.model.layers)==32
    config=read(ROOT/'DATA_ADMISSION.json');tasks=config['tasks']
    template=read(P/'private/M13_SOURCE_PACKET.json')['original_tasks'][4]
    seed=template['seed'];bindings=read(P/'private/EVAL_BINDINGS.json')
    assert runtime.generation_config==next(iter(bindings.values()))['generation']
    def count(*_):
        budget();assert counts['forwards']<80000;counts['forwards']+=1
    fh=runtime.model.register_forward_pre_hook(count)
    vh=runtime.model.get_vision_tower().register_forward_pre_hook(lambda *_:counts.__setitem__('vision_forwards',counts['vision_forwards']+1))

    def prepare(rows,training):
        runtime.model.half()
        try:items=r.s.prepare(runtime,bindings,template,rows,training)
        finally:r.old.fp32(runtime)
        for row,_,raw,expanded,binding in items:
            assert all(v==1 for v in binding['attention_mask'])
            assert binding['question']==row['question'] and binding['reference']==row['reference']
        return items

    def make(arm):
        e=TT(seed).to(runtime.device)
        e.load_state_dict(r.mm.canonicalize(snapshot(e))[0])
        return e,Hook(runtime.get_module(TARGETS[arm]),e,True)

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
            assert len(trace)==2 and trace[0]['active_predictor_count']==first
            assert trace[1]['active_predictor_count']==1 and all(t['active_residual_norm']>0 for t in trace)
            return dict(status='PASS',zero_logits_max_abs=float((base-zero).abs().max()),
                        teacher_prefill_max_abs=float((actual-expected[0]).abs().max()),
                        teacher_KV_decode_max_abs=float((actual_next-expected[1]).abs().max()),trace=trace,
                        source_only=True,real_generations=0)
        finally:hook.detach();del expert,hook

    try:
        write(OUT/'MODEL_READY.json',dict(time=time.time(),gpu=os.environ['CUDA_VISIBLE_DEVICES']))
        items=prepare([row for t in tasks for row in t['train_rows']],True)
        batch=r.old.batch_on(runtime,items[0][1])
        real=read(ROOT/'engineering_failure1/0/REAL_ADMISSION.json') if (ROOT/'ENGINEERING_REPAIR1.json').exists() else mechanical('S',batch)
        write(OUT/'REAL_ADMISSION.json',real)
        expert,hook=make('S');hook.attach()
        try:
            for item in items:
                row,_,raw,expanded,binding=item
                dest=OUT/'base'/(row['query_id']+'.json')
                attempt=OUT/'attempts'/(row['query_id']+'.json')
                assert not dest.exists() and not attempt.exists()
                write(attempt,dict(time=time.time(),attempt=1))
                counts['generations']+=1
                answer,trace=r.s.fit.short.generate(runtime,None,raw,expanded)
                tokens=list(answer.raw_token_ids)
                write(dest,dict(binding=binding,raw_answer=answer.decoded_text,raw_token_ids=tokens,
                               EOS=bool(tokens) and tokens[-1]==runtime.adapter.tokenizer.eos_token_id,
                               at_cap=len(tokens)>=1024,zero_writer=True))
        finally:hook.detach()
        write(OUT/'BASE_QUALIFICATION_READY.json',dict(status='AWAIT_FIXED_REFERENCE_REVIEW',rows=6,counts=counts,time=time.time()))
    finally:fh.remove();vh.remove()

if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        code=1
        write(OUT/'FAILURE.json',dict(error=traceback.format_exc(),counts=counts,time=time.time(),automatic_retry=False))
        if not (ROOT/'STOP').exists():write(ROOT/'STOP',dict(reason='QUALIFICATION_FAILURE',time=time.time()))
        raise
    finally:write(OUT/'EXIT.json',dict(code=code,counts=counts,seconds=time.time()-START+PRIOR_SECONDS))
