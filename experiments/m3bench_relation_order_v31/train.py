"""Frozen two-arm relation experiment after Base opportunity qualification."""
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
from types import SimpleNamespace

ROOT=Path(os.environ['V31_ROOT']);PART=int(os.environ['PARTITION']);OUT=ROOT/str(PART)
P=Path(os.environ['PARENT_RUN']);OLD=Path(os.environ['V31_PRIOR_ROOT']);START=time.time()
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
    assert counts['forwards']<=160000 and counts['backwards']<=4000 and counts['updates']<=1920
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
    GEN_CAP=180
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
        budget();assert counts['forwards']<160000;counts['forwards']+=1
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
        return e,Hook(runtime.get_module(TARGETS['S']),e,True)

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

    from gradient import combine
    def emit(hook,item,folder):
        row,_,raw,expanded,binding=item
        dest=folder/(row['query_id']+'.json');attempt=folder.parent/(folder.name+'_attempts')/dest.name
        assert not dest.exists() and not attempt.exists()
        budget();write(attempt,dict(time=time.time(),attempt=1));counts['generations']+=1
        answer,trace=r.s.fit.short.generate(runtime,hook,raw,expanded);tokens=list(answer.raw_token_ids)
        write(dest,dict(binding=binding,raw_answer=answer.decoded_text,raw_token_ids=tokens,
                       EOS=bool(tokens) and tokens[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(tokens)>=1024))
    try:
        write(OUT/'MODEL_READY.json',dict(time=time.time(),gpu=os.environ['CUDA_VISIBLE_DEVICES']))
        first=prepare(tasks[0]['train_rows'],True)
        real=mechanical('S',r.old.batch_on(runtime,first[0][1]))
        write(OUT/'REAL_ADMISSION.json',real)
        # Check actual paired-gradient composition before formal updates.
        ex,h=make('S');h.attach()
        try:
            grads=[]
            for item in first:
                b=r.old.batch_on(runtime,item[1]);teacher(h,b);loss=runtime.compute_loss(b)
                gs=torch.autograd.grad(loss,tuple(ex.parameters()));counts['backwards']+=1
                grads.append(torch.cat([g.detach().cpu().double().flatten() for g in gs]))
            j=torch.tensor(combine(*[g.tolist() for g in grads],False),dtype=torch.float64)
            torch.testing.assert_close(j,(grads[0]+grads[1])/2,rtol=0,atol=0)
            v=torch.tensor(combine(*[g.tolist() for g in grads],True),dtype=torch.float64)
            assert torch.isfinite(v).all()
            assert float(v@grads[1])>=-1e-10*max(1.,float(v.norm()*grads[1].norm()))
            assert sum(x.numel() for x in ex.parameters())==7168
            write(OUT/'PAIRED_GRADIENT_ADMISSION.json',dict(status='PASS',parameters=sum(x.numel() for x in ex.parameters()),joint_exact=True,relation_dot=float(v@grads[1])))
        finally:h.detach()
        alt_first=prepare(tasks[0]['alternate_train_rows'],True)
        ex,h=make('S');h.attach()
        try:
            for item in alt_first:
                b=r.old.batch_on(runtime,item[1]);teacher(h,b);loss=runtime.compute_loss(b)
                gs=torch.autograd.grad(loss,tuple(ex.parameters()));counts['backwards']+=1
                assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in gs) and sum(float(g.square().sum()) for g in gs)>0
        finally:h.detach()
        write(OUT/'EXPRESSION_ADMISSION.json',dict(status='PASS',two_alternate_losses_finite=True,nonzero_gradients=True))
        write(OUT/'TRAINING_RELEASE.json',dict(status='PASS',time=time.time()))
        for task in tasks:
            folder=OUT/task['key'];items=prepare(task['train_rows'],True)
            batches=[r.old.batch_on(runtime,x[1]) for x in items]
            alt=prepare(task['alternate_train_rows'],True)
            alt_batches=[r.old.batch_on(runtime,x[1]) for x in alt]
            evaluation=prepare(task['eval_rows'],False)
            ex,h=make('S')
            try:
                for item in evaluation:
                    emit(None,item,folder/'outputs'/'BASE')
            finally:h.detach()
            for arm in ('EXPR_FORWARD','EXPR_REVERSE'):
                dest=folder/arm;ex,h=make('S');h.attach();scalar=dict(step=0,second_moments=[0.,0.]);curve=[]
                try:
                    with PrefixCache(runtime,batches+alt_batches):
                        for step in range(1,161):
                            budget();before=snapshot(ex);grads=[];losses=[]
                            chosen=alt_batches if ((arm=='EXPR_FORWARD' and step%2==0) or (arm=='EXPR_REVERSE' and step%2==1)) else batches
                            for b in chosen:
                                teacher(h,b);loss=runtime.compute_loss(b);assert torch.isfinite(loss)
                                gs=torch.autograd.grad(loss,tuple(ex.parameters()));counts['backwards']+=1
                                grads.append(torch.cat([g.detach().cpu().double().flatten() for g in gs]));losses.append(float(loss.detach()))
                            gradient=torch.tensor(combine(*[g.tolist() for g in grads],False),dtype=torch.float64)
                            if arm=='EXPR_FORWARD':
                                old=read(OLD/'1'/task['key']/'JOINT_EXPR'/'TRAIN_COMPLETE.json')['curve'][step-1]
                                torch.testing.assert_close(torch.tensor(losses,dtype=torch.float64),torch.tensor([old['update_NLL'],old['preserve_NLL']],dtype=torch.float64),rtol=1e-5,atol=5e-4)
                            state,scalar=next_step(r,before,gradient,scalar);ex.load_state_dict(state);counts['updates']+=1
                            curve.append(dict(step=step,update_NLL=losses[0],preserve_NLL=losses[1],gradient_dot=float(grads[0]@grads[1]),alternate_expression=((arm=='EXPR_FORWARD' and step%2==0) or (arm=='EXPR_REVERSE' and step%2==1)),operator_change=map_change(before,state,r.mm)))
                            save(dest/'active.pt',dict(state=snapshot(ex),scalar=scalar,curve=curve,counts=counts))
                            if step%10==0:write(OUT/'PROGRESS.json',dict(case=task['key'],arm=arm,step=step,counts=counts,time=time.time()))
                    save(dest/'final.pt',dict(state=snapshot(ex)))
                    loaded=torch.load(dest/'final.pt',map_location=runtime.device,weights_only=True)['state']
                    assert all(torch.equal(v,loaded[k]) for k,v in ex.state_dict().items())
                    write(dest/'TRAIN_COMPLETE.json',dict(status='PASS',steps=160,parameters=7168,curve=curve,save_load_exact=True,reconstruction160='PASS' if arm=='EXPR_FORWARD' else 'NOT_APPLICABLE_NEW_ORDER',counts=counts))
                    for item in evaluation:emit(h,item,folder/'outputs'/arm)
                finally:h.detach()
                del ex,h
            replay_count=0
            for q in task['eval_rows']:
                for arm in ('BASE','EXPR_FORWARD'):
                    current=read(folder/'outputs'/arm/(q['query_id']+'.json'))
                    prior=read(OLD/'1'/task['key']/'outputs'/('JOINT_EXPR' if arm=='EXPR_FORWARD' else 'BASE')/(q['query_id']+'.json'))
                    assert current['binding']==prior['binding'],'Replay binding mismatch'
                    assert current['raw_token_ids']==prior['raw_token_ids'] and current['raw_answer']==prior['raw_answer'],'V30 control replay token/answer mismatch'
                    replay_count+=1
            write(folder/'REPLAY_ADMISSION.json',dict(status='PASS',consumers=replay_count,exact_tokens_and_answers=True))
            files=[folder/a/n for a in ('EXPR_FORWARD','EXPR_REVERSE') for n in ('active.pt','final.pt')]
            assert len(list((folder/'outputs'/'EXPR_FORWARD').glob('*.json')))==10 and len(list((folder/'outputs'/'EXPR_REVERSE').glob('*.json')))==10
            assert all(x.is_file() and not x.is_symlink() and x.resolve().is_relative_to(ROOT.resolve()) for x in files)
            retired=[dict(file=str(x.relative_to(ROOT)),bytes=x.stat().st_size) for x in files]
            write(folder/'RETIREMENT_PREPARED.json',dict(files=retired,last_consumers='both-arm outputs, source bindings, save/load, complete tokens',no_followup_weight_consumers=True))
            for x in files:x.unlink()
            write(folder/'LIFECYCLE.json',dict(status='RETIRED',files=retired,no_copies=True,reconstruction='requires retraining'))
        write(OUT/'MODEL_COMPLETE.json',dict(status='PASS',counts=counts,time=time.time()))
    finally:fh.remove();vh.remove()

if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        code=1;write(OUT/'FAILURE.json',dict(error=traceback.format_exc(),counts=counts,time=time.time(),automatic_retry=False))
        if not (ROOT/'STOP').exists():write(ROOT/'STOP',dict(reason='TRAIN_FAILURE',time=time.time()))
        raise
    finally:write(OUT/'EXIT.json',dict(code=code,counts=counts,seconds=time.time()-START+PRIOR_SECONDS))
