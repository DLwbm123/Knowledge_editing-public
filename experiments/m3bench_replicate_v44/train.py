"""V44: one paired COVER/ADAPT replication at the prespecified next seed."""
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
from types import SimpleNamespace

ROOT=Path(os.environ['V44_ROOT']);PART=int(os.environ['PARTITION']);OUT=ROOT/str(PART)
P=Path(os.environ['PARENT_RUN']);START=time.time()
counts=dict(forwards=0,backwards=0,updates=0,generations=0,vision_forwards=0,query_keys=0,candidate_sequences=0)
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
    assert time.time()-START+PRIOR_SECONDS < 3600-60
    assert time.time()<HARD_END-300
    assert counts['forwards']<=40000 and counts['backwards']<=2600 and counts['updates']<=640
    assert counts['generations']<=GEN_CAP and counts['candidate_sequences']==0
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
    remaining=3600-60-PRIOR_SECONDS
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
    GEN_CAP=90 if PART==0 else 96
    r.c.budget=budget
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.matmul.allow_tf32=False
    budget();runtime=load_real_runtime(SimpleNamespace(cpu_gate=P/'private/cpu_gate'))
    runtime.model.eval();r.old.fp32(runtime)
    assert not any(v.requires_grad for v in runtime.model.parameters())
    assert len(runtime.model.model.layers)==32
    config=read(ROOT/'DATA_ADMISSION.json');tasks=config['tasks'][PART::2];assert len(tasks)==2
    template=read(P/'private/M13_SOURCE_PACKET.json')['original_tasks'][4]
    seed=template['seed']+1;assert seed!=template['seed'];bindings=read(P/'private/EVAL_BINDINGS.json')
    assert runtime.generation_config==next(iter(bindings.values()))['generation']
    def count(*_):
        budget();assert counts['forwards']<40000;counts['forwards']+=1
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

    from scoring import answer_scores
    from protocol import brief_row, verify_original, gate_ready, source_rows, adaptive_gradient, paired_gradient
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
        first=prepare(source_rows(tasks[0]),True)
        real=mechanical('S',r.old.batch_on(runtime,first[0][1]))
        write(OUT/'REAL_ADMISSION.json',real)
        # Check actual paired-gradient composition before formal updates.
        ex,h=make('S');h.attach()
        try:
            grads=[];losses=[]
            for item in first:
                b=r.old.batch_on(runtime,item[1]);teacher(h,b);loss=runtime.compute_loss(b)
                gs=torch.autograd.grad(loss,tuple(ex.parameters()));counts['backwards']+=1
                grads.append(torch.cat([g.detach().cpu().double().flatten() for g in gs]));losses.append(float(loss.detach()))
            j,weights=adaptive_gradient(grads,losses)
            assert len(grads)==4 and torch.isfinite(j).all()
            expected=sum(g*w for g,w in zip(grads,torch.softmax(torch.tensor(losses,dtype=torch.float64),0)))
            torch.testing.assert_close(j,expected,rtol=1e-12,atol=1e-12)
            assert sum(x.numel() for x in ex.parameters())==7168
            gc,wc=paired_gradient(grads,losses,'COVER2');torch.testing.assert_close(gc,torch.stack(grads).mean(0),rtol=0,atol=0)
            write(OUT/'FOUR_SOURCE_GRADIENT_ADMISSION.json',dict(status='PASS',parameters=7168,source_rows=4,objective="paired_seed_plus_one",seed=seed,cover_equal_mean=True,weights=weights.tolist(),finite=True,heldout_training_rows=0))
        finally:h.detach()
        b=r.old.batch_on(runtime,first[0][1])
        with torch.no_grad():
            out=runtime.model(**b.forward_kwargs())
            scored=answer_scores(out.logits,b.labels,runtime.adapter.tokenizer.eos_token_id)
            assert abs(scored['mean_with_eos']+float(out.loss))<1e-4
        write(OUT/'SCORING_ADMISSION.json',dict(status='PASS',loss_matches_mean_token_nll=True,finite=True,source_only=True))
        # All original/SHORT Base outputs must reproduce before augmented training.
        for task in tasks:
            folder=OUT/task['key']
            for arm in ('BASE','BASE_SHORT'):
                qs=task['eval_rows'] if arm=='BASE' else [brief_row(q) for q in task['eval_rows']]
                for item in prepare(qs,False):emit(None,item,folder/'outputs'/arm)
                for q in task['eval_rows']:
                    path=Path(str(PART))/task['key']/'outputs'/arm/(q['query_id']+'.json')
                    verify_original(read(ROOT/path),read(Path(config['prior_root'])/path))
            write(folder/'ORIGINAL_PARITY.json',dict(status='PASS',outputs=2*len(task['eval_rows']),tokens_exact=True,text_exact=True))
        expected=30 if PART==0 else 32
        assert counts['generations']==expected
        write(OUT/'ORIGINAL_PARITY.json',dict(status='PASS',outputs=expected,time=time.time()))
        while not gate_ready(ROOT):budget();time.sleep(1)
        write(OUT/'TRAINING_RELEASE.json',dict(status='PASS',base_original_and_short_exact=62,time=time.time()))
        for task in tasks:
            folder=OUT/task['key'];items=prepare(source_rows(task),True)
            batches=[r.old.batch_on(runtime,x[1]) for x in items]
            evaluation=prepare(task['eval_rows'],False)
            short=prepare([brief_row(q) for q in task['eval_rows']],False)
            paired_initial=None
            for arm in ('COVER2','ADAPT2'):
                dest=folder/arm;ex,h=make('S');h.attach();scalar=dict(step=0,second_moments=[0.,0.]);curve=[]
                initial=snapshot(ex)
                if paired_initial is None:paired_initial=initial
                else:assert all(torch.equal(v,paired_initial[k]) for k,v in initial.items())
                try:
                    with PrefixCache(runtime,batches):
                        for step in range(1,161):
                            budget();before=snapshot(ex);grads=[];losses=[]
                            for batch in batches:
                                teacher(h,batch);loss=runtime.compute_loss(batch);assert torch.isfinite(loss)
                                gs=torch.autograd.grad(loss,tuple(ex.parameters()));counts['backwards']+=1
                                grads.append(torch.cat([g.detach().cpu().double().flatten() for g in gs]));losses.append(float(loss.detach()))
                            gradient,weights=paired_gradient(grads,losses,arm)
                            state,scalar=next_step(r,before,gradient,scalar);ex.load_state_dict(state);counts['updates']+=1
                            norms=torch.stack([g.norm() for g in grads]);cosines=[float(torch.dot(grads[i],grads[j])/(norms[i]*norms[j]).clamp_min(1e-30)) for i in range(4) for j in range(i+1,4)]
                            curve.append(dict(step=step,source_NLL=losses,source_weights=weights.tolist(),pair_gradient_cosines=cosines,operator_change=map_change(before,state,r.mm)))
                            save(dest/'active.pt',dict(state=snapshot(ex),scalar=scalar,curve=curve,counts=counts))
                            if step%10==0:write(OUT/'PROGRESS.json',dict(case=task['key'],arm=arm,step=step,counts=counts,time=time.time()))
                    save(dest/'final.pt',dict(state=snapshot(ex)))
                    loaded=torch.load(dest/'final.pt',map_location=runtime.device,weights_only=True)['state']
                    assert all(torch.equal(v,loaded[k]) for k,v in ex.state_dict().items())
                    write(dest/'TRAIN_COMPLETE.json',dict(status='PASS',seed=seed,arm=arm,paired_initial_exact=True,steps=160,parameters=7168,source_rows=4,heldout_training_rows=0,curve=curve,save_load_exact=True,counts=counts))
                    for item in evaluation:emit(h,item,folder/'outputs'/arm)
                    for item in short:emit(h,item,folder/'outputs'/(arm+'_SHORT'))
                finally:h.detach()
                del ex,h
                files=[dest/n for n in ('active.pt','final.pt')]
                assert all(len(list((folder/'outputs'/arm).glob('*.json')))==len(task['eval_rows']) for arm in ('BASE','BASE_SHORT',arm,arm+'_SHORT'))
                assert all(x.is_file() and not x.is_symlink() and x.resolve().is_relative_to(ROOT.resolve()) for x in files)
                retired=[dict(file=str(x.relative_to(ROOT)),bytes=x.stat().st_size) for x in files]
                write(dest/'RETIREMENT_PREPARED.json',dict(files=retired,last_consumers='All four-source training, original and SHORT outputs, bindings, save/load',no_followup_weight_consumers=True))
                for x in files:x.unlink()
                write(dest/'LIFECYCLE.json',dict(status='RETIRED',files=retired,no_copies=True,reconstruction='requires retraining'))
        assert counts['updates']==640 and counts['generations']==GEN_CAP and counts['candidate_sequences']==0
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
