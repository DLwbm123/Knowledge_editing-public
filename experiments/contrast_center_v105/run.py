"""Two fixed TT writers on the existing source pair; one shared model load."""
import json
import os
import random
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT=Path(os.environ['V105_ROOT'])
P=Path(os.environ['PARENT_RUN'])
START=time.time()
counts=dict(forwards=0,backwards=0,updates=0,generations=0,vision_forwards=0)


def read(path):
    return json.loads(Path(path).read_text())


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


lock=read(ROOT/'PROTOCOL_LOCK.json')


def budget():
    assert time.time()<lock['hard_end']-15 and not (ROOT/'STOP.json').exists()
    assert time.time()-START<lock['new_gpu_cap_seconds']-15
    assert counts['backwards']<=672 and counts['updates']<=320 and counts['generations']<=36 and counts['forwards']<40000
    assert time.time()-read(ROOT/'GUARD_HEALTH.json')['time']<30


def main():
    with (ROOT/'WORKER_ONCE').open('x'):pass
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Original shared budget')))
    signal.alarm(max(1,int(min(lock['new_gpu_cap_seconds']-10,lock['hard_end']-time.time()-15))))
    sys.path[:0]=[str(ROOT/'tools'),str(P/'private/tools'),str(P/'private/source'),str(P/'private/official_llava')]
    import torch
    import module_run as r
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from types import SimpleNamespace
    from alignment import TT,Hook,snapshot
    from path_v18 import proposal,construct
    from v51_cache import PrefixCache
    from centered import CenteredTT,normalize
    r.c.budget=budget
    random.seed(lock['seed']);torch.manual_seed(lock['seed']);torch.cuda.manual_seed_all(lock['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.matmul.allow_tf32=False
    runtime=load_real_runtime(SimpleNamespace(cpu_gate=P/'private/cpu_gate'))
    runtime.model.eval();r.old.fp32(runtime)
    assert not any(v.requires_grad for v in runtime.model.parameters())
    cfg=read(ROOT/'DATA_ADMISSION.json')
    template=read(P/'private/M13_SOURCE_PACKET.json')['original_tasks'][4]
    bindings=read(P/'private/EVAL_BINDINGS.json')
    assert runtime.generation_config==next(iter(bindings.values()))['generation']
    module=runtime.get_module('model.layers.21.mlp.down_proj')
    def count(*_):budget();counts['forwards']+=1
    fh=runtime.model.register_forward_pre_hook(count)
    vh=runtime.model.get_vision_tower().register_forward_pre_hook(lambda *_:counts.__setitem__('vision_forwards',counts['vision_forwards']+1))
    def prepare(rows,training=False):
        runtime.model.half()
        try:return r.s.prepare(runtime,bindings,template,rows,training)
        finally:r.old.fp32(runtime)
    assert [x['query_id'] for x in cfg['train_rows']]==['S00_C','S00_A']
    assert cfg['train_rows'][0]['question']==cfg['train_rows'][1]['question']
    train=prepare(cfg['train_rows'],True)
    batches=[r.old.batch_on(runtime,x[1]) for x in train]
    diag_rows=[dict(x,reference=cfg['train_rows'][0]['reference'],query_id='DIAG_'+x['query_id']) for x in cfg['train_rows']]
    diagnostic=[r.old.batch_on(runtime,x[1]) for x in prepare(diag_rows,True)]
    evaluation=prepare(cfg['eval_rows'])
    assert len(evaluation)==17
    for item in evaluation:assert item[4]==cfg['Base_outputs'][item[0]['query_id']]['binding']
    features=[]
    for b in batches:
        position=int(torch.where(b.labels[0]!=-100)[0][0]);captured=[]
        def capture(_,args):captured.append(args[0][0,position-1].detach().clone())
        hook=module.register_forward_pre_hook(capture)
        kw={k:v for k,v in b.forward_kwargs().items() if k!='labels'}
        for k in ['inputs_embeds','attention_mask','position_ids']:
            if k in kw:kw[k]=kw[k][:,:position]
        kw['use_cache']=False
        try:
            with torch.no_grad():out=runtime.model(**kw)
            del out
        finally:hook.remove()
        assert len(captured)==1
        features.append(normalize(captured[0]))
    pair=torch.stack(features);center=pair.mean(0).detach()
    assert torch.isfinite(center).all() and (pair[0]-pair[1]).norm()>1e-8
    torch.save(center.cpu(),ROOT/'CENTER.pt')
    write(ROOT/'CENTER_FROZEN.json',dict(source_prefix_only=True,features=2,trainable_extra_parameters=0,
        fixed_floats=center.numel(),norms=[float(v.norm()) for v in pair],difference_norm=float((pair[0]-pair[1]).norm()),
        centered_norms=[float((v-center).norm()) for v in pair],time=time.time()))
    for arm in lock['arms']:
        dest=ROOT/arm;dest.mkdir()
        ex=(TT(lock['seed']) if arm=='JOINT_CE' else CenteredTT(lock['seed'],center.cpu())).to(runtime.device)
        ex.load_state_dict(r.mm.canonicalize(snapshot(ex))[0]);initial=snapshot(ex)
        h=Hook(module,ex,True);h.attach()
        def value(b):
            h.teacher(b.labels,b.forward_kwargs().get('attention_mask'))
            loss=runtime.compute_loss(b);assert torch.isfinite(loss)
            gs=torch.autograd.grad(loss,tuple(ex.parameters()));counts['backwards']+=1
            return float(loss.detach()),torch.cat([g.detach().cpu().double().flatten() for g in gs])
        def cosine():
            v=[value(b)[1] for b in diagnostic];norms=[float(g.norm()) for g in v]
            return dict(cosine=float(v[0]@v[1])/(norms[0]*norms[1]) if min(norms)>0 else None,norms=norms)
        def emit(item,smoke=False):
            counts['generations']+=1
            if smoke:
                with h.generation_request():answer,trace=r.s.fit.short.generate(runtime,None,item[2],item[3])
                assert h.last_generation_trace and all(x['active_residual_norm']==0 for x in h.last_generation_trace)
            else:answer,trace=r.s.fit.short.generate(runtime,h,item[2],item[3])
            tokens=list(answer.raw_token_ids)
            output=dict(binding=item[4],raw_answer=answer.decoded_text,raw_token_ids=tokens,
                        EOS=bool(tokens) and tokens[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(tokens)>=1024)
            write(dest/('smoke' if smoke else 'outputs')/(item[0]['query_id']+'.json'),output)
            return output
        try:
            smoke=emit(evaluation[0],True);base=cfg['Base_outputs'][evaluation[0][0]['query_id']]
            assert smoke['raw_token_ids']==base['raw_token_ids'] and smoke['raw_answer']==base['raw_answer']
            for nonzero in [False,True]:
                ex.load_state_dict(initial)
                if nonzero:
                    with torch.no_grad():ex.G1.fill_(.001)
                before=snapshot(ex);refs=[value(b) for b in batches]
                with PrefixCache(runtime,batches,21):
                    for _ in range(2):
                        for b,ref in zip(batches,refs):
                            cur=value(b);assert cur[0]==ref[0] and torch.equal(cur[1],ref[1])
                assert all(torch.equal(v,before[k]) for k,v in snapshot(ex).items())
            ex.load_state_dict(initial)
            initial_cosine=cosine();curve=[];scalar=dict(step=0,second_moments=[0.,0.])
            write(dest/'ADMISSION.json',dict(zero_Base_exact=True,zero_and_nonzero_cache_loss_gradient_exact=True,counts=dict(counts)))
            with PrefixCache(runtime,batches,21):
                for step in range(1,161):
                    budget();before=snapshot(ex);values=[value(b) for b in batches]
                    gradient=(values[0][1]+values[1][1])/2;keys=list(before)
                    raw,scalar,direction=proposal(r,before,gradient,scalar,keys)
                    ex.load_state_dict(construct(r,before,raw,direction,direction,1.,keys));counts['updates']+=1
                    curve.append(dict(step=step,losses=[v[0] for v in values]))
                    if step%20==0:write(ROOT/'PROGRESS.json',dict(arm=arm,step=step,counts=dict(counts),time=time.time()))
            final_cosine=cosine()
            torch.save(snapshot(ex),dest/'delta.pt')
            loaded=torch.load(dest/'delta.pt',map_location=runtime.device,weights_only=True)
            assert all(torch.equal(v,loaded[k]) for k,v in snapshot(ex).items());ex.load_state_dict(loaded)
            for item in evaluation:emit(item)
            write(dest/'COMPLETE.json',dict(curve=curve,initial_same_target_gradient=initial_cosine,
                final_same_target_gradient=final_cosine,save_load_exact=True,counts=dict(counts),time=time.time()))
        finally:h.detach()
    assert counts['backwards']==672 and counts['updates']==320 and counts['generations']==36
    write(ROOT/'COMPLETE.json',dict(status='GENERATED_REVIEW_PENDING',counts=counts,time=time.time()))
    fh.remove();vh.remove()


if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        code=1;write(ROOT/'FAILURE.json',dict(error=traceback.format_exc(),counts=counts,time=time.time()));raise
    finally:write(ROOT/'WORKER_EXIT.json',dict(code=code,seconds=time.time()-START,counts=counts,time=time.time()))
