"""One frozen source-only endpoint restoration; original C/D training is reused."""
import os
import time
import traceback
import torch
import scope as s
s.q.GPUS=(6,7)
PARENT=s.Path(os.environ['SCOPE_PARENT'])
_original_baseline=s.baseline

def baseline(t,row):
    path=_original_baseline(t,row)
    return PARENT/path.relative_to(s.RUN) if path.is_relative_to(s.RUN) else path
s.baseline=baseline
s.jobs=lambda protected=None:[(t,a) for t in s.tasks() for a in (('C','D') if protected is None else ('D',) if protected else ('C',))]

def source_stats(runtime,hook,batches):
    losses=[];margins=[]
    for b in batches:
        logits,target=s.q.logits(runtime,hook,b)
        losses.append(torch.nn.functional.cross_entropy(logits.double(),target))
        correct=logits.gather(1,target[:,None]).squeeze(1)
        other=logits.clone();other.scatter_(1,target[:,None],float('-inf'))
        margins.append(correct-other.max(dim=1).values)
    return torch.stack(losses),margins

def qualifies(losses,margins,base_losses,reference_losses,reference_margins):
    gain=base_losses-losses;required=.9*(base_losses-reference_losses).clamp_min(0)
    gains_ok=bool(torch.all(gain>=required-1e-8))
    # Only reference-correct teacher prefixes are constrained; wrong targets are not repaired by definition.
    tokens_ok=all(bool(torch.all(m[r>0]>=.9*r[r>0]-1e-8)) for m,r in zip(margins,reference_margins))
    return gains_ok and tokens_ok

def selfcheck():
    base=torch.tensor([2.,2.],dtype=torch.float64);ref=torch.tensor([1.,1.],dtype=torch.float64);m=[torch.tensor([1.,-1.])]
    assert qualifies(ref,m,base,ref,m)
    assert not qualifies(torch.tensor([1.2,.8]),m,base,ref,m)
    assert not qualifies(ref,[torch.tensor([.8,2.])],base,ref,m)
    assert qualifies(ref,[torch.tensor([.95,-20.])],base,ref,m)
    return dict(status='PASS',individual_gain_and_reference_correct_token_guards=True)

def restore_worker():
    part=int(os.environ['PARTITION']);gpu=s.q.GPUS[part];ts=s.tasks()[part::2]
    s.old.calibration.require_memory(gpu)
    counts=dict(forwards=0,backwards=0,updates=0,generation_attempts=0)
    with s.p.lease(gpu):
        runtime,bindings=s.c.load(gpu);frozen={t['order']:s.prepare(runtime,bindings,t,s.source(t),True) for t in ts};s.old.fp32(runtime)
        h=runtime.model.register_forward_pre_hook(lambda *_:counts.__setitem__('forwards',counts['forwards']+1))
        try:
            for t in ts:
                c=torch.load(s.point(t,'C'),map_location='cpu',weights_only=True);d=torch.load(s.point(t,'D'),map_location='cpu',weights_only=True)
                assert all(torch.equal(c['states']['BASE'][k],d['states']['BASE'][k]) for k in s.q.KEYS)
                expert=s.p.expert(c['states']['BASE'],t['seed'],runtime.device).requires_grad_(False)
                hook=s.q.scoped.replay.MedTraceLayerHook(runtime.get_module(s.c.LAYER),expert);hook.attach()
                try:
                    batches=[s.old.batch_on(runtime,v[1]) for v in frozen[t['order']]]
                    def stats(state):
                        expert.load_state_dict(state);return source_stats(runtime,hook,batches)
                    base_loss,_=stats(c['states']['BASE']);ref_loss,ref_margin=stats(c['states']['FINAL'])
                    start={k:v.to(runtime.device) for k,v in d['states']['FINAL'].items()};ref={k:v.to(runtime.device) for k,v in c['states']['FINAL'].items()}
                    assert qualifies(ref_loss,ref_margin,base_loss,ref_loss,ref_margin),'Reference itself fails: preserve evidence, no automatic continuation'
                    losses,margins=stats(start);ok=qualifies(losses,margins,base_loss,ref_loss,ref_margin);trace=[];chosen=start;alpha=0.
                    if not ok:
                        lo,hi=0.,1.;chosen=ref;alpha=1.
                        for _ in range(16):
                            a=(lo+hi)/2;state=s.q.add(start,a*(s.q.flatten(ref)-s.q.flatten(start)))
                            losses,margins=stats(state);valid=qualifies(losses,margins,base_loss,ref_loss,ref_margin);trace.append(dict(alpha=a,qualified=valid))
                            if valid:hi,chosen,alpha=a,state,a
                            else:lo=a
                    losses,margins=stats(chosen);assert qualifies(losses,margins,base_loss,ref_loss,ref_margin)
                    s.c.save(s.point(t,'E'),dict(states=dict(BASE=c['states']['BASE'],FINAL=s.previous.weights(chosen)),arm='E',order=t['order'],lock=s.c.digest(s.c.read(s.RUN/'private/SCOPE_LOCK.json'))))
                    s.c.write(s.RUN/'private/results/E'/f"{t['order']}.json",dict(alpha=alpha,reference_qualified=True,initial_D_qualified=ok,endpoint_qualified=True,trace=trace,base_losses=base_loss.tolist(),reference_losses=ref_loss.tolist(),final_losses=losses.tolist(),reference_positive_tokens=sum(int((m>0).sum()) for m in ref_margin)))
                    for item in frozen[t['order']]:s.emit(runtime,hook,t,'E',item,counts)
                finally:hook.detach()
        finally:h.remove();s.c.write(s.RUN/'private'/f'COUNTS_endpoint_{part}.json',counts)

def verify_reference():
    for t in s.tasks():
        for arm in ('C','D'):
            for row in s.source(t):
                current=s.c.read(s.output(arm,t['order'],row));prior=s.c.read(PARENT/'private/outputs'/arm/str(t['order'])/(row['query_id']+'.json'))
                assert current['binding']==prior['binding'] and current['R0']==prior['R0'],'Rebuilt source mismatch: stop before successor'
    s.c.write(s.RUN/'private/REFERENCE_EXACT.json',dict(source_outputs=112,status='PASS',training_reconstruction_charged=True))

def evaluate():
    assert s.c.read(s.RUN/'private/CANDIDATES_FROZEN.json')['candidates']==24
    part=int(os.environ['PARTITION']);gpu=s.q.GPUS[part];ts=s.tasks()[part::2];s.old.calibration.require_memory(gpu)
    counts=dict(forwards=0,backwards=0,updates=0,generation_attempts=0)
    with s.p.lease(gpu):
        runtime,bindings=s.c.load(gpu);frozen={t['order']:s.prepare(runtime,bindings,t,s.events(t)+s.previous.data.held()) for t in ts};s.old.fp32(runtime)
        h=runtime.model.register_forward_pre_hook(lambda *_:counts.__setitem__('forwards',counts['forwards']+1))
        try:
            for t in ts:
                saved=torch.load(s.point(t,'E'),map_location='cpu',weights_only=True);expert=s.p.expert(saved['states']['FINAL'],t['seed'],runtime.device).requires_grad_(False)
                hook=s.q.scoped.replay.MedTraceLayerHook(runtime.get_module(s.c.LAYER),expert);hook.attach()
                try:
                    for item in frozen[t['order']]:s.emit(runtime,hook,t,'E',item,counts)
                finally:hook.detach()
        finally:h.remove();s.c.write(s.RUN/'private'/f'COUNTS_endpoint_eval_{part}.json',counts)

def controller():
    import pipeline
    for action in ('scope_raw','scope_protected'):
        pipeline.wait([pipeline.launch('endpoint.py',action,g,i) for i,g in enumerate(s.q.GPUS)])
    verify_reference()
    pipeline.wait([pipeline.launch('endpoint.py','endpoint_restore',g,i) for i,g in enumerate(s.q.GPUS)])
    assert all(s.point(t,a).exists() for t in s.tasks() for a in ('C','D','E'))
    s.c.write(s.RUN/'private/CANDIDATES_FROZEN.json',dict(candidates=24,epoch=time.time(),selection=False))
    pipeline.wait([pipeline.launch('endpoint.py','endpoint_eval',g,i) for i,g in enumerate(s.q.GPUS)])
    pipeline.wait([pipeline.launch('endpoint_route.py','endpoint_route',6,0)])
    counts=[s.c.read(p) for p in (s.RUN/'private').glob('COUNTS_*.json')]
    assert sum(c['updates'] for c in counts)==2560 and sum(c['backwards'] for c in counts)<=88352
    outputs=[s.c.read(p) for p in (s.RUN/'private/outputs').glob('*/*/*.json')]
    assert len(outputs)==1270 and sum(c['generation_attempts'] for c in counts)<=1270
    nong=sum(c['forwards'] for c in counts)-sum(o['forwards'] for o in outputs if not o.get('alias_of'))
    assert nong<=300000
    s.c.write(s.RUN/'private/ENDPOINT_GENERATION_COMPLETE.json',dict(status='GENERATED_SCORING_PENDING',outputs=1270,counts=counts,non_generation_forwards=nong,epoch=time.time()))

if __name__=='__main__':
    try:
        selfcheck()
        {'scope_raw':s.train,'scope_protected':s.train,'endpoint_restore':restore_worker,'endpoint_eval':evaluate,'endpoint_controller':controller}[os.environ['ACTION']]()
    except BaseException as e:
        s.c.write(s.RUN/'private'/f"FAILURE_{os.environ.get('ACTION')}_{os.environ.get('PARTITION')}.json",dict(error=repr(e),traceback=traceback.format_exc(),epoch=time.time()));raise
