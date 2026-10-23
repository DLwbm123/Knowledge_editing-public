"""Fixed source-supported sequence distribution, original TT operator training."""
import math
import os
import signal
import time
import traceback
import v81 as v
from diagnosis import short_context


def source_items(task, candidates, arm):
    rows=v.read(v.ROOT/'SOURCE_ROWS.json')[str(task['order'])]
    expanded=[dict(row,reference=answer) for row in rows for answer in
              (candidates[:1] if arm=='S0' else candidates)]
    return v.prepare(task,expanded,True)


def score(task):
    torch=v.torch;owner=str(task['order']);fact=v.read(v.ROOT/'FACTS.json')[owner]
    candidates=v.read(v.ROOT/'CANDIDATES_FROZEN.json')[owner]
    source=v.read(v.ROOT/'SOURCE_ROWS.json')[owner]
    native=source_items(task,candidates,'S1');records=[]
    for view,row in enumerate(source):
        names=['Base','S1','S2','S3']
        rows=[row]+[dict(row,question=short_context(fact,a)+'\n\nQuestion: '+row['question']) for a in names[1:]]
        items=v.prepare(task,rows)
        values={}
        for name,item in zip(names,items):
            pref=v.pref(item[3]);n=pref['inputs_embeds'].shape[1]
            if name!='Base':
                tokenized=v.runtime.adapter.tokenizer.decode([i for i in item[2]['input_ids'][0].tolist() if i>=0])
                assert all(x.lower() in tokenized.lower() for x in fact['medical_record']['values'])
            scores=[];sequence=[];lengths=[]
            for j in range(3):
                frozen=native[view*3+j][1];labels=frozen[1]
                ids=labels[labels!=-100].tolist()
                assert ids[-1]==v.runtime.adapter.tokenizer.eos_token_id
                with torch.inference_mode():
                    y=torch.tensor([ids],device=v.runtime.device)
                    embed=v.runtime.model.get_input_embeddings()(y).half().float()
                    out=v.runtime.model(inputs_embeds=torch.cat([pref['inputs_embeds'],embed],1),use_cache=False,return_dict=True)
                    logits=out.logits[0,n-1:n+len(ids)-1].float()
                    logp=logits.log_softmax(-1).gather(1,y[0,:,None]).squeeze(1)
                    assert torch.isfinite(logp).all()
                    scores.append(float(logp.mean()));sequence.append(float(logp.sum()));lengths.append(len(ids))
                    if view==0 and name=='Base' and j==0:
                        batch=v.r.old.batch_on(v.runtime,frozen)
                        direct=v.runtime.model(**batch.forward_kwargs())
                        start=int(torch.where(batch.labels[0]!=-100)[0][0])
                        torch.testing.assert_close(logits,direct.logits[0,start-1:start+len(ids)-1],rtol=1e-5,atol=5e-4)
                        del batch,direct
                    del out,logits,logp,embed
            probability=torch.softmax(torch.tensor(scores,dtype=torch.float64),0).tolist()
            values[name]=dict(mean_log_probability=scores,sequence_log_probability=sequence,
                token_counts=lengths,restricted_distribution=probability,
                native_support_log_mass=float(torch.logsumexp(torch.tensor(sequence,dtype=torch.float64),0)))
        q={}
        for arm in names[1:]:
            q[arm]=[(a+b)*.5 for a,b in zip(values['Base']['restricted_distribution'],values[arm]['restricted_distribution'])]
            assert all(x>0 and math.isfinite(x) for x in q[arm]) and abs(sum(q[arm])-1)<1e-12
        records.append(dict(view=view,conditions=values,q=q))
        v.progress(owner=int(owner),view=view,teacher_sequence_scores=(view+1)*12)
    dest=v.ROOT/'scores'/f'{owner}.json';assert not dest.exists()
    v.write(dest,dict(status='PASS',owner=int(owner),records=records,candidates=3,source_only=True,
        candidate_review_before_scores=True,sequence_scores=60,Base_reference_path_parity=True))


def gradient(expert,hook,batches,step,arm,score_rows):
    expert.zero_grad(set_to_none=True);losses=[]
    for view,outer in ((0,.5),(1+(step-1)%4,.5)):
        weights=[1.] if arm=='S0' else [.5+.5*score_rows[view]['q'][arm][0]]+[.5*x for x in score_rows[view]['q'][arm][1:]]
        assert abs(sum(weights)-1)<1e-12
        for j,weight in enumerate(weights):
            batch=batches[view*(1 if arm=='S0' else 3)+j]
            hook.set_teacher_routing(batch.labels)
            loss=v.runtime.compute_loss(batch)
            assert v.torch.isfinite(loss)
            (outer*weight*loss).backward();v.counts['backwards']+=1
            losses.append(float(loss.detach()))
    return {k:p.grad.detach().clone() for k,p in expert.named_parameters()},losses


def train(task,fisher):
    from m3bench_repro.editors.llava_runtime import seed_everything
    torch=v.torch;r=v.r;owner=task['order']
    candidates=v.read(v.ROOT/'CANDIDATES_FROZEN.json')[str(owner)]
    score_rows=v.read(v.ROOT/'scores'/f'{owner}.json')['records']
    stepsizes=[x['operator_step_norm'] for x in v.read(v.PARENT/'private/results/C'/f'{owner}.json')['curve']]
    regularization=v.read(v.PARENT/'private/calibration146'/f'{owner}.json')['coefficient']
    original=r.p.load_state(v.PARENT/'private/candidates/C'/f'{owner}.pt')['states']['BASE']
    assert sum(x.numel() for x in original.values())==7168
    for arm in ('S0','S1','S2','S3'):
        point=v.ROOT/'candidates'/arm/f'{owner}.pt';assert not point.exists()
        seed_everything(task['seed']);items=source_items(task,candidates,arm)
        batches=[r.old.batch_on(v.runtime,x[1]) for x in items]
        expert=r.p.expert(original,task['seed'],v.runtime.device)
        expert.load_state_dict(r.mm.canonicalize(original)[0])
        hook=r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(r.c.LAYER),expert);hook.attach()
        curve=[];scalar=dict(step=0,second_moments=[0.,0.])
        try:
            if arm=='S0':
                test=r.old.snapshot(expert);test['G1'].fill_(1e-5);expert.load_state_dict(test)
                check=r.s.fit.short.mechanical(v.runtime,hook,batches[0])
                v.write(v.ROOT/'mechanical'/f'{owner}.json',check)
                expert.load_state_dict(r.mm.canonicalize(original)[0])
            with v.cache_class(v.runtime,batches) as cache:
                for step in range(1,161):
                    v.budget();before=r.old.snapshot(expert)
                    grads,losses=gradient(expert,hook,batches,step,arm,score_rows)
                    selected,scalar,audit=v.training_math.step(r,before,grads,fisher,regularization,scalar,stepsizes[step-1],'M13P')
                    expert.load_state_dict(selected);v.counts['operator_backwards']+=audit['operator_backwards'];v.counts['updates']+=1
                    curve.append(dict(step=step,source_NLL=losses,operator_step_norm=r.fm.map_norm(before,selected),**audit))
                    if step%20==0:
                        v.save_state(v.ROOT/'resume'/arm/f'{owner}.pt',dict(state=r.old.snapshot(expert),scalar=scalar,step=step,
                            curve=curve,counts=dict(v.counts),rng=r.c.rng()))
                        v.progress(owner=owner,arm=arm,step=step)
                assert cache.misses==(5 if arm=='S0' else 15)
                assert cache.hits==(315 if arm=='S0' else 945)
            final=r.old.snapshot(expert)
            v.save_state(point,dict(state=final,scalar=scalar,owner=owner,arm=arm,seed=task['seed']))
            loaded=r.p.load_state(point)['state'];assert all(torch.equal(loaded[k].cpu(),final[k].cpu()) for k in final)
            if arm=='S0':
                old=v.read(v.PARENT/'private/results/M13P'/f'{owner}.json')['curve'];assert len(old)==len(curve)==160
                for a,b in zip(old,curve):
                    assert a['source_NLL']==b['source_NLL'],'S0 trajectory drift'
                    assert a['operator_step_norm']==b['operator_step_norm'],'S0 operator drift'
            v.write(v.ROOT/'training'/arm/f'{owner}.json',dict(status='PASS',owner=owner,arm=arm,steps=160,
                curve=curve,save_load_exact=True,original_S0_curve_exact=arm=='S0',regularization=regularization,
                frozen_answer_support=True,exact_expectation=True))
        finally:hook.detach()
        del batches,items,expert


def main():
    with (v.ROOT/f'ONCE_{v.PHASE}_{v.PART}').open('x'):pass
    assert v.read(v.ROOT/'CANDIDATE_REVIEW_FROZEN.json')['status']=='PASS'
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Original deadline')))
    signal.alarm(max(1,int(v.limits['hard_end']-time.time()-300)))
    v.setup();tasks=v.read(v.ROOT/'TASKS.json')[v.PART::2]
    if v.PHASE=='teacher':
        for task in tasks:score(task)
    elif v.PHASE=='train':
        assert v.read(v.ROOT/'SUPERVISION_FROZEN.json')['status']=='PASS'
        fisher={k:x.to(v.runtime.device) if isinstance(x,v.torch.Tensor) else x for k,x in v.r.p.load_state(v.PARENT/'private/OPERATOR_FISHER.pt').items()}
        for task in tasks:train(task,fisher)
    elif v.PHASE=='evaluate':
        assert v.read(v.ROOT/'ALL_CANDIDATES_FROZEN.json')['count']==32
        for task in tasks:v.evaluate(task)
    else:raise ValueError(v.PHASE)
    v.write(v.ROOT/f'COMPLETE_{v.PHASE}_{v.PART}.json',dict(status='PASS',counts=v.counts,time=time.time()))


if __name__=='__main__':
    code=0
    try:main()
    except BaseException as e:
        code=1;v.write(v.ROOT/f'FAILURE_{v.PHASE}_{v.PART}.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time(),counts=v.counts,no_implicit_retry=True))
        if not (v.ROOT/'STOP').exists():v.write(v.ROOT/'STOP',dict(reason='WORKER_FAILURE',phase=v.PHASE,partition=v.PART,time=time.time()))
        raise
    finally:
        v.progress(exit_code=code);v.write(v.ROOT/f'EXIT_{v.PHASE}_{v.PART}.json',dict(exit_code=code,counts=v.counts,time=time.time()))
