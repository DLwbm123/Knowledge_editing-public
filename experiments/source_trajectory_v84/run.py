"""One source-only student trajectory per input and fixed sequence-score audit."""
import os
from pathlib import Path
import signal
import time
import traceback
import v81 as v
from diagnosis import short_context


def sample(item,hook,seed):
    torch=v.torch;g=torch.Generator(device='cpu').manual_seed(seed%(2**63));tokens=[]
    v.generating=True;v.counts['evaluation_generations']+=1
    with torch.inference_mode(),hook.generation_request():
        out=v.runtime.model(**v.pref(item[3]))
        for _ in range(128):
            probability=(out.logits[0,-1].float()/.7).softmax(-1).cpu()
            token=int(torch.multinomial(probability,1,generator=g));tokens.append(token)
            if token==v.runtime.adapter.tokenizer.eos_token_id:break
            past=out.past_key_values;del out
            out=v.runtime.model(input_ids=torch.tensor([[token]],device=v.runtime.device),past_key_values=past,use_cache=True,return_dict=True)
            del past
        del out
    v.generating=False
    return tokens


def score_prefix(item,tokens,hook=None):
    torch=v.torch;pref=v.pref(item[3]);n=pref['inputs_embeds'].shape[1]
    y=torch.tensor([tokens],device=v.runtime.device)
    with torch.inference_mode():
        embedding=v.runtime.model.get_input_embeddings()(y).half().float()
        embeds=torch.cat([pref['inputs_embeds'],embedding],1)
        if hook is not None:
            labels=torch.tensor([[-100]*n+tokens],device=v.runtime.device)
            hook.set_teacher_routing(labels)
        out=v.runtime.model(inputs_embeds=embeds,use_cache=False,return_dict=True)
        logp=out.logits[0,n-1:n+len(tokens)-1].float().log_softmax(-1).gather(1,y[0,:,None]).squeeze(1)
        assert v.torch.isfinite(logp).all()
        value=dict(mean_log_probability=float(logp.mean()),sequence_log_probability=float(logp.sum()),tokens=len(tokens))
        del out,logp,embedding,embeds
    return value


def main():
    with (v.ROOT/f'ONCE_{v.PHASE}_{v.PART}').open('x'):pass
    v.allocations=dict(v.limits['counts'])  # one worker owns all eight original edits
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Original audit deadline')))
    signal.alarm(max(1,int(v.limits['hard_end']-time.time()-300)))
    v.setup();state_run=Path(os.environ['SOURCE_STATE_RUN'])
    candidates=v.read(v.ROOT/'CANDIDATES_FROZEN.json')
    for task in v.read(v.ROOT/'TASKS.json'):
        owner=task['order'];fact=v.read(v.ROOT/'FACTS.json')[str(owner)]
        state=v.r.p.load_state(state_run/'candidates/S0'/f'{owner}.pt')['state']
        expert=v.r.p.expert(state,task['seed'],v.runtime.device)
        rows=v.read(v.ROOT/'SOURCE_ROWS.json')[str(owner)]
        for view,row in enumerate(rows):
            dest=v.ROOT/'trajectories'/str(owner)/f'{view}.json';assert not dest.exists()
            attempt=v.ROOT/'attempts'/str(owner)/f'{view}.json';assert not attempt.exists()
            v.write(attempt,dict(owner=owner,view=view,time=time.time()))
            source=v.prepare(task,[row],True)[0]
            hook=v.r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(v.r.c.LAYER),expert);hook.attach()
            try:
                if owner==1 and view==0:
                    batch=v.r.old.batch_on(v.runtime,source[1])
                    check=v.r.s.fit.short.mechanical(v.runtime,hook,batch)
                    v.write(v.ROOT/'SOURCE_PREFIX_MECHANICAL.json',check)
                    del batch
                tokens=sample(source,hook,task['seed']+1009*view)
            finally:hook.detach()
            text=v.runtime.adapter.tokenizer.decode(tokens,skip_special_tokens=True).strip()
            # Persist the paid trajectory before probability scoring; no hidden resampling on failure.
            v.write(v.ROOT/'sampled'/str(owner)/f'{view}.json',dict(owner=owner,view=view,question=row['question'],reference=row['reference'],response=text,
                raw_token_ids=tokens,EOS=tokens[-1]==v.runtime.adapter.tokenizer.eos_token_id,at_cap=len(tokens)==128))
            candidate_items=v.prepare(task,[dict(row,reference=y) for y in candidates[str(owner)]],True)
            sequences=[x[1][1][x[1][1]!=-100].tolist() for x in candidate_items]+[tokens]
            hook=v.r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(v.r.c.LAYER),expert);hook.attach()
            try:student=[score_prefix(source,t,hook) for t in sequences]
            finally:hook.detach()
            teacher=v.prepare(task,[dict(row,question=short_context(fact,'S3')+'\n\nQuestion: '+row['question'])])[0]
            tokenized=v.runtime.adapter.tokenizer.decode([i for i in teacher[2]['input_ids'][0].tolist() if i>=0])
            assert all(x.lower() in tokenized.lower() for x in fact['medical_record']['values'])
            teacher_scores=[score_prefix(teacher,t) for t in sequences]
            v.write(dest,dict(owner=owner,view=view,question=row['question'],reference=row['reference'],response=text,
                raw_token_ids=tokens,EOS=tokens[-1]==v.runtime.adapter.tokenizer.eos_token_id,at_cap=len(tokens)==128,
                student_scores=student,teacher_scores=teacher_scores,temperature=.7,source_only=True,
                first3_candidates_frozen=True,sample_seed=task['seed']+1009*view))
            v.progress(owner=owner,view=view)
        del expert
    assert v.counts['evaluation_generations']==40 and v.counts['non_generation_forwards']==322
    v.write(v.ROOT/'MAIN_COMPLETE.json',dict(status='SOURCE_AUDIT_GENERATION_COMPLETE_REVIEW_PENDING',samples=40,sequence_scores=320,counts=v.counts,time=time.time()))


if __name__=='__main__':
    code=0
    try:main()
    except BaseException as e:
        code=1;v.write(v.ROOT/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),counts=v.counts,time=time.time()))
        if not (v.ROOT/'STOP').exists():v.write(v.ROOT/'STOP',dict(reason='SOURCE_AUDIT_FAILURE',time=time.time()))
        raise
    finally:
        v.progress(exit_code=code);v.write(v.ROOT/'EXIT.json',dict(exit_code=code,counts=v.counts,time=time.time()))
        v.write(v.ROOT/'RESOURCE_SCOPE_CLOSED.json',dict(time=time.time()))
