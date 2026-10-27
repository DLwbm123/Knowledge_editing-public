"""Evaluation-only continuation: frozen states, retained answers, exact zero-write checks."""
import os,signal,time,traceback
import route_common as c
import canonedit
v=c.v


def zero(expert):
    if hasattr(expert,'G1'):return bool(v.torch.count_nonzero(expert.G1)==0)
    return bool(v.torch.count_nonzero(expert.z@expert.vectors)==0)


def mechanical(hook,batch,is_zero):
    if not is_zero:return v.r.s.fit.short.mechanical(v.runtime,hook,batch)
    torch=v.torch;runtime=v.runtime
    teacher,_=v.r.s.fit.short.q.logits(runtime,hook,batch)
    first=int(torch.where(batch.labels[0]!=-100)[0][0]);kwargs=dict(batch.forward_kwargs());kwargs.pop('labels',None)
    for name in ('inputs_embeds','attention_mask','position_ids'):
        if isinstance(kwargs.get(name),torch.Tensor):kwargs[name]=kwargs[name][:,:first]
    with torch.inference_mode(),hook.generation_request():out=runtime.model(**kwargs)
    trace=hook.last_generation_trace
    assert len(trace)==1 and trace[0]['active_predictor_count']==1 and trace[0]['first_active_predictor']==first-1
    assert trace[0]['active_residual_norm']==0
    generated=out.logits[0,-1].detach().cpu().double()
    torch.testing.assert_close(teacher[0],generated,rtol=1e-5,atol=5e-4)
    hook.clear_request_routing()
    with torch.inference_mode():baseline=runtime.model(**kwargs).logits[0,-1].detach().cpu().double()
    assert torch.equal(generated,baseline),'Zero write must exactly preserve Base logits'
    return dict(status='PASS',zero_write=True,exact_Base_logits=True,same_prefix=True,
                generation_trace=list(trace),logits_max_abs=float((teacher[0]-generated).abs().max()),rtol=1e-5,atol=5e-4)


def main():
    part=v.PART
    with (v.ROOT/f'EVAL_RECOVERY_ONCE_{part}').open('x'):pass
    v.counts.update(v.read(v.ROOT/f'EVALUATION_CARRIED_COUNTS_{part}.json'))
    v.allocations={k:n//2 for k,n in v.limits['counts'].items()}
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Original deadline')))
    signal.alarm(max(1,int(v.limits['hard_end']-time.time()-300)))
    v.setup();torch=v.torch;R=canonedit.classes()[0]
    tasks=v.read(v.ROOT/'TASKS.json');assert (v.ROOT/'ALL_CANDIDATES_FROZEN.json').exists()
    for task in tasks[part::2]:
        owner=task['order'];plans=[p for p in v.read(v.ROOT/'EVAL_PLAN.json') if p['owner']==owner]
        routes=v.read(v.ROOT/'ROUTE_BINDINGS_FROZEN.json')[str(owner)]
        items=v.prepare(task,[p['query'] for p in plans])
        native=v.prepare(task,[v.read(v.ROOT/'SOURCE_ROWS.json')[str(owner)][0]],True)[0]
        batch=v.r.old.batch_on(v.runtime,native[1])
        for arm in ['Base']+canonedit.ARMS:
            expert=None;is_zero=False
            if arm!='Base':
                state=v.r.p.load_state(v.ROOT/'candidates'/arm/f'{owner}.pt')['state']
                if arm in ('CE','CANON_TT'):expert=v.r.p.expert(state,owner,v.runtime.device)
                else:
                    expert=R(state['key'].to(v.runtime.device),state['vectors'].to(v.runtime.device));expert.z.data.copy_(state['z'])
                is_zero=zero(expert)
                hook=v.r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(v.r.c.LAYER),expert);hook.attach()
                try:v.write(v.ROOT/'mechanical'/f'recovery1_{arm}_{owner}.json',mechanical(hook,batch,is_zero))
                finally:hook.detach()
            for index,(plan,item,route) in enumerate(zip(plans,items,routes)):
                assert route['question']==item[4]['question'] and route['query_id']==plan['query']['query_id'] and route['image_sha256']==item[4]['image_sha256']
                path=v.ROOT/'outputs'/arm/str(owner)/f'{index}.json'
                if path.exists():
                    old=v.read(path)
                    assert old['query']==plan['query'] and old['binding']==item[4] and old['original_route_preserved']
                    assert old['raw_token_ids'] and old['at_cap']==(len(old['raw_token_ids'])>=1024)
                    if is_zero or route['effective_expert'] is None:
                        assert old['raw_token_ids']==v.read(v.ROOT/'outputs/Base'/str(owner)/f'{index}.json')['raw_token_ids']
                    continue
                hook=None
                if expert is not None and route['effective_expert'] is not None:
                    hook=v.r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(v.r.c.LAYER),expert);hook.attach()
                try:answer=c.generate(hook,item)
                finally:
                    if hook:hook.detach()
                tokens=list(answer.raw_token_ids)
                if is_zero or arm!='Base' and route['effective_expert'] is None:
                    assert tokens==v.read(v.ROOT/'outputs/Base'/str(owner)/f'{index}.json')['raw_token_ids'],'Zero/off-route write must exactly preserve Base tokens'
                v.write(path,dict(owner=owner,arm=arm,query=plan['query'],binding=item[4],raw_answer=answer.decoded_text,raw_token_ids=tokens,
                    EOS=tokens[-1]==v.runtime.adapter.tokenizer.eos_token_id,at_cap=len(tokens)>=1024,original_route_preserved=True))
                v.progress(stage='EVALUATION_RECOVERY1',owner=owner,arm=arm,index=index)
    v.write(v.ROOT/f'EVAL_COMPLETE_{part}.json',dict(counts=v.counts,time=time.time()))


if __name__=='__main__':
    code=0
    try:main()
    except BaseException as e:
        code=1;v.write(v.ROOT/f'FAILURE_worker_{v.PART}.json',dict(error=repr(e),traceback=traceback.format_exc(),counts=v.counts,time=time.time()))
        v.write(v.ROOT/'STOP',dict(reason='EVALUATION_RECOVERY1_FAILURE',time=time.time()));raise
    finally:
        v.progress(exit_code=code);v.write(v.ROOT/f'EXIT_worker_{v.PART}.json',dict(exit_code=code,counts=v.counts,time=time.time()))
