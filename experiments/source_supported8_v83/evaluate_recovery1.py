"""Evaluation-only repair: restore frozen original routing; reuse trained states."""
import signal
import time
import traceback
import v81 as v


def evaluate(task):
    torch=v.torch;owner=task['order']
    plans=[x for x in v.read(v.ROOT/'EVAL_PLAN.json') if x['owner']==owner]
    routes=v.read(v.ROOT/'ROUTE_BINDINGS_FROZEN.json')[str(owner)]
    items=v.prepare(task,[p['query'] for p in plans])
    for arm in ('S0','S1','S2','S3'):
        saved=v.r.p.load_state(v.ROOT/'candidates'/arm/f'{owner}.pt')
        expert=v.r.p.expert(saved['state'],task['seed'],v.runtime.device)
        for index,(plan,item,binding) in enumerate(zip(plans,items,routes)):
            dest=v.ROOT/'outputs'/arm/str(owner)/f'{index}.json'
            if dest.exists():
                assert str(dest.relative_to(v.ROOT)) in v.read(v.ROOT/'RECOVERY1_REUSED_OUTPUTS.json')['paths']
                continue
            assert binding['index']==index and binding['query_id']==plan['query']['query_id']
            assert binding['image_sha256']==item[4]['image_sha256']
            assert binding['question']==item[4]['question']
            selected=binding['effective_expert'];hook=None
            if selected is not None:
                assert selected==task['edit_id']
                hook=v.r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(v.r.c.LAYER),expert)
                hook.attach()
            try:
                v.generating=True;v.counts['evaluation_generations']+=1
                with torch.inference_mode():answer,trace=v.r.s.fit.short.generate(v.runtime,hook,item[2],item[3])
                v.generating=False;tokens=list(answer.raw_token_ids)
                out=dict(owner=owner,arm=arm,query=plan['query'],binding=item[4],raw_answer=answer.decoded_text,
                    raw_token_ids=tokens,EOS=tokens[-1]==v.runtime.adapter.tokenizer.eos_token_id,
                    at_cap=len(tokens)>=1024,route=binding['route'],effective_expert=selected,
                    original_route_preserved=True,evaluation_repair='RESTORE_ORIGINAL_ROUTER',
                    generation_trace=trace)
                if arm=='S0':
                    historical=v.read(plan['historical_path'])
                    if historical['R0']['raw_token_ids']!=tokens:
                        v.write(v.ROOT/'failed_evaluation_recovery1'/str(owner)/f'{index}.json',out)
                        raise AssertionError('S0 historical generation drift after original routing restoration')
                    out['historical_token_parity']=True
                v.write(dest,out);v.progress(owner=owner,arm=arm,evaluation_index=index)
            finally:
                if hook is not None:hook.detach()
        del expert


if __name__=='__main__':
    code=0
    try:
        with (v.ROOT/f'ONCE_{v.PHASE}_{v.PART}').open('x'):pass
        assert v.read(v.ROOT/'ALL_CANDIDATES_FROZEN.json')['count']==32
        signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Original deadline')))
        signal.alarm(max(1,int(v.limits['hard_end']-time.time()-300)))
        v.setup()
        for task in v.read(v.ROOT/'TASKS.json')[v.PART::2]:evaluate(task)
        v.write(v.ROOT/f'COMPLETE_{v.PHASE}_{v.PART}.json',dict(status='PASS',counts=v.counts,time=time.time()))
    except BaseException as e:
        code=1;v.write(v.ROOT/f'FAILURE_{v.PHASE}_{v.PART}.json',dict(error=repr(e),traceback=traceback.format_exc(),counts=v.counts,time=time.time()))
        if not (v.ROOT/'STOP').exists():v.write(v.ROOT/'STOP',dict(reason='RECOVERY_EVALUATION_FAILURE',time=time.time()))
        raise
    finally:
        v.progress(exit_code=code);v.write(v.ROOT/f'EXIT_{v.PHASE}_{v.PART}.json',dict(exit_code=code,counts=v.counts,time=time.time()))
