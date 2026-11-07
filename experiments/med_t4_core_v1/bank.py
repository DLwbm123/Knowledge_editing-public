"""Evaluate single edits and the real final T4 expert bank with unchanged R0."""
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT=Path(os.environ['T4_REMOTE_ROOT'])
CLOCK=json.loads((ROOT/'CLOCK.json').read_text())
COUNTS=dict(forwards=0,generations=0,consumers=0,reused=0)


def write(name,value):
    p=ROOT/name;p.parent.mkdir(exist_ok=True,parents=True)
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False));temp.replace(p)


def budget():
    assert time.time()<CLOCK['hard_end']-60 and not (ROOT/'EVAL_STOP.json').exists()
    assert time.time()-json.loads((ROOT/'EVAL_GUARD_HEALTH.json').read_text())['time']<45
    assert COUNTS['generations']<=3000


def main():
    with (ROOT/'EVAL_WORKER_ONCE').open('x'):pass
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('guard stop')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original clock')))
    signal.alarm(max(1,int(CLOCK['hard_end']-time.time()-60)))
    parent=Path(os.environ['PARENT_RUN'])
    sys.path[:0]=[str(ROOT/'tools'),os.environ['MED150_TOOLS'],str(parent/'private/tools'),str(parent/'private/source'),str(parent/'private/official_llava')]
    import numpy as np
    import torch
    import module_run as r
    import matched_train as mt
    from types import SimpleNamespace
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from m3bench_repro.editors.routing import MemoryRouter,decision_as_json
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);r.c.budget=budget
    data=json.loads((ROOT/'TASK_INPUTS.private.json').read_text());tasks=data['tasks'];byid={t['edit_id']:t for t in tasks}
    config=json.loads((ROOT/'BASE_CONFIGURATION.private.json').read_text())
    bindings={'configuration_only':config}
    runtime=load_real_runtime(SimpleNamespace(cpu_gate=parent/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime)
    assert runtime.generation_config==config['generation'] and not any(p.requires_grad for p in runtime.model.parameters())
    runtime.run_root=ROOT/'runtime';runtime.run_root.mkdir(exist_ok=True)
    states,routers,hashes={},{},{}
    for task in tasks:
        owner=task['order'];shard=ROOT/data['owner_shards'][str(owner)]
        state=torch.load(shard/'weights'/f'{owner}_MDU.pt',map_location='cpu',weights_only=True)
        assert state['arm']=='MDU' and state['steps']==160
        states[task['edit_id']]=state['state'];hashes[task['edit_id']]=mt.state_digest(state['state'])
        route=torch.load(shard/'routers'/f'{owner}.pt',map_location='cpu',weights_only=True)
        assert len(route['entries'])==1 and route['entries'][0]['logical_edit_id']==task['edit_id']
        routers[task['edit_id']]=route['entries'][0]
    ordered=[routers[t['edit_id']] for t in tasks]
    torch.save(dict(tasks=tasks,states=states,R0_entries=ordered,state_hashes=hashes),ROOT/'T4_CANONICAL.pt')
    rng=torch.load(Path(os.environ['RAW_MED_ROOT'])/'GENERATION_RNG.pt',map_location='cpu',weights_only=False)
    handle=runtime.model.register_forward_pre_hook(lambda *_:COUNTS.__setitem__('forwards',COUNTS['forwards']+1))
    features,cache,route_cache={},{},{}
    for mode in ('single','bank'):
        for row in data['plan']:
            budget();task=byid[row['edit_id']];query=row['query'];owner=task['order']
            runtime.model.half()
            try:
                qkey=r.c.digest(dict(question=query['question'],image_sha256=query['image_sha256']))
                if qkey not in features:
                    b=runtime.build_question_batch(r.s.record(task,query))
                    with torch.inference_mode():
                        features[qkey]=runtime.extract_layer_input_key(b,module_path=runtime.target_lock['balancedit']['targets'][0],pooling='mean').detach().cpu()
                routerkey=task['edit_id'] if mode=='single' else 'BANK'
                if routerkey not in route_cache:
                    route_cache[routerkey]=MemoryRouter.from_state(dict(distance='euclidean',entries=[routers[task['edit_id']]] if mode=='single' else ordered),device=runtime.device)
                decision=route_cache[routerkey].route(features[qkey].to(runtime.device))
                chosen=decision.logical_edit_id if decision.activated else None
                item=r.s.prepare(runtime,bindings,task,[query])[0]
            finally:r.old.fp32(runtime)
            signature=dict(input=item[4],effective_state=hashes[chosen] if chosen else 'UNEDITED_BASE')
            key=r.c.digest(signature);reused=key in cache
            if not reused:
                hook=None
                try:
                    if chosen:
                        expert=r.p.expert(states[chosen],byid[chosen]['seed'],runtime.device).requires_grad_(False)
                        hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert);hook.attach()
                    r.c.restore_rng(rng['rng']);np.random.set_state(rng['numpy_rng']);COUNTS['generations']+=1
                    value,trace=r.s.fit.short.generate(runtime,hook,item[2],item[3])
                    cache[key]=dict(raw_answer=value.decoded_text,raw_token_ids=list(value.raw_token_ids))
                    assert 0<len(cache[key]['raw_token_ids'])<=1024
                finally:
                    if hook:hook.detach()
            else:COUNTS['reused']+=1
            answer=cache[key];cid=r.c.digest([mode,task['edit_id'],row['role'],query['query_id']])
            path=ROOT/'outputs'/f'{cid}.json';assert not path.exists()
            write('outputs/'+path.name,dict(consumer_id=cid,mode=mode,owner=owner,edit_id=task['edit_id'],role=row['role'],
                query_id=query['query_id'],judge_input=item[4],answer=answer,route=decision_as_json(decision),
                effective_expert=chosen,state_hash=signature['effective_state'],prediction_reused=reused,
                EOS=answer['raw_token_ids'][-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(answer['raw_token_ids'])>=1024))
            COUNTS['consumers']+=1
            if COUNTS['consumers']%10==0:write('EVAL_PROGRESS.json',dict(time=time.time(),counts=COUNTS,total=2*len(data['plan'])))
    handle.remove();assert COUNTS['consumers']==2*len(data['plan'])
    write('EVAL_GENERATION_COMPLETE.json',dict(status='ALL_CONSUMERS_FROZEN',time=time.time(),counts=COUNTS,
        experts=len(tasks),expected_consumers=2*len(data['plan']),original_R0=True,separate_T4_bank=True))


if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        code=1;write('EVAL_FAILURE.json',dict(time=time.time(),error=traceback.format_exc(),counts=COUNTS));raise
    finally:write('EVAL_WORKER_EXIT.json',dict(code=code,time=time.time(),counts=COUNTS))
