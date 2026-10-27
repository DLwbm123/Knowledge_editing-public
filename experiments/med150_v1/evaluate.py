"""Evaluate new single edits, insertion, and actual 150-expert R0 bank."""
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT=Path(os.environ['EXPERIMENT_ROOT']);PARENT=Path(os.environ['PARENT_RUN']);OLD=Path(os.environ['RAW_MED_ROOT'])
LOCK=json.loads((ROOT/'EVAL_LOCK.json').read_text());CLOCK=json.loads((ROOT/'CLOCK.json').read_text())
COUNTS=dict(forwards=0,generations=0,backwards=0,updates=0,vision_forwards=0)


def write(name,value):
    path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def budget():
    assert time.time()<CLOCK['hard_end']-30 and not (ROOT/'EVAL_STOP.json').exists()
    assert time.time()-json.loads((ROOT/'EVAL_GUARD_HEALTH.json').read_text())['time']<45
    for k,v in LOCK['budgets'].items():
        if k in COUNTS:assert COUNTS[k]<=v,(k,COUNTS[k],v)


def main():
    with (ROOT/'EVAL_WORKER_ONCE').open('x'):pass
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('Resource guard stop')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Original clock')))
    signal.alarm(max(1,int(CLOCK['hard_end']-time.time()-30)))
    sys.path[:0]=[str(ROOT/'tools'),str(PARENT/'private/tools'),str(PARENT/'private/source'),str(PARENT/'private/official_llava')]
    import numpy as np
    import torch
    import module_run as r
    import med133_evaluate
    from m3bench_repro.editors.routing import MemoryRouter,decision_as_json
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from types import SimpleNamespace
    r.c.budget=budget;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(LOCK['physical_GPU']);torch.cuda.set_per_process_memory_fraction(.78)
    assert json.loads((ROOT/'TRAIN_COMPLETE.json').read_text())['status']=='FOUR_MED_AND_FOUR_C_COMPLETE'
    assert json.loads((ROOT/'BASE_MASK_FROZEN.json').read_text())['status']=='FROZEN'
    inputs=json.loads((ROOT/'EVAL_PLAN_FROZEN.json').read_text());assert r.c.digest(inputs)==LOCK['plan_binding']
    package=torch.load(ROOT/'retained/MED150.pt',map_location='cpu',weights_only=True)
    tasks=package['tasks'];states=package['states'];entries=package['R0_entries'];hashes=package['state_hashes']
    assert len(tasks)==len(states)==len(entries)==150 and len(inputs['plan'])==2207
    assert all(med133_evaluate.state_id(v)==hashes[k] for k,v in states.items())
    byid={t['edit_id']:t for t in tasks};assert len(byid)==150
    runtime=load_real_runtime(SimpleNamespace(cpu_gate=PARENT/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime)
    assert not any(p.requires_grad for p in runtime.model.parameters())
    bindings=r.c.read(PARENT/'private/EVAL_BINDINGS.json');assert runtime.generation_config==next(iter(bindings.values()))['generation']
    rng=torch.load(OLD/'GENERATION_RNG.pt',map_location='cpu',weights_only=False)
    def count(_m,args,kwargs):budget();COUNTS['forwards']+=1
    fh=runtime.model.register_forward_pre_hook(count,with_kwargs=True)
    vh=runtime.model.get_vision_tower().register_forward_pre_hook(lambda *_:COUNTS.__setitem__('vision_forwards',COUNTS['vision_forwards']+1))
    layer=runtime.target_lock['balancedit']['targets'][0];routers={};features={};answers={}
    for begin in range(0,len(inputs['plan']),32):
        budget();runtime.model.half();prepared=[]
        try:
            for row in inputs['plan'][begin:begin+32]:
                task=tasks[row['owner']-1] if row['owner'] else tasks[0];query=row['query'];qkey=r.c.digest(query)
                if qkey not in features:
                    b=runtime.build_question_batch(r.s.record(task,query))
                    with torch.inference_mode():features[qkey]=runtime.extract_layer_input_key(b,module_path=layer,pooling='mean').detach().cpu()
                routerkey=(row['mode'],row['prefix'],row['owner'] if row['mode']=='single' else 0)
                allowed=[entries[row['owner']-1]] if row['mode']=='single' else entries[:row['prefix']]
                if routerkey not in routers:routers[routerkey]=MemoryRouter.from_state(dict(distance='euclidean',entries=allowed),device=runtime.device)
                decision=routers[routerkey].route(features[qkey].to(runtime.device));selected=decision.logical_edit_id if decision.activated else None
                assert selected is None or selected in {e['logical_edit_id'] for e in allowed}
                item=r.s.prepare(runtime,bindings,task,[query])[0]
                prepared.append((row,item,decision_as_json(decision),selected))
        finally:r.old.fp32(runtime)
        for row,item,route,selected in prepared:
            budget();cid=json.dumps([row[k] for k in ('method','mode','prefix','owner')]+[row['query']['query_id']],separators=(',',':'))
            dest=ROOT/'eval_outputs'/(r.c.digest(cid)+'.json');assert not dest.exists(),'No implicit prediction retry'
            binding=dict(input=item[4],effective_state=hashes[selected] if selected else 'UNEDITED_BASE',
                source_commit=LOCK['source_commit'],runtime_reference=LOCK['runtime_reference'],rng_reference=LOCK['rng_reference'],deterministic_greedy=True)
            key=r.c.digest(binding);reused=key in answers
            if reused:answer=answers[key]
            else:
                hook=None
                try:
                    if selected is not None:
                        expert=r.p.expert(states[selected],byid[selected]['seed'],runtime.device).requires_grad_(False)
                        hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert);hook.attach()
                    r.c.restore_rng(rng['rng']);np.random.set_state(rng['numpy_rng']);COUNTS['generations']+=1
                    value,trace=r.s.fit.short.generate(runtime,hook,item[2],item[3]);observed=r.c.rng()
                    assert observed['python_rng']==rng['rng']['python_rng']
                    for k in ('torch_rng','cuda_rng'):assert torch.equal(observed[k],rng['rng'][k])
                    answer=dict(raw_answer=value.decoded_text,raw_token_ids=list(value.raw_token_ids),generation_trace=trace);assert answer['raw_token_ids']
                    answers[key]=answer
                finally:
                    if hook is not None:hook.detach()
            write(str(dest.relative_to(ROOT)),dict(method='MED',mode=row['mode'],prefix=row['prefix'],owner=row['owner'],consumer_id=cid,
                binding=dict(input=row['query'],judge_input=item[4]),R0={k:answer[k] for k in ('raw_answer','raw_token_ids')},route=route,
                effective_expert=selected,effective_state_hash=binding['effective_state'],prediction_binding=binding,prediction_cache_key=key,
                prediction_reused=reused,actual_Base_R0_evaluated=True,generation_trace=answer['generation_trace'],
                EOS=answer['raw_token_ids'][-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(answer['raw_token_ids'])>=1024))
        write('EVAL_PROGRESS.json',dict(time=time.time(),consumers=min(begin+32,len(inputs['plan'])),total=len(inputs['plan']),counts=COUNTS,unique_features=len(features),unique_predictions=len(answers)))
        print('EVAL',min(begin+32,len(inputs['plan'])),len(inputs['plan']),flush=True)
    fh.remove();vh.remove();assert len(list((ROOT/'eval_outputs').glob('*.json')))==len(inputs['plan'])
    write('EVAL_GENERATION_COMPLETE.json',dict(time=time.time(),consumers=len(inputs['plan']),single=55,insertion=4,bank150=2148,counts=COUNTS,actual_R0=True,cuda_max_memory_allocated=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        code=1;write('EVAL_FAILURE.json',dict(time=time.time(),error=traceback.format_exc(),counts=COUNTS));raise
    finally:write('EVAL_WORKER_EXIT.json',dict(time=time.time(),code=code,counts=COUNTS))
