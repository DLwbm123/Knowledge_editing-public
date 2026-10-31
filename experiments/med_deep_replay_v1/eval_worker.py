"""Generate a fixed single-owner shard after all replay training succeeds."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT=Path(os.environ['EXPERIMENT_ROOT'])
PARENT=Path(os.environ['PARENT_RUN'])
OLD=Path(os.environ['RAW_MED_ROOT'])
FIXED=Path(os.environ['FIXED_DEPTH_ROOT'])
PART=int(os.environ['PARTITION'])
PARTITIONS=int(os.environ['PARTITIONS'])
DEADLINE=float(os.environ['EXPERIMENT_DEADLINE'])
COUNTS=dict(forwards=0,generations=0,predictions=0,reused_off=0)


def write_once(name,value):
    path=ROOT/name
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(value,f,ensure_ascii=False)
    path.chmod(0o600)


def budget():
    if time.time()>=DEADLINE-60 or (ROOT/'EVAL_STOP.json').exists():
        raise TimeoutError('Deadline or independent guard stop')
    health=json.loads((ROOT/'EVAL_GUARD_HEALTH.json').read_text())
    if time.time()-health['time']>45 or health['hard_end']!=DEADLINE:
        raise RuntimeError('Guard unavailable or deadline mismatch')
    if COUNTS['generations']>508 or COUNTS['predictions']>508:
        raise RuntimeError('Frozen generation budget exceeded')


def main():
    if PARTITIONS not in (4,5) or PART not in range(PARTITIONS):
        raise ValueError('Invalid shard')
    with (ROOT/f'EVAL_ONCE_{PART}').open('x'):
        pass
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('Guard stop')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Deadline')))
    if time.time()>=DEADLINE-60:
        raise TimeoutError('No remaining time')
    signal.alarm(max(1,int(DEADLINE-time.time()-60)))
    sys.path[:0]=[str(ROOT/'tools'),str(PARENT/'private/tools'),str(PARENT/'private/source'),
                  str(PARENT/'private/official_llava')]
    import torch
    import module_run as r
    import generate
    import train_pair
    import runtime_binding
    from first_eval import rng_digest
    from types import SimpleNamespace
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    r.c.budget=budget
    guard=json.loads((ROOT/'TRAIN_GUARD_EXIT.json').read_text())
    if guard['codes'] != [0]*PARTITIONS:
        raise ValueError('Training guard failed')
    states={}
    for i in range(PARTITIONS):
        if json.loads((ROOT/f'EXIT_{i}.json').read_text())['code']!=0:
            raise ValueError('Training shard failed')
        for item in json.loads((ROOT/f'COMPLETE_{i}.json').read_text())['results']:
            if item['owner'] in states:
                raise ValueError('Duplicate owner state')
            states[item['owner']]=item
    admission=json.loads((ROOT/'ROLE_ADMISSION.json').read_text())
    tasks=json.loads((ROOT/'TASK_IDENTITIES_PRIVATE.json').read_text())
    if len(tasks)!=16 or [x['order'] for x in tasks]!=admission['owners'] or set(states)!=set(admission['owners']):
        raise ValueError('Owner coverage mismatch')
    assigned={x['order']:x for x in tasks[PART::PARTITIONS]}
    rows=[x for x in json.loads((ROOT/'EVAL_INPUTS_PRIVATE.json').read_text()) if x['owner'] in assigned]
    prior={(x['owner'],x['query_id']):x for x in json.loads((ROOT/'PRIOR_INPUT_BINDINGS_PRIVATE.json').read_text())}
    if not rows or len(prior)!=508:
        raise ValueError('Input coverage mismatch')
    with (PARENT/'private/EVAL_BINDINGS.json').open('rb') as f:
        config=runtime_binding.extract(f)['configuration_only']
    runtime=load_real_runtime(SimpleNamespace(cpu_gate=PARENT/'private/cpu_gate'))
    runtime.model.eval()
    r.old.fp32(runtime)
    if runtime.generation_config!=config['generation']:
        raise ValueError('Generation configuration mismatch')
    rng=torch.load(OLD/'GENERATION_RNG.pt',map_location='cpu',weights_only=False)
    reference=json.loads((OLD/'PROTOCOL_LOCK.json').read_text())['source_reference']
    expected_runtime=dict(inherited_runtime=config['runtime'],actual_precision='MATH_FP32_FROZEN_FP16_PREFILL',
                          frozen_FP16_prefill=True)
    def count(_module,_args,_kwargs):
        budget()
        COUNTS['forwards']+=1
    handle=runtime.model.register_forward_pre_hook(count,with_kwargs=True)
    try:
        for row in rows:
            owner,q=row['owner'],row['query']
            original=prior[(owner,q['query_id'])]
            if original['input']['runtime']!=expected_runtime or original['prediction_binding']['model_source']!=reference:
                raise ValueError('Original runtime or model binding mismatch')
            if original['prediction_binding']['generation_rng_binding']!=rng_digest(rng):
                raise ValueError('Original RNG binding mismatch')
            old_route=original['route']
            route=dict(activated=old_route['activated'],logical_edit_id=old_route['logical_edit_id'] if old_route['activated'] else None)
            binding={k:original['input'][k] for k in ('question','image_sha256','prompt_ids','attention_mask','generation')}
            state_path=ROOT/'weights'/f'{owner}.pt'
            state=torch.load(state_path,map_location='cpu',weights_only=True)
            if train_pair.state_digest(state['state'])!=states[owner]['final_digest']:
                raise ValueError('Candidate state mismatch')
            key=hashlib.sha256(json.dumps([owner,q['query_id'],'REPLAY'],separators=(',',':')).encode()).hexdigest()
            write_once('eval_attempts/'+key+'.json',dict(time=time.time()))
            if route['activated']:
                COUNTS['generations']+=1
                value=generate.generate_one(runtime,r,assigned[owner],q,route,binding,route,state_path,'CDEEP',rng,budget)
                value.update(arm='REPLAY',prediction_reused_from=None)
            else:
                old_key=hashlib.sha256(json.dumps([owner,q['query_id'],'CDEEP'],separators=(',',':')).encode()).hexdigest()
                candidates=[p for p in (FIXED/'first_eval_outputs'/f'{old_key}.json',FIXED/'batch_eval_outputs'/f'{old_key}.json') if p.is_file()]
                if len(candidates)!=1:
                    raise ValueError('Original OFF prediction missing or ambiguous')
                value=json.loads(candidates[0].read_text())
                if value['owner']!=owner or value['query_id']!=q['query_id'] or value['route']!=route:
                    raise ValueError('Original OFF identity mismatch')
                if any(value[k]!=binding[k] for k in binding) or value['original_execution_binding']!=original['prediction_binding']:
                    raise ValueError('Original OFF input binding mismatch')
                value.update(arm='REPLAY',prediction_reused_from='CDEEP_OFF_BASE',state_file=state_path.name)
                COUNTS['reused_off']+=1
            value.update(owner=owner,original_consumer_id=original['consumer_id'],
                         original_execution_binding=original['prediction_binding'],original_route=old_route)
            COUNTS['predictions']+=1
            write_once('eval_outputs/'+key+'.json',value)
        if COUNTS['predictions']!=len(rows) or COUNTS['generations']+COUNTS['reused_off']!=len(rows):
            raise ValueError('Prediction counts mismatch')
        write_once(f'EVAL_COMPLETE_{PART}.json',dict(time=time.time(),events=len(rows),counts=COUNTS))
    finally:
        handle.remove()


if __name__=='__main__':
    code=0
    try:
        main()
    except BaseException:
        code=1
        write_once(f'EVAL_FAILURE_{PART}.json',dict(time=time.time(),error=traceback.format_exc(),counts=COUNTS))
        raise
    finally:
        write_once(f'EVAL_EXIT_{PART}.json',dict(time=time.time(),code=code,counts=COUNTS))
