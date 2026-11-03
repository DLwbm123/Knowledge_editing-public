"""Evaluate one shard of the frozen MED150 R0 bank without decoding references."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

root = Path(os.environ['EXPERIMENT_ROOT'])
parent = Path(os.environ['PARENT_RUN'])
original = Path(os.environ['ORIGINAL_ROOT'])
old150 = Path(os.environ['ADDITIONS_ROOT'])
fixed = Path(os.environ['FIXED_ROOT'])
shard = int(os.environ['SHARD'])
deadline = float(os.environ['EXPERIMENT_DEADLINE'])
arm=os.environ['ARM']
if arm not in ('CE','UNIFORM'): raise ValueError('invalid arm')
counts = dict(forwards=0, generations=0, reused=0)


def write(name, value, once=False):
    path = root/name
    path.parent.mkdir(parents=True, exist_ok=True)
    if once:
        with path.open('x') as f:
            json.dump(value, f, ensure_ascii=False)
        path.chmod(0o600)
    else:
        temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(value,ensure_ascii=False))
        temp.replace(path)


def budget():
    if time.time() >= deadline-120 or (root/'EVAL_STOP.json').exists():
        raise TimeoutError('original deadline or guard stop')
    h=json.loads((root/'EVAL_GUARD_HEALTH.json').read_text())
    if time.time()-h['time']>45 or h['hard_end']!=deadline:
        raise RuntimeError('evaluation guard stale')
    if counts['generations']>2148:
        raise ValueError('generation cap exceeded')


def main():
    with (root/f'EVAL_WORKER_{shard}_ONCE').open('x'):
        pass
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('guard stop')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original deadline')))
    signal.alarm(max(1,int(deadline-time.time()-120)))
    sys.path[:0]=[str(root/'tools'),str(Path(os.environ['FULL_D_ROOT'])/'tools'),str(fixed/'tools'),str(parent/'private/tools'),
                  str(parent/'private/source'),str(parent/'private/official_llava')]
    import torch
    import module_run as r
    import fixed_generate
    import runtime_binding
    from types import SimpleNamespace
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    r.c.budget=budget
    for n in ('TRAIN_GUARD_EXIT.json',):
        if json.loads((root/n).read_text()).get('code') != 0:
            raise ValueError('training guard did not succeed')
    if any(json.loads((root/f'TRAIN_EXIT_{i}.json').read_text()).get('code')!=0 for i in range(4)):
        raise ValueError('training shard failed')
    if json.loads((root/'SCORING_CHAIN_ADMISSION.json').read_text()).get('status')!='SCORING_CHAIN_ADMITTED_WITH_UNKNOWN_PRESERVED':
        raise ValueError('scoring dependency missing')
    inputs=json.loads((root/'BANK_INPUTS_PRIVATE.json').read_text())
    if inputs['status']!='BANK_INPUTS_BOUND' or len(inputs['records'])!=2148 or inputs['candidate_count']!=int(os.environ['EXPECTED_GENERATIONS']):
        raise ValueError('bank input coverage mismatch')
    entries=json.loads((root/'TRAIN_SOURCE_BOUND_PRIVATE.json').read_text())
    lock=json.loads((root/'SPAN_LOCK.private.json').read_text())
    owners={x['order'] for x in lock['records'] if x['status']=='WEAK_TOKEN_MATCH'}
    tokens=json.loads((root/'TOKEN_LOCK.private.json').read_text())
    if len(owners)!=25 or not owners.issubset({x['order'] for x in tokens['records'] if x['status']=='TOKEN_BOUND'}):
        raise ValueError('fixed 25 token identities differ')
    tasks={x['task']['edit_id']:x['task'] for x in entries if x['task']['order'] in owners}
    if len(tasks)!=25:
        raise ValueError('candidate task binding mismatch')
    with (parent/'private/EVAL_BINDINGS.json').open('rb') as f:
        config=runtime_binding.extract(f)['configuration_only']
    runtime=load_real_runtime(SimpleNamespace(cpu_gate=parent/'private/cpu_gate'))
    runtime.model.eval();r.old.fp32(runtime)
    if runtime.generation_config != config['generation']:
        raise ValueError('generation configuration differs')
    rng=torch.load(original/'GENERATION_RNG.pt',map_location='cpu',weights_only=False)
    def count(_module,_args,_kwargs):
        budget();counts['forwards']+=1
    handle=runtime.model.register_forward_pre_hook(count,with_kwargs=True)
    try:
        selected=[x for i,x in enumerate(inputs['records']) if i%4==shard]
        for index,row in enumerate(selected):
            cid=row['consumer_id']
            key=hashlib.sha256(cid.encode()).hexdigest()
            dest=f'predictions/{key}.json'
            route=row['route'];selected_expert=row['effective_expert']
            if selected_expert not in tasks:
                counts['reused']+=1
                value={'consumer_id':cid,'query_id':row['query']['query_id'],'route':route,
                       'prediction_reused_from':row['original_filename'],'candidate':False}
            else:
                task=tasks[selected_expert]
                normalized={'activated':bool(route['activated']),'logical_edit_id':selected_expert}
                if not normalized['activated'] or route['logical_edit_id']!=selected_expert:
                    raise ValueError('original R0 effective expert mismatch')
                q={k:row['query'][k] for k in ('query_id','question','image_path','image_sha256')}
                b={k:row['judge_input'][k] for k in ('question','image_sha256','prompt_ids','attention_mask','generation')}
                write(f'attempts/{key}.json',{'consumer_id':cid,'time':time.time()},once=True)
                counts['generations']+=1
                value=fixed_generate.generate_one(runtime,r,task,q,normalized,b,normalized,
                    root/'weights'/f"{task['order']}_{arm}.pt",'C21',rng,budget)
                value.update(consumer_id=cid,original_route=route,candidate=True,prediction_reused_from=None)
            write(dest,value,once=True)
            write(f'EVAL_PROGRESS_{shard}.json',{'completed':index+1,'expected':len(selected),'counts':counts})
        if counts['generations']+counts['reused']!=len(selected):
            raise ValueError('evaluation count mismatch')
        write(f'EVAL_COMPLETE_{shard}.json',{'status':'SHARD_GENERATION_COMPLETE','events':len(selected),'counts':counts})
    finally:
        handle.remove()


if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        code=1;write(f'EVAL_FAILURE_{shard}.private.json',{'traceback':traceback.format_exc(),'counts':counts});raise
    finally:write(f'EVAL_EXIT_{shard}.json',{'code':code,'counts':counts})
