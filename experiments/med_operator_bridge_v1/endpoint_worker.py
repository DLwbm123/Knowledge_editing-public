"""Train a disjoint fixed shard of matched experts using the original MED update."""
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
extra = Path(os.environ['ADDITIONS_ROOT'])
arm = os.environ['ARM']
if arm not in ('D','DU'): raise ValueError('invalid arm')
shard = int(os.environ['SHARD'])
deadline = float(os.environ['EXPERIMENT_DEADLINE'])
counts = dict(forwards=0, backwards=0, updates=0, generations=0, vision_forwards=0, operator_backwards=0)


def write(name, obj):
    path = root / name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False))
    temp.replace(path)


def budget():
    if time.time() >= deadline - 120 or (root / 'ENDPOINT_STOP.json').exists():
        raise TimeoutError('original experiment deadline or resource stop')
    health = json.loads((root / 'ENDPOINT_GUARD_HEALTH.json').read_text())
    if time.time() - health['time'] > 45 or health['hard_end'] != deadline:
        raise RuntimeError('endpoint guard stale')
    if counts['updates'] != 0 or counts['generations'] != 0 or counts['forwards'] > 101:
        raise ValueError('frozen training count exceeded')


def main():
    with (root / f'ENDPOINT_WORKER_{shard}_ONCE').open('x'):
        pass
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError('guard stop')))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('experiment deadline')))
    signal.alarm(max(1, int(deadline - time.time() - 120)))
    fixed = Path(os.environ['FIXED_ROOT'])
    sys.path[:0] = [str(root/'tools'), str(Path(os.environ['FULL_D_ROOT'])/'tools'), str(fixed/'tools'), str(parent/'private/tools'),
                    str(parent/'private/source'), str(parent/'private/official_llava')]
    import torch
    import module_run as r
    import runtime_binding
    import operator_endpoint as endpoint
    from train_pair import state_digest
    from types import SimpleNamespace
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    r.c.budget, r.COUNTS = budget, counts
    admission = json.loads((root/'ROLE_ADMISSION.json').read_text())
    entries = json.loads((root/'TRAIN_SOURCE_BOUND_PRIVATE.json').read_text())
    lock = json.loads((root/'SPAN_LOCK.private.json').read_text())
    scoring = json.loads((root/'SCORING_CHAIN_ADMISSION.json').read_text())
    if scoring.get('status') != 'SCORING_CHAIN_ADMITTED_WITH_UNKNOWN_PRESERVED':
        raise ValueError('scoring dependency missing')
    if admission.get('status') != 'TRAINING_SOURCE_ROLE_ADMITTED' or lock.get('status') != 'WEAK_TOKEN_PROXY_ONLY':
        raise ValueError('training identity or token lock missing')
    if [x['task']['order'] for x in entries] != admission['owners'] or [x['order'] for x in lock['records']] != admission['owners']:
        raise ValueError('source and token identities differ')
    if arm == 'DU':
        lock = json.loads((root/'TOKEN_LOCK.private.json').read_text())
        span = {x['order']:x for x in lock['records'] if x['status'] == 'TOKEN_BOUND'}
    else:
        span = {x['order']:x for x in lock['records'] if x['status'] == 'WEAK_TOKEN_MATCH'}
    if len(span) != (101 if arm == 'DU' else 25):
        raise ValueError('frozen coverage count changed')
    verified = json.loads((root/'BRIDGE_ADMISSION.private.json').read_text())
    if verified.get('status') != 'STATE_IDENTITIES_BOUND': raise ValueError('state identity admission missing')
    states = {x['owner']: x for x in verified['owners']}
    if len(states) != 101: raise ValueError('state owner coverage mismatch')
    selected = [x for x in entries if x['task']['order'] in span]
    selected = [x for i,x in enumerate(selected) if i % 4 == shard]
    with (parent/'private/EVAL_BINDINGS.json').open('rb') as f:
        config = runtime_binding.extract(f)['configuration_only']
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=parent/'private/cpu_gate'))
    runtime.model.eval()
    r.old.fp32(runtime)
    if runtime.generation_config != config['generation']:
        raise ValueError('original generation configuration changed')
    def count(_module, _args, _kwargs):
        budget()
        counts['forwards'] += 1
    handle = runtime.model.register_forward_pre_hook(count, with_kwargs=True)
    completed = []
    try:
        for entry in selected:
            owner = entry['task']['order']
            task = entry['task']; expected = states[owner]
            if task['edit_id'] != expected['edit_id'] or task['seed'] != expected['seed']: raise ValueError('owner identity changed')
            med = torch.load((original if owner <= 146 else extra)/'candidates'/f'MED_{owner}.pt', map_location='cpu', weights_only=True)
            du = torch.load(Path(os.environ['DU_ROOT'])/'weights'/f'{owner}_DU.pt', map_location='cpu', weights_only=True)['state']
            if state_digest(med) != expected['med_digest'] or state_digest(du) != expected['du_digest']: raise ValueError('state digest changed')
            completed.append(endpoint.check(runtime, r, task, entry['source_rows'], config, med, du, budget))
            write(f'ENDPOINT_PROGRESS_{shard}.json', {'completed':len(completed),'expected':len(selected),'counts':counts})
        if counts['updates'] != 0 or counts['generations'] != 0 or counts['forwards'] != len(selected):
            raise ValueError('endpoint forward counts differ')
        write(f'ENDPOINT_COMPLETE_{shard}.json', {'status':'SHARD_ENDPOINT_COMPLETE','results':completed,'counts':counts})
    finally:
        handle.remove()


if __name__ == '__main__':
    code = 0
    try:
        main()
    except BaseException:
        code = 1
        write(f'ENDPOINT_FAILURE_{shard}.private.json', {'error':traceback.format_exc(),'counts':counts})
        raise
    finally:
        write(f'ENDPOINT_EXIT_{shard}.json', {'code':code,'counts':counts})
