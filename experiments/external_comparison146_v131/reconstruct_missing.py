"""Reconstruct only retired M13P experts needed for previously absent answers.

The original 2956 single generations and 4102 bank consumers are reused. Each
missing expert is trained once with the unchanged160 recipe; every source-NLL
step must reproduce its historical curve before new evaluation. Its complete
registered future-query set is consumed before the rebuilt state is retired.
"""
import copy
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['V131_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])
OUT = ROOT / 'M13P_160'
lock = json.loads(Path(os.environ.get('V131_PROTOCOL_FILE', ROOT / 'PROTOCOL_LOCK.json')).read_text())
START = time.time()
counts = dict(forwards=0, backwards=0, operator_backwards=0, updates=0,
              generation_attempts=0, formal_consumers=0, rebuilt_experts=0)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def budget():
    assert time.time() < lock['hard_end'] - 20 and not (OUT / 'STOP.json').exists()
    assert time.time() - read(OUT / 'GUARD_HEALTH.json')['time'] < 60
    for key, maximum in lock['maximum_counts']['M13P_160'].items():
        assert counts[key] <= maximum, (key, counts[key], maximum)
        if 'maximum_actual_counts' in lock:
            assert counts[key]+lock['prior_counts']['M13P_160'].get(key,0) <= lock['maximum_actual_counts']['M13P_160'][key]


def main():
    with (OUT / 'WORKER_ONCE').open('x'):
        pass
    assert read(OUT / 'REUSE_FROZEN.json')['status'] == 'FROZEN_REUSE_WITH_EXPLICIT_GAPS'
    requests = read(OUT / 'MISSING_REQUESTS.json')['requests']
    assert len(requests) == 287 and len({r['owner'] for r in requests}) == 94
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Original wall')))
    signal.alarm(max(1, int(lock['hard_end'] - time.time() - 20)))
    sys.path[:0] = [str(ROOT / 'tools'), str(PARENT / 'private/tools'),
                   str(PARENT / 'private/source'), str(PARENT / 'private/official_llava')]
    import torch
    import module_run as r
    import trajectory
    import contract
    from reuse import consumer_id, name
    from prefix_cache import PrefixCache
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from m3bench_repro.editors.llava_runtime import seed_everything
    from types import SimpleNamespace
    r.c.budget = budget
    r.COUNTS = counts
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    packet = read(PARENT / 'private/M13_SOURCE_PACKET.json')
    assert contract.source_admission(packet)['status'] == 'PASS'
    tasks = packet['original_tasks']
    bindings = read(PARENT / 'private/EVAL_BINDINGS.json')
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=PARENT / 'private/cpu_gate'))
    runtime.model.eval()
    # Original math_load sets the complete math backend before the first vision
    # prefill, including disabling cuDNN SDPA as well as flash/memory-efficient.
    r.old.fp32(runtime)
    assert runtime.generation_config == next(iter(bindings.values()))['generation']
    assert not any(v.requires_grad for v in runtime.model.parameters())
    def count(*_):
        budget()
        counts['forwards'] += 1
    handle = runtime.model.register_forward_pre_hook(count)
    fisher = {k: v.to(runtime.device) if isinstance(v, torch.Tensor) else v
              for k,v in r.p.load_state(PARENT / 'private/OPERATOR_FISHER.pt').items()}
    def prepare(task, rows, training=False):
        # The prepared image/prompt uses only these unchanged input modules.
        # Keep any attached trained expert and decoder math in FP32 throughout.
        modules = [runtime.model.model.embed_tokens, runtime.model.model.mm_projector,
                   runtime.model.get_vision_tower()]
        for module in modules:
            module.half()
        try:
            return r.s.prepare(runtime, bindings, task, rows, training)
        finally:
            r.old.fp32(runtime)
    for owner in sorted({v['owner'] for v in requests}):
        task = tasks[owner-1]
        seed_everything(task['seed'])
        rows = trajectory.source_rows(packet, owner)
        items = prepare(task, rows, True)
        batches = [r.old.batch_on(runtime, item[1]) for item in items]
        initial = r.p.tr.TT4(task['seed'],8,8).state_dict()
        expert = r.p.expert(initial, task['seed'], runtime.device)
        # The original worker resets raw initial here. trajectory.run performs
        # the single canonicalization; doing it twice changes finite precision.
        expert.load_state_dict(initial)
        hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER), expert)
        hook.attach()
        binding = dict(arm='M13P', seed=task['seed'], steps=160, parameters=7168,
            source_recipe=contract.SOURCE_RECIPE, source_order=[v['query_id'] for v in rows],
            source_binding=r.c.digest(contract.original.source_binding(rows)),
            scientific_lock=r.c.digest(lock), source_version=lock['execution_identity'],
            budget_binding=r.c.digest(lock['maximum_counts']))
        targets = [v['operator_step_norm'] for v in read(PARENT / 'private/results/C' / f'{owner}.json')['curve']]
        coefficient = read(PARENT / 'private/calibration146' / f'{owner}.json')['coefficient']
        def checkpoint(saved):
            torch.save(saved, OUT / 'resume.tmp')
            (OUT / 'resume.tmp').replace(OUT / 'resume.pt')
            write(OUT / 'PROGRESS.json', dict(phase='REBUILD', owner=owner, step=saved['step'],
                counts=counts, time=time.time()))
        try:
            with PrefixCache(runtime,batches):
                final = trajectory.run(r,runtime,hook,expert,batches,'M13P',fisher,
                    coefficient,targets,binding,checkpoint)
            old = read(PARENT / 'private/results/M13P' / f'{owner}.json')['curve']
            errors = [max(abs(a-b) for a,b in zip(now['source_NLL'],prior['source_NLL']))
                      for now,prior in zip(final['curve'],old)]
            assert len(errors) == 160 and max(errors) <= 1e-6, ('Frozen M13P source parity',owner,max(errors))
            path = OUT / 'active_delta.pt'
            state = r.old.snapshot(expert)
            torch.save(state,path)
            loaded = torch.load(path,map_location=runtime.device,weights_only=True)
            assert all(torch.equal(v,loaded[k]) for k,v in state.items())
            expert.load_state_dict(loaded)
            counts['rebuilt_experts'] += 1
            write(OUT / 'training' / f'{owner}.json', dict(owner=owner,curve=final['curve'],
                all160_source_NLL_reproduced=True,maximum_NLL_error=max(errors),save_load_exact=True,
                old_recipe_unchanged=True,purpose='Only previously absent bank-selected query answers',
                counts=counts,time=time.time()))
            owned = [v for v in requests if v['owner'] == owner]
            for request in owned:
                # The expert hook does not affect multimodal input expansion.
                item = prepare(task,[request['query']])[0]
                assert item[4] == request['judge_binding']
                budget()
                counts['generation_attempts'] += 1
                answer,trace = r.s.fit.short.generate(runtime,hook,item[2],item[3])
                prediction = dict(binding=item[4],R0=dict(raw_answer=answer.decoded_text,
                    raw_token_ids=list(answer.raw_token_ids)),at_cap=len(answer.raw_token_ids)>=1024,
                    generation_trace=trace,expert_owner=owner,execution=lock['execution_identity'])
                prediction_path = OUT / 'new_unique_predictions' / (name([owner,request['query']['query_id']])+'.json')
                assert not prediction_path.exists()
                write(prediction_path,prediction)
                for consumer in request['consumers']:
                    row = consumer['row']
                    route = read(consumer['route_path'])
                    assert route['effective_expert'] == request['selected']
                    assert route['binding']['judge_input'] == item[4]
                    value = copy.deepcopy(route)
                    value.update(method='M13P_160',consumer_id=consumer_id(row),R0=prediction['R0'],
                        generation_trace=trace,at_cap=prediction['at_cap'],
                        prediction_reused=False,new_GPU_generation=True,functional_bank_composition=True,
                        execution=lock['execution_identity'],
                        provenance=dict(kind='PREVIOUSLY_ABSENT_SELECTED_EXPERT_PREDICTION',
                            prediction=str(prediction_path),actual_routing=consumer['route_path'],
                            routing_execution=route['execution'],no_C_or_M13_prediction_substitution=True))
                    destination = OUT / 'outputs' / (name(consumer_id(row))+'.json')
                    assert not destination.exists()
                    write(destination,value)
                    counts['formal_consumers'] += 1
                write(OUT / 'PROGRESS.json',dict(phase='FILL_MISSING',owner=owner,counts=counts,time=time.time()))
            files = [OUT/'resume.pt',path]
            manifest = []
            for file in files:
                assert file.is_file() and not file.is_symlink() and file.resolve().is_relative_to(OUT.resolve())
                manifest.append(dict(path=str(file),bytes=file.stat().st_size))
            write(OUT/'retirement'/f'{owner}.plan.json',dict(files=manifest,
                last_consumers='All registered queries at every future prefix for this expert are durable; source parity/save-load complete'))
            for file in files:
                file.unlink()
            write(OUT/'retirement'/f'{owner}.receipt.json',dict(files=2,
                bytes=sum(v['bytes'] for v in manifest),retained_copies=0,
                reconstruction='Retrain unchanged original M13P-160 recipe',time=time.time()))
        finally:
            hook.detach()
        del expert,batches,items,final,loaded,state
    assert counts['updates'] == counts['operator_backwards'] == 94*160
    assert counts['backwards'] == 94*320 and counts['rebuilt_experts'] == 94
    assert counts['generation_attempts'] == 287 and counts['formal_consumers'] == 436
    expected = {consumer_id(row) for row in read(ROOT/'EVAL_PLAN.json') if row['method']=='M13P_160'}
    actual = {read(path)['consumer_id'] for path in (OUT/'outputs').glob('*.json')}
    assert actual == expected and len(actual) == 7494
    write(OUT/'COMPLETE.json',dict(status='ALL7494_CONSUMERS_BOUND_JUDGE_PENDING',
        prior_single_generations_repeated=0,existing_bank_consumers_reused=4102,
        new_unique_generations=287,new_full_bank_GPU_run=False,
        original_R0_functional_composition=True,counts=counts,time=time.time()))
    handle.remove()


if __name__ == '__main__':
    code = 0
    try:
        main()
    except BaseException:
        code = 1
        write(OUT/'FAILURE.json',dict(error=traceback.format_exc(),counts=counts,time=time.time()))
        raise
    finally:
        write(OUT/'WORKER_EXIT.json',dict(code=code,seconds=time.time()-START,counts=counts,time=time.time()))
