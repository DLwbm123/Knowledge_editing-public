"""Bounded read-only-state GPU equivalence and paired timing; no formal updates."""
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
from prefix_cache import PrefixCache


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def cpu(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().clone()
    if isinstance(x, dict): return {k: cpu(v) for k, v in x.items()}
    if isinstance(x, list): return [cpu(v) for v in x]
    if isinstance(x, tuple): return tuple(cpu(v) for v in x)
    return copy.deepcopy(x)


def main():
    import runtime146 as w
    import selected_training146 as training
    sys.path.insert(0, str(Path(training.__file__).parent))
    import trajectory
    import training_math
    from types import SimpleNamespace
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from m3bench_repro.editors.llava_runtime import seed_everything
    r, c, parent = w.r, w.c, w.RUN
    out = Path(os.environ['BENCH_RUN']); out.mkdir(exist_ok=True)
    write = lambda name, obj: c.write(out / name, obj)
    with (out / 'ONCE').open('x'): pass
    started = time.time()
    steps = 0
    original_budget = c.budget
    def budget():
        original_budget()
        assert time.time() - started < 1800
        assert r.COUNTS['forwards'] <= 500 and r.COUNTS['backwards'] <= 256 and steps <= 128
    c.budget = budget
    w.configure()
    entry = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split()[21],
        gpu_uuid=os.environ['M3BENCH_FORMAL_EXPECTED_GPU_UUID'], started_epoch=time.time(),
        action='prefix_cache_validation1', part='2')
    with c.resources() as ledger: ledger['gpu_sessions'].append(entry)
    success = False
    try:
        runtime = load_real_runtime(SimpleNamespace(cpu_gate=parent/'private/cpu_gate'))
        runtime.model.eval()
        assert not any(p.requires_grad for p in runtime.model.parameters())
        bindings = c.read(parent/'private/EVAL_BINDINGS.json')
        assert runtime.generation_config == next(iter(bindings.values()))['generation']
        runtime.run_root = out/'work'; runtime.run_root.mkdir(exist_ok=True)
        counter = runtime.model.register_forward_pre_hook(lambda *args: r.count(*args))
        r.old.fp32(runtime)
        rows_out = []; times = []; builds = []
        fisher = {k: v.to(runtime.device) if isinstance(v, torch.Tensor) else v
                  for k, v in r.p.load_state(parent/'private/OPERATOR_FISHER.pt').items()}
        for owner in (1, 2):
            task = w.tasks()[owner-1]
            runtime.model.half()
            prepared = r.s.prepare(runtime, bindings, task, trajectory.source_rows(w.packet(), owner), True)
            r.old.fp32(runtime)
            batches = [r.old.batch_on(runtime, v[1]) for v in prepared]
            ref = c.read(parent/f'private/results/C/{owner}.json')
            cal = c.read(parent/f'private/calibration146/{owner}.json')
            for arm in ('C', 'M13'):
                retained = r.p.load_state(w.point(task, arm))['states']
                saved = torch.load(parent/f'private/resume146/{arm}_{owner}.pt', map_location='cpu', weights_only=False)
                assert saved['step'] == 160 and equal(saved['state'], retained['FINAL'])
                expert = r.p.expert(retained['BASE'], task['seed'], runtime.device)
                hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER), expert); hook.attach()
                cache = PrefixCache(runtime, batches)
                seed_everything(task['seed'])
                rng = c.rng(); numpy_rng = np.random.get_state()
                def reset(which):
                    c.restore_rng(rng); np.random.set_state(numpy_rng)
                    expert.load_state_dict(retained['BASE'] if which == 'BASE' else retained['FINAL'])
                    expert.zero_grad(set_to_none=True)
                    if arm == 'C':
                        optimizer = r.p.optimizer(expert, runtime.model)
                        if which == 'FINAL': optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
                        return optimizer, None
                    return None, copy.deepcopy(saved['scalar'] if which == 'FINAL' else dict(step=0, second_moments=[0., 0.]))
                try:
                    cache.__enter__()
                    reset('BASE'); torch.cuda.synchronize(); start = time.perf_counter()
                    with torch.no_grad():
                        for batch in batches:
                            hook.set_teacher_routing(batch.labels); runtime.compute_loss(batch)
                    torch.cuda.synchronize()
                    builds.append(dict(case=owner, arm=arm, seconds=time.perf_counter()-start,
                        bytes=sum(v.numel()*v.element_size() for v in cache.cache.values())))
                    assert equal(rng, c.rng()) and equal(numpy_rng, np.random.get_state())
                    cached_loss = runtime.compute_loss
                    def block(which, use_cache, collect):
                        nonlocal steps
                        optimizer, scalar = reset(which)
                        runtime.compute_loss = cached_loss if use_cache else cache.original_loss
                        trace = []
                        torch.cuda.synchronize(); start = time.perf_counter()
                        for i in range(4):
                            budget(); before = r.old.snapshot(expert)
                            _, losses = trajectory.source_gradients(r, runtime, hook, expert, batches, i+1)
                            gradients = {k: v.grad.detach().clone() for k, v in expert.named_parameters()}
                            if arm == 'C':
                                selected, audit = training_math.raw_step(r, expert, optimizer, gradients, None, 0., None, 'C')
                            else:
                                selected, scalar, audit = training_math.step(r, before, gradients, fisher,
                                    cal['coefficient'], scalar, ref['curve'][i]['operator_step_norm'], 'M13')
                                expert.load_state_dict(selected)
                                r.COUNTS['operator_backwards'] += 1
                            steps += 1
                            if collect:
                                trace.append(cpu(dict(loss=losses, gradient=gradients, state=selected,
                                    optimizer=optimizer.state_dict() if optimizer else None, scalar=scalar,
                                    audit=audit, rng=c.rng(), numpy_rng=np.random.get_state(), mask=hook.token_mask)))
                        torch.cuda.synchronize()
                        return trace, time.perf_counter()-start
                    for which in ('BASE', 'FINAL'):
                        reset(which)
                        for batch in batches:
                            outputs = []
                            capture = runtime.model.register_forward_hook(lambda mod, args, result: outputs.append(result.logits.detach().cpu()))
                            try:
                                hook.set_teacher_routing(batch.labels)
                                with torch.no_grad():
                                    baseline = cache.original_loss(batch).detach().cpu()
                                    candidate = cached_loss(batch).detach().cpu()
                                assert equal(baseline, candidate) and equal(outputs[0], outputs[1]), 'Logits/loss mismatch'
                            finally: capture.remove()
                        a, _ = block(which, False, True); b, _ = block(which, True, True)
                        assert equal(a, b), f'{arm} {which} state/gradient/optimizer/RNG mismatch'
                        rows_out.append(dict(case=owner, arm=arm, state=which, steps=4, all_exact=True))
                        write('PROGRESS.json', dict(comparisons=rows_out, counts=r.COUNTS, shadow_updates=steps))
                    for mode in (False, True, True, False):
                        _, seconds = block('FINAL', mode, False)
                        times.append(dict(case=owner, arm=arm, cached=mode, steps=4, seconds=seconds))
                finally:
                    cache.__exit__(); hook.detach()
        counter.remove()
        summary = {}
        for arm in ('C', 'M13'):
            original = sum(t['seconds'] for t in times if t['arm']==arm and not t['cached'])
            cached = sum(t['seconds'] for t in times if t['arm']==arm and t['cached'])
            build = sum(t['seconds'] for t in builds if t['arm']==arm)
            summary[arm] = dict(original_seconds=original, cached_seconds=cached, speedup=original/cached,
                measured_step_reduction_percent=100*(1-cached/original), build_seconds=build,
                projected160_original_seconds=original/4/4*160,
                projected160_cached_seconds=cached/4/4*160+build/2)
        passed = all(v['cached_seconds'] < v['original_seconds'] and v['projected160_cached_seconds'] < v['projected160_original_seconds'] for v in summary.values())
        assert steps == 128 and r.COUNTS['backwards'] == 256 and r.COUNTS['operator_backwards'] == 64
        write('RESULTS.json', dict(status='PASS' if passed else 'NO_SPEEDUP', exact=rows_out, timing=times,
            cache_build=builds, summary=summary, counts=r.COUNTS, shadow_updates=steps, formal_updates=0,
            generation=0, Judge=0, deployment_allowed=passed, projected_not_full160_benchmark=True))
        success = True
    except BaseException as e:
        write('FAILURE.json', dict(error=repr(e), traceback=traceback.format_exc(), counts=r.COUNTS,
            shadow_updates=steps, deployment_allowed=False, automatic_retry=False))
        raise
    finally:
        with c.resources() as ledger:
            session = next(x for x in ledger['gpu_sessions'] if x['pid']==entry['pid'] and x['started_epoch']==entry['started_epoch'])
            seconds = time.time()-entry['started_epoch']; session.update(ended_epoch=time.time(), resident_seconds=seconds)
            ledger['gpu_seconds_used'] += seconds
        write('EXIT.json', dict(exit_code=0 if success else 1, gpu_seconds=seconds, counts=r.COUNTS, shadow_updates=steps))


if __name__ == '__main__':
    main()
