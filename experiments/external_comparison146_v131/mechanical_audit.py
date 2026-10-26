"""Source-only replay and physical-write checks; no formal answers or Judge."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    import torch
    root = Path(os.environ['V131_ROOT'])
    out = Path(os.environ['AUDIT_OUTPUT'])
    assert not (out / 'RESULT.json').exists()
    free, total = torch.cuda.mem_get_info()
    assert free >= 36 * 1024**3
    torch.cuda.set_per_process_memory_fraction((free - 4 * 1024**3) / total)
    for name in ('V131_MEMIT_RESUME_FILE', 'V131_PARALLEL_DEVICES'):
        os.environ.pop(name, None)
    source = Path(os.environ['AUDIT_PARAMETER_SOURCE'])
    spec = importlib.util.spec_from_file_location('audit_parameter_run', source)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.OUT = out
    m.lock['aborted_consumers'] = []
    def budget():
        health = json.loads((root / 'MEMIT_MM/GUARD_HEALTH.json').read_text())
        assert time.time() - health['time'] < 60 and health['accounting_error'] is None
        assert health['new_bytes'] < 47_000_000_000
        assert m.counts['forwards'] <= 2500 and m.counts['backwards'] <= 1000
        assert m.counts['updates'] <= 150 and m.counts['edits'] <= 6
        if m.counts['forwards'] % 32 == 0:
            write(out/'LIVE_COUNTS.json',dict(time=time.time(),counts=m.counts))
    m.budget = budget
    # Reuse the production initializer and input closures through inputs();
    # none of its formal edit/generation/retirement loops are executed.
    backend = os.environ.get('AUDIT_INVERSE_BACKEND','CPU_FLOAT64')
    precomputed = {}
    if backend == 'GPU_BEFORE_MODEL':
        torch.set_num_threads(8)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        for layer in m.lock['parameter_edit_layers']:
            covariance = torch.load(root/'STATISTICS'/f'covariance_{layer}.pt',
                                    map_location='cuda:0',weights_only=True).double()
            inverse = torch.linalg.inv(m.lock['MEMIT_MM']['mom2_update_weight'] * covariance)
            assert torch.isfinite(inverse).all()
            precomputed[layer] = inverse.cpu()
            del covariance,inverse
            torch.cuda.empty_cache()
            write(out/'INVERSE_PRELOAD.json',dict(time=time.time(),completed_layer=layer,backend=backend))
        m.__dict__['__audit_saved_inverse'] = lambda layer: precomputed[layer]
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    cut = next(i for i,n in enumerate(function.body)
               if isinstance(n,ast.FunctionDef) and n.name == 'inputs') + 1
    function.body = function.body[:cut] + [ast.Return(ast.Call(ast.Name('locals',ast.Load()),[],[]))]
    # Initialization-only CPU LU avoids a second model's CUDA workspace; the
    # covariance, double precision, coefficient and mathematical solve stay fixed.
    moved = 0
    for node in ast.walk(function):
        if isinstance(node,ast.keyword) and node.arg == 'map_location' and isinstance(node.value,ast.Attribute) and isinstance(node.value.value,ast.Name) and node.value.value.id == 'runtime' and node.value.attr == 'device':
            node.value = ast.Constant('cpu')
            moved += 1
    assert moved == 1
    if backend == 'GPU_BEFORE_MODEL':
        replaced = 0
        for node in ast.walk(function):
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='inverse':
                node.value = ast.Call(ast.Name('__audit_saved_inverse',ast.Load()),[ast.Name('layer',ast.Load())],[])
                replaced += 1
        assert replaced == 1
    code = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(code, str(source), 'exec'), m.__dict__)
    frame = m.main()
    runtime, editor = frame['runtime'], frame['editor']
    original_capture, original_optimize = editor.capture, editor.optimize_z
    active = dict(writes=False, layers=[], before={}, observed={})
    def capture(*args, **kwargs):
        result = original_capture(*args, **kwargs)
        if active['writes']:
            layer = args[2][0]
            assert len(args[2]) == 1
            active['layers'].append(layer)
            active['before'][layer] = frame['weights'][layer].detach().cpu().clone()
            active['observed'][layer] = (result[0][layer].clone(), result[1].detach().cpu().clone())
        return result
    def optimize(*args, **kwargs):
        result = original_optimize(*args, **kwargs)
        active['target'] = result[0].detach().cpu().clone()
        active['writes'] = True
        return result
    editor.capture, editor.optimize_z = capture, optimize
    records, generation_checks, solve_records = [], [], []
    try:
        from m3bench_repro.editors.llava_runtime import seed_everything
        for task in frame['tasks'][:6]:
            seed_everything(task['seed'])
            batches, kl = frame['inputs'](task)
            # Teacher-forced and truncated causal prefix must locate the same vector.
            batch = batches[0]
            kwargs, pos = editor.prefix(batch)
            captured = []
            h = runtime.get_module('model.layers.8').register_forward_hook(
                lambda _m,_i,o: captured.append(o[0][0,pos].detach().cpu().clone()))
            try:
                with torch.no_grad():
                    runtime.model(**kwargs)
                    full = dict(batch.forward_kwargs()); full.pop('labels')
                    runtime.model(**full)
            finally:
                h.remove()
            difference = float((captured[0]-captured[1]).abs().max())
            assert torch.allclose(captured[0],captured[1],rtol=1e-5,atol=5e-4)
            assert all(b.target_token_ids[-1] == runtime.adapter.tokenizer.eos_token_id for b in batches)
            active.update(writes=False,layers=[],before={},observed={})
            audit = editor.edit(runtime,batches,kl,frame['layers'],frame['hp'],
                                frame['statistics'],m.counts,budget)
            active['writes'] = False
            assert active['layers'] == [4,5,6,7,8]
            errors = []
            for index,layer in enumerate(frame['layers']):
                key,current = active['observed'][layer]
                inverse = frame['statistics'][layer]
                adjusted = inverse @ key
                adjusted /= 1 + torch.dot(key,adjusted)
                if task['order'] in (1,6):
                    solve_records.append(dict(owner=task['order'],layer=layer,key=key.clone(),adjusted=adjusted.clone()))
                residual = (active['target']-current).double() / (5-index)
                expected = active['before'][layer] + torch.outer(residual,adjusted).float()
                actual = frame['weights'][layer].detach().cpu()
                err = float((actual-expected).abs().max())
                assert torch.allclose(actual,expected,rtol=1e-6,atol=1e-7), (layer,err)
                errors.append(dict(layer=layer,max_abs=err,exact=torch.equal(actual,expected)))
            owner = task['order']
            old = json.loads((root/'MEMIT_MM/training'/f'sequential_{owner}.json').read_text())
            fields = ('initial_norm','delta_norm')
            replay = {k:dict(current=audit['latent'][k], historical=old['latent'][k],
                        relative_error=abs(audit['latent'][k]-old['latent'][k])/max(abs(old['latent'][k]),1e-30)) for k in fields}
            record = dict(owner=owner,prefix_vs_teacher_max_abs=difference,physical_writes=errors,
                          replay=replay,latent=audit['latent'])
            records.append(record)
            # This is a separate 16-token mechanical check, never a formal consumer.
            row = frame['trajectory'].source_rows(frame['packet'],owner)[0]
            item = frame['prepare'](task,[row])[0]
            cfg = runtime.generation_config
            runtime.generation_config = dict(cfg,max_new_tokens=16)
            try:
                answer,_ = frame['r'].s.fit.short.generate(runtime,None,item[2],item[3])
            finally:
                runtime.generation_config = cfg
            ids = list(answer.raw_token_ids)
            registered = next(q for q in frame['plan'] if q['mode']=='insertion' and q['prefix']==owner)
            old_path = root/'MEMIT_MM/outputs'/(frame['r'].c.digest(frame['contract'].consumer_id(registered))+'.json')
            old_ids = json.loads(old_path.read_text())['R0']['raw_token_ids']
            generation_checks.append(dict(owner=owner,diagnostic_tokens=len(ids),
                prefix_equal=ids==old_ids[:len(ids)],EOS=runtime.adapter.tokenizer.eos_token_id in ids))
            write(out/'PROGRESS.json',dict(time=time.time(),completed_edits=owner,counts=m.counts,
                  replay=records,generation_checks=generation_checks))
            print('AUDIT_EDIT_COMPLETE',owner,flush=True)
        torch.save(solve_records,out/'SOLVE_RECORDS.pt')
        result = dict(status='COMPLETE',time=time.time(),counts=m.counts,replay=records,
                      generation_checks=generation_checks,formal_generations=0,Judge_calls=0,
                      diagnostic_generation_attempts=len(generation_checks),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),inverse_initialization_backend=backend,
                      maximum_write_error=max(q['max_abs'] for r in records for q in r['physical_writes']))
        write(out/'RESULT.json',result)
    finally:
        editor.capture,editor.optimize_z = original_capture,original_optimize
        write(out/'COUNTS.json',m.counts)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        write(Path(os.environ['AUDIT_OUTPUT'])/'FAILURE.json',
              dict(time=time.time(),error=traceback.format_exc()))
        raise
