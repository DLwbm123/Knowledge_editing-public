"""Extend a retained cumulative MEMIT state on the four frozen MED150 additions."""
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['RUN_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])
OLD = Path(os.environ['MEMIT_ROOT'])
MED = Path(os.environ['MED150_ROOT'])
LOCK = json.loads((ROOT / 'LOCK.json').read_text())
CLOCK = json.loads((ROOT / 'CLOCK.json').read_text())
COUNTS = dict(forwards=0, backwards=0, updates=0, edits=0, inserted=0,
              generation_attempts=0, formal_consumers=0, vision_forwards=0, LM_input_positions=0)


def read(path):
    return json.loads(Path(path).read_text())


def write(name, value):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    t.replace(p)


def budget():
    assert time.time() < CLOCK['hard_end'] - 30 and not (ROOT / 'STOP.json').exists()
    assert time.time() - read(ROOT / 'GUARD_HEALTH.json')['time'] < 60
    for key, maximum in LOCK['max_counts'].items():
        assert COUNTS[key] <= maximum, (key, COUNTS[key], maximum)


def main():
    with (ROOT / 'WORKER_ONCE').open('x'):
        pass
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError('Guard stop')))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Hard deadline')))
    signal.alarm(max(1, int(CLOCK['hard_end'] - time.time() - 30)))
    sys.path[:0] = [str(OLD / 'tools'), str(PARENT / 'private/tools'), str(PARENT / 'private/source'), str(PARENT / 'private/official_llava')]
    import torch
    import module_run as r
    import parameter_edit as editor
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from m3bench_repro.editors.llava_runtime import seed_everything
    from types import SimpleNamespace

    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(LOCK['gpu'])
    r.c.budget = budget
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    oldlock = read(OLD / 'MEMIT_MM/COMPLETION_LOCK.json')
    layers = oldlock['parameter_edit_layers']
    assert layers == [4, 5, 6, 7, 8]
    saved = torch.load(OLD / 'MEMIT_MM/sequential_resume.pt', map_location='cpu', weights_only=True)
    assert saved['source'] == oldlock['execution_identity'] and saved['mode'] == 'sequential' and saved['owner'] == saved['counts']['inserted'] == 146
    assert set(saved['weights']) == set(layers)
    extra = read(MED / 'ADDITIONAL_INPUTS.json')['tasks']
    plan = read(MED / 'EVAL_PLAN_FROZEN.json')['plan']
    assert [x['task']['order'] for x in extra] == [147, 148, 149, 150]
    assert len(plan) == 2207 and sum(x['mode'] == 'single' for x in plan) == 55 and sum(x['mode'] == 'bank' for x in plan) == 2148
    assert read(MED / 'BASE_MASK_FROZEN.json')['status'] == 'FROZEN'
    bindings = read(PARENT / 'private/EVAL_BINDINGS.json')
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=PARENT / 'private/cpu_gate'))
    runtime.model.eval()
    r.old.fp32(runtime)
    assert not any(p.requires_grad for p in runtime.model.parameters())
    assert runtime.generation_config == next(iter(bindings.values()))['generation']
    weights = {layer: runtime.get_module(f'model.layers.{layer}.mlp.down_proj').weight for layer in layers}
    base = {layer: value.detach().cpu().clone() for layer, value in weights.items()}
    for layer, value in weights.items():
        old = saved['weights'][layer]
        assert old.shape == value.shape and old.dtype == value.dtype == torch.float32 and torch.isfinite(old).all()
    covariance = {}
    for layer in layers:
        budget()
        matrix = torch.load(OLD / 'STATISTICS' / f'covariance_{layer}.pt', map_location=runtime.device, weights_only=True).double()
        inverse = torch.linalg.inv(oldlock['MEMIT_MM']['mom2_update_weight'] * matrix)
        assert torch.isfinite(inverse).all()
        covariance[layer] = inverse.cpu()
        del matrix, inverse
    torch.cuda.empty_cache()

    def count(_module, args, kwargs):
        budget()
        COUNTS['forwards'] += 1
        x = kwargs.get('inputs_embeds', kwargs.get('input_ids'))
        if x is None and args:
            x = args[0]
        if isinstance(x, torch.Tensor):
            COUNTS['LM_input_positions'] += int(x.shape[0] * x.shape[1])

    fh = runtime.model.register_forward_pre_hook(count, with_kwargs=True)
    vh = runtime.model.get_vision_tower().register_forward_pre_hook(lambda *_: COUNTS.__setitem__('vision_forwards', COUNTS['vision_forwards'] + 1))

    def prepare(task, rows, training=False):
        modules = [runtime.model.model.embed_tokens, runtime.model.model.mm_projector, runtime.model.get_vision_tower()]
        for module in modules:
            module.half()
        try:
            return r.s.prepare(runtime, bindings, task, rows, training)
        finally:
            for module in modules:
                module.float()
            assert all(value.dtype == torch.float32 for value in weights.values())

    def restore(state):
        with torch.no_grad():
            for layer, value in weights.items():
                value.copy_(state[layer].to(runtime.device))
        assert all(torch.equal(value.detach().cpu(), state[layer]) for layer, value in weights.items())

    def inputs(entry):
        rows, task = entry['source_rows'], entry['task']
        assert len(rows) == 5
        items = prepare(task, rows, True)
        batches = [r.old.batch_on(runtime, item[1]) for item in items]
        kl_row = dict(rows[0], question='Describe the image.', query_id=f"V131_KL_{task['order']}")
        item = prepare(task, [kl_row])[0]
        expanded = item[3]
        kwargs = dict(inputs_embeds=expanded[4].to(runtime.device, dtype=torch.float32),
                      attention_mask=expanded[2].to(runtime.device) if expanded[2] is not None else None,
                      position_ids=expanded[1].to(runtime.device) if expanded[1] is not None else None,
                      use_cache=False, return_dict=True)
        return batches, (kwargs, kwargs['inputs_embeds'].shape[1] - 1)

    # Confirm FP16 image prefill does not round the FP32 edited matrices.
    first = weights[layers[0]]
    with torch.no_grad():
        first[0, 0] += 1e-5
    sentinel = first[0, 0].clone()
    source_expected = read(MED / 'EXPECTED_SOURCE_BINDINGS.json')
    assert len(source_expected) == 4
    for entry, expected in zip(extra, source_expected):
        prepared = prepare(entry['task'], entry['source_rows'][:1], True)[0]
        assert entry['task']['edit_id'] == expected['query_id']
        assert all(prepared[4][k] == expected[k] for k in ('prompt_ids', 'attention_mask', 'image_sha256'))
    assert torch.equal(first[0, 0], sentinel), 'FP16 prefill changed cumulative physical weights'
    restore(base)
    batches0, kl0 = inputs(extra[0])
    keys0, _ = editor.capture(runtime, batches0, layers, layers[-1])
    from algebra import memit_key
    residuals = []
    for layer in layers:
        budget()
        mat = torch.load(OLD / 'STATISTICS' / f'covariance_{layer}.pt', map_location='cpu', weights_only=True).double()
        key = keys0[layer]
        adjusted = memit_key(covariance[layer], key)
        error = (oldlock['MEMIT_MM']['mom2_update_weight'] * (mat @ adjusted) + key * torch.dot(key, adjusted) - key).norm() / key.norm()
        assert torch.isfinite(error) and float(error) < 1e-5
        residuals.append(float(error))
    del batches0, kl0, keys0, mat

    def edit(entry, kind):
        task = entry['task']
        seed_everything(task['seed'])
        batches, kl = inputs(entry)
        audit = editor.edit(runtime, batches, kl, layers, oldlock['MEMIT_MM'], covariance, COUNTS, budget)
        write(f'training/{kind}_{task["order"]}.json', dict(audit=audit, counts=COUNTS, time=time.time()))

    def evaluate(rows, task, prefix):
        for row in rows:
            budget()
            query = row['query']
            cid = json.dumps(['MEMIT_MM', row['mode'], prefix, row['owner'], query['query_id']], separators=(',', ':'))
            path = ROOT / 'outputs' / (r.c.digest(cid) + '.json')
            assert not path.exists(), 'No generation retry'
            item = prepare(task, [query])[0]
            COUNTS['generation_attempts'] += 1
            answer, trace = r.s.fit.short.generate(runtime, None, item[2], item[3])
            assert answer.raw_token_ids
            write(str(path.relative_to(ROOT)), dict(method='MEMIT_MM', mode=row['mode'], prefix=prefix,
                  owner=row['owner'], consumer_id=cid, binding=dict(input=query, judge_input=item[4]),
                  R0=dict(raw_answer=answer.decoded_text, raw_token_ids=list(answer.raw_token_ids)),
                  EOS=answer.raw_token_ids[-1] == runtime.adapter.tokenizer.eos_token_id,
                  at_cap=len(answer.raw_token_ids) >= 1024, actual_inserted_prefix=prefix,
                  generation_trace=trace))
            COUNTS['formal_consumers'] += 1
            write('PROGRESS.json', dict(counts=COUNTS, time=time.time()))

    write('ADMISSION.json', dict(status='PASS', parent_owner=146, source_binding=True, layers=layers,
                                 prefill_preserves_fp32=True, native_matrix_residuals=residuals, time=time.time()))
    for entry in extra:
        task = entry['task']
        restore(base)
        edit(entry, 'single')
        evaluate([x for x in plan if x['mode'] == 'single' and x['owner'] == task['order']], task, 1)
    restore(saved['weights'])
    for entry in extra:
        task = entry['task']
        edit(entry, 'sequential')
        COUNTS['inserted'] += 1
        evaluate([x for x in plan if x['mode'] == 'insertion' and x['owner'] == task['order']], task, task['order'])
    snapshot = {layer: value.detach().cpu().clone() for layer, value in weights.items()}
    torch.save(snapshot, ROOT / 'final_state.tmp')
    (ROOT / 'final_state.tmp').replace(ROOT / 'final_state.pt')
    reloaded = torch.load(ROOT / 'final_state.pt', map_location='cpu', weights_only=True)
    assert all(torch.equal(snapshot[k], reloaded[k]) for k in layers)
    restore(reloaded)
    evaluate([x for x in plan if x['mode'] == 'bank'], extra[0]['task'], 150)
    fh.remove()
    vh.remove()
    assert COUNTS['inserted'] == 4 and COUNTS['formal_consumers'] == COUNTS['generation_attempts'] == 2207
    write('GENERATION_COMPLETE.json', dict(status='COMPLETE', counts=COUNTS, time=time.time()))


if __name__ == '__main__':
    code = 0
    try:
        main()
    except BaseException:
        code = 1
        write('FAILURE.json', dict(error=traceback.format_exc(), counts=COUNTS, time=time.time()))
        raise
    finally:
        write('WORKER_EXIT.json', dict(code=code, counts=COUNTS, time=time.time()))
