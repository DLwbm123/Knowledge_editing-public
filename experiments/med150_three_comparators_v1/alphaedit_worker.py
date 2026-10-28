"""Rebuild frozen AlphaEdit statistics and its real 146+4 sequential state."""
import gc
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['EXTENSION_ROOT'])
MED = Path(os.environ['MED150_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])
ALPHA = Path(os.environ['ALPHA_ROOT'])
LOCK = json.loads((ROOT / 'LOCK.json').read_text())
CLOCK = json.loads((ROOT / 'CLOCK.json').read_text())
COUNTS = dict(edits=0, inserted=0, generation_attempts=0, formal_consumers=0)


def read(p):
    return json.loads(Path(p).read_text())


def write(name, value):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)


def budget():
    assert time.time() < CLOCK['hard_end'] - 30 and not (ROOT / 'STOP.json').exists()
    assert time.time() - read(ROOT / 'GUARD_HEALTH.json')['time'] < 60
    for key, maximum in LOCK['max_counts'].items():
        assert COUNTS[key] <= maximum, (key, COUNTS[key], maximum)


def local_path(value):
    value = str(value)
    if '/derived_inputs/' in value:
        return str(PARENT / 'private/derived_inputs' / value.split('/derived_inputs/', 1)[1])
    if '/DataP/' in value:
        return str(Path(os.environ['DATA_ROOT']) / value.split('/DataP/', 1)[1])
    return value


def main():
    with (ROOT / 'WORKER_ONCE').open('x'):
        pass
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError('guard stop')))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('hard deadline')))
    signal.alarm(max(1, int(CLOCK['hard_end'] - time.time() - 30)))
    sys.path.insert(0, str(ALPHA / 'source'))
    import torch
    from experiments.alphaedit_stage17 import run as original
    from experiments.alphaedit_stage17 import runtime as native
    from experiments.alphaedit_stage17.algebra import update
    from experiments.alphaedit_vlm.core import capture, optimize_z, prefix_inputs

    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(LOCK['gpu'])
    original.configure()
    c = read(ALPHA / 'CONFIG.private.json')
    assert len(c['tasks']) == 146 and c['statistics_documents'] == 10000
    assert c['arms']['multi_layer']['layers'] == c['statistics_layers'] == [19, 20, 21, 22, 23]
    c['deadline_epoch'] = CLOCK['hard_end']
    original_tasks = read(Path(os.environ['MED146_ROOT']) / 'INPUTS_FROZEN.json')['tasks']
    assert [t['edit_id'] for t in c['tasks']] == [t['edit_id'] for t in original_tasks]
    extra = read(MED / 'ADDITIONAL_INPUTS.json')['tasks']
    plan = read(MED / 'EVAL_PLAN_FROZEN.json')['plan']
    assert len(extra) == 4 and len(plan) == 2207 and read(MED / 'BASE_MASK_FROZEN.json')['status'] == 'FROZEN'
    expected = {}
    for p in (MED / 'eval_outputs').glob('*.json'):
        value = read(p)
        q, binding = value['binding']['input'], value['binding']['judge_input']
        item = (binding['question'], binding['reference'], binding['image_sha256'], binding['prompt_ids'], binding['attention_mask'])
        if q['query_id'] in expected:
            assert expected[q['query_id']] == item
        expected[q['query_id']] = item
    assert len(expected) == 2148
    additions = []
    for index, entry in enumerate(extra):
        task = entry['task']
        assert task['order'] == 147 + index
        query = dict(entry['source_rows'][0], query_id=task['edit_id'])
        query['dataset'] = task['native']['dataset']
        query['image_path'] = local_path(query['image_path'])
        native_rows = [v['query'] for v in plan if v['mode'] == 'single' and v['owner'] == task['order'] and v['query']['audit_role'] == 'T0']
        assert len(native_rows) == 1 and all(native_rows[0][k] == query[k] for k in ('question', 'reference', 'image_sha256'))
        query['expected_prompt_ids'] = expected[native_rows[0]['query_id']][3]
        additions.append(dict(edit_id=task['edit_id'], order=task['order'], native=query,
                              reference=c['tasks'][index]['reference']))
    statistics_root = ROOT / 'statistics'
    statistics_root.mkdir(exist_ok=True)
    corpus = ROOT / 'wikipedia.jsonl.gz'
    assert corpus.is_symlink() and corpus.resolve() == (ALPHA / 'wikipedia.jsonl.gz').resolve()

    class StatisticsReady(Exception):
        pass

    def stats_progress(status):
        budget()
        if status.get('state') == 'BASE_AUDIT_LOADING':
            raise StatisticsReady()
        write('PROGRESS.json', dict(phase='STATISTICS', detail=status, counts=COUNTS, time=time.time()))

    try:
        original.statistics(c, statistics_root, stats_progress)
    except StatisticsReady:
        pass
    assert read(statistics_root / 'READY.json')['documents'] == 10000
    assert all((statistics_root / f'basis-{layer}.private.pt').is_file() for layer in c['statistics_layers'])
    write('STATISTICS_COMPLETE.json', dict(status='PASS', documents=10000, layers=c['statistics_layers'], time=time.time()))

    edit_root = ROOT / 'edit'
    edit_root.mkdir(exist_ok=True)
    runtime = native.load(c, edit_root)
    model = runtime.model
    layers = c['arms']['multi_layer']['layers']
    weights = {layer: model.get_submodule(f'model.layers.{layer}.mlp.down_proj').weight for layer in layers}
    originals = {layer: value.detach().cpu().clone() for layer, value in weights.items()}
    bases = {layer: torch.load(statistics_root / f'basis-{layer}.private.pt', map_location='cpu', weights_only=True) for layer in layers}
    names = {f'model.layers.{layer}.mlp.down_proj.weight' for layer in layers}
    versions = {n: p._version for n, p in model.named_parameters() if n not in names}

    def verify():
        assert not original.hooks(model)
        assert versions == {n: p._version for n, p in model.named_parameters() if n not in names}

    def restore():
        with torch.no_grad():
            for layer, weight in weights.items():
                weight.copy_(originals[layer].to(weight.device))

    def edit(query, ref, history, sequential):
        budget()
        target = native.prepared(runtime, query)
        reference = native.prepared(runtime, ref)
        z, latent = optimize_z(model, target, reference, layers[-1], c['hparams'])
        inputs, pos = prefix_inputs(target)
        writes = []
        for offset, layer in enumerate(layers):
            key, _ = capture(model, inputs, layer, pos)
            _, current = capture(model, inputs, layers[-1], pos)
            residual = (z - current) / (len(layers) - offset)
            basis = bases[layer].cuda()
            delta = update(key, residual, basis, history[layer], c['hparams']['L2'])
            weight = weights[layer]
            before = weight.detach().float().clone()
            with torch.no_grad():
                weight.copy_((before + delta).to(weight.dtype))
            assert torch.isfinite(weight).all()
            writes.append(dict(layer=layer, requested_norm=float(delta.norm()), actual_norm=float((weight.detach().float()-before).norm())))
            del basis, delta, before
        if sequential:
            for layer in layers:
                key, _ = capture(model, inputs, layer, pos)
                history[layer] = torch.cat((history[layer], key[:, None]), 1)
        COUNTS['edits'] += 1
        verify()
        return dict(latent=latent, writes=writes)

    def query_for(row):
        q = dict(row['query'])
        q['dataset'] = original_tasks[0]['native']['dataset']
        q['image_path'] = local_path(q['image_path'])
        q['expected_prompt_ids'] = expected[q['query_id']][3]
        return q

    def evaluate(rows, prefix):
        import hashlib
        for row in rows:
            budget()
            q = query_for(row)
            cid = json.dumps(['ALPHAEDIT_MM', row['mode'], prefix, row['owner'], q['query_id']], separators=(',', ':'))
            digest = hashlib.sha256(json.dumps(cid, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
            dest = ROOT / 'outputs' / (digest + '.json')
            assert not dest.exists(), 'No implicit retry'
            raw = runtime.adapter.prepare_inputs(q['image_path'], q['question'])
            realized = (q['question'], q['reference'], raw['image_sha256'], raw['input_ids'][0].tolist(), raw['attention_mask'][0].tolist())
            assert realized == expected[q['query_id']]
            COUNTS['generation_attempts'] += 1
            value = native.generated(runtime, q)
            binding = dict(question=q['question'], reference=q['reference'], image_sha256=raw['image_sha256'],
                           prompt_ids=raw['input_ids'][0].tolist(), attention_mask=raw['attention_mask'][0].tolist(),
                           runtime={'precision': 'float16', 'source': LOCK['runtime_source']}, generation=runtime.generation_config)
            write(str(dest.relative_to(ROOT)), dict(method='ALPHAEDIT_MM', mode=row['mode'], prefix=prefix,
                  owner=row['owner'], consumer_id=cid, binding=dict(input=row['query'], judge_input=binding),
                  R0=dict(raw_answer=value['raw_answer'], raw_token_ids=value['raw_token_ids']),
                  EOS=value['eos_reached'], at_cap=value['hit_token_cap'], actual_inserted_prefix=prefix))
            COUNTS['formal_consumers'] += 1
            write('PROGRESS.json', dict(phase='GENERATING', counts=COUNTS, time=time.time()))

    def empty_history():
        return {layer: torch.empty(weights[layer].shape[1], 0, device='cuda') for layer in layers}

    write('ADMISSION.json', dict(status='PASS', method='ALPHAEDIT_MM', tasks=150, statistics_documents=10000,
                                 layers=layers, new_reference_policy='reuse first four approved references in order', time=time.time()))
    for task in additions:
        restore()
        audit = edit(task['native'], task['reference'], empty_history(), False)
        write(f'training/single_{task["order"]}.json', dict(audit=audit, time=time.time()))
        evaluate([v for v in plan if v['mode'] == 'single' and v['owner'] == task['order']], 1)
    restore()
    history = empty_history()
    tasks = c['tasks'] + additions
    for task in tasks:
        query = c['queries'][task['edit_id']] if task['order'] <= 146 else task['native']
        audit = edit(query, task['reference'], history, True)
        COUNTS['inserted'] += 1
        assert task['order'] == COUNTS['inserted']
        state = dict(owner=task['order'], weights={k: v.detach().cpu().clone() for k, v in weights.items()},
                     history={k: v.detach().cpu() for k, v in history.items()})
        torch.save(state, ROOT / 'resume.tmp')
        (ROOT / 'resume.tmp').replace(ROOT / 'resume.pt')
        write(f'training/sequential_{task["order"]}.json', dict(audit=audit, time=time.time()))
        if task['order'] >= 147:
            evaluate([v for v in plan if v['mode'] == 'insertion' and v['owner'] == task['order']], task['order'])
        write('PROGRESS.json', dict(phase='SEQUENTIAL', owner=task['order'], counts=COUNTS, time=time.time()))
    saved = torch.load(ROOT / 'resume.pt', map_location='cpu', weights_only=True)
    assert saved['owner'] == 150 and all(torch.equal(saved['weights'][k], v.detach().cpu()) for k, v in weights.items())
    evaluate([v for v in plan if v['mode'] == 'bank'], 150)
    assert COUNTS['edits'] == 154 and COUNTS['inserted'] == 150 and COUNTS['formal_consumers'] == COUNTS['generation_attempts'] == 2207
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
