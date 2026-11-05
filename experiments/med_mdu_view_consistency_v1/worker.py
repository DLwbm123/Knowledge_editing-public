"""Compare MDU and two-view JS gradients on the fixed training sources only."""
import json
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(os.environ['DIAG_ROOT'])
SOURCE = Path(os.environ['SOURCE_ROOT'])
STATES = Path(os.environ['STATE_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])
FIXED = Path(os.environ['FIXED_ROOT'])
END = float(os.environ['EXPERIMENT_DEADLINE'])
COUNTS = dict(forwards=0, backwards=0, pairs=0, updates=0, generations=0, judge=0)


def write(name, obj):
    path = ROOT / name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, allow_nan=False))
    temp.replace(path)


def budget():
    if time.time() >= END - 60 or (ROOT / 'STOP.json').exists():
        raise TimeoutError('original deadline or guard stop')
    health = json.loads((ROOT / 'GUARD_HEALTH.json').read_text())
    if time.time() - health['time'] > 45 or health['hard_end'] != END:
        raise RuntimeError('guard unavailable')
    if COUNTS['forwards'] >= 2000 or COUNTS['backwards'] >= 1000:
        raise RuntimeError('preregistered compute ceiling')


def answer_logits(runtime, hook, batch, row, lock, view, torch):
    hook.set_teacher_routing(batch.labels)
    out = runtime.model(**batch.forward_kwargs())
    labels = batch.labels[:, 1:]
    mask = labels != -100
    ids = labels[mask]
    raw = runtime._tokenize_prompt(runtime.adapter._prompt(row['question'], row['reference']))[0]
    start, end = view['answer_start_label'], view['answer_end_label']
    n = lock['answer_tokens']
    if not 0 <= start < end <= raw.numel() or end - start != n:
        raise ValueError('answer lock invalid')
    target = raw[start:end].cpu()
    hits = [j for j in range(ids.numel() - n + 1) if torch.equal(ids[j:j+n].cpu(), target)]
    if len(hits) != 1:
        raise ValueError('answer labels do not uniquely bind')
    offset = hits[0]
    decoded = runtime.adapter.tokenizer.decode(ids[offset:offset+n].tolist(), skip_special_tokens=False).strip()
    if decoded != row['reference'].strip():
        raise ValueError('answer text differs')
    return out.logits[:, :-1, :][mask][offset:offset+n].float(), target


def symmetric_js(logits0, logits1, torch):
    if logits0.ndim != 2 or logits0.shape != logits1.shape or not logits0.numel():
        raise ValueError('JS inputs must be nonempty aligned token distributions')
    a, b = logits0.float().log_softmax(-1), logits1.float().log_softmax(-1)
    m = torch.logaddexp(a, b) - math.log(2.)
    result = (.5 * ((a.exp() * (a-m)).sum(-1) + (b.exp() * (b-m)).sum(-1))).mean()
    if not torch.isfinite(result):
        raise ValueError('nonfinite JS')
    return result


def main():
    with (ROOT / 'WORKER_ONCE').open('x'):
        pass
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError('guard stop')))
    sys.path[:0] = [str(ROOT/'tools'), str(FIXED/'tools'), str(PARENT/'private/tools'),
                    str(PARENT/'private/source'), str(PARENT/'private/official_llava')]
    import torch
    import module_run as r
    import runtime_binding
    from robust_objective import weights_and_value, view_loss
    from span_train import terms
    from train_pair import state_digest
    from types import SimpleNamespace
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from m3bench_repro.editors.llava_runtime import seed_everything

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    r.COUNTS, r.c.budget = COUNTS, budget
    admission = json.loads((SOURCE/'ROLE_ADMISSION.json').read_text())
    weak = json.loads((SOURCE/'SPAN_LOCK.private.json').read_text())
    token = json.loads((SOURCE/'TOKEN_LOCK.private.json').read_text())
    source = json.loads((SOURCE/'TRAIN_SOURCE_BOUND_PRIVATE.json').read_text())
    owners = {x['order'] for x in weak['records'] if x['status'] == 'WEAK_TOKEN_MATCH'}
    locks = {x['order']: x for x in token['records'] if x['order'] in owners and x['status'] == 'TOKEN_BOUND'}
    if admission.get('status') != 'TRAINING_SOURCE_ROLE_ADMITTED' or len(owners) != 25 or set(locks) != owners:
        raise ValueError('source or token admission differs')
    if len(source) != 25 or {x['task']['order'] for x in source} != owners:
        raise ValueError('source selection differs')
    if source != json.loads((STATES/'TRAIN_SOURCE_BOUND_PRIVATE.json').read_text()):
        raise ValueError('training source differs from the original MDU run')
    old_token = json.loads((STATES/'TOKEN_LOCK.private.json').read_text())
    old_locks = {x['order']: x for x in old_token['records'] if x['order'] in owners}
    if old_locks != locks:
        raise ValueError('answer lock differs from the original MDU run')
    if any(json.loads((STATES/f'TRAIN_EXIT_{i}.json').read_text())['code'] != 0 for i in range(2)):
        raise ValueError('historical worker exit differs')
    if json.loads((STATES/'TRAIN_GUARD_EXIT.json').read_text())['code'] != 0:
        raise ValueError('historical guard exit differs')
    if json.loads((STATES/'TRAIN_FROZEN.json').read_text()).get('status') != 'ALL_EXPERTS_TRAINED':
        raise ValueError('historical states are not frozen')
    receipts = [x for i in range(2) for x in json.loads((STATES/f'TRAIN_COMPLETE_{i}.json').read_text())['results']]
    if len(receipts) != 25 or {x['owner'] for x in receipts} != owners:
        raise ValueError('historical expert coverage differs')
    receipts = {x['owner']: x for x in receipts}
    with (PARENT/'private/EVAL_BINDINGS.json').open('rb') as f:
        config = runtime_binding.extract(f)['configuration_only']
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=PARENT/'private/cpu_gate'))
    runtime.model.eval()
    r.old.fp32(runtime)
    if runtime.generation_config != config['generation'] or any(p.requires_grad for p in runtime.model.parameters()):
        raise ValueError('runtime binding differs')
    versions = {name: p._version for name, p in runtime.model.named_parameters()}
    def count(_module, _args, _kwargs):
        budget()
        COUNTS['forwards'] += 1
    handle = runtime.model.register_forward_pre_hook(count, with_kwargs=True)
    output = []
    try:
        for entry in source:
            task, rows = entry['task'], entry['source_rows']
            owner, lock = task['order'], locks[task['order']]
            if [x['audit_role'] for x in rows] != ['NATIVE', 'FIT', 'FIT', 'FIT', 'FIT']:
                raise ValueError('training role mismatch')
            saved = torch.load(STATES/'weights'/f'{owner}_MDU.pt', map_location='cpu', weights_only=True)
            if saved['arm'] != 'MDU' or saved['steps'] != 160 or receipts[owner]['final_digest'] != state_digest(saved['state']):
                raise ValueError('expert receipt mismatch')
            runtime.model.half()
            try:
                prepared = r.s.prepare(runtime, {'configuration_only': config}, task, rows, True)
            finally:
                r.old.fp32(runtime)
            batches = [r.old.batch_on(runtime, x[1]) for x in prepared]
            seed_everything(task['seed'])
            expert = r.p.expert(saved['state'], task['seed'], runtime.device)
            hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module('model.layers.21.mlp.down_proj'), expert)
            hook.attach()
            before = state_digest(expert.state_dict())
            try:
                for fit in range(1, 5):
                    runs = []
                    cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
                    for repeat in range(2):
                        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(gpu_rng)
                        expert.zero_grad(set_to_none=True)
                        logits0, target0 = answer_logits(runtime, hook, batches[0], rows[0], lock, lock['views'][0], torch)
                        logits1, target1 = answer_logits(runtime, hook, batches[fit], rows[fit], lock, lock['views'][fit], torch)
                        if not torch.equal(target0, target1) or logits0.shape != logits1.shape:
                            raise ValueError('views have different target token sequences')
                        js = symmetric_js(logits0, logits1, torch)
                        budget(); js.backward(); COUNTS['backwards'] += 1
                        gj = torch.cat([p.grad.detach().double().reshape(-1).cpu() for p in expert.parameters()])
                        del logits0, logits1
                        expert.zero_grad(set_to_none=True)
                        indices = (0, fit)
                        with torch.no_grad():
                            first = [terms(runtime, hook, batches[i], rows[i], lock['answer_tokens'], lock['spans'], lock['views'][i]) for i in indices]
                        weights, _ = weights_and_value(first[0][1], first[1][1], 1.0)
                        for j, i in enumerate(indices):
                            ce, spans = terms(runtime, hook, batches[i], rows[i], lock['answer_tokens'], lock['spans'], lock['views'][i])
                            if not torch.allclose(ce.detach(), first[j][0], atol=1e-5, rtol=1e-5) or not torch.allclose(spans.detach(), first[j][1], atol=1e-5, rtol=1e-5):
                                raise ValueError('MDU recomputation differs')
                            budget(); view_loss(ce, spans, weights[j], 1.0).backward(); COUNTS['backwards'] += 1
                        gm = torch.cat([p.grad.detach().double().reshape(-1).cpu() for p in expert.parameters()])
                        if not torch.isfinite(gj).all() or not torch.isfinite(gm).all():
                            raise ValueError('nonfinite gradient')
                        runs.append((float(js.detach()), gj, gm))
                    if not math.isclose(runs[0][0], runs[1][0], abs_tol=1e-7, rel_tol=1e-5) or any(not torch.allclose(runs[0][i], runs[1][i], atol=1e-7, rtol=1e-5) for i in (1, 2)):
                        raise ValueError('repeatability failed')
                    values = []
                    for js, gj, gm in runs:
                        nj, nm = float(torch.linalg.vector_norm(gj)), float(torch.linalg.vector_norm(gm))
                        cos = float(torch.dot(gj, gm) / (nj*nm)) if nj > 0 and nm > 0 else None
                        values.append({'js': js, 'js_norm': nj, 'mdu_norm': nm, 'cosine': cos})
                    signal_hit = all(v['js'] > 1e-6 and v['js_norm'] > 1e-8 and v['cosine'] is not None and abs(v['cosine']) < .999 for v in values)
                    output.append({'owner': owner, 'fit': fit, 'repeats': values, 'signal': signal_hit,
                                   'repeat_js_abs': abs(runs[0][0]-runs[1][0]),
                                   'repeat_js_gradient_max': float((runs[0][1]-runs[1][1]).abs().max()),
                                   'repeat_mdu_gradient_max': float((runs[0][2]-runs[1][2]).abs().max())})
                    COUNTS['pairs'] += 1
                    if state_digest(expert.state_dict()) != before:
                        raise ValueError('expert changed')
                    write('PROGRESS.json', {'pairs': COUNTS['pairs'], 'expected': 100, 'counts': COUNTS})
            finally:
                hook.detach()
        if len(output) != 100 or any(p._version != versions[name] for name, p in runtime.model.named_parameters()):
            raise ValueError('coverage or frozen backbone differs')
        write('PAIRS.private.json', output)
        valid = [v for row in output for v in row['repeats'] if v['cosine'] is not None]
        write('RESULTS.json', {'status': 'COMPLETE', 'counts': COUNTS, 'pairs': len(output),
                              'signal_pairs': sum(row['signal'] for row in output),
                              'zero_norm_repeats': 200-len(valid),
                              'mean_js': sum(v['js'] for row in output for v in row['repeats'])/200,
                              'mean_js_norm': sum(v['js_norm'] for row in output for v in row['repeats'])/200,
                              'mean_mdu_norm': sum(v['mdu_norm'] for row in output for v in row['repeats'])/200,
                              'mean_cosine': sum(v['cosine'] for v in valid)/len(valid) if valid else None,
                              'max_repeat_js_abs': max(row['repeat_js_abs'] for row in output),
                              'max_repeat_js_gradient': max(row['repeat_js_gradient_max'] for row in output),
                              'max_repeat_mdu_gradient': max(row['repeat_mdu_gradient_max'] for row in output)})
    finally:
        handle.remove()


if __name__ == '__main__':
    code = 0
    try:
        main()
    except BaseException:
        code = 1
        write('WORKER_FAILURE.private.json', {'error': traceback.format_exc(), 'counts': COUNTS})
        raise
    finally:
        write('WORKER_EXIT.json', {'code': code, 'counts': COUNTS})
