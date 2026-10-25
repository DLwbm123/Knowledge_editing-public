"""Read-only EOS and literal natural-prefix audit on seventy source training rows."""
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['V123_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])
TRAIN_ROOT = Path(os.environ['V122_PARENT'])
Q_ROOT = Path(os.environ['PARENT_Q_ROOT'])
START = time.time()
counts = dict(forwards=0, backwards=0, operator_backwards=0, updates=0,
              generations=0, vision_forwards=0)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


lock = read(ROOT / 'PROTOCOL_LOCK.json')


def budget():
    assert time.time() < lock['hard_end'] - 15 and not (ROOT / 'STOP.json').exists()
    assert time.time() - read(ROOT / 'GUARD_HEALTH.json')['time'] < 30
    for key, maximum in lock['maximum_counts'].items():
        assert counts[key] <= maximum, (key, counts[key], maximum)


def main():
    with (ROOT / 'WORKER_ONCE').open('x'):
        pass
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Frozen wall limit')))
    signal.alarm(max(1, int(lock['hard_end'] - time.time() - 15)))
    sys.path[:0] = [str(ROOT / 'tools'), str(PARENT / 'private/tools'),
                   str(PARENT / 'private/source'), str(PARENT / 'private/official_llava')]
    import torch
    import kernel
    import boundary
    import module_run as r
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from types import SimpleNamespace
    r.c.budget = budget
    write(ROOT / 'BOUNDARY_SELF_CHECK.json', boundary.selfcheck())
    torch.manual_seed(lock['seed'])
    torch.cuda.manual_seed_all(lock['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    tasks = read(ROOT / 'TASKS.json')['tasks']
    assert len(tasks) == 14 and sum(len(t['train']) for t in tasks) == 70
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=PARENT / 'private/cpu_gate'))
    runtime.model.eval()
    r.old.fp32(runtime)
    assert not any(p.requires_grad for p in runtime.model.parameters())
    tokenizer = runtime.adapter.tokenizer
    eos = tokenizer.eos_token_id
    bindings = read(PARENT / 'private/EVAL_BINDINGS.json')
    template = read(PARENT / 'private/M13_SOURCE_PACKET.json')['original_tasks'][4]
    module = runtime.get_module(r.c.LAYER)
    def count(*_):
        budget()
        counts['forwards'] += 1
    fh = runtime.model.register_forward_pre_hook(count)
    vh = runtime.model.get_vision_tower().register_forward_pre_hook(
        lambda *_: counts.__setitem__('vision_forwards', counts['vision_forwards'] + 1))

    def prepare(rows):
        runtime.model.half()
        try:
            return r.s.prepare(runtime, bindings, template, rows, True)
        finally:
            r.old.fp32(runtime)

    def metrics(logits):
        logits = logits.detach().double()
        assert logits.ndim == 1 and torch.isfinite(logits).all()
        non_eos = logits.clone()
        non_eos[eos] = -torch.inf
        top = torch.topk(logits, 5)
        return dict(EOS_NLL=float(torch.logsumexp(logits, 0) - logits[eos]),
            EOS_probability=float(torch.softmax(logits, 0)[eos]),
            EOS_rank=1 + int((logits > logits[eos]).sum()),
            EOS_margin=float(logits[eos] - non_eos.max()), argmax_token=int(logits.argmax()),
            top5_token_ids=top.indices.tolist(), top5_logits=top.values.tolist())

    def forward(kw, positions, content_positions, q, expert):
        captured = []
        def capture(_module, args):
            captured.append(args[0][0, positions + content_positions].detach().clone())
        handle = module.register_forward_pre_hook(capture)
        try:
            with torch.no_grad():
                result = runtime.model(**kw)
                eos_result = metrics(result.logits[0, positions[0]])
            assert len(captured) == 1 and torch.isfinite(captured[0]).all()
            x = captured[0].float()
            x = (x / (x.square().mean(-1, keepdim=True).sqrt() + 1e-6)).double()
            residual = x - (x @ q) @ q.T
            ratios = (residual.norm(dim=1) / x.norm(dim=1)).tolist()
            with torch.no_grad():
                writer_norms = expert.residual(captured[0]).float().norm(dim=-1).tolist()
            eos_result.update(EOS_predictor_Q_retained_ratio=ratios[0],
                content_predictor_Q_retained_ratios=ratios[1:],
                EOS_predictor_writer_norm=writer_norms[0],
                content_predictor_writer_norms=writer_norms[1:])
            return eos_result, result
        finally:
            handle.remove()

    audits = []
    for task in tasks:
        tid = task['task_id']
        q = torch.load(Q_ROOT / tid / 'basis.pt', map_location=runtime.device, weights_only=True)
        dest = TRAIN_ROOT / tid / 'SOURCE_PRIORITY_PATH'
        assert read(dest / 'TRAIN_COMPLETE.json')['save_load_exact']
        state = torch.load(dest / 'delta.pt', map_location=runtime.device, weights_only=True)
        expert = r.p.expert(state, lock['seed'], runtime.device)
        before = r.old.snapshot(expert)
        prepared = prepare(task['train'])
        for item in prepared:
            row, _, _, expanded, binding = item
            original = read(dest / 'outputs' / (row['query_id'] + '.json'))
            assert original['binding'] == binding
            cut = boundary.boundary(tokenizer, original['raw_token_ids'], row['reference'])
            batch = r.old.batch_on(runtime, item[1])
            labels = batch.labels[0]
            active = torch.where(labels != -100)[0]
            assert int((labels[active] == eos).sum()) == 1 and int(labels[active[-1]]) == eos
            p = int(active[0])
            teacher_kw = {k: v for k, v in batch.forward_kwargs().items() if k != 'labels'}
            assert torch.equal(teacher_kw['inputs_embeds'][:, :p].cpu(), expanded[4].float())
            teacher_positions = [int(active[-1]) - 1]
            content_positions = (active[:-1] - 1).tolist()
            natural_kw = None
            if cut['matched']:
                prefix = torch.tensor(original['raw_token_ids'][:cut['prefix_tokens']],
                                      device=runtime.device, dtype=torch.long)[None]
                embeddings = torch.cat((expanded[4].to(runtime.device).float(),
                    runtime.model.get_input_embeddings()(prefix)), dim=1)
                natural_kw = dict(inputs_embeds=embeddings, attention_mask=torch.ones(
                    embeddings.shape[:2], device=runtime.device, dtype=torch.long),
                    use_cache=False, return_dict=True)
                assert embeddings.shape[1] < runtime.model.config.max_position_embeddings
            observations = {}
            for condition in ('BASE', 'TRAINED_Q', 'TRAINED_EMPTY_Q'):
                kernel.install(expert, q if condition != 'TRAINED_EMPTY_Q' else q[:, :0])
                hook = r.q.scoped.replay.MedTraceLayerHook(module, expert)
                if condition != 'BASE':
                    hook.attach()
                try:
                    if condition != 'BASE':
                        hook.set_teacher_routing(batch.labels)
                    teacher, result = forward(teacher_kw, teacher_positions, content_positions, q, expert)
                    shifted = result.logits[0, active - 1].double()
                    target = labels[active]
                    nll = torch.logsumexp(shifted, dim=-1) - shifted.gather(1, target[:, None]).squeeze(1)
                    assert torch.isfinite(nll).all()
                    teacher.update(content_NLL_mean=float(nll[:-1].mean()),
                                   total_NLL_mean=float(nll.mean()), target_tokens=len(active))
                    del result, shifted, nll
                    natural = None
                    if natural_kw is not None:
                        mask = torch.zeros(natural_kw['inputs_embeds'].shape[:2],
                                           dtype=torch.bool, device=runtime.device)
                        mask[:, p - 1:] = True
                        if condition != 'BASE':
                            hook.set_request_routing(mask)
                        natural, result = forward(natural_kw,
                            [natural_kw['inputs_embeds'].shape[1] - 1], [], q, expert)
                        natural['original_next_token_matches'] = natural['argmax_token'] == cut['original_next_token']
                        del result
                    observations[condition] = dict(teacher=teacher, natural=natural)
                finally:
                    hook.detach()
            assert all(torch.equal(v, before[k]) for k, v in r.old.snapshot(expert).items())
            result = dict(task_id=tid, role=task['role'], query_id=row['query_id'],
                original_at_cap=original['at_cap'], boundary=cut, conditions=observations,
                weights_unchanged_exact=True)
            audits.append(result)
            write(ROOT / 'rows' / tid / (row['query_id'] + '.json'), result)
            write(ROOT / 'PROGRESS.json', dict(completed=len(audits), counts=dict(counts), time=time.time()))
        del prepared, q, expert
    matched = [v for v in audits if v['boundary']['matched']]
    parity_failures = [dict(task_id=v['task_id'], query_id=v['query_id']) for v in matched
        if not v['conditions']['TRAINED_Q']['natural']['original_next_token_matches']]
    caps = [v for v in audits if v['original_at_cap']]
    assert len(audits) == 70 and len(caps) == 5
    assert counts['forwards'] == 3 * (70 + len(matched))
    assert counts['updates'] == counts['generations'] == counts['backwards'] == 0
    budget()
    gaps = [dict(task_id=v['task_id'], query_id=v['query_id']) for v in caps
        if v['boundary']['matched'] and v['conditions']['TRAINED_Q']['natural']['original_next_token_matches']
        and v['conditions']['TRAINED_Q']['teacher']['EOS_margin'] > 0
        and v['conditions']['TRAINED_Q']['natural']['EOS_margin'] < 0]
    write(ROOT / 'AUDIT_FROZEN.json', dict(rows=audits, time=time.time()))
    write(ROOT / 'COMPLETE.json', dict(status='EOS_AUDIT_COMPLETE', rows=70, matched=len(matched),
        cap_training_rows=5, parity_failures=parity_failures, prefix_gap_rows=gaps,
        source_prefix_completion_test_admissible=bool(gaps) and not parity_failures,
        counts=dict(counts), weights_unchanged_exact=True, time=time.time()))
    fh.remove()
    vh.remove()


if __name__ == '__main__':
    code = 0
    try:
        main()
    except BaseException:
        code = 1
        write(ROOT / 'FAILURE.json', dict(error=traceback.format_exc(), counts=counts, time=time.time()))
        raise
    finally:
        write(ROOT / 'WORKER_EXIT.json', dict(code=code, counts=counts, seconds=time.time() - START, time=time.time()))
