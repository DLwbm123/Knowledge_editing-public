"""Extend a fixed TT endpoint's protection span once; no TT training or weight copies."""
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['V119_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])
PARENT_PATH = Path(os.environ['V118_PARENT'])
START = time.time()
STYLE = 'Please answer the following question about this image: '
counts = dict(forwards=0, backwards=0, operator_backwards=0, updates=0,
              generations=0, vision_forwards=0, path_positions=0)


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
    import extend
    import module_run as r
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from types import SimpleNamespace
    r.c.budget = budget
    torch.manual_seed(lock['seed'])
    torch.cuda.manual_seed_all(lock['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    write(ROOT / 'EXTENSION_SELF_CHECK.json', extend.selfcheck())
    tasks = read(ROOT / 'TASKS.json')['tasks']
    assert len(tasks) == 14 and sum(len(t['protect']) for t in tasks) == 28
    assert read(PARENT_PATH / 'COMPLETE.json')['status'] == 'BLIND_REVIEW_PENDING'
    assert read(PARENT_PATH / 'WORKER_EXIT.json')['code'] == read(PARENT_PATH / 'GUARD_EXIT.json')['code'] == 0
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=PARENT / 'private/cpu_gate'))
    runtime.model.eval()
    r.old.fp32(runtime)
    assert not any(v.requires_grad for v in runtime.model.parameters())
    bindings = read(PARENT / 'private/EVAL_BINDINGS.json')
    assert runtime.generation_config == next(iter(bindings.values()))['generation']
    template = read(PARENT / 'private/M13_SOURCE_PACKET.json')['original_tasks'][4]
    def count(*_):
        budget()
        counts['forwards'] += 1
    fh = runtime.model.register_forward_pre_hook(count)
    vh = runtime.model.get_vision_tower().register_forward_pre_hook(
        lambda *_: counts.__setitem__('vision_forwards', counts['vision_forwards'] + 1))

    def prepare(rows):
        runtime.model.half()
        try:
            return r.s.prepare(runtime, bindings, template, rows)
        finally:
            r.old.fp32(runtime)

    def emit(item, hook, path):
        budget()
        counts['generations'] += 1
        answer, _ = r.s.fit.short.generate(runtime, hook, item[2], item[3])
        tokens = list(answer.raw_token_ids)
        value = dict(binding=item[4], raw_answer=answer.decoded_text, raw_token_ids=tokens,
                     EOS=bool(tokens) and tokens[-1] == runtime.adapter.tokenizer.eos_token_id,
                     at_cap=len(tokens) >= 1024)
        write(path, value)
        return value

    audits = {}
    for task in tasks:
        dest = ROOT / task['task_id']
        dest.mkdir()
        vectors = []
        augmented = [dict(row, query_id=row['query_id'] + '_STYLE', question=STYLE + row['question'])
                     for row in task['protect']]
        for item in prepare(augmented):
            captured = []
            def capture(_module, args):
                assert args[0].shape[0] == 1
                captured.append(args[0][0, -1:].detach().cpu().double())
            handle = runtime.get_module(r.c.LAYER).register_forward_pre_hook(capture)
            try:
                answer = emit(item, None, dest / 'BASE_STYLE' / (item[0]['query_id'] + '.json'))
            finally:
                handle.remove()
            assert answer['EOS'] and not answer['at_cap'] and len(captured) == len(answer['raw_token_ids'])
            counts['path_positions'] += len(captured)
            budget()
            vectors.extend(captured)
        old_q = torch.load(PARENT_PATH / task['task_id'] / 'basis.pt', map_location='cpu', weights_only=True)
        q, audit = extend.basis(old_q, vectors)
        assert sum(v['new_rank'] for v in audits.values()) + audit['new_rank'] <= 4096
        audits[task['task_id']] = audit
        torch.save(q, dest / 'basis.pt')
        restored = torch.load(dest / 'basis.pt', map_location='cpu', weights_only=True)
        assert torch.equal(q, restored)
        write(dest / 'BASIS_ADMISSION.json', dict(**audit, save_load_exact=True, counts=dict(counts), time=time.time()))
        write(ROOT / 'PROGRESS.json', dict(phase='EXTENSION', task=task['task_id'],
            completed=len(audits), counts=dict(counts), time=time.time()))
        del old_q, q, restored, vectors
    assert counts['generations'] == 28
    write(ROOT / 'ALL_BASES_FROZEN.json', dict(audits=audits, counts=dict(counts), time=time.time()))
    mechanism = []
    for task in tasks:
        dest = ROOT / task['task_id']
        old_q = torch.load(PARENT_PATH / task['task_id'] / 'basis.pt', map_location=runtime.device, weights_only=True)
        q = torch.load(dest / 'basis.pt', map_location=runtime.device, weights_only=True)
        state = torch.load(PARENT_PATH / task['task_id'] / 'PATH_NULL/delta.pt', map_location=runtime.device, weights_only=True)
        expert = r.p.expert(state, lock['seed'], runtime.device)
        assert all(torch.equal(v, state[k]) for k, v in r.old.snapshot(expert).items())
        b, a = expert.factors()
        def effective_norm(basis):
            projected = a.double() - (a.double() @ basis) @ basis.T
            return float((((b.double().T @ b.double()) * (projected @ projected.T)).sum()).clamp_min(0).sqrt())
        operator_audit = dict(parent_effective_norm=effective_norm(old_q), extended_effective_norm=effective_norm(q))
        kernel.install(expert, q)
        hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER), expert)
        hook.attach()
        evaluation = prepare(task['train'] + task['heldout'] + task['protect'] + task['protect_extra'])
        try:
            for item in evaluation:
                answer = emit(item, hook, dest / 'outputs' / (item[0]['query_id'] + '.json'))
                if item[0]['query_id'] in task['Base_protect']:
                    old = task['Base_protect'][item[0]['query_id']]
                    mechanism.append(dict(task_id=task['task_id'], query_id=item[0]['query_id'],
                        Base_tokens_exact=answer['raw_token_ids'] == old['raw_token_ids'],
                        Base_fulltext_exact=answer['raw_answer'] == old['raw_answer'],
                        maximum_active_residual_norm=max(v['active_residual_norm'] for v in hook.last_generation_trace)))
            assert all(torch.equal(v, state[k]) for k, v in r.old.snapshot(expert).items())
            write(dest / 'GENERATION_COMPLETE.json', dict(rows=len(evaluation), TT_weights_unchanged_exact=True,
                **operator_audit, time=time.time()))
            write(ROOT / 'PROGRESS.json', dict(phase='EVALUATION', task=task['task_id'], counts=dict(counts), time=time.time()))
        finally:
            hook.detach()
    assert len(mechanism) == 28 and counts['generations'] == 210
    assert counts['backwards'] == counts['updates'] == counts['operator_backwards'] == 0
    write(ROOT / 'PATH_MECHANISM_AUDIT.json', mechanism)
    write(ROOT / 'COMPLETE.json', dict(status='BLIND_REVIEW_PENDING', counts=dict(counts), time=time.time()))
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
