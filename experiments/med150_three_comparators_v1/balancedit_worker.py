"""Rebuild independent BalancEdit writers and evaluate the actual 150-entry bank."""
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['EXTENSION_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])
MED = Path(os.environ['MED150_ROOT'])
ORIGINAL = Path(os.environ['MED146_ROOT'])
SOURCE = Path(os.environ['FROZEN_SOURCE'])
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


def main():
    with (ROOT / 'WORKER_ONCE').open('x'):
        pass
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError('guard stop')))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('hard deadline')))
    signal.alarm(max(1, int(CLOCK['hard_end'] - time.time() - 30)))
    sys.path[:0] = [str(SOURCE), str(PARENT / 'private/tools'), str(PARENT / 'private/official_llava')]
    import torch
    import common as c
    from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor
    from m3bench_repro.editors.routing import MemoryRouter, decision_as_json
    from m3bench_repro.editors.llava_runtime import seed_everything
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from types import SimpleNamespace

    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(LOCK['gpu'])
    c.budget = budget
    torch.set_num_threads(4)
    old = read(ORIGINAL / 'INPUTS_FROZEN.json')['tasks']
    extra = read(MED / 'ADDITIONAL_INPUTS.json')['tasks']
    tasks = old + [x['task'] for x in extra]
    plan = read(MED / 'EVAL_PLAN_FROZEN.json')['plan']
    assert len(tasks) == 150 and [x['order'] for x in tasks] == list(range(1, 151))
    assert len(plan) == 2207 and read(MED / 'BASE_MASK_FROZEN.json')['status'] == 'FROZEN'
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=PARENT / 'private/cpu_gate'))
    runtime.model.eval()
    runtime.run_root = ROOT / 'work'
    runtime.run_root.mkdir(exist_ok=True)
    assert next(runtime.model.parameters()).dtype == torch.float16
    seed_everything(20260912)
    editor = BalanceEditPaperSpecEditor(runtime)
    actual = editor.config_lock()
    for key, value in LOCK['method_config'].items():
        if key in actual and key != 'config_sha256':
            assert actual[key] == value, ('method configuration mismatch', key)
    assert actual['steps_per_edit'] == 50 and actual['learning_rate'] == .01
    assert not any(p.requires_grad for p in editor.wrapper.base.parameters())
    expected_queries = {}
    for p in (MED / 'eval_outputs').glob('*.json'):
        v = read(p)
        q, b = v['binding']['input'], v['binding']['judge_input']
        identity = (b['question'], b['reference'], b['image_sha256'], b['prompt_ids'], b['attention_mask'])
        if q['query_id'] in expected_queries:
            assert expected_queries[q['query_id']] == identity
        expected_queries[q['query_id']] = identity
    assert len(expected_queries) == 2148
    entries, writers = [], {}
    loaded = None

    def evaluate(rows, task, prefix):
        nonlocal loaded
        for row in rows:
            budget()
            q = row['query']
            cid = json.dumps(['BALANCEDIT', row['mode'], prefix, row['owner'], q['query_id']], separators=(',', ':'))
            dest = ROOT / 'outputs' / (c.digest(cid) + '.json')
            assert not dest.exists(), 'No implicit retry'
            query = replace(c.record(task), record_id='query', question=q['question'], target='', official_rephrase='', image_path=Path(c.local_path(q['image_path'])))
            raw = runtime.adapter.prepare_inputs(query.image_path, query.question, None)
            identity = (q['question'], q['reference'], raw['image_sha256'], raw['input_ids'][0].tolist(), raw['attention_mask'][0].tolist())
            assert identity == expected_queries[q['query_id']]
            decision = editor._route(query)
            selected = decision.logical_edit_id if decision.activated else None
            if selected is not None:
                assert selected in writers
                if loaded != selected:
                    editor.wrapper.clear()
                    state = torch.load(writers[selected], map_location='cpu', weights_only=False)
                    assert state['edit_id'] == selected
                    editor.wrapper.load_exported_state(state['wrapper'])
                    loaded = selected
            COUNTS['generation_attempts'] += 1
            with editor._activated(selected):
                with torch.inference_mode():
                    answer = runtime.adapter.generate_prepared_with_result(raw, runtime.generation_config)
            tokens = list(answer.raw_token_ids)
            assert tokens
            binding = dict(question=q['question'], reference=q['reference'], image_sha256=raw['image_sha256'],
                           prompt_ids=raw['input_ids'][0].tolist(), attention_mask=raw['attention_mask'][0].tolist(),
                           runtime={'precision': 'float16', 'source': LOCK['runtime_source']}, generation=runtime.generation_config)
            write(str(dest.relative_to(ROOT)), dict(method='BALANCEDIT', mode=row['mode'], prefix=prefix,
                  owner=row['owner'], consumer_id=cid, binding=dict(input=q, judge_input=binding),
                  R0=dict(raw_answer=answer.decoded_text, raw_token_ids=tokens), route=decision_as_json(decision),
                  EOS=tokens[-1] == runtime.adapter.tokenizer.eos_token_id, at_cap=len(tokens) >= 1024,
                  actual_inserted_prefix=prefix))
            COUNTS['formal_consumers'] += 1
            write('PROGRESS.json', dict(phase='GENERATING', counts=COUNTS, time=time.time()))

    write('ADMISSION.json', dict(status='PASS', method='BALANCEDIT', tasks=150, method_config=actual, time=time.time()))
    for task in tasks:
        budget()
        editor.reset_editor_state()
        loaded = None
        audit = editor.apply_edit(c.record(task))
        COUNTS['edits'] += 1
        assert audit['steps'] == 50 and audit['finite_losses'] and audit['finite_gradients']
        editor.router.labels = [()] * len(editor.router)
        state = dict(edit_id=task['edit_id'], router=editor.router.export_state(), wrapper=editor.wrapper.export_state())
        assert len(state['router']['entries']) == 1 and state['router']['entries'][0]['logical_edit_id'] == task['edit_id']
        path = ROOT / 'writers' / f"{task['order']}.pt"
        path.parent.mkdir(exist_ok=True)
        assert not path.exists()
        torch.save(state, path.with_suffix('.tmp'))
        path.with_suffix('.tmp').replace(path)
        if task['order'] == 1:
            check = torch.load(path, map_location='cpu', weights_only=False)
            assert check['edit_id'] == task['edit_id'] and all(torch.equal(v, check['wrapper']['edits'][task['edit_id']][k]) for k, v in state['wrapper']['edits'][task['edit_id']].items())
        entries.append(dict(state['router']['entries'][0], label=[]))
        writers[task['edit_id']] = path
        write(f'training/{task["order"]}.json', dict(audit=audit, time=time.time()))
        if task['order'] >= 147:
            loaded = task['edit_id']
            evaluate([x for x in plan if x['mode'] == 'single' and x['owner'] == task['order']], task, 1)
        write('PROGRESS.json', dict(phase='TRAINING', owner=task['order'], counts=COUNTS, time=time.time()))
        del state
    editor.reset_editor_state()
    loaded = None
    assert len(entries) == len(writers) == 150
    for owner in range(147, 151):
        editor.router = MemoryRouter.from_state(dict(distance='euclidean', entries=entries[:owner]), device=runtime.device)
        COUNTS['inserted'] = owner
        evaluate([x for x in plan if x['mode'] == 'insertion' and x['owner'] == owner], tasks[owner-1], owner)
    editor.router = MemoryRouter.from_state(dict(distance='euclidean', entries=entries), device=runtime.device)
    assert len(editor.router) == 150
    evaluate([x for x in plan if x['mode'] == 'bank'], tasks[0], 150)
    assert COUNTS['edits'] == COUNTS['inserted'] == 150 and COUNTS['formal_consumers'] == COUNTS['generation_attempts'] == 2207
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
