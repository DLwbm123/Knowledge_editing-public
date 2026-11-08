"""Fixed comparators on the already frozen T4/T5 cohorts; no method selection."""
from contextlib import nullcontext
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['COMPARE_ROOT'])
METHOD = os.environ['COMPARE_METHOD']
JOB = ROOT / METHOD


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p = Path(p);p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix('.tmp');temp.write_text(json.dumps(value, ensure_ascii=False, indent=2));temp.replace(p)


def budget():
    assert time.time() < read(ROOT / 'CLOCK.json')['hard_end'] - 1200
    assert not (ROOT / 'STOP.json').exists()
    assert time.time() - read(ROOT / os.environ.get('COMPARE_HEALTH_FILE','HEALTH.json'))['time'] < 60


def main():
    import torch
    import numpy as np
    cfg = read(ROOT / 'CONFIG.private.json')
    sys.path[:0] = [str(ROOT / 'source'), cfg['parent_tools'], str(Path(os.environ['PARENT_RUN']) / 'private/official_llava'), str(ROOT / 'tools')]
    import common as c
    import module_run as r
    c.budget = budget;r.c.budget = budget
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor, GracePaperSpecEditor, LoraPaperSpecEditor
    from m3bench_repro.editors.routing import MemoryRouter, decision_as_json
    from cache_editor import QueryAnchoredLinear
    from types import SimpleNamespace
    with (JOB / 'WORKER_ONCE').open('x'):pass
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('original clock')))
    signal.alarm(max(1, int(read(ROOT / 'CLOCK.json')['hard_end'] - time.time() - 1200)))
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=Path(os.environ['PARENT_RUN']) / 'private/cpu_gate'))
    r.old.fp32(runtime);runtime.model.eval();runtime.run_root = JOB / 'runtime';runtime.run_root.mkdir()
    backbone = [(p, p._version) for p in runtime.model.parameters()]
    assert all(not p.requires_grad for p, _ in backbone)
    if METHOD == 'BALANCEDIT':editor = BalanceEditPaperSpecEditor(runtime)
    elif METHOD == 'LORA':editor = LoraPaperSpecEditor(runtime)
    elif METHOD == 'GRACE':
        editor = GracePaperSpecEditor(runtime)
        base = editor.wrapper.base
        editor.wrapper = QueryAnchoredLinear(base)
        editor.wrapper.replacement = 'replace_prompt'
        runtime.replace_module(editor.target, editor.wrapper)
        # Preserve the benchmark's 100-step/cosine/last-MLP recipe. Only cache
        # handling changes: intervention stays on the same prompt positions.
        from cache_editor import CorrectedGraceEditor
        editor.route_generation = CorrectedGraceEditor.route_generation.__get__(editor)
    else:raise ValueError(METHOD)
    actual = editor.config_lock()
    if METHOD == 'LORA':
        import importlib.metadata
        actual['source'] = 'PEFT '+importlib.metadata.version('peft')+'; compatibility dependency, unchanged fixed LoRA equations/hyperparameters'
    write(JOB / 'METHOD_CONFIG.json', dict(config=actual, actual_decoder_precision='float32',
        multimodal_prefill='frozen FP16, identical to MDU input construction',
        cache_repair=METHOD == 'GRACE', training='native only; BE positive anchor uses first frozen FIT rewrite'))
    rng = torch.load(Path(os.environ['RAW_MED_ROOT']) / 'GENERATION_RNG.pt', map_location='cpu', weights_only=False)
    counts = dict(edits=0, updates=0, consumers=0, generations=0, diagnostics=0)
    counter = runtime.model.register_forward_pre_hook(lambda *_: budget())

    def record(task, query=None):
        rec = c.record(task)
        if query is not None:
            rec = replace(rec, record_id='query', question=query['question'], target='', official_rephrase='', image_path=Path(c.local_path(query['image_path'])))
        return rec

    def prepare(task, q, expected):
        # Only frozen multimodal-prefix modules change precision; trained LM
        # editors are never rounded to half precision during evaluation.
        model = runtime.llava_model();inner = model.get_model()
        prefix_modules = [inner.embed_tokens, inner.mm_projector, model.get_vision_tower()]
        for module in prefix_modules:module.half()
        try:
            raw = runtime.adapter.prepare_inputs(Path(c.local_path(q['image_path'])), q['question'], None)
            raw['images'] = raw['images'].half()
            with torch.no_grad():expanded = model.prepare_inputs_labels_for_multimodal(raw['input_ids'], None, raw['attention_mask'], None, None, raw['images'], image_sizes=None)
            assert expanded[4].dtype == torch.float16
        finally:
            for module in prefix_modules:module.float()
        binding = dict(question=q['question'], reference=q['reference'], image_sha256=raw['image_sha256'],
            prompt_ids=raw['input_ids'][0].tolist(), attention_mask=raw['attention_mask'][0].tolist())
        assert all(binding[k] == expected[k] for k in binding), 'MDU/comparator input mismatch'
        return raw, expanded, binding

    def generate(task, q, expected, cap=None, use_cache=True):
        raw, expanded, binding = prepare(task, q, expected)
        rec = record(task, q)
        context = nullcontext(None) if METHOD == 'LORA' else editor.route_generation(rec)
        old_config = runtime.generation_config
        runtime.generation_config = dict(old_config, use_cache=use_cache, **({'max_new_tokens':cap} if cap else {}))
        try:
            r.c.restore_rng(rng['rng']);np.random.set_state(rng['numpy_rng'])
            with context as decision:
                answer, _ = r.s.fit.short.generate(runtime, None, raw, expanded)
            counts['generations'] += 1
            if cap:counts['diagnostics'] += 1
        finally:runtime.generation_config = old_config
        return dict(answer=dict(raw_answer=answer.decoded_text, raw_token_ids=list(answer.raw_token_ids)),
            judge_input=binding, route=decision_as_json(decision) if decision is not None else dict(activated=True, kind='GLOBAL_LORA_ALWAYS_ACTIVE'),
            EOS=answer.raw_token_ids[-1] == runtime.adapter.tokenizer.eos_token_id, at_cap=len(answer.raw_token_ids) >= 1024)

    def train(task, folder, mode):
        budget();audit = editor.apply_edit(record(task))
        assert audit['finite_losses'] and audit['finite_gradients']
        steps = audit.get('steps', audit.get('epochs'))
        assert steps == {'GRACE':100, 'BALANCEDIT':50, 'LORA':5}[METHOD]
        counts['edits'] += 1;counts['updates'] += steps
        assert counts['updates'] <= {'GRACE':42600, 'BALANCEDIT':10750, 'LORA':2130}[METHOD]
        write(folder / 'training' / f"{mode}_{task['order']}.json", audit)

    def save_state(path):
        temp = path.with_suffix('.tmp')
        if METHOD == 'BALANCEDIT':
            torch.save(dict(wrapper=editor.wrapper.export_state(), router=editor.router.export_state(), history=editor.edit_history), temp)
        elif METHOD == 'GRACE':
            torch.save(dict(wrapper=editor.wrapper.export_state(), codebook=editor.codebook.export_state(),
                mapping=editor.requested_to_effective, history=editor.edit_history), temp)
        else:torch.save(dict(adapter=editor._capture_adapter_state(), history=editor.edit_history), temp)
        temp.replace(path)

    def restore_state(path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        if METHOD == 'BALANCEDIT':
            editor.wrapper.clear();editor.wrapper.load_exported_state(state['wrapper'])
        elif METHOD == 'GRACE':
            from m3bench_repro.editors.routing import GraceCodebook
            editor.wrapper.load_exported_state(state['wrapper'])
            editor.codebook = GraceCodebook.from_state(state['codebook'], device=runtime.device)
            editor.requested_to_effective = state['mapping']
        else:
            params = dict(editor.peft_model.named_parameters())
            with torch.no_grad():
                for name, value in state['adapter'].items():params[name].copy_(value.to(params[name]))
            editor._set_enabled(True)
        editor.edit_history = state['history']

    for cohort in ('T4','T5'):
        folder = JOB / cohort;folder.mkdir()
        data = read(ROOT / f'{cohort}_INPUTS.private.json');tasks = data['tasks']
        assert runtime.generation_config == data['generation']
        plan = data['plan'];expected = data['expected'];byid = {t['edit_id']:t for t in tasks}
        seq = [t for t in tasks if cohort == 'T4' or t['sequential_role_safe']]
        bankids = {t['edit_id'] for t in seq}
        assert len(tasks) == (67 if cohort == 'T4' else 148)
        editor.reset_editor_state()
        native = tasks[0]['native']
        base_answer = generate(tasks[0], native, expected[native['query_id']])
        counts['diagnostics'] += 1
        assert base_answer['answer']['raw_token_ids'] == data['base_native_tokens'][native['query_id']], 'Fixed Base precision parity failed'
        write(folder / 'BASE_PARITY.json', dict(status='PASS', actual_precision='FP32_FROZEN_FP16_PREFILL'))
        entries = [];writers = {};loaded = None

        def evaluate(rows, mode, prefix):
            nonlocal loaded
            for row in rows:
                budget();task = byid[row['edit_id']];q = row['query']
                if METHOD == 'BALANCEDIT' and mode == 'bank':
                    decision = editor._route(record(task, q));chosen = decision.logical_edit_id if decision.activated else None
                    if chosen and chosen != loaded:
                        restore_state(writers[chosen]);loaded = chosen
                result = generate(task, q, expected[q['query_id']])
                cid = c.digest([METHOD,cohort,mode,row['edit_id'],row['role'],q['query_id']])
                dest = folder / 'outputs' / (cid + '.json');assert not dest.exists()
                write(dest, dict(result, consumer_id=cid, method=METHOD, cohort=cohort, mode=mode,
                    owner=task['order'], edit_id=task['edit_id'], role=row['role'], query_id=q['query_id'], prefix=prefix))
                counts['consumers'] += 1
                assert counts['consumers'] <= 2696
                write(JOB / 'PROGRESS.json', dict(phase=mode, cohort=cohort, owner=task['order'], counts=counts, time=time.time()))

        for index, task in enumerate(tasks):
            editor.reset_editor_state();train(task, folder, 'single');loaded = task['edit_id']
            state_path = folder / (f"writer_{task['order']}.pt" if METHOD == 'BALANCEDIT' else 'single_resume.pt')
            save_state(state_path)
            if METHOD == 'BALANCEDIT':
                entries.append(editor.router.export_state()['entries'][0]);writers[task['edit_id']] = state_path
            if index == 0:
                q = task['native'];a = generate(task,q,expected[q['query_id']],16)
                if METHOD == 'GRACE':
                    b = generate(task,q,expected[q['query_id']],16,False)
                    assert a['answer']['raw_token_ids'] == b['answer']['raw_token_ids'], 'cache/no-cache parity failed'
                restore_state(state_path);b = generate(task,q,expected[q['query_id']],16)
                assert a['answer']['raw_token_ids'] == b['answer']['raw_token_ids'], 'saved state generation parity failed'
                write(folder / 'MECHANICAL.json', dict(status='PASS', save_load_tokens_equal=True,
                    cache_tokens_equal=True if METHOD == 'GRACE' else None))
            evaluate([x for x in plan if x['edit_id'] == task['edit_id']], 'single', 1)
        editor.reset_editor_state();loaded = None
        if METHOD == 'BALANCEDIT':
            editor.router = MemoryRouter.from_state(dict(distance='euclidean', entries=[e for e in entries if e['logical_edit_id'] in bankids]), device=runtime.device)
        else:
            for task in seq:
                train(task,folder,'bank');save_state(folder / 'bank_resume.pt')
                write(JOB / 'PROGRESS.json', dict(phase='bank_training', cohort=cohort, owner=task['order'], counts=counts, time=time.time()))
        evaluate([x for x in plan if x['edit_id'] in bankids], 'bank', len(seq))
        assert all(p._version == version for p, version in backbone), 'Backbone changed'
        write(folder / 'GENERATION_COMPLETE.json', dict(status='COMPLETE', single_edits=len(tasks), bank_edits=len(seq),
            consumers=len(list((folder/'outputs').glob('*.json'))), actual_bank=True, counts=counts, time=time.time()))
        editor.reset_editor_state()
        # Every declared weight consumer is complete. Retire this comparator's
        # exact generated state files before constructing the next cohort bank.
        retired = []
        for path in folder.glob('*.pt'):
            retired.append(dict(name=path.name, bytes=path.stat().st_size));path.unlink()
        write(folder / 'STATE_RETIREMENT.private.json', dict(files=retired, reason='single, real final bank and source save/load consumers completed', raw_data_untouched=True))
    counter.remove()
    assert counts['consumers'] == 2696
    write(JOB / 'COMPLETE.json', dict(status='ALL_GENERATION_COMPLETE', counts=counts, time=time.time()))


if __name__ == '__main__':
    code = 0
    try:main()
    except BaseException:
        code = 1;write(JOB / 'FAILURE.json', dict(error=traceback.format_exc(), time=time.time()));raise
    finally:write(JOB / 'EXIT.json', dict(code=code, time=time.time()))
