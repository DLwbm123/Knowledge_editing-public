"""Real model preparation shared by the isolated canary and formal workers."""
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import torch
from freshstart.runtime import ROOT, LAYER, read, write, check_budget, gpu_session, load_runtime, epoch
from m3bench_repro.editors.llava_runtime import EditorRecord
from m3bench_repro.editors.routing import MemoryRouter, balanced_radius, route_dict_equal
from methods.medtrace.selective_write import LowRankExpert
from scripts.medtrace import stage15
from scripts.medtrace.stage17_prepare import digest, PROMPT, PROTOCOL
from scripts.medtrace.stage18_cfact import state_hash, assert_base_off
from scripts.medtrace.run_selective_write import save


def record(task):
    row = task['native']
    return EditorRecord(task['canonical_edit_id'], row['dataset'], row['question'], row['reference'],
        task['fit_questions'][0], Path(row['image_path']), '', task['order'], 'SOURCE_LABEL', 'FRESH_DEV16')


def input_key(row):
    return digest([row.get('image_sha256'), row['question']])


def score_request(row, output, consumer, kind='SOURCE_ANSWER_AGREEMENT'):
    payload = dict(question=row['question'], gold_answer=row['reference'], raw_base_answer=output['raw_answer'])
    key = digest(dict(payload=payload, model='gpt-6-astra', prompt=PROMPT, protocol=PROTOCOL,
                      normalizer='NONE', namespace=ROOT.resolve().name,kind=kind))
    path = ROOT/'private/judge/pending'/f'{key}.json'
    if not path.exists():
        write(path, dict(key=key, record=dict(opaque_query_id=key, **payload), protocol=PROTOCOL,kind=kind))
    return dict(judge_key=key, consumer_id=consumer)


def key_for(runtime, rec):
    assert_base_off(runtime)
    with torch.inference_mode():
        return runtime.extract_layer_input_key(runtime.build_question_batch(rec),
            module_path=runtime.target_lock['balancedit']['targets'][0], pooling='mean')


def prepare(runtime, tasks):
    destination = ROOT/'private/preparation'
    destination.mkdir(parents=True, exist_ok=True)
    router = MemoryRouter('euclidean')
    queries, bases, routes = {}, {}, []
    evaluation = read(ROOT/'private/FRESH_EVALUATION.json')['candidate_packages']
    for task, package in zip(tasks, evaluation):
        check_budget(p0=True)
        rec = record(task)
        key = key_for(runtime, rec)
        positive = key_for(runtime, replace(rec, question=task['fit_questions'][0]))
        black = runtime.make_black_image(rec, destination/'black'/str(task['order']))
        negative = key_for(runtime, replace(rec, image_path=black))
        radius = balanced_radius(key, positive, negative, alpha=.2, distance='euclidean')
        router.add(task['canonical_edit_id'], key, radius)
        rows = [task['native']] + task['H_fit'] + task['U_fit'] + package['evaluation']
        for row in rows:
            fp = input_key(row)
            if fp not in queries:
                queries[fp] = key_for(runtime, replace(rec, question=row['question'], image_path=Path(row['image_path']))) .cpu()
            output, _, _, _ = stage15.base_output(runtime, ROOT/'fresh_base', row, rec)
            score = score_request(row, output, f'base/{task["order"]}/{fp}/{digest(row["reference"])}')
            bases[digest([fp, row['reference']])] = dict(row=row, output=output, **score)
        save(destination/'router.pt', router.export_state())
        routes.append(dict(prefix=task['order'], native_route=asdict(router.route(key))))
        print('PREPARED_BASE', task['order'], flush=True)
    save(destination/'query_keys.pt', queries)
    write(destination/'BASE_OUTPUTS.json', bases)
    write(destination/'ROUTE_CHECKS.json', routes)
    restored = MemoryRouter.from_state(torch.load(destination/'router.pt', weights_only=True), device=runtime.device)
    for query in queries.values():
        if not route_dict_equal(asdict(router.route(query.to(runtime.device))), asdict(restored.route(query.to(runtime.device))), radius_mode='float32'):
            raise ValueError('Real router save/load changed routing')
    write(destination/'BASE_READY.json', dict(status='PASS', tasks=len(tasks), inputs=len(queries), scored_inputs=len(bases),
        runtime='official_native', candidate_outputs_seen=False, base_unchanged=runtime.base_guard.verify()['unchanged']))
    return bases


def initialize(runtime, task, run, seed, *, p0=False):
    """Original native CP -> 80 A2 -> 320 CP W0, then exact low-rank conversion."""
    check_budget(training=True, p0=p0)
    frozen = read(ROOT/'RUN_MANIFEST.json')
    limit = epoch(frozen['p0_deadline_at'] if p0 else frozen['no_new_training_after'])
    cfg = dict(campaign_epoch=epoch(frozen['first_started_at']), train_seconds=limit-epoch(frozen['first_started_at']))
    t = dict(task, probes=[dict(task['native'], status='AVAILABLE')], U=task['U_fit'])
    cp = stage15.initialize(runtime, run, cfg, t, record=record(task), seed_base=seed, layer_path=LAYER)
    expert = LowRankExpert(cp, task['seed'], rank=4).to(runtime.device)
    with torch.no_grad():
        x = torch.linspace(-1, 1, 14336, device=runtime.device).reshape(1,14336)
        error = float((cp.residual(x)-expert.residual(x)).abs().max())
        if not torch.allclose(cp.residual(x), expert.residual(x), rtol=2e-4, atol=2e-5):
            raise ValueError('CP-to-low-rank conversion mismatch')
    path = run/'private/edits'/f'e{task["order"]:03d}'/'SHARED_W0.pt'
    binding = dict(run=str(run), seed=seed, edit=task['canonical_edit_id'], layer=LAYER,
        recipe='original native CP -> A2 80 -> CP W0 320 -> exact free rank4', parameter_hash=state_hash(expert), conversion_max_abs_error=error)
    save(path, dict(expert=expert.state_dict(), binding=binding))
    restored = copy.deepcopy(expert)
    restored.load_state_dict(torch.load(path, map_location=runtime.device, weights_only=True)['expert'])
    if state_hash(restored) != state_hash(expert):
        raise ValueError('W0 real save/reload mismatch')
    write(path.with_suffix('.json'), binding)
    return expert


def preflight():
    tasks = read(ROOT/'private/FRESH_TASKS.json')['tasks']
    result = dict(status='STARTING', actual_initializers=0, canary_training_passed=False)
    write(ROOT/'P0_PREPARATION.json', result)
    try:
        with gpu_session('P0_RUNTIME'):
            runtime = load_runtime(ROOT/'canary', 20260923)
            def boundary(_module, _args):
                # Applies to the native/A2 legacy loops too; no hidden unbounded initializer.
                check_budget(training=True, p0=True)
            handle = runtime.model.register_forward_pre_hook(boundary)
            prepare(runtime, tasks)
            for task in tasks[:7]:
                expert = initialize(runtime, task, ROOT/'canary', 20260923, p0=True)
                del expert
                result['actual_initializers'] += 1
                write(ROOT/'P0_PREPARATION.json', result)
            handle.remove()
            result.update(status='BASE_AND_CANARY_W0_READY', base_unchanged=runtime.base_guard.verify()['unchanged'])
    except Exception as error:
        result.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write(ROOT/'P0_PREPARATION.json', result)


if __name__ == '__main__':
    preflight()
