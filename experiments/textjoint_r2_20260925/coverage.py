"""Freeze outcome-blind SLAKE pressure candidates and score their Base outputs."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(os.environ['RUN_ROOT'])
DATA = Path(os.environ['SLAKE_ROOT'])


def write(name, value):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def freeze():
    tasks = json.loads((ROOT / 'private/TASKS_MATRIX.json').read_text())['tasks']
    aux = json.loads((ROOT / 'private/AUXILIARY_POOL.json').read_text())
    used = [r for t in tasks for r in [t['native'], *t['evaluation'], *t['U_fit'], *t['U_expanded']]] + aux
    used_groups = {r['source_group'] for r in used}
    used_text = {' '.join(r['question'].casefold().split()) for r in used}
    pools = {}
    for split in ('validation', 'test'):
        source = json.loads((DATA / f'{split}.json').read_text())
        eligible = [r for r in source if r['q_lang'] == 'en'
                    and 'SLAKE:' + r['img_name'].split('/')[0] not in used_groups
                    and ' '.join(r['question'].casefold().split()) not in used_text]
        by_group = {}
        for r in eligible:
            by_group.setdefault(r['img_name'].split('/')[0], []).append(r)
        rng = random.Random(20260925 if split == 'validation' else 20260926)
        rows = []
        for group in sorted(by_group):
            candidates = by_group[group][:]
            rng.shuffle(candidates)
            candidates.sort(key=lambda r: (r['content_type'], r['answer_type']))
            # A deterministic, outcome-blind round robin covers distinct QA types.
            chosen = []
            types = set()
            for r in candidates:
                label = (r['content_type'], r['answer_type'])
                if label not in types:
                    chosen.append(r)
                    types.add(label)
            chosen += [r for r in candidates if r not in chosen]
            for rank, r in enumerate(chosen):
                image = DATA / 'imgs' / r['img_name']
                if not image.is_file():
                    raise FileNotFoundError(image)
                rows.append(dict(query_id=f'{split.upper()}-{r["qid"]}', dataset='SLAKE',
                                 image_path=str(image), image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                                 source_group='SLAKE:' + group, question=r['question'], reference=r['answer'],
                                 task='T2L_PRESSURE', split=split, source_qid=r['qid'],
                                 answer_type=r['answer_type'], content_type=r['content_type'],
                                 initial=rank < 4))
        pools[split] = rows
    assert {r['source_group'] for r in pools['validation']}.isdisjoint(
        {r['source_group'] for r in pools['test']})
    assert not any(r['source_group'] in used_groups for v in pools.values() for r in v)
    write('private/PRESSURE_CANDIDATES.json', dict(rule='split isolation; English QA; exclude all old task and U sources and text; 4 per source first; seed 20260925/26; no edit outputs', pools=pools))
    write('COVERAGE_PREFLIGHT.json', dict(validation_initial=sum(r['initial'] for r in pools['validation']),
                                          validation_total=len(pools['validation']), validation_sources=len({r['source_group'] for r in pools['validation']}),
                                          test_initial=sum(r['initial'] for r in pools['test']), test_total=len(pools['test']),
                                          test_sources=len({r['source_group'] for r in pools['test']}), candidate_outputs_seen=False))


def generate(split='validation', initial=True):
    import torch
    sys.path.insert(0, str(ROOT))
    import worker_v3 as worker
    from freshstart import runtime as rt
    rows = json.loads((ROOT / 'private/PRESSURE_CANDIDATES.json').read_text())['pools'][split]
    if initial:
        rows = [r for r in rows if r['initial']]
    task = json.loads((ROOT / 'private/TASKS_MATRIX.json').read_text())['tasks'][0]
    rt.check_budget()
    with rt.gpu_session('BASE_PRESSURE_' + split.upper()):
        runtime = rt.load_runtime(ROOT / ('runtime_pressure_' + split), 20260924)
        record = worker.record(task)
        for i, row in enumerate(rows, 1):
            rt.check_budget()
            worker.base(runtime, row, record)
            if i % 10 == 0 or i == len(rows):
                write(f'PRESSURE_{split.upper()}_PROGRESS.json', dict(done=i, total=len(rows), epoch=time.time()))
        if not runtime.base_guard.verify()['unchanged']:
            raise RuntimeError('Base backbone changed')
    write(f'PRESSURE_{split.upper()}_GENERATED.json', dict(status='GPU_COMPLETE', inputs=len(rows), epoch=time.time()))


def audit(split='validation', target=50, min_sources=15):
    sys.path.insert(0, str(ROOT))
    import worker_v3 as worker
    rows = json.loads((ROOT / 'private/PRESSURE_CANDIDATES.json').read_text())['pools'][split]
    scored = []
    expected = 0
    missing = 0
    failed = 0
    blocked_path = ROOT / 'private/judge/JUDGE_MISSING_LOCK.json'
    blocked = set(json.loads(blocked_path.read_text())['keys']) if blocked_path.exists() else set()
    for row in rows:
        base_path = ROOT / 'private/base' / f'{worker.input_id(row)}.json'
        if not base_path.exists():
            continue
        expected += 1
        key = worker.request(row, json.loads(base_path.read_text()))
        score_path = ROOT / 'private/judge/scores' / f'{key}.json'
        if not score_path.exists():
            missing += 1
            failed += key in blocked
            continue
        verdict = json.loads(score_path.read_text())['is_correct']
        scored.append((row, verdict))
    correct = [r for r, v in scored if v]
    groups = {r['source_group'] for r in correct}
    pending = missing-failed
    result = dict(split=split, generated=expected, scored=len(scored), missing=missing,
                  failed_no_retry=failed, pending=pending,
                  base_correct=len(correct), base_wrong=len(scored)-len(correct),
                  base_correct_sources=len(groups), target=target, min_sources=min_sources,
                  status='PENDING' if pending else 'READY' if len(correct) >= target and len(groups) >= min_sources else 'INSUFFICIENT')
    write(f'BASE_COVERAGE_{split.upper()}.json', result)
    if result['status'] != 'READY':
        return result
    by_group = {}
    for row in correct:
        by_group.setdefault(row['source_group'], []).append(row)
    rng = random.Random(20260927 if split == 'validation' else 20260928)
    groups = list(by_group)
    rng.shuffle(groups)
    selected = []
    while len(selected) < target:
        progressed = False
        for group in groups:
            if by_group[group] and len(selected) < target:
                selected.append(by_group[group].pop(0))
                progressed = True
        if not progressed:
            break
    assert len(selected) == target
    assert len({(r['image_sha256'],r['question']) for r in selected}) == target
    assert len({r['source_group'] for r in selected}) >= min_sources
    write(f'private/PRESSURE_{split.upper()}_FROZEN.json', dict(selection='Base-correct only; source round robin; seed fixed before edit outcomes',
                                                                 rows=selected, created_epoch=time.time()))
    return result


if __name__ == '__main__':
    if sys.argv[1:] == ['freeze']:
        freeze()
    elif sys.argv[1:] == ['generate']:
        generate()
    elif sys.argv[1:] == ['audit']:
        print(json.dumps(audit()))
    else:
        raise SystemExit('Usage: coverage.py freeze|generate|audit')
