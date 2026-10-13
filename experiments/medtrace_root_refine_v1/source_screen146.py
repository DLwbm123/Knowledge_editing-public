"""Bounded source-only retrieval packet; lexical coverage never admits training."""
import collections
import json
import os
from pathlib import Path
import re
import subprocess
import time


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def image_key(row):
    text = str(row.get('source_group') or row.get('image_id') or row.get('original_image_path') or row['image_path'])
    matches = re.findall(r'xmlab\d+|synpic\d+', text.lower())
    return (row['dataset'], matches[0]) if matches else (row['dataset'], text)


def normalized(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def terms(text):
    stop = {'the', 'a', 'an', 'of', 'in', 'and', 'with', 'to', 'or', 'is', 'are', 'this', 'that', 'it', 'what'}
    return set(normalized(text).split()) - stop


def candidates(task, pool, excluded):
    target = task['native']['reference']
    ranked = []
    for row in pool:
        key = image_key(row)
        if key in excluded or key == image_key(task['native']):
            continue
        exact = normalized(target) == normalized(row['answer'])
        overlap = terms(target) & terms(row['answer'])
        if not exact and not overlap:
            continue
        score = len(overlap) / max(1, len(terms(target) | terms(row['answer'])))
        ranked.append((not exact, -score, key, str(row['qid']), row))
    selected, seen = [], set()
    for _, _, key, _, row in sorted(ranked, key=lambda x: x[:-1]):
        if key not in seen:
            selected.append(row)
            seen.add(key)
        if len(selected) == 6:
            break
    return selected


def selfcheck():
    task = dict(native=dict(dataset='SLAKE', source_group='SLAKE:xmlab1', reference='Brain edema'))
    pool = [dict(dataset='SLAKE', image_id=f'xmlab{i}', qid=str(i), answer='Brain edema') for i in range(1, 12)]
    rows = candidates(task, pool + pool, {('SLAKE', 'xmlab2')})
    assert len(rows) == 6 and len({image_key(r) for r in rows}) == 6
    assert not {image_key(r) for r in rows} & {('SLAKE', 'xmlab1'), ('SLAKE', 'xmlab2')}
    assert not candidates(task, [dict(pool[0], answer='Kidney')], set())
    return dict(status='PASS', distinct_images=True, excluded_and_native_rejected=True,
                maximum_six_retrieval_candidates=True, automatic_source_admission=False)


def main():
    checks = selfcheck()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    command = subprocess.check_output(['ps', '-ww', '-p', str(os.getpid()), '-o', 'args='], text=True).strip()
    assert not any(x in command.lower() for x in ('wangbomin', 'knowledge_editing', 'medtrace'))
    up = Path(os.environ['RUN_ROOT']).resolve()
    out = Path(os.environ['FOLLOWUP_ROOT']).resolve()
    assert out != up and out.parent.parent == up.parent.parent
    with (out/'private/SOURCE_SCREEN_ONCE').open('x'):
        pass
    env = read(up/'private/LAUNCH_ENV.json')
    original = read(Path(env['PROTECTION_PARENT'])/'private/BENCHMARK146_QUEUE.json')['tasks']
    assert original == read(out/'private/ORIGINAL146_TASKS.json') and len(original) == 146
    existing = read(out/'private/EXISTING_SOURCE_BINDINGS.json')
    reusable = set(existing['exact_original_task_matches'])
    assert len(reusable) == 7
    parent = up.parent.parent/'medtrace-scope-anchor-audit-v1-20261009/run'
    screen = read(parent/'public/SOURCE_SCREEN.json')
    assert screen['source_only'] and screen['known_case_exclusion']
    pool = read(parent/'private/ELIGIBLE_SOURCE_POOL.json')
    assert len(pool) == screen['eligible_QA'] == 268
    grouped = collections.defaultdict(list)
    raw_files = {}
    for row in pool:
        path = Path(row['source_file'])
        if path not in raw_files:
            raw_files[path] = {str(x['qid']): x for x in read(path)}
        raw = raw_files[path][str(row['qid'])]
        assert raw == row['raw_source'] and raw['question'] == row['question'] and str(raw['answer']) == row['answer']
        if row['dataset'] == 'SLAKE':
            assert path.name == 'train.json' and raw['q_lang'] == 'en'
        else:
            assert row['dataset'] == 'VQA-RAD' and raw['evaluation'] == 'evaluated' and raw['phrase_type'] == 'freeform'
        assert Path(row['image_path']).is_file()
        grouped[image_key(row)].append(row)
    queries = read(Path(env['GATE_PREVIOUS'])/'private/PAPER_QUERIES.json')['146']
    qualification = read(Path(env['FP32_BASELINE_PARENT'])/'private/FP32_SEMANTIC_QUALIFICATION.json')['rows']
    excluded = {image_key(row) for row in queries.values()}
    excluded.update(image_key(row['row']) for row in qualification)
    excluded.update(image_key(t['native']) for t in original)
    excluded.update(image_key(row) for row in existing['anchors'])
    packet = []
    for task in original:
        if task['order'] in reusable:
            continue
        selected = candidates(task, pool, excluded)
        packet.append(dict(owner=task['order'], native=task['native'],
            candidates=[dict(source=row, same_image_source_QA=grouped[image_key(row)],
                relation_status='UNREVIEWED', image_review='NOT_RUN', basis_conflict_review='NOT_RUN',
                admitted=False) for row in selected],
            external_search_slots_remaining=6-len(selected),
            status='SOURCE_EVIDENCE_REVIEW_REQUIRED' if selected else 'EXTERNAL_SOURCE_SEARCH_REQUIRED'))
    assert len(packet) == 139 and all(len(p['candidates']) <= 6 for p in packet)
    result = dict(status='BOUNDED_LOCAL_RETRIEVAL_COMPLETE_SOURCE_ADMISSION_PENDING',
        original_edits=146, original_version_reuse_candidates=7, owners_requiring_qualification=139,
        eligible_source_QA=268, eligible_source_images=len(grouped),
        candidate_count_histogram=dict(collections.Counter(len(p['candidates']) for p in packet)),
        owners_with_two_or_more_local_candidates=sum(len(p['candidates']) >= 2 for p in packet),
        retrieval_is_not_source_evidence=True, new_qualified_owners=0,
        external_source_review_not_complete=True, training_ready=False,
        new_model_calls=0, new_Judge=0, downloaded_bytes=0, epoch=time.time())
    write(out/'private/SOURCE_REVIEW_PACKET_LOCAL1.json', packet)
    write(out/'public/SOURCE_RETRIEVAL_LOCAL1.json', result)
    write(out/'private/SOURCE_SCREEN_EXECUTION1.json', dict(status='COMPLETE_CPU_RETRIEVAL',
        pid=os.getpid(), argv=command, selfcheck=checks, source_QA_verified=268,
        full_evaluation_image_exclusion=True, source_pool_known_case_exclusion_inherited=True,
        semantic_review_complete=False, training_started=False, epoch=time.time()))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
