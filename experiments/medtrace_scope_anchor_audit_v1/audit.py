"""Read-only paired outputs and source-role screening; no model or judge calls."""
import collections
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def transition(before, after):
    assert before in (0, 1) and after in (0, 1)
    return {(1, 1): 'both_correct', (1, 0): 'lost',
            (0, 1): 'gained', (0, 0): 'both_wrong'}[before, after]


def divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def source_key(dataset, image):
    matches = set(re.findall(r'xmlab\d+|synpic\d+', str(image).lower()))
    assert len(matches) == 1, (dataset, image)
    return dataset, next(iter(matches))


def row_key(row):
    return row['expert_order'], row['role'], row['binding']['input']['query_id']


def image_key(row):
    return source_key(row['dataset'], row.get('source_group') or row.get('original_image_path') or row['image_path'])


def selfcheck():
    assert [transition(a, b) for a, b in [(1, 1), (1, 0), (0, 1), (0, 0)]] == ['both_correct', 'lost', 'gained', 'both_wrong']
    assert divergence([1, 2], [1, 3]) == 1
    assert divergence([1], [1, 2]) == 1
    assert divergence([1], [1]) is None
    assert source_key('SLAKE', '/a/xmlab12/source.jpg') == ('SLAKE', 'xmlab12')
    assert image_key(dict(dataset='SLAKE', source_group='SLAKE:xmlab12')) == ('SLAKE', 'xmlab12')
    assert not admitted([dict(qualified_images=2, basis_conflict_review='PASS')]*7 + [dict(qualified_images=0, basis_conflict_review='NOT_RUN')])
    assert not admitted([dict(qualified_images=2, basis_conflict_review='UNKNOWN')]*8)
    assert admitted([dict(qualified_images=2, basis_conflict_review='PASS')]*8)


def admitted(owners):
    return len(owners) == 8 and all(x['qualified_images'] >= 2 and x['basis_conflict_review'] == 'PASS' for x in owners)


def pair_outputs(parent, env, out):
    damage = Path(env['DAMAGE_PARENT'])
    raw = {}
    for item in read(damage/'private/DAMAGE_SEMANTIC.json'):
        d = read(item['path'])
        if d['arm'] == 'RAW':
            key = row_key(d)
            assert key not in raw
            raw[key] = (d, item['correct'], item['path'])
    bases = read(parent/'private/HELD_BASE.json')
    base_scores = {v['path']: v['correct'] for v in bases.values()}
    rows = []
    for item in read(parent/'private/PROTECTION_SEMANTIC.json'):
        d = read(item['path'])
        if d['role'] in ('NATIVE', 'FIT'):
            continue
        key = row_key(d)
        a, before, raw_path = raw.pop(key)
        assert a['binding']['input'] == d['binding']['input']
        x, y = a['R0'], d['R0']
        rows.append(dict(owner=key[0], role=key[1], query_id=key[2],
            RAW=before, PROTECTED=item['correct'], transition=transition(before, item['correct']),
            same_text=x['raw_answer'] == y['raw_answer'],
            same_tokens=x['raw_token_ids'] == y['raw_token_ids'],
            first_divergence_token=divergence(x['raw_token_ids'], y['raw_token_ids']),
            Base_correct=base_scores.get(d.get('baseline_path')),
            original_primary=d.get('original_primary', False),
            RAW_path=raw_path, PROTECTED_path=item['path']))
    assert not raw and len(rows) == 883
    panels = {}
    for role in ('T1G', 'T2G', 'T1L', 'T2L', 'HELDOUT', 'FP32_CORRECT62', 'BASE_WRONG34', 'ORIGINAL63'):
        selected = [r for r in rows if r['role'] == role or
                    r['role'] == 'HELDOUT' and (
                        role == 'FP32_CORRECT62' and r['Base_correct'] == 1 or
                        role == 'BASE_WRONG34' and r['Base_correct'] == 0 or
                        role == 'ORIGINAL63' and r['original_primary'])]
        def summarize(rs):
            return dict(n=len(rs), RAW_correct=sum(x['RAW'] for x in rs),
                PROTECTED_correct=sum(x['PROTECTED'] for x in rs),
                transitions=dict(collections.Counter(x['transition'] for x in rs)),
                same_text=sum(x['same_text'] for x in rs), same_tokens=sum(x['same_tokens'] for x in rs),
                first_token_divergences=sum(x['first_divergence_token'] == 0 for x in rs))
        panels[role] = dict(total=summarize(selected), owners=[dict(owner=i, **summarize([x for x in selected if x['owner'] == i])) for i in range(1, 9)])
    assert panels['T2G']['total']['RAW_correct'] == 26 and panels['T2G']['total']['PROTECTED_correct'] == 22
    assert panels['HELDOUT']['total']['transitions'] == dict(both_correct=490, both_wrong=231, lost=32, gained=15)
    assert panels['FP32_CORRECT62']['total']['n'] == 496
    assert panels['ORIGINAL63']['total']['n'] == 504
    write(out/'private/PAIRED_ROWS.json', rows)
    write(out/'public/PAIRED_RESULTS.json', dict(status='PASS', panels=panels, model_calls=0, new_Judge=0,
        interpretation='Text/token identities and accepted scores, not clinical labels or causal attribution'))


def screen(parent, env, out):
    root = parent.parents[1]
    old = root/'medtrace-full-method-comparison-20261003/run/private/legacy_stage17'
    roles_path = root/'scope-m3bench-exposure-audit-20261001/run/private/IDENTITY_ROLE_LEDGER.json'
    source_path = root/'scope-m3bench-candidate-audit-20261001/run/private/SOURCE_QA_ROLE_AUDIT.json'
    join, roles, sources = read(old/'ROLE_BOUNDARY_JOIN.json'), read(roles_path), read(source_path)
    helper = Path(os.environ['QUESTION_HELPER'])
    spec = importlib.util.spec_from_file_location('source_helpers', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    family, normalized = module.family, module.p0.normalized
    module.selfcheck()
    groups = collections.defaultdict(set)
    for row in roles:
        groups[row['dataset'], row['image_id']].update(row['roles'])
    forbidden = {'PROTECTED_SPLIT', 'FORMAL_OR_RESERVED_SOURCE', 'CAL_CHECK_SUPPORT_RESERVED', 'FROZEN_FIT_SOURCE'}
    excluded = {k for k, v in groups.items() if v & forbidden}
    excluded.update(source_key(d, p) for d, p in join['reserved_source_groups'])
    queries = read(Path(env['GATE_PREVIOUS'])/'private/PAPER_QUERIES.json')['146']
    excluded.update(image_key(x) for x in queries.values())
    for role in ('FIT', 'CHECK'):
        excluded.update(image_key(x) for x in read(Path(env['REPLAY_PARENT'])/f'private/REPLAY_{role}.json'))
    qualification = read(Path(env['ANSWER_PARENT'])/'private/SEMANTIC_QUALIFICATION.json')['rows']
    excluded.update(image_key(x['row']) for x in qualification)
    data = Path(env['DATA_ROOT'])/'knowledge_editing/data/m3bench'
    slake = read(data/'SLAKE/train.json')
    vqa = read(data/'VQA-RAD/VQA_RAD Dataset Public.json')
    test_images = {source_key('VQA-RAD', x['image_name']) for x in vqa if x['phrase_type'].startswith('test_')}
    excluded.update(test_images)
    cases = {x['image_case_url'] for x in vqa if source_key('VQA-RAD', x['image_name']) in excluded} - {None, '', 'NULL'}
    index = {('SLAKE', str(x['qid'])): x for x in slake}
    index.update({('VQA-RAD', str(x['qid'])): x for x in vqa})
    pool, rejected = [], collections.Counter()
    for row in sources:
        key = row['dataset'], row['image_id']
        if key in excluded:
            rejected['excluded_image_role'] += 1
            continue
        if not groups[key]:
            rejected['unknown_role'] += 1
            continue
        raw = index[row['dataset'], str(row['qid'])]
        assert raw == row['raw_source']
        assert raw['question'] == row['question'] and str(raw['answer']) == row['answer']
        if row['dataset'] == 'VQA-RAD' and (raw['image_case_url'] in cases or raw['evaluation'] != 'evaluated' or raw['phrase_type'] != 'freeform'):
            rejected['case_or_annotation_restriction'] += 1
            continue
        if row['dataset'] == 'SLAKE' and raw['q_lang'] != 'en':
            rejected['non_english'] += 1
            continue
        if not Path(row['image_path']).is_file():
            rejected['missing_image'] += 1
            continue
        pool.append(row)
    tasks = read(parent/'private/BENCHMARK146_QUEUE.json')['tasks'][:8]
    packet, coverage = [], []
    stopwords = {'the', 'a', 'an', 'and', 'of', 'in', 'on', 'with', 'is', 'are', 'to', 'or', 'by'}
    def words(text):
        return set(re.findall(r'[a-z0-9]+', normalized(text))) - stopwords
    for task in tasks:
        native = task['native']
        eligible = [r for r in pool if source_key(r['dataset'], r['image_id']) != source_key(native['dataset'], native['original_image_path'])]
        exact_answer = [r for r in eligible if normalized(r['answer']) == normalized(native['reference'])]
        family_answer = [r for r in exact_answer if family(r['question']) == family(native['question'])]
        terms = words(native['reference'])
        ranked = sorted([r for r in eligible if terms & words(r['answer'])],
                        key=lambda r: (-len(terms & words(r['answer'])) / max(1, len(terms | words(r['answer']))), r['dataset'], r['image_id'], str(r['qid'])))
        candidates = {(r['dataset'], str(r['qid'])): r for r in exact_answer + ranked[:12]}
        for r in candidates.values():
            packet.append(dict(owner=task['order'], native=native, source=r,
                same_answer=normalized(r['answer']) == normalized(native['reference']),
                same_family=family(r['question']) == family(native['question']),
                relation_status='UNREVIEWED', admitted=False))
        coverage.append(dict(owner=task['order'], same_answer_QA=len(exact_answer),
            same_answer_images=len({(r['dataset'], r['image_id']) for r in exact_answer}),
            same_family_answer_QA=len(family_answer),
            same_family_answer_images=len({(r['dataset'], r['image_id']) for r in family_answer}),
            review_candidates=len(candidates)))
    write(out/'private/SOURCE_REVIEW_PACKET.json', packet)
    write(out/'private/ELIGIBLE_SOURCE_POOL.json', pool)
    write(out/'public/SOURCE_SCREEN.json', dict(status='REVIEW_REQUIRED', source_QA=len(sources),
        eligible_QA=len(pool), eligible_images=len({(r['dataset'], r['image_id']) for r in pool}),
        excluded_images=len(excluded), rejected=dict(rejected), coverage=coverage,
        known_case_exclusion=True, new_image_hashes=0, source_only=True,
        automatic_positive_labels=False, independent_confirmation=False))


def main():
    began = time.time()
    selfcheck()
    parent, out = Path(os.environ['AUDIT_PARENT']), Path(os.environ['AUDIT_ROOT'])
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    command = subprocess.check_output(['ps', '-p', str(os.getpid()), '-o', 'args='], text=True).strip()
    assert not any(x in command.casefold() for x in ('wangbomin', 'knowledge_editing', 'medtrace', 'scope_anchor'))
    env = read(parent/'private/LAUNCH_ENV.json')
    assert read(parent/'public/REVIEW_AUDIT.json')['status'] == 'PASS'
    if os.environ.get('AUDIT_PHASE') == 'closeout':
        closeout(parent, out, command)
        return
    pair_outputs(parent, env, out)
    screen(parent, env, out)
    write(out/'private/EXECUTION.json', dict(status='COMPLETE_CPU_SCREEN', pid=os.getpid(), command=command,
        seconds=time.time()-began, model_calls=0, new_Judge=0))
    print(json.dumps(read(out/'public/SOURCE_SCREEN.json')))


def closeout(parent, out, command):
    review = read(out/'private/RELATION_REVIEW.json')
    screen_result = read(out/'public/SOURCE_SCREEN.json')
    pairs = read(out/'public/PAIRED_RESULTS.json')
    owners = review['owners']
    assert [x['owner'] for x in owners] == list(range(1, 9))
    assert review['reviewed_pool_QA'] == screen_result['eligible_QA']
    assert review['reviewed_pool_images'] == screen_result['eligible_images']
    assert not admitted(owners), 'Passing data requires a separate frozen training protocol, not an implicit launch'
    assert any(x['status'] == 'NO_SUPPORTED_ANCHOR_IN_AUDITED_POOL' for x in owners)
    prior = read(parent/'public/RESULTS.json')['resource']
    result = dict(status='COMPLETE', decision='DATA_GATE_NOT_MET',
        paired_observations=sum(pairs['panels'][k]['total']['n'] for k in ('T1G', 'T2G', 'T1L', 'T2L', 'HELDOUT')),
        source_QA=screen_result['source_QA'], eligible_QA=screen_result['eligible_QA'],
        eligible_images=screen_result['eligible_images'], owner_coverage=owners,
        source_review='All eligible source QA inspected; same-answer and lexical candidates are not automatic scope labels',
        training_started=False, six_GPUs_authorized=True, GPUs_used=0,
        natural_route_evaluation='NOT_STARTED_DATA_GATE_NOT_MET',
        basis_conflict_review='NOT_COMPLETED_NO_ADMITTED_FULL_EIGHT_OWNER_ANCHOR_SET',
        resource=dict(new_GPU_process_hours=0, new_Judge=0, new_generations=0, new_checkpoints=0,
            cumulative_GPU_process_hours=prior['cumulative_GPU_process_hours'], cumulative_Judge=prior['cumulative_Judge']),
        engineering_failures=[dict(error='Missing legacy original_image_path field',
            correction='Use existing source_group identity; paired output files preserved', new_model_calls=0)],
        limitations=['Audited local source inventory only, not proof no valid anchors exist elsewhere',
            'No clinical qualification or independent confirmation',
            'Partial candidates and unknown relationships were not promoted or used for training',
            'No causal inference from text/token divergence'])
    write(out/'public/RESULTS.json', result)
    write(out/'public/REVIEW_AUDIT.json', dict(status='PASS', paired_binding_identity=True,
        expected_PR54_aggregate_reproduced=True, reviewed_source_rows=review['reviewed_pool_QA'],
        training_gate_passed=False, source_questions_published=False,
        historical_outputs_modified=False, checkpoints_created_or_deleted=0))
    write(out/'private/COMPLETE.json', dict(status='COMPLETE', decision=result['decision'],
        epoch=time.time(), pid=os.getpid(), command=command, new_GPU_process_hours=0, new_Judge=0))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
