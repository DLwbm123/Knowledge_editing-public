"""Complete frozen M13P panels and matched C contrasts; no new model calls."""
import collections
import json
import os
from pathlib import Path
import random
import sqlite3
import time


def read(path):
    return json.loads(Path(path).read_text())


def summarize(owners):
    rows = [xs for xs in owners.values() if xs]
    n = sum(map(len, rows))
    missing = sum(v is None for xs in rows for v in xs)
    return dict(observations=n, edits=len(rows), correct=sum(v == 1 for xs in rows for v in xs),
        missing=missing,
        macro=None if not n or missing else 100 * sum(sum(xs)/len(xs) for xs in rows)/len(rows),
        micro=None if not n or missing else 100 * sum(map(sum, rows))/n)


def interval(values, groups=None):
    if not values or any(v is None for v in values):
        return None
    packages = [[v] for v in values]
    if groups is not None:
        grouped = collections.defaultdict(list)
        for group, value in zip(groups, values):
            grouped[group].append(value)
        packages = list(grouped.values())
    rng = random.Random(13120261025)
    samples = []
    for _ in range(10000):
        draw = [v for _ in packages for v in rng.choice(packages)]
        samples.append(100 * sum(draw)/len(draw))
    samples.sort()
    return [samples[250], samples[9749]]


def selfcheck():
    assert summarize({1: [1], 2: [0, 0, 0]})['macro'] == 50
    assert summarize({1: [1], 2: [0, 0, 0]})['micro'] == 25
    assert summarize({1: [1, None]})['macro'] is None
    assert summarize({1: []})['macro'] is None
    assert interval([0., 0.]) == [0., 0.]
    return dict(status='PASS', unequal_support_not_micro=True, missing_or_empty_not_zero=True)


def main():
    root, parent, cm = [Path(os.environ[k]) for k in ('V131_ROOT', 'PARENT_RUN', 'CM_RUN')]
    assert read(root/'M13P_160/COMPLETE.json')['counts']['formal_consumers'] == 436
    assert read(root/'M13P_160/GUARD_EXIT.json')['code'] == 0
    include_memit = os.environ.get('INCLUDE_COMPLETED_MEMIT') == '1'
    wanted = ['M13P_160', 'ALPHAEDIT_MM']
    expected = dict(M13P_160=7494, ALPHAEDIT_MM=5722)
    if include_memit:
        assert read(root/'MEMIT_MM/COMPLETE.json')['counts']['formal_consumers'] == 7494
        assert read(root/'MEMIT_MM/GUARD_EXIT.json')['code'] == 0
        wanted.append('MEMIT_MM')
        expected['MEMIT_MM'] = 7494
    packet = read(parent/'private/M13_SOURCE_PACKET.json')
    tasks, mask = packet['original_tasks'], packet['base_mask']
    plan = read(root/'EVAL_PLAN.json')
    queries = {row['query']['query_id']: row['query'] for row in plan}
    scores, bindings = {}, {}
    with sqlite3.connect('file:'+str(root/'judge_v131/queue.sqlite')+'?mode=ro', uri=True) as db:
        db.execute('BEGIN')
        for method, mode, prefix, owner, qid, state, correct, binding in db.execute(
                'SELECT c.method,c.mode,c.prefix,c.owner,c.query_id,p.state,p.correct,p.binding '
                'FROM consumer c JOIN payload p ON p.key=c.payload_key '
                'WHERE c.method IN ('+','.join('?' for _ in wanted)+')', wanted):
            assert state in ('FORMAT_VALID', 'REUSED') and correct in (0, 1)
            key = (method, mode, prefix, owner, qid)
            assert key not in scores
            scores[key], bindings[key] = correct, json.loads(binding)['binding']
        attempts = dict(db.execute("SELECT key,value FROM meta WHERE key!='prompt'"))
    assert collections.Counter(k[0] for k in scores) == expected
    with sqlite3.connect('file:'+str(cm/'private/judge_cm146_astra_medium/queue.sqlite')+'?mode=ro', uri=True) as db:
        for method, mode, prefix, owner, qid, state, correct, binding in db.execute(
                'SELECT c.method,c.mode,c.prefix,c.edit_order,c.query_id,p.status,p.correct,p.binding '
                'FROM consumer c JOIN payload p ON p.key=c.payload_key WHERE c.method IN (\'C\',\'BASE\')'):
            assert state == 'FORMAT_VALID' and correct in (0, 1)
            key = (method, mode, prefix, owner, qid)
            assert key not in scores
            scores[key], bindings[key] = correct, json.loads(binding)
    assert collections.Counter(k[0] for k in scores) == dict(expected, C=7494, BASE=2093)
    for row in plan:
        if row['method'] != 'M13P_160':
            continue
        tail = (row['mode'], row['prefix'], row['owner'], row['query']['query_id'])
        left, right = bindings[('C', *tail)], bindings[('M13P_160', *tail)]
        for field in ('question', 'reference', 'image_sha256', 'prompt_ids', 'runtime', 'generation'):
            assert left[field] == right[field], ('Matched C/M13P identity', tail, field)
    methods = ('C', *wanted)
    panels, owners, protection, primary_locality = {m: {} for m in methods}, {}, {}, {}
    for mode, prefix in [('single', 1)] + [('bank', n) for n in (1, 50, 100, 146)]:
        subset = tasks if mode == 'single' else tasks[:prefix]
        active = {(t['native']['image_sha256'], t['native']['question']) for t in subset}
        label = f'{mode}_{prefix}'
        for method in methods:
            panels[method][label] = {}
            for role in ('T0', 'FIT', 'T1G', 'T2G', 'T1L', 'T2L'):
                if method == 'ALPHAEDIT_MM' and role == 'FIT':
                    panels[method][label][role] = dict(status='NOT_MEASURED_NOT_ZERO')
                    continue
                values = {}
                for task in subset:
                    qids = ([f"edit_{task['order']}_{i}" for i in range(1, 5)] if role == 'FIT' else
                        list(dict.fromkeys(q for e in task['events'] if e['task'] == role for q in e['all_probe_query_ids'])))
                    if role not in ('T0', 'FIT'):
                        qids = [q for q in qids if mask[q] == role.endswith('L')]
                    if mode == 'bank' and role.endswith('L'):
                        qids = [q for q in qids if (queries[q]['image_sha256'], queries[q]['question']) not in active]
                    values[task['order']] = [scores[method, mode, prefix,
                        task['order'] if mode == 'single' else 0, q] for q in qids]
                owners[method, label, role] = values
                panels[method][label][role] = summarize(values)
            local = [panels[method][label][role] for role in ('T1L', 'T2L')]
            n, correct = sum(v['observations'] for v in local), sum(v['correct'] for v in local)
            primary_locality[method+'_'+label] = dict(observations=n, correct=correct,
                lost=n-correct, missing=0, retention_percent=100*correct/n if n else None,
                reference='Frozen original Base-correct locality mask; not the fresh medium Base audit')
            if method in ('ALPHAEDIT_MM', 'MEMIT_MM'):
                continue  # The matched Base uses medium effort; no high/medium protection conflation.
            protected = []
            for task in subset:
                qids = {q for e in task['events'] for q in e['all_probe_query_ids']}
                for q in qids:
                    if mode == 'bank' and (queries[q]['image_sha256'], queries[q]['question']) in active:
                        continue
                    protected.append((scores['BASE', 'base', 0, 0, q],
                        scores[method, mode, prefix, task['order'] if mode == 'single' else 0, q]))
            protection[method+'_'+label] = dict(observations=len(protected),
                Base_correct=sum(a == 1 for a, b in protected),
                new_damage=sum(a == 1 and b == 0 for a, b in protected),
                repairs=sum(a == 0 and b == 1 for a, b in protected), missing=0)
    assert panels['M13P_160']['single_1']['T2G']['correct'] == 427
    assert abs(panels['M13P_160']['single_1']['T2G']['macro']-76.48401826484019) < 1e-9
    assert protection['M13P_160_single_1']['new_damage'] == 2
    assert protection['M13P_160_single_1']['Base_correct'] == 85
    assert primary_locality['C_bank_146']['lost'] == 8
    assert primary_locality['M13P_160_bank_146']['lost'] == 0
    contrasts = {}
    for label in ('single_1', 'bank_1', 'bank_50', 'bank_100', 'bank_146'):
        ts = tasks if label == 'single_1' else tasks[:int(label.split('_')[1])]
        contrasts[label] = {}
        for role in ('FIT', 'T2G', 'T2L'):
            left, right = owners['C', label, role], owners['M13P_160', label, role]
            supported = [t for t in ts if left[t['order']]]
            delta = [(sum(right[t['order']])-sum(left[t['order']]))/len(left[t['order']]) for t in supported]
            contrasts[label][role] = dict(M13P_minus_C_macro_pp=100*sum(delta)/len(delta) if delta else None,
                edit_CI_pp=interval(delta), source_CI_pp=interval(delta, [t['native']['source_group'] for t in supported]),
                M13P_better=sum(v > 0 for v in delta), same=sum(v == 0 for v in delta), M13P_worse=sum(v < 0 for v in delta))
    result = dict(status='M13P_C_ALPHA_MEMIT_COMPLETE' if include_memit else
            'M13P_ALL7494_ACCEPTED_ALPHA_ALL5722_ACCEPTED_MEMIT_PENDING',
        snapshot_time=time.time(), panels=panels, matched_C_M13P=contrasts, protection=protection,
        common_original_Base_mask_locality=primary_locality,
        missing_consumers=0, bootstrap=dict(samples=10000, seed=13120261025, source_groups_not_patients=True),
        scoring_attempts_all_methods=attempts, selfcheck=selfcheck(),
        cost=dict(M13P_new_unique_generations=287, M13P_reused_bank_consumers=4102,
            M13P_repeated_single_generations=0, M13P_guard_seconds_including_failed_attempt=read(root/'M13P_160/GUARD_EXIT.json')['total_method_process_seconds'],
            M13P_new_updates=15040, M13P_prior_failed_updates=160),
        limitations=['Original146 exposed DEV, not independent confirmation or clinical safety.',
            'M13P bank is functional composition of actual frozen R0 routing and independent expert predictions; no new end-to-end bank GPU run.',
            'C/M13P have matched scientific input, runtime and medium Judge contract; original scoring epochs retained.',
            'AlphaEdit primary multi-layer19-23 MM adaptation retains historical FP16/high effort; information and compute budgets and mechanism differ.',
            'AlphaEdit FIT not measured; absent panels not zero. T1L only one edit/two probes, T2L31 edits/54 probes.',
            'The optional MEMIT report requires complete generation, normal guard exit and every accepted score; it never consumes partial results for selection.',
            'BalancEdit and GRACE retain their separately completed historical reports; this file does not relabel those configurations as matched runs.'])
    prefix = 'M13P_C_ALPHA_MEMIT' if include_memit else 'M13P_COMPLETE'
    if include_memit:
        result['cost']['MEMIT_guard_seconds'] = read(root/'MEMIT_MM/GUARD_EXIT.json')['process_seconds']
    private = root/(prefix+'_RESULTS.private.json')
    private.write_text(json.dumps(result, indent=2)+'\n')
    public = json.loads(json.dumps(result))
    for method in methods:
        public['panels'][method]['bank_1'] = dict(status='PRIVATE_SMALL_GROUP_AUDIT_ONLY')
        public['protection'].pop(method+'_bank_1', None)
        public['common_original_Base_mask_locality'].pop(method+'_bank_1', None)
    public['matched_C_M13P']['bank_1'] = dict(status='PRIVATE_SMALL_GROUP_AUDIT_ONLY')
    (root/(prefix+'_RESULTS.json')).write_text(json.dumps(public, indent=2)+'\n')
    receipt = 'M13P_C_ALPHA_MEMIT_REPORT_COMPLETE.json' if include_memit else 'M13P_REPORT_COMPLETE.json'
    (root/receipt).write_text(json.dumps(dict(status=result['status'], time=time.time()))+'\n')
    print(json.dumps(dict(panels={m: panels[m]['bank_146'] for m in methods},
        protection={m: protection[m+'_bank_146'] for m in ('C', 'M13P_160')},
        matched_C_M13P=contrasts['bank_146'], cost=result['cost']), indent=2))


if __name__ == '__main__':
    main()
