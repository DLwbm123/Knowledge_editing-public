"""CPU-only closeout of the original146; never trains, judges or edits the queue."""
from collections import Counter, defaultdict
from fractions import Fraction
import csv
import json
import os
from pathlib import Path
import random
import sqlite3
import sys
import time

METHODS = ('medtrace', 'balancedit', 'belora')
NAMES = dict(medtrace='MedTRACE_AVAILABLE_H_R0', balancedit='BalancEdit_locked_adaptation',
             belora='BELoRA_V2_effect_repaired')
PREFIXES = (1, 50, 100, 146)


def read(p):
    return json.loads(Path(p).read_text())


def write_new(p, value):
    p = Path(p)
    with p.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def combine(units, counts=None):
    result = defaultdict(Fraction)
    counts = counts or {i: 1 for i in range(len(units))}
    for i, count in counts.items():
        for key, coefficient in units[i].items():
            result[key] += coefficient * count
    return {k: v for k, v in result.items() if v}


def bounds(coefficients, scores):
    """One binary variable per full shared payload identity, with rational weights."""
    known = sum((v * scores[k] for k, v in coefficients.items() if scores[k] is not None), Fraction())
    missing = [v for k, v in coefficients.items() if scores[k] is None]
    return [float(known + sum((v for v in missing if v < 0), Fraction())),
            float(known + sum((v for v in missing if v > 0), Fraction()))]


def units_for(rows):
    edits = defaultdict(list)
    for row in rows:
        edits[row['edit']].append(row['key'])
    return {e: dict(Counter(keys)) for e, keys in edits.items()}


def normalized(units):
    return {e: {k: Fraction(n, sum(counts.values())) for k, n in counts.items()}
            for e, counts in units.items()}


def metric(rows, scores):
    units = normalized(units_for(rows))
    if not rows:
        return dict(edits=0, probes=0, known_correct=0, missing_occurrences=0,
                    missing_unique_keys=0, macro=None, micro=None, macro_bounds=None, micro_bounds=None), units
    weights = [dict(Counter(r['key'] for r in rows))]
    micro = bounds({k: Fraction(v, len(rows)) for k, v in weights[0].items()}, scores)
    macro = bounds({k: v / len(units) for k, v in combine(list(units.values())).items()}, scores)
    missing = sum(scores[r['key']] is None for r in rows)
    return dict(edits=len(units), probes=len(rows), known_correct=sum(scores[r['key']] == 1 for r in rows),
        missing_occurrences=missing, missing_unique_keys=len({r['key'] for r in rows if scores[r['key']] is None}),
        macro=macro[0] if not missing else None, micro=micro[0] if not missing else None,
        macro_bounds=macro, micro_bounds=micro), units


def bootstrap(units, scores, groups, repetitions=10000):
    """Original seeded edit/source resampling, with a shared-missing envelope per draw."""
    if not units:
        return dict(edit_ci95_envelope=None, source_cluster_ci95_envelope=None, source_groups=0)
    edits = list(units)
    clusters = defaultdict(list)
    for e in edits:
        clusters[groups[e]].append(e)
    def sample(packages):
        known = []
        unknown = []
        for es in packages:
            weights = combine([units[e] for e in es])
            known.append(sum(float(v) * scores[k] for k, v in weights.items() if scores[k] is not None))
            unknown.append({k: float(v) for k, v in weights.items() if scores[k] is None})
        rng = random.Random(20260912)
        lows, highs = [], []
        for _ in range(repetitions):
            counts = Counter(rng.choices(range(len(packages)), k=len(packages)))
            denom = sum(len(packages[i]) * n for i, n in counts.items())
            value = sum(known[i] * n for i, n in counts.items())
            coefficients = defaultdict(float)
            for i, n in counts.items():
                for key, weight in unknown[i].items():
                    coefficients[key] += weight * n
            lows.append((value + sum(v for v in coefficients.values() if v < 0)) / denom)
            highs.append((value + sum(v for v in coefficients.values() if v > 0)) / denom)
        lows.sort(); highs.sort()
        return [lows[int(repetitions * .025) - 1], highs[int(repetitions * .975) - 1]]
    return dict(edit_ci95_envelope=sample([[e] for e in edits]),
        source_cluster_ci95_envelope=sample(list(clusters.values())), source_groups=len(clusters))


def paired(a, b, scores, groups):
    assert list(a) == list(b), 'Paired support must match in original edit order'
    delta = {e: combine([a[e], {k: -v for k, v in b[e].items()}]) for e in a}
    interval = bounds({k: v / len(delta) for k, v in combine(list(delta.values())).items()}, scores) if delta else None
    return dict(edits=len(delta), macro_delta_bounds=interval,
                macro_delta=interval[0] if interval and interval[0] == interval[1] else None,
                **bootstrap(delta, scores, groups))


def micro_delta(a, b, scores):
    assert [(r['edit'], r['query_id']) for r in a] == [(r['edit'], r['query_id']) for r in b]
    if not a:
        return None
    coefficients = combine([dict(Counter(r['key'] for r in a)),
                            {k: -v for k, v in Counter(r['key'] for r in b).items()}])
    return bounds({k: v / len(a) for k, v in coefficients.items()}, scores)


def trajectory(pairs, scores):
    """Exact insertion-retention ratio bounds; independent query identities per native."""
    used = set(); states = {(0, 0)}; transitions = Counter(); missing = 0
    for before, after in pairs:
        assert not used.intersection({before, after}), 'Native query identities unexpectedly overlap'
        used.update({before, after})
        keys = list(dict.fromkeys([before, after]))
        assignments = [{}]
        for k in keys:
            assignments = [{**a, k: v} for a in assignments for v in
                           ([0, 1] if scores[k] is None else [scores[k]])]
        possibilities = {(a[before], a[after]) for a in assignments}
        if len(possibilities) == 1:
            v, w = next(iter(possibilities)); transitions[f'{v}_to_{w}'] += 1
        else:
            missing += 1
        additions = {(v * w, v) for v, w in possibilities}
        states = {(num + n, den + d) for num, den in states for n, d in additions}
    ratios = [Fraction(n, d) for n, d in states if d]
    return dict(native_denominator=len(pairs), known_transitions=dict(transitions),
        missing_native_pairs=missing, insertion_correct_bounds=[min(d for n, d in states), max(d for n, d in states)],
        retention_given_insertion_correct_bounds=[float(min(ratios)), float(max(ratios))] if ratios else None,
        zero_insertion_correct_possible=any(d == 0 for n, d in states))


def load(root, complete=False):
    sys.path.insert(0, str(root / 'private/source'))
    from scripts.medtrace.stage17_prepare import digest, PROMPT, PROTOCOL
    from scripts.medtrace.stage17_campaign import role_map, query_ids
    ledger = read(root / 'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json')
    assert digest({k: v for k, v in ledger.items() if k != 'freeze_id'}) == ledger['freeze_id']
    tasks = {t['edit_id']: t for t in ledger['tasks']}
    ordered = [tasks[e] for e in ledger['main_T0']]
    assert len(ordered) == 146
    roles = role_map(ledger)
    # Current original map has no changed gold references; fail closed if this ceases to hold.
    assert all(ledger['queries'][q]['reference'] == ref for p in roles.values() for q, ref in p.items())
    for name in ('CPU_ADMISSION', 'GPU_MECHANICAL', 'BASELINE_ADMISSION', 'BASELINE_METHOD_LOCK_ADMISSION'):
        assert read(root / 'private' / (name + '.json'))['status'] == 'PASS', name
    epoch = read(root / 'private/judge_common/EPOCH_MANIFEST.json')
    assert digest({k: v for k, v in epoch.items() if k != 'config_binding'}) == epoch['config_binding']
    assert (epoch['epoch'], epoch['model'], epoch['reasoning_effort'], epoch['prompt'], epoch['protocol']) == (
        'MEDTRACE_ORIGINAL146_COMMON_ASTRA_20261003_V1', 'gpt-6-astra', 'high', PROMPT, PROTOCOL)
    assert epoch['source_commit'] == read(root / 'private/SOURCE_COMMIT.json')['commit']
    db = sqlite3.connect('file:' + str(root / 'private/judge_common/queue.sqlite') + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row; db.execute('BEGIN')
    pending = dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status').fetchall())
    support = {}
    for task in ('T0', 'T1G', 'T2G', 'T1L', 'T2L', 'T4G'):
        rows = [(t['edit_id'], q) for t in ordered for ev in t['events'] if ev['task'] == task for q in ev['all_probe_query_ids']]
        eligible = [(e, q) for e, q in rows if ledger['Base_correctness'][q] == task.endswith('L')]
        support[task] = dict(all_probes=len(rows), original_eligible_probes=len(eligible), original_eligible_edits=len({e for e, q in eligible}))
    if not complete:
        result = dict(status='METADATA_PREFLIGHT_PASS_NO_PERFORMANCE_READ', N=146, support=support,
            active_target_counts={p: len(r) for p, r in roles.items()}, changed_active_references=0,
            original_masks_preserved=True, payload_states=pending, generation_complete=(root / 'private/GENERATION_COMPLETE.json').exists(),
            consumer_counts=[dict(r) for r in db.execute('SELECT method,mode,count(*) AS count FROM consumer GROUP BY method,mode')],
            GPU_jobs=0, Judge_attempts=0)
        db.close(); return result
    assert read(root / 'private/GENERATION_COMPLETE.json')['status'] == 'GENERATED_NOT_SCORED'
    assert read(root / 'private/CONTROLLER_COMPLETE.json')['status'] == 'GENERATION_COMPLETE'
    assert read(root / 'private/judge_common/SCORER_DONE.json')['status'] == 'COMMON_SCORING_COMPLETE_WITH_MISSING'
    assert not pending.get('PENDING', 0) and not pending.get('RESERVED', 0), 'Nonterminal payloads'
    assert db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone()
    payloads = {r['key']: dict(r) for r in db.execute('SELECT * FROM payload')}
    scores = {}
    for key, row in payloads.items():
        assert row['status'] in ('FORMAT_VALID', 'MISSING')
        assert (row['correct'] in (0, 1)) if row['status'] == 'FORMAT_VALID' else row['correct'] is None
        binding = json.loads(row['binding']); record = json.loads(row['record'])
        assert digest(binding) == key
        assert binding['judge'] == {k: epoch[k] for k in ('model', 'reasoning_effort', 'protocol', 'prompt')}
        assert record == dict(opaque_query_id=digest([epoch['epoch'], key]), question=binding['question'],
                              gold_answer=binding['reference'], raw_base_answer=binding['output']['raw_answer'])
        scores[key] = row['correct']
    resource = read(root / 'RESOURCE_LEDGER.json')
    assert not any(s.get('ended_epoch') is None for s in resource['gpu_sessions']), 'GPU consumers still active'
    attempted = [key for batch in resource['Judge_batches'] for key in batch['keys']]
    assert len(attempted) == len(set(attempted)) == resource['Judge_attempts'] == len(payloads)
    from scripts.medtrace.astra_judge_bundle import validate
    for batch in resource['Judge_batches']:
        saved = read(root / 'private/judge_common/evidence' / (batch['id'] + '.json'))
        assert saved['batch']['records'] == [json.loads(payloads[k]['record']) for k in batch['keys']]
        if batch['status'] == 'FORMAT_VALID':
            ev = saved['evidence']
            assert ev['actual_model'] == 'gpt-6-astra' and ev['reasoning_effort'] == 'high'
            assert ev['input_binding'] == digest(saved['batch']) and ev['exit_code'] == 0
            assert not ev.get('errors') and not ev['tool_event_types'] and all(ev['isolation_checks'].values())
            decisions = validate(saved['batch'], saved['response'])
            assert [int(d['is_correct']) for d in decisions] == [scores[k] for k in batch['keys']]
        else:
            assert batch['status'] == 'FAILED_NO_RETRY' and all(scores[k] is None for k in batch['keys'])
    bases = read(root / 'private/legacy_stage17/BINDINGS.json'); lookup = {}; counts = Counter()
    for row in db.execute('SELECT * FROM consumer'):
        qid = row['query_id']; q = ledger['queries'][qid]; base = bases[q['opaque_Base_id']]
        if row['method'] == 'Base':
            output = dict(raw_answer=base['output']['model_answer_raw'], raw_token_ids=base['output']['raw_generated_token_ids'])
            assert digest(base) == row['output_binding']
        else:
            raw = read(row['path'])
            assert digest(raw) == row['output_binding'] and raw['binding']['input'] == q
            assert raw['Base_cache_id'] == q['opaque_Base_id']
            arm = 'NATIVE' if row['method'] == 'belora' else 'R0'
            output = raw.get('modes', raw)[arm]
        full = dict(query_id=qid, question=q['question'], reference=q['reference'], image_sha256=q['image_sha256'],
            image_path=base['image_path'], prompt_ids=base['prompt_ids'], attention_mask=base['attention_mask'],
            runtime=base['runtime'], generation=base['generation'], output=output,
            judge={k: epoch[k] for k in ('model', 'reasoning_effort', 'protocol', 'prompt')})
        assert digest(full) == row['payload_key']
        key = (row['method'], row['mode'], row['prefix'], row['edit_order'], row['folder'], qid)
        assert key not in lookup, 'Duplicate consumer slot'
        lookup[key] = row['payload_key']; counts[row['method'], row['mode']] += 1
    db.close()
    def get(method, mode, prefix, edit_order, folder, qid):
        return lookup[method, mode, prefix, edit_order, folder, qid]
    assert counts['Base', 'base'] == len(ledger['queries'])
    expected_single = sum(len(query_ids(t)) for t in ordered)
    assert counts['medtrace', 'single'] == expected_single
    expected_seq = 146 + sum(len(set(q for t in ordered[:p] for q in query_ids(t))) for p in PREFIXES)
    assert counts['medtrace', 'sequential'] == expected_seq
    baseline_counts = Counter((r['method'], r['mode']) for r in read(root / 'private/BASELINE_RAW_CONSUMERS.json'))
    assert all(counts[key] == n for key, n in baseline_counts.items())
    groups = {t['edit_id']: t['native']['source_group'] for t in ordered}
    allrows = []; insertion = {}
    for method in METHODS:
        insertion[method] = trajectory([(get(method, 'sequential', i, i, 'native', t['edit_id']),
            get(method, 'sequential', 146, 146, 'panel', t['edit_id'])) for i, t in enumerate(ordered, 1)], scores)
        for mode, prefix in [('single', 1)] + [('sequential', p) for p in PREFIXES]:
            for t in (ordered if mode == 'single' else ordered[:prefix]):
                for event in t['events']:
                    for qid in event['all_probe_query_ids']:
                        allrows.append(dict(method=method, mode=mode, prefix=prefix, edit=t['edit_id'], task=event['task'],
                            key=get(method, mode, prefix, t['order'] if mode == 'single' else prefix, 'panel', qid),
                            base_key=get('Base', 'base', 0, 0, 'panel', qid), query_id=qid,
                            original_base=ledger['Base_correctness'][qid], active=mode == 'sequential' and qid in roles[str(prefix)]))
    return ledger, allrows, scores, groups, insertion, resource, epoch, support


def report(root):
    started = time.time()
    ledger, rows, scores, groups, insertion, resources, epoch, support = load(root, complete=True)
    GPU_by_action = defaultdict(float)
    for session in resources['gpu_sessions']:
        GPU_by_action[session.get('action', 'unclassified_initialization')] += session['ended_epoch'] - session['started_epoch']
    panels = []; aggregates = {}; metric_rows = {}; basemask = {}
    keys = list(dict.fromkeys((r['mode'], r['prefix'], r['task']) for r in rows))
    base_missing = sum(scores[r['base_key']] is None for r in rows)
    for mode, prefix, task in keys:
        for mask in (('original', 'new_Base_sensitivity') if not base_missing else ('original',)):
            for method in METHODS:
                raw = [r for r in rows if (r['method'], r['mode'], r['prefix'], r['task']) == (method, mode, prefix, task)]
                eligible = [r for r in raw if not (task.endswith('L') and r['active']) and (task == 'T0' or
                    (r['original_base'] if mask == 'original' else bool(scores[r['base_key']])) == task.endswith('L'))]
                primary, units = metric(eligible, scores)
                primary.update(bootstrap(units, scores, groups))
                post, _ = metric([r for r in raw if not (task.endswith('L') and r['active'])], scores)
                if task.endswith('L'):
                    primary['c2w_micro_bounds'] = [1 - x for x in reversed(primary['micro_bounds'])] if primary['micro_bounds'] else None
                aggregates[mask, method, mode, prefix, task] = units
                metric_rows[mask, method, mode, prefix, task] = eligible
                panels.append(dict(mask=mask, method=NAMES[method], mode=mode, prefix=prefix, task=task,
                    primary_metric='PostAcc' if task == 'T0' else 'Retention' if task.endswith('L') else 'Fix',
                    primary=primary, post_accuracy=post, active_locality_exclusions=sum(r['active'] for r in raw) if task.endswith('L') else 0))
        # Count Base mask disagreement on the same original panel support, once per method-independent occurrence.
        raw = [r for r in rows if (r['method'], r['mode'], r['prefix'], r['task']) == ('medtrace', mode, prefix, task)]
        basemask[f'{mode}/{prefix}/{task}'] = dict(probes=len(raw), missing=sum(scores[r['base_key']] is None for r in raw),
            original_wrong_new_correct=sum(not r['original_base'] and scores[r['base_key']] == 1 for r in raw),
            original_correct_new_wrong=sum(r['original_base'] and scores[r['base_key']] == 0 for r in raw))
    contrasts = []
    for mask, method, mode, prefix, task in aggregates:
        if method != 'medtrace':
            continue
        for other in ('balancedit', 'belora'):
            contrasts.append(dict(mask=mask, contrast=NAMES['medtrace'] + ' minus ' + NAMES[other], mode=mode,
                prefix=prefix, task=task, micro_delta_bounds=micro_delta(metric_rows[mask, method, mode, prefix, task],
                    metric_rows[mask, other, mode, prefix, task], scores),
                **paired(aggregates[mask, method, mode, prefix, task],
                                                aggregates[mask, other, mode, prefix, task], scores, groups)))
    criteria = {}
    for other in ('balancedit', 'belora'):
        selected = [c for c in contrasts if c['mask'] == 'original' and c['mode'] == 'sequential' and c['prefix'] == 146
                    and c['contrast'] == NAMES['medtrace'] + ' minus ' + NAMES[other] and c['task'] in ('T0', 'T1G', 'T2G', 'T2L')]
        checks = {c['task']: c['macro_delta_bounds'] is not None and c['macro_delta_bounds'][0] >=
                  (.05 if c['task'] == 'T2L' else -.01) - 1e-12 for c in selected}
        criteria[other] = dict(checks=checks, worth_separate_validation=len(checks) == 4 and all(checks.values()),
                               automatic_next_experiment_authorized=False, not_a_superiority_test=True)
    result = dict(status='REPORTED_PUBLICATION_PENDING', N=146, qualified_H_edits=8, qualified_H_relations=12,
        new_H=0, H_eval=None, original_support=support, panels=panels, paired=contrasts,
        insertion_to_final={NAMES[m]: value for m, value in insertion.items()}, Base_mask_disagreement=basemask,
        new_Base_sensitivity_status='NA_MISSING_BASE_NO_ELIGIBILITY_IMPUTATION' if base_missing else 'COMPLETE',
        validation_readiness=criteria, judge=dict(epoch=epoch['epoch'], model=epoch['model'], reasoning_effort='high',
        immutable_snapshot=None, old_success_scores_in_primary=0, payloads=len(scores), format_valid=sum(v is not None for v in scores.values()),
        permanent_missing=sum(v is None for v in scores.values()), attempts=resources['Judge_attempts'], batches=len(resources['Judge_batches'])),
        statistics=dict(bootstrap_repetitions=10000, seed=20260912, unit='paired edit',
            source_sensitivity='native source-group cluster resampling, edit-weighted; not patient independence',
            missing='exact shared-key linear point bounds; per-draw envelope for descriptive bootstrap intervals',
            bootstrap_missing_envelope_is_not_a_single_joint_completion=True),
        cost=dict(GPU_hours=resources['gpu_seconds_used'] / 3600, active_GPU_leases=sum(s.get('ended_epoch') is None for s in resources['gpu_sessions']),
            GPU_resident_seconds_by_action=dict(GPU_by_action),
            summed_Judge_batch_wall_seconds=sum(b['ended_epoch'] - b['started_epoch'] for b in resources['Judge_batches']),
            original_first_clock_wall_seconds=time.time() - read(root / 'RUN_MANIFEST.json')['starting_epoch'],
            failed_attempts=resources['failed_attempts'], failed_initializations=resources['failed_initializations'],
            Judge_preflight_failures=resources.get('Judge_preflight_failures'), historical_baseline_compute_reused=True,
            baseline_historical_cost_is_not_matched_hardware_cost=True,
            exact_storage_peak_bytes=None, storage_peak_limitation='Hourly snapshots are not continuous peak instrumentation; final owned inventory and cleanup receipt are separate',
            CPU_reporting_seconds=time.time() - started),
        limitations=['Original146 exposed diagnostic/regression cohort and one order, not independent confirmation',
            '8 edits use qualified H; 138 retain original C_NO_H; not146 full C_FACT', 'H_eval unsupported; no ten-task overall',
            'Patient independence UNKNOWN; original U source shared and future metadata overlap disclosed',
            'BalancEdit locked adaptation; BELoRA effect-repaired rank16/alpha16/50steps, not paper exact',
            'No immutable Judge model snapshot; common new scoring epoch does not erase prior exposure'])
    destination = root / 'public/final_metrics'; destination.mkdir(exist_ok=False)
    write_new(destination / 'COMPARISON_RESULTS.json', result)
    with (destination / 'RESULTS.csv').open('x', newline='') as f:
        fields = ['mask', 'method', 'mode', 'prefix', 'task', 'primary_metric', 'edits', 'probes', 'known_correct',
                  'missing_occurrences', 'macro', 'micro', 'macro_bounds', 'micro_bounds']
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for panel in panels:
            writer.writerow({k: panel[k] if k in panel else panel['primary'][k] for k in fields})
    text = '# 原146完整流程比较\n\n本表使用共同新Astra/high评分，保留原Base资格掩码。数值按完整合格分母计算；方括号为共享missing精确点界，不是置信区间。独立H_eval为NA。\n\n'
    text += '|方法|任务|编辑数|probe分母|已知正确|missing出现|宏平均界|微平均界|\n|---|---|---:|---:|---:|---:|---|---|\n'
    for p in panels:
        if p['mask'] == 'original' and p['mode'] == 'sequential' and p['prefix'] == 146 and p['task'] in ('T0', 'T1G', 'T2G', 'T1L', 'T2L'):
            m = p['primary']; fmt = lambda v: 'NA' if v is None else '[' + ', '.join(f'{x*100:.2f}%' for x in v) + ']'
            text += f"|{p['method']}|{p['task']} {p['primary_metric']}|{m['edits']}|{m['probes']}|{m['known_correct']}|{m['missing_occurrences']}|{fmt(m['macro_bounds'])}|{fmt(m['micro_bounds'])}|\n"
    text += '\n全部single/prefix、配对bootstrap、来源组敏感性、Base新掩码、插入保持和消耗见同目录JSON/CSV。\n\n' + '\n'.join('- ' + x for x in result['limitations']) + '\n'
    (destination / 'COMPARISON_REPORT_ZH.md').write_text(text)
    return dict(status='REPORTED_PUBLICATION_PENDING', destination=str(destination), N=146,
                performance_not_used_for_training_or_queue_changes=True)


def selfcheck():
    from itertools import product
    scores = dict(shared=None, a=None, b=1)
    samples = [dict(shared=Fraction(1, 3), a=Fraction(-1, 2), b=Fraction(1, 5)),
               combine([dict(shared=Fraction(1)), dict(shared=Fraction(-1))])]
    for coefficients in samples:
        values = [sum(float(v) * {**scores, 'shared': x, 'a': y}[k] for k, v in coefficients.items()) for x, y in product((0, 1), repeat=2)]
        assert bounds(coefficients, scores) == [min(values), max(values)]
    rows = [dict(edit='e1', key='shared'), dict(edit='e1', key='shared'), dict(edit='e2', key='b')]
    value, units = metric(rows, scores)
    assert value['micro_bounds'] == [1/3, 1] and value['macro_bounds'] == [.5, 1]
    assert paired(units, units, scores, dict(e1='g', e2='g'))['macro_delta_bounds'] == [0, 0]
    assert metric([], scores)[0]['macro'] is None
    assert trajectory([('shared', 'shared')], scores)['retention_given_insertion_correct_bounds'] == [1, 1]
    assert trajectory([('b', 'a')], scores)['retention_given_insertion_correct_bounds'] == [0, 1]
    print(json.dumps(dict(status='PASS', checks=['shared-key cancellation', 'weighted macro/micro',
        'brute force missing extrema', 'empty support NA', 'paired support', 'shared insertion ratio'], GPU_jobs=0, Judge=0)))


if __name__ == '__main__':
    action = os.environ.get('REPORT_ACTION', 'preflight')
    if action == 'selfcheck':
        selfcheck()
    else:
        root = Path(os.environ['RUN_ROOT'])
        print(json.dumps(report(root) if action == 'report' else load(root), ensure_ascii=False))
