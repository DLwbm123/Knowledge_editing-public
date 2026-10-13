"""All registered contrasts, including the prior E endpoint; no model selection."""
import json
import os
import sqlite3
import time
from collections import Counter
from pathlib import Path
import full_queue as queue
import scope_report as prior
import probe_report as stats
ARMS = ('REFINE',)

q, RUN = queue.q, queue.RUN
q.ROOT = RUN / 'private/judge_full_astra_medium'
ROLES = prior.ROLES
PAIRS = (('C', 'REFINE'), ('M1', 'REFINE'), ('M3', 'REFINE'), ('E', 'REFINE'), ('M13', 'REFINE'))


def decision(complete, source, generalization, less_damage, no_owner_worse, positive_intervals, natural):
    if not complete:
        return 'INCOMPLETE_SEMANTIC_EVIDENCE'
    if not source or not generalization:
        return 'EDIT_OR_GENERALIZATION_NOT_PRESERVED'
    if not less_damage or not no_owner_worse or not positive_intervals or not natural:
        return 'NO_JOINT_GAIN_OVER_E'
    return 'STRONGER_DEVELOPMENT_SIGNAL_NOT_INDEPENDENT_CONFIRMATION'


def compatibility_decision(complete, gates):
    return "M1_GENERALIZATION_M3_PROTECTION_COMPATIBLE_DEV" if complete and all(gates.values()) else "JOINT_COMPATIBILITY_NOT_ESTABLISHED"


def selfcheck():
    assert compatibility_decision(True, {"source": True, "damage": True}) == "M1_GENERALIZATION_M3_PROTECTION_COMPATIBLE_DEV"
    assert compatibility_decision(False, {"source": True}) == "JOINT_COMPATIBILITY_NOT_ESTABLISHED"
    assert compatibility_decision(True, {"source": True, "damage": False}) == "JOINT_COMPATIBILITY_NOT_ESTABLISHED"
    assert decision(True, True, True, True, True, True, True) == 'STRONGER_DEVELOPMENT_SIGNAL_NOT_INDEPENDENT_CONFIRMATION'
    for index in range(7):
        gates = [True] * 7; gates[index] = False
        assert decision(*gates) != 'STRONGER_DEVELOPMENT_SIGNAL_NOT_INDEPENDENT_CONFIRMATION'
    assert ('M1', 'REFINE') in PAIRS and ('M3', 'REFINE') in PAIRS and ('E', 'REFINE') in PAIRS
    return dict(status='PASS', missing_and_each_failed_gate_block_success=True, M3_increment_and_E_reference_reported=True)


def main():
    checks = selfcheck(); env = q.read(RUN / 'private/LAUNCH_ENV.json')
    assert (q.ROOT / 'ALL_WORKERS_COMPLETE.json').exists()
    judge = {k: q.read(q.ROOT / 'EPOCH_MANIFEST.json')[k] for k in ('model', 'reasoning_effort', 'protocol', 'prompt')}
    db = sqlite3.connect('file:' + str(q.ROOT / 'queue.sqlite') + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
    scores, cache = {}, {}
    def read(path):
        path = str(path)
        if path not in cache:
            cache[path] = q.read(path)
        return cache[path]
    for key, name in (('DAMAGE_PARENT', 'DAMAGE_SEMANTIC.json'), ('OPTIMIZER160_PARENT', 'OPTIMIZER160_SEMANTIC.json'),
                      ('SCOPE_PARENT', 'SCOPE_SEMANTIC.json'), ('ENDPOINT_PARENT', 'ENDPOINT_SEMANTIC.json'),
                      ('MODULE_PARENT', 'MODULE_SEMANTIC.json'), ('M13_PARENT', 'MODULE_SEMANTIC.json')):
        for row in q.read(Path(env[key]) / 'private' / name):
            scores[row['path']] = row['correct']
    for row in q.read(Path(env['GENERATION_PARENT']) / 'private/EDIT_BASELINE_QUALIFICATION.json')['rows']:
        if row['arm'] == 'BASE':
            scores[row['path']] = row['correct']
    for row in q.read(RUN / 'private/HELD_BASE.json').values():
        scores[row['path']] = row['correct']
    current = []
    for row in db.execute('SELECT c.path,c.output_binding,p.status,p.correct FROM consumer c JOIN payload p ON c.payload_key=p.key'):
        data = read(row['path']); assert q.digest(data) == row['output_binding']
        score = row['correct'] if row['status'] == 'FORMAT_VALID' else None
        scores[row['path']] = score
        current.append(dict(path=row['path'], d=data, score=score))
    assert len(current) == queue.CONSUMERS
    groups = {arm: [r for r in current if r['d']['arm'] == arm] for arm in ARMS + tuple('NATURAL_' + a for a in ARMS)}
    for key, arms in (('MODULE_PARENT', ('C', 'M1', 'M3', 'NATURAL_C', 'NATURAL_M1', 'NATURAL_M3')),
                      ('ENDPOINT_PARENT', ('E', 'NATURAL_E')), ('M13_PARENT', ('M13', 'NATURAL_M13'))):
        for arm in arms:
            groups[arm] = [dict(path=str(p), d=read(p), score=scores[str(p)]) for p in sorted((Path(env[key]) / 'private/outputs' / arm).glob('*/*.json'))]
    forced = ('C', 'M1', 'M3', 'M13', 'E', 'REFINE')
    assert all(len(groups[a]) == 939 and len(groups['NATURAL_' + a]) == 219 for a in forced)
    def base(row):
        path = row['d']['baseline_path']; assert path in scores
        assert prior.input_identity(row['d'], judge) == prior.input_identity(read(path), judge)
        return scores[path]
    def select(rows, role, owners=None):
        result = []
        for row in rows:
            data = row['d']; match = data['role'] == role
            if role == 'ORIGINAL_PRIMARY':
                match = data['role'] == 'HELDOUT' and data.get('original_primary', False)
            if role == 'FP32_QUALIFIED':
                match = data['role'] == 'HELDOUT' and base(row) == 1
            if role.startswith('T1L_'):
                match = data['role'] == 'T1L' and data.get('same_reference', False) == (role == 'T1L_SAME_REFERENCE')
            if match and (owners is None or data['expert_order'] in owners):
                result.append(row)
        return result
    def summary(rows):
        return prior.helpers.summary([(base(r), r['score'], r['d']['R0']['raw_answer'] == read(r['d']['baseline_path'])['R0']['raw_answer']) for r in rows])
    panels = {a: {role: summary(select(rows, role)) for role in ROLES} for a, rows in groups.items()}
    owners = {a: [dict(owner=i, panels={role: summary(select(groups[a], role, {i})) for role in ROLES}) for i in range(1, 9)] for a in forced}
    def mapped(arm, role):
        return {(r['d']['expert_order'], r['d']['binding']['input']['query_id']): r for r in select(groups[arm], role)}
    def pair(left, right, role):
        a, b = mapped(left, role), mapped(right, role); assert a.keys() == b.keys()
        for key in a:
            assert prior.input_identity(a[key]['d'], judge) == prior.input_identity(b[key]['d'], judge)
        return dict(left_arm=left, right_arm=right, **prior.paired([(a[k]['score'], b[k]['score']) for k in a]))
    comparisons = {a + '_' + b: {role: pair(a, b, role) for role in ROLES} for a, b in PAIRS}
    complete = all(r['score'] is not None and base(r) is not None for rows in groups.values() for r in rows)
    intervals = {}
    if complete:
        for left, right in PAIRS:
            a, b = mapped(left, 'ORIGINAL_PRIMARY'), mapped(right, 'ORIGINAL_PRIMARY')
            rows = [dict(owner=k[0], group=a[k]['d']['binding']['input']['source_group'], delta=b[k]['score'] - a[k]['score']) for k in a]
            assert len(rows) == 504
            intervals[left + '_' + right] = dict(mean=stats.mean([r['delta'] for r in rows]),
                expert_CI=stats.interval([stats.mean([r['delta'] for r in rows if r['owner'] == i]) for i in range(1, 9)]),
                source_CI=stats.interval([stats.mean([r['delta'] for r in rows if r['group'] == group]) for group in sorted({r['group'] for r in rows})]))
    def preserved(roles):
        return all(owners['REFINE'][i]['panels'][role]['candidate_correct'] >= owners['E'][i]['panels'][role]['candidate_correct'] for i in range(8) for role in roles)
    natural = all(panels['NATURAL_REFINE'][role]['candidate_correct'] >= panels['NATURAL_E'][role]['candidate_correct'] for role in ('NATIVE', 'T1G', 'T2G'))
    natural = natural and panels['NATURAL_REFINE']['FP32_QUALIFIED']['new_damage'] <= panels['NATURAL_E']['FP32_QUALIFIED']['new_damage']
    gates = dict(complete=complete, source=preserved(('NATIVE', 'FIT', 'GFIT')), generalization=preserved(('T1G', 'T2G')),
        less_damage=panels['REFINE']['FP32_QUALIFIED']['new_damage'] < panels['E']['FP32_QUALIFIED']['new_damage'],
        no_owner_worse=all(owners['REFINE'][i]['panels']['FP32_QUALIFIED']['new_damage'] <= owners['E'][i]['panels']['FP32_QUALIFIED']['new_damage'] for i in range(8)),
        positive_intervals=complete and intervals['E_REFINE']['expert_CI'][0] > 0 and intervals['E_REFINE']['source_CI'][0] > 0,
        natural=natural)
    mechanism = []
    for arm in ARMS:
        for owner in range(1, 9):
            data = q.read(RUN / 'private/results' / arm / f'{owner}.json'); assert len(data['curve']) == 160
            reference = q.read(RUN / 'private/results/C' / f'{owner}.json')['curve']
            ratios = [v['operator_step_norm'] / old['operator_step_norm'] for v, old in zip(data['curve'], reference) if old['operator_step_norm'] > 0]
            mechanism.append(dict(arm=arm, owner=owner, steps=160, save_load_exact=data['save_load_exact'],
                final_teacher_all_argmax_correct=sum(v['all_argmax_correct'] for v in data['values']),
                finite_modes=dict(Counter(v.get('finite', {}).get('status', 'REFERENCE') for v in data['curve'])),
                projection_modes=dict(Counter((v.get('projection') or {}).get('status', 'NONE') for v in data['curve'])),
                mean_operator_step_ratio=sum(ratios) / len(ratios), minimum_operator_step_ratio=min(ratios),
                maximum_operator_step_ratio=max(ratios), operator_Fisher=data.get('mechanism'),
                scalar_optimizer_state=data.get('scalar_optimizer_state')))
    ledger = q.read(RUN / 'RESOURCE_LEDGER.json'); inherited = q.read(RUN / 'private/INHERITED_COST.json')
    recovery = RUN / 'private/import_recovery1/ADMISSION.json'
    failed_leases = q.read(recovery)['extra_failed_GPU_leases'] if recovery.exists() else 0
    assert len(ledger['gpu_sessions']) == 11 + failed_leases and all(row.get('ended_epoch') for row in ledger['gpu_sessions'])
    resource = dict(new_GPU_process_hours=(ledger['gpu_seconds_used'] - inherited['gpu_seconds_used']) / 3600,
        cumulative_GPU_process_hours=ledger['gpu_seconds_used'] / 3600,
        new_Judge=ledger['Judge_attempts'] - inherited['Judge_attempts'], cumulative_Judge=ledger['Judge_attempts'])
    resource['zero_work_failed_GPU_leases'] = failed_leases
    resource['GPU_time_limit_enabled'] = q.read(RUN / 'RUN_MANIFEST.json').get('GPU_time_limit_enabled', True)
    assert resource['new_Judge'] <= queue.MAX_JUDGE
    assert not resource['GPU_time_limit_enabled']
    compatibility = dict(M1_source_and_generalization=all(owners['REFINE'][i]['panels'][role]['candidate_correct'] >= owners['M1'][i]['panels'][role]['candidate_correct'] for i in range(8) for role in ('NATIVE','FIT','GFIT','T1G','T2G')),
        M3_damage=all(owners['REFINE'][i]['panels']['FP32_QUALIFIED']['new_damage'] <= owners['M3'][i]['panels']['FP32_QUALIFIED']['new_damage'] for i in range(8)),
        natural_M1_preserved=all(panels['NATURAL_REFINE'][role]['candidate_correct'] >= panels['NATURAL_M1'][role]['candidate_correct'] for role in ('NATIVE','T1G','T2G')),
        natural_M3_damage=panels['NATURAL_REFINE']['FP32_QUALIFIED']['new_damage'] <= panels['NATURAL_M3']['FP32_QUALIFIED']['new_damage'])
    result = dict(status='COMPLETE', decision=compatibility_decision(complete,compatibility), E_joint_decision=decision(**gates), primary='M1 generalization plus M3 protection', gates=gates, compatibility=compatibility,
        panels=panels, per_owner=owners, paired=comparisons, original63_intervals=intervals, mechanism=mechanism,
        selfcheck=checks, independent_confirmation=False, reference_C_M1_M3_M13_E_reused=True,
        version_panels={a: {v: {role: summary(select(groups[a], role, ids)) for role in ROLES} for v, ids in
            [('ORIGINAL_SEVEN', set(range(1, 8))), ('REVISED_OWNER8', {8})]} for a in forced},
        queue=q.read(q.ROOT / 'READY.json'), payload_status=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status')),
        missing_consumers=sum(r['score'] is None for r in current), resource=resource,
        generation=q.read(RUN / 'private/MODULE_GENERATION_COMPLETE.json'), reference=q.read(RUN / 'private/REFERENCE_EXACT.json'),
        fisher=q.read(RUN / 'public/FISHER_ADMISSION.json'))
    q.write(RUN / 'private/MODULE_SEMANTIC.json', [dict(path=r['path'], correct=r['score']) for r in current])
    q.write(RUN / 'private/FULL_RESULTS_WITH_OWNER_AUDIT.json', result)
    # Clinical per-owner panels remain private; publish aggregate comparisons only.
    public={k:v for k,v in result.items() if k not in ('per_owner','mechanism','version_panels','generation','reference')}
    public['training_counts']={k:sum(row[k] for row in result['generation']['counts']) for k in result['generation']['counts'][0]}
    q.write(RUN / 'public/FULL_RESULTS.json', public)
    lines = ['# 修正机制完整160步性能比较', '', f"执行状态：{result['decision']}。主条件为同时保留M1泛化和M3保护，并完整比较M13/E；本次全部属于已暴露开发数据。", '',
        '| 方法 | Native | FIT | GFIT | T1G | T2G | 原 63 正确 | Base 正确题新损伤 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in forced:
        values = [str(panels[arm][role]['candidate_correct']) for role in ('NATIVE', 'FIT', 'GFIT', 'T1G', 'T2G', 'ORIGINAL_PRIMARY')]
        lines.append('| ' + ' | '.join([arm] + values + [str(panels[arm]['FP32_QUALIFIED']['new_damage'])]) + ' |')
    lines += ['', f"本次 GPU 用量 {resource['new_GPU_process_hours']:.4f} 小时；新增 Judge 尝试 {resource['new_Judge']}，缺失消费者 {result['missing_consumers']}。", '',
        '匿名自然 R0、成对变化及双聚类置信区间见 FULL_RESULTS.json；逐owner审计仅在私有目录。', '',
        '未增加新种子或扩大到 146 个编辑；观察到开发集收益也不能视为独立确认。']
    (RUN / 'public/FULL_REPORT_ZH.md').write_text('\n'.join(lines) + '\n')
    q.write(RUN / 'private/FULL_REPORT_COMPLETE.json', dict(decision=result['decision'], epoch=time.time()))
    q.write(RUN / 'public/PROGRESS.json', dict(status='REPORTED_PUBLICATION_PENDING', decision=result['decision'],
        execution_complete=True, publication_complete=False, independent_confirmation=False))
    print(json.dumps(dict(decision=result['decision'], gates=gates, resource=resource)))


if __name__ == '__main__':
    main()
