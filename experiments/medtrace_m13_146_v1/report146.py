"""Frozen T2L macro/support and paired FIT/T2G non-regression, without selection."""
import collections
import json
import os
from pathlib import Path
import random
import sqlite3
import time
import astra_queue as a
import reporting
import contract
from fine_queue import full_identity
q,RUN=a.q,a.RUN
q.ROOT=RUN/'private/judge_m13_146_astra_medium'
interval,summarize,preserved=reporting.interval,reporting.summarize,reporting.preserved










def protection_pass(rows):
    return (sum(v['fresh_Base_correct'] for v in rows) > 0
            and all(v['missing'] == 0 and v['new_damage'] == 0 for v in rows))


def selfcheck():
    unequal = {1: [1], 2: [0, 0, 0]}
    assert summarize(unequal)['macro'] == 50 and summarize(unequal)['micro'] == 25
    assert summarize({1: [1, None]})['macro_bounds'] == [50, 100]
    assert not preserved({1: [1], 2: [0]}, {1: [0], 2: [1]})
    assert not preserved({1: [1]}, {1: [None]})
    assert interval([0., 0.])[0] == 0 and interval([.1, .1], ['a', 'b'])[0] > 0
    assert not protection_pass([dict(fresh_Base_correct=0, missing=0, new_damage=0)])
    return dict(status='PASS', macro_not_micro=True, missing_bounds=True,
                average_does_not_hide_owner_regression=True, zero_interval_not_positive=True,
                unsupported_Base_not_success=True)


def main():
    for worker in range(4):assert (q.ROOT/'workers'/str(worker)/'SCORER_DONE.json').exists()
    checks = selfcheck()
    assert (q.ROOT/'ALL_WORKERS_COMPLETE.json').exists()
    db = sqlite3.connect('file:'+str(q.ROOT/'queue.sqlite')+'?mode=ro', uri=True)
    assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
    judge = {k: q.read(q.ROOT/'EPOCH_MANIFEST.json')[k] for k in ('model', 'reasoning_effort', 'protocol', 'prompt')}
    records, scores = {}, {}
    for path, binding, status, correct in db.execute('SELECT c.path,c.output_binding,p.status,p.correct FROM consumer c JOIN payload p ON c.payload_key=p.key'):
        data = q.read(Path(path)); assert q.digest(data) == binding
        key = (data['method'], data['mode'], data['prefix'], data['eval_owner'], data['binding']['input']['query_id'])
        assert key not in records
        records[key] = data; scores[key] = correct if status == 'FORMAT_VALID' else None
    assert len(records) == 24867
    tasks = q.read(RUN/'private/ORIGINAL146_TASKS.json')
    base_mask = q.read(RUN/'private/ORIGINAL_BASE_MASK146.json')
    def keys(method, mode, prefix, task, role):
        owner = task['order'] if mode == 'single' else 0
        if role == 'FIT':
            qids = [r['query_id'] for r in contract.fit_rows(task)]
        else:
            qids = list(dict.fromkeys(v for e in task['events'] if e['task'] == role for v in e['all_probe_query_ids']))
            if role != 'T0':
                qids = [v for v in qids if base_mask[v] == role.endswith('L')]
        result = [(method, mode, prefix, owner, v) for v in qids]
        if mode == 'bank' and role.endswith('L'):
            result = [k for k in result if not records[k]['active_target']]
        return result
    panels, per_owner, protection, damage_owners = {}, {}, {}, {}
    for mode, prefix in [('single', 1)] + [('bank', n) for n in contract.PREFIXES]:
        ts = tasks if mode == 'single' else tasks[:prefix]
        label = f'{mode}_{prefix}'
        for method in ('M1', 'M3', 'M13'):
            per_owner[method, label] = {}
            panels.setdefault(method, {})[label] = {}
            for role in ('T0', 'FIT', 'T1G', 'T2G', 'T1L', 'T2L'):
                owner_scores = {t['order']: [scores[k] for k in keys(method, mode, prefix, t, role)] for t in ts}
                per_owner[method, label][role] = owner_scores
                panels[method][label][role] = summarize(owner_scores)
            protected = {}
            for task in ts:
                # Keep the original primary mask; fresh Base is a separate protection audit.
                qids = {v for e in task['events'] if e['task'] in ('T1L', 'T2L') for v in e['all_probe_query_ids']}
                for v in qids:
                    key = (method, mode, prefix, task['order'] if mode == 'single' else 0, v)
                    baseline = ('BASE', 'base', 0, 0, v)
                    left, right = full_identity(records[baseline], judge), full_identity(records[key], judge)
                    assert {k: x for k, x in left.items() if k != 'output'} == {k: x for k, x in right.items() if k != 'output'}
                    if not records[key]['active_target']:
                        protected[task['order'], v] = (scores[baseline], scores[key])
            protection[method, label] = dict(
                observations=len(protected), fresh_Base_correct=sum(b == 1 for b, s in protected.values()),
                new_damage=sum(b == 1 and s == 0 for b, s in protected.values()),
                repairs=sum(b == 0 and s == 1 for b, s in protected.values()),
                missing=sum(b is None or s is None for b, s in protected.values()))
            damage_owners[method,label]={t['order']:dict(fresh_Base_correct=sum(b==1 for (owner,qid),(b,s) in protected.items() if owner==t['order']),new_damage=sum(b==1 and s==0 for (owner,qid),(b,s) in protected.items() if owner==t['order']),missing=sum(b is None or s is None for (owner,qid),(b,s) in protected.items() if owner==t['order'])) for t in ts}
            audit = protection[method, label]
            audit['correct_retention_rate'] = (1-audit['new_damage']/audit['fresh_Base_correct']) if audit['fresh_Base_correct'] and not audit['missing'] else None
    terminal = panels['M13']['bank_146']['T2L']
    assert terminal['observations'] == 54 and terminal['edit_units'] == 31
    a, b = per_owner['M1', 'bank_146']['T2L'], per_owner['M13', 'bank_146']['T2L']
    delta, groups = [], []
    for task in tasks:
        owner = task['order']
        assert len(a[owner]) == len(b[owner])
        if a[owner]:
            value = None if any(v is None for v in a[owner]+b[owner]) else (sum(b[owner])-sum(a[owner]))/len(a[owner])
            delta.append(value); groups.append(task['native']['source_group'])
    edit_ci, source_ci = interval(delta), interval(delta, groups)
    complete = all(v is not None for v in scores.values())
    gates = dict(complete=complete,
        T2L_better_than_M1=complete and terminal['macro'] > panels['M1']['bank_146']['T2L']['macro'],
        T2L_edit_CI_positive=edit_ci is not None and edit_ci[0] > 0,
        T2L_source_CI_positive=source_ci is not None and source_ci[0] > 0,
        T2L_historical_floor=complete and terminal['macro'] >= 52.688,
        FIT_per_owner=all(preserved(per_owner['M1', label]['FIT'], per_owner['M13', label]['FIT']) for label in ('single_1', 'bank_146')),
        T2G_per_owner=all(preserved(per_owner['M1', label]['T2G'], per_owner['M13', label]['T2G']) for label in ('single_1', 'bank_146')),
        prefix_FIT_T2G=all(panels['M13'][f'bank_{n}'][role]['missing'] == 0 and
            panels['M1'][f'bank_{n}'][role]['missing'] == 0 and
            panels['M13'][f'bank_{n}'][role]['macro'] >= panels['M1'][f'bank_{n}'][role]['macro']
            for n in (1, 50, 100) for role in ('FIT', 'T2G')),
        Base_correct_no_new_damage=protection_pass([v for (method, label), v in protection.items() if method == 'M13']),
        M3_damage_per_owner=all(reporting.damage_preserved(damage_owners['M3',label],damage_owners['M13',label]) for label in ['single_1']+[f'bank_{n}' for n in contract.PREFIXES]))
    ledger = q.read(RUN/'RESOURCE_LEDGER.json'); inherited = q.read(RUN/'private/INHERITED_COST.json')
    assert all(v.get('ended_epoch') for v in ledger['gpu_sessions'])
    cost = dict(new_GPU_process_hours=(ledger['gpu_seconds_used']-inherited['gpu_seconds_used'])/3600,
        cumulative_GPU_process_hours=ledger['gpu_seconds_used']/3600,
        new_Judge=ledger['Judge_attempts']-inherited['Judge_attempts'], cumulative_Judge=ledger['Judge_attempts'])
    assert cost['new_Judge'] <= 24867
    result = dict(status='COMPLETE', decision='COMPATIBLE146_DEVELOPMENT_SIGNAL' if reporting.success(gates) else '146_COMPATIBILITY_NOT_ESTABLISHED',
        gates=gates, panels=panels, T2L=dict(original_mask_probes=54, original_mask_edits=31,
            delta_macro_pp=None if any(v is None for v in delta) else 100*sum(delta)/len(delta),
            edit_CI_pp=edit_ci, source_CI_pp=source_ci,
            source_cluster_unit='Original native edit source group; historical grouping preserved',
            historical_reference_different_recipe=True),
        protection={method+'_'+label: v for (method, label), v in protection.items()},
        missing_consumers=sum(v is None for v in scores.values()), cost=cost,
        selfcheck=checks, independent_confirmation=False, medical_safety_conclusion=False)
    q.write(RUN/'private/M13_146_RESULTS_WITH_OWNER_AUDIT.json', dict(result,
        owner_panels={method+'_'+label: values for (method, label), values in per_owner.items()}))
    public=json.loads(json.dumps(result))
    for method in ('M1','M3','M13'):
        public['panels'][method]['bank_1']=dict(status='PRIVATE_SMALL_GROUP_AUDIT_ONLY',edit_units=1)
        public['protection'][method+'_bank_1']=dict(status='PRIVATE_SMALL_GROUP_AUDIT_ONLY')
    q.write(RUN/'public/M13_146_RESULTS.json', public)
    lines = ['# 原146编辑完整验证', '', '全部为已暴露DEV；保持原队列、主掩码、种子及真实逐次R0插入。', '',
        f"执行决策：{result['decision']}；缺失消费者：{result['missing_consumers']}。", '',
        '| 方法 | T2L编辑宏平均 | T2L正确/探针 | FIT宏平均 | T2G宏平均 |', '|---|---:|---:|---:|---:|']
    for method in ('M1', 'M3', 'M13'):
        p = panels[method]['bank_146']
        lines.append(f"| {method} | {p['T2L']['macro']} | {p['T2L']['known_correct']}/{p['T2L']['observations']} | {p['FIT']['macro']} | {p['T2G']['macro']} |")
    lines += ['', '逐编辑不退化门槛、各prefix、双聚类成对区间、fresh Base损伤/修复及成本见匿名JSON；逐题及owner审计保持私有。',
        '旧W0的52.688%仅为不同配方的历史开发诊断参考；没有独立确认或医学安全结论。']
    (RUN/'public/M13_146_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    q.write(RUN/'private/M13_146_REPORT_COMPLETE.json', dict(status='COMPLETE', decision=result['decision'], epoch=time.time()))


if __name__ == '__main__':
    main()
