"""Aggregate the fixed panel without publishing questions, answers or identities."""
import json
import os
from pathlib import Path
import statistics
import time

ROOT = Path(os.environ['FIXED_SCORE_ROOT'])
WORKTREE = Path(os.environ['FIXED_WORKTREE'])
ARMS = ('BASE', 'MED', 'C21', 'CDEEP')
FIELDS = ('core_correct', 'full_supported', 'joint_supported')
ROLES = ('T0', 'FIT', 'T1G', 'T2G', 'T1L', 'T2L')


def aggregate(rows):
    total = len(rows)
    missing = sum(v is None for _, v in rows)
    by = {}
    for owner, value in rows:
        by.setdefault(owner, []).append(value)
    return dict(events=total, owners=len(by), correct=sum(v is True for _, v in rows), unknown=missing,
                micro_percent=100 * sum(v is True for _, v in rows) / total if total and not missing else None,
                macro_percent=100 * statistics.mean(statistics.mean(v) for v in by.values()) if total and not missing else None)


def contrast(pairs):
    support = [(o, a, b) for o, a, b in pairs if a is not None and b is not None]
    by = {}
    for owner, a, b in support:
        by.setdefault(owner, []).append(int(b) - int(a))
    return dict(events=len(pairs), complete=len(support), unknown=len(pairs)-len(support),
                gains=sum(not a and b for _, a, b in support) if support else None,
                losses=sum(a and not b for _, a, b in support) if support else None,
                owners_with_gain=sum(sum(v) > 0 for v in by.values()) if support else None,
                macro_delta_pp=100 * statistics.mean(statistics.mean(v) for v in by.values())
                if len(support) == len(pairs) and support else None)


def main():
    complete = json.loads((ROOT / 'COMPLETE.json').read_text())
    if json.loads((ROOT / 'WORKER_EXIT.json').read_text())['code'] != 0 or json.loads((ROOT / 'GUARD_EXIT.json').read_text())['code'] != 0:
        raise ValueError('Scoring worker or guard failed')
    packet = json.loads((ROOT / 'PACKET_PRIVATE.json').read_text())
    scored = json.loads((ROOT / 'DECISIONS.json').read_text())['decisions']
    private = WORKTREE / '.private/med_fixed_depth_v1_20261030'
    old = json.loads((private / 'BASE_M0_SCORE_BINDINGS.json').read_text())['rows']
    roles = json.loads((private / 'EVALUATION_ROLES.json').read_text())
    if len(old) != 508 or len(packet['consumers']) != 1016 or len(roles) != 16:
        raise ValueError('Fixed panel coverage mismatch')
    values = {}
    for row in old:
        for arm in ('BASE', 'MED'):
            score = {k: row[arm][k] for k in ('core_correct', 'full_supported')}
            score['joint_supported'] = score['core_correct'] and score['full_supported']
            values[(row['owner'], row['query_id'], arm)] = score
    for row in packet['consumers']:
        key = (row['owner'], row['query_id'], row['arm'])
        if key in values:
            raise ValueError('Duplicate candidate result')
        value = scored.get(row['key'])
        score = {k: value.get(k) if value else None for k in ('core_correct', 'full_supported')}
        score['joint_supported'] = None if None in score.values() else score['core_correct'] and score['full_supported']
        values[key] = score
    if len(values) != 508 * 4:
        raise ValueError('Four-arm event coverage mismatch')
    candidate_meta = {(r['owner'], r['query_id'], r['arm']): r for r in packet['consumers']}
    generation = {}
    for arm in ('C21', 'CDEEP'):
        rows = [r for r in packet['consumers'] if r['arm'] == arm]
        generation[arm] = dict(predictions=len(rows), empty=sum(r['empty'] for r in rows),
                               at_cap=sum(r['at_cap'] for r in rows), eos=sum(r['eos'] for r in rows),
                               reused_off=sum(bool(r['prediction_reused_from']) for r in rows))
    panels = {}
    for role in ROLES:
        events = [(int(owner), qid) for owner, groups in roles.items() for qid in groups[role]]
        for field in FIELDS:
            key = role + '_' + field
            panels[key] = {'arms': {arm: aggregate([(o, values[(o, q, arm)][field]) for o, q in events]) for arm in ARMS},
                           'deep_vs_c21': contrast([(o, values[(o, q, 'C21')][field], values[(o, q, 'CDEEP')][field]) for o, q in events]),
                           'deep_vs_med': contrast([(o, values[(o, q, 'MED')][field], values[(o, q, 'CDEEP')][field]) for o, q in events])}
    all_events = [(row['owner'], row['query_id']) for row in old]
    native = {(int(owner), q) for owner, groups in roles.items() for role in ('T0', 'FIT') for q in groups[role]}
    protection = {}
    for field in FIELDS:
        eligible = [(o, q) for o, q in all_events if (o, q) not in native and values[(o, q, 'BASE')][field] is True]
        common = [(o, q) for o, q in eligible if values[(o, q, 'MED')][field] is True]
        protection[field] = {}
        for arm in ('MED', 'C21', 'CDEEP'):
            selected = [(o, values[(o, q, 'BASE')][field], values[(o, q, arm)][field]) for o, q in eligible]
            protection[field][arm] = contrast(selected)
        protection[field]['deep_vs_med_common'] = contrast([(o, values[(o, q, 'MED')][field], values[(o, q, 'CDEEP')][field]) for o, q in common])
    t2g = panels['T2G_core_correct']
    d1, d0 = t2g['deep_vs_c21']['macro_delta_pp'], t2g['deep_vs_med']['macro_delta_pp']
    safety = [panels[role + '_core_correct']['deep_vs_med'] for role in ('T0', 'FIT', 'T2L')]
    full = contrast([(o, values[(o, q, 'MED')]['full_supported'], values[(o, q, 'CDEEP')]['full_supported']) for o, q in all_events])
    t2g_events = [(int(owner), q) for owner, groups in roles.items() for q in groups['T2G']]
    valid_gains = {}
    for reference in ('MED', 'C21'):
        valid_gains[reference] = sum(values[(o, q, reference)]['core_correct'] is False
            and values[(o, q, 'CDEEP')]['core_correct'] is True
            and not candidate_meta[(o, q, 'CDEEP')]['at_cap']
            and not candidate_meta[(o, q, 'CDEEP')]['empty'] for o, q in t2g_events)
    complete_results = complete['status'] == 'COMPLETE' and all(x['unknown'] == 0 for x in safety) and full['unknown'] == 0
    eligible = bool(complete_results and d1 is not None and d0 is not None and d1 >= 2 and d0 >= 2
                    and all(x['events'] and x['losses'] == 0 for x in safety) and full['losses'] == 0
                    and all(valid_gains.values())
                    and protection['core_correct']['deep_vs_med_common']['events']
                    and protection['core_correct']['deep_vs_med_common']['losses'] == 0)
    result = dict(status='DEV_SCREEN_PASS' if eligible else 'DEV_SCREEN_NOT_PASSED' if complete_results else 'INCOMPLETE',
                  owners=16, single_events=508, scoring=complete, generation=generation, panels=panels, protection=protection,
                  full_all_events_deep_vs_med=full, frozen_t2g_threshold_pp=2,
                  nonempty_untruncated_t2g_gains=valid_gains,
                  limitations=['exposed development cohort; not an independent clinical validation',
                               'C21 and CDEEP omit original MED protection curvature',
                               'empty support is NA; comparison-method score decisions remain unchanged'])
    output = WORKTREE / 'reports/med_fixed_depth_v1_20261030'
    (output / 'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# MED 固定层位开发实验', '', f"状态：{result['status']}。16个编辑，508个single事件。", '',
             '| 面板 | BASE CORE | MED CORE | C21 CORE | CDEEP CORE | CDEEP−MED |',
             '|---|---:|---:|---:|---:|---:|']
    for role in ROLES:
        panel = panels[role + '_core_correct']
        def fmt(v): return 'NA' if v is None else f'{v:.4f}%'
        lines.append('| ' + role + ' | ' + ' | '.join(fmt(panel['arms'][a]['macro_percent']) for a in ARMS)
                     + ' | ' + fmt(panel['deep_vs_med']['macro_delta_pp']) + ' |')
    lines += ['', '详细 CORE/FULL、分母、未知、配对增损及保护统计见 RESULTS.json。',
              '本研究使用已曝光开发数据，不证明患者独立或临床安全；两条新臂未使用原MED保护曲率。',
              '失败及工程恢复成本应与独立执行审计一并披露。']
    (output / 'REPORT_ZH.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=result['status'], owners=result['owners'], events=result['single_events'])))


if __name__ == '__main__':
    main()
