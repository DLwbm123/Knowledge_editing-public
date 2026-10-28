"""Stratify completed BalancEdit comparisons and audit paired Base-correct damage."""
import collections
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/medical_proposition_v133'))
from med133_report import summary, interval


def read(p):
    return json.loads(Path(p).read_text())


def main():
    br, mr, old, base = (Path(os.environ[k]) for k in ('COMPARATOR_SCORE_ROOT', 'MED150_SCORE_ROOT', 'MED146_SCORE_ROOT', 'MED150_BASE_SCORE_ROOT'))
    assert read(br / 'COMPLETE.json')['generation_and_scoring_complete'] and read(br / 'GUARD_EXIT.json')['code'] == 0
    bcons = read(br / 'PACKET_PRIVATE.json')['consumers']
    bdec = read(br / 'DECISIONS.json')['decisions']
    mcons = read(mr / 'PACKET_PRIVATE.json')['consumers']
    mdec = read(mr / 'DECISIONS.json')['decisions']
    oldcons = read(old / 'CONSUMERS_B_FROZEN.json')['consumers']
    olddec = read(old / 'MERGED_DECISIONS.json')['decisions']
    meta = read(old / 'PACKET_METADATA_B.json')
    masks = read(base / 'BASE_MASK_FROZEN.json')['consumers']
    bm = {(x['mode'], x['prefix'], x['owner'], x['query_id']): x for x in bcons}
    mm = {(x['mode'], x['prefix'], x['owner'], x['query_id']): x for x in mcons}
    prior = {x['query_id']: x for x in oldcons if x['arm'] == 'MED' and x['mode'] == 'bank' and x['prefix'] == 146}
    prior_base = {x['query_id']: x for x in oldcons if x['arm'] == 'BASE' and x['mode'] == 'bank' and x['prefix'] == 146}
    assert len(bm) == len(mm) == 2207 and len(prior) == len(prior_base) == 2093
    candidates = read(os.environ['MED150_CANDIDATES'])['items']
    active = {(x['task']['native']['image_sha256'], x['task']['native']['question']) for x in candidates}
    newly_active = {q for q, v in prior.items() if (v['image_sha256'], v['question']) in active}

    def roleids(task, role):
        if role == 'FIT':
            return [f"edit_{task['order']}_{i}" for i in range(1, 5)]
        ids = list(dict.fromkeys(q for e in task['events'] if e['task'] == role for q in e['all_probe_query_ids']))
        return ids if role == 'T0' else [q for q in ids if meta['base_mask'][q] == role.endswith('L')]

    panels = {}
    for field in ('core_correct', 'full_supported'):
        panels[field] = {}
        for role in ('T0', 'FIT', 'T1G', 'T2G', 'T1L', 'T2L'):
            per = {'old146': ({}, {}, {}), 'new4': ({}, {}, {})}
            for task in meta['tasks']:
                owner = task['order']
                ids = [q for q in roleids(task, role) if not (role.endswith('L') and (prior[q]['active_target'] or q in newly_active))]
                med, bal, groups = per['old146']
                med[owner] = [mdec.get(mm[('bank',150,0,q)]['key'], {}).get(field) for q in ids]
                bal[owner] = [bdec.get(bm[('bank',150,0,q)]['key'], {}).get(field) for q in ids]
                groups[owner] = task['native'].get('source_group', str(owner))
            for owner in range(147, 151):
                ids = [x['query_id'] for x in masks if x['owner'] == owner and x['role'] == role and (role in ('T0','FIT') or x['eligible'] is True)]
                if role == 'T1G': ids = []
                med, bal, groups = per['new4']
                med[owner] = [mdec.get(mm[('bank',150,0,q)]['key'], {}).get(field) for q in ids]
                bal[owner] = [bdec.get(bm[('bank',150,0,q)]['key'], {}).get(field) for q in ids]
                groups[owner] = str(owner)
            panels[field][role] = {}
            for cohort, (med, bal, groups) in per.items():
                diffs, cluster = [], []
                for owner, values in med.items():
                    if values:
                        other = bal[owner]
                        assert len(values) == len(other)
                        diffs.append(None if any(v is None for v in values + other) else statistics.mean(values)-statistics.mean(other))
                        cluster.append(groups[owner])
                panels[field][role][cohort] = dict(MED=summary(med), BALANCEDIT=summary(bal),
                    delta_pp=100*statistics.mean(diffs) if diffs and all(v is not None for v in diffs) else None,
                    edit_ci95_pp=interval(diffs), source_ci95_pp=interval(diffs, cluster))
    protection = {}
    for field in ('core_correct', 'full_supported'):
        eligible = set()
        for task in meta['tasks']:
            for event in task['events']:
                eligible.update(q for q in event['all_probe_query_ids'] if not prior[q]['active_target'] and q not in newly_active)
        count = collections.Counter()
        for q in eligible:
            baseline = olddec.get(prior_base[q]['payload_key'], {}).get(field)
            med = mdec.get(mm[('bank',150,0,q)]['key'], {}).get(field)
            bal = bdec.get(bm[('bank',150,0,q)]['key'], {}).get(field)
            if None in (baseline, med, bal): count['unknown'] += 1; continue
            count['known'] += 1
            if baseline:
                count['base_correct'] += 1
                count['MED_damage'] += not med
                count['BALANCEDIT_damage'] += not bal
                if med:
                    count['base_and_MED_correct'] += 1
                    count['BALANCEDIT_new_damage_vs_MED'] += not bal
                if bal:
                    count['base_and_BALANCEDIT_correct'] += 1
                    count['MED_new_damage_vs_BALANCEDIT'] += not med
        protection[field] = {k: count[k] for k in ('known','unknown','base_correct','MED_damage','BALANCEDIT_damage','base_and_MED_correct','BALANCEDIT_new_damage_vs_MED','base_and_BALANCEDIT_correct','MED_new_damage_vs_BALANCEDIT')}
    retention = {}
    for field in ('core_correct','full_supported'):
        result = collections.Counter()
        for owner in range(147,151):
            q = next(x['query_id'] for x in masks if x['owner'] == owner and x['role'] == 'T0')
            a = bdec.get(bm[('insertion',owner,owner,q)]['key'],{}).get(field)
            b = bdec.get(bm[('bank',150,0,q)]['key'],{}).get(field)
            if a is None or b is None: result['unknown'] += 1; continue
            result['known'] += 1; result['insert_correct'] += a; result['final_correct'] += b
            result['forgotten'] += a and not b; result['repaired'] += not a and b
        retention[field] = {k:result[k] for k in ('known','unknown','insert_correct','final_correct','forgotten','repaired')}
    target = Path(os.environ['REPORT_OUTPUT']) / 'STRATIFIED_AUDIT.json'
    assert not target.exists()
    target.write_text(json.dumps(dict(panels=panels,protection=protection,new_insertion_retention=retention,
        protection_scope='deduplicated original146 nonnative queries, active targets excluded'),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(T2G={f:panels[f]['T2G'] for f in panels},protection=protection,retention=retention)))


if __name__ == '__main__':
    main()
