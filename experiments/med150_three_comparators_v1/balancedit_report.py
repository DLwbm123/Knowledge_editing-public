"""Compare completed BALANCEDIT150 and MED150 using the original frozen masks."""
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
    scored, med, old, base = (Path(os.environ[k]) for k in ('COMPARATOR_SCORE_ROOT', 'MED150_SCORE_ROOT', 'MED146_SCORE_ROOT', 'MED150_BASE_SCORE_ROOT'))
    assert read(scored / 'COMPLETE.json')['generation_and_scoring_complete']
    assert read(scored / 'GUARD_EXIT.json')['code'] == 0
    assert read(med / 'COMPLETE.json')['generation_and_scoring_complete']
    rows = read(scored / 'PACKET_PRIVATE.json')['consumers']
    dec = read(scored / 'DECISIONS.json')['decisions']
    mrows = read(med / 'PACKET_PRIVATE.json')['consumers']
    mdec = read(med / 'DECISIONS.json')['decisions']
    prior = read(old / 'CONSUMERS_B_FROZEN.json')['consumers']
    prior_dec = read(old / 'MERGED_DECISIONS.json')['decisions']
    metadata = read(old / 'PACKET_METADATA_B.json')
    masks = read(base / 'BASE_MASK_FROZEN.json')['consumers']
    oldmed = {(x['mode'], x['prefix'], x['owner'], x['query_id']): x for x in prior if x['arm'] == 'MED'}
    oldbank = {x['query_id']: x for x in prior if x['arm'] == 'MED' and x['mode'] == 'bank' and x['prefix'] == 146}
    memit = {(x['mode'], x['prefix'], x['owner'], x['query_id']): x for x in rows}
    newmed = {(x['mode'], x['prefix'], x['owner'], x['query_id']): x for x in mrows}
    assert len(memit) == len(rows) == 2207 and len(newmed) == 2207 and len(oldbank) == 2093
    candidates = read(os.environ['MED150_CANDIDATES'])['items']
    active = {(x['task']['native']['image_sha256'], x['task']['native']['question']) for x in candidates}
    newly_active = {q for q, x in oldbank.items() if (x['image_sha256'], x['question']) in active}

    def ids(task, role):
        if role == 'FIT':
            return [f"edit_{task['order']}_{i}" for i in range(1, 5)]
        qs = list(dict.fromkeys(q for e in task['events'] if e['task'] == role for q in e['all_probe_query_ids']))
        return qs if role == 'T0' else [q for q in qs if metadata['base_mask'][q] == role.endswith('L')]

    panels = {}
    for mode, prefix in (('bank', 150),):
        panels[mode] = {}
        for field in ('core_correct', 'full_supported'):
            panels[mode][field] = {}
            for role in ('T0', 'FIT', 'T1G', 'T2G', 'T1L', 'T2L'):
                by_m, by_b, groups = {}, {}, {}
                for task in metadata['tasks']:
                    owner = task['order']
                    qs = ids(task, role)
                    if mode == 'bank' and role.endswith('L'):
                        qs = [q for q in qs if not oldbank[q]['active_target'] and q not in newly_active]
                    by_m[owner], by_b[owner] = [], []
                    groups[owner] = task['native'].get('source_group', str(owner))
                    for q in qs:
                        key = (mode, prefix, owner if mode == 'single' else 0, q)
                        v = memit[key]
                        if mode == 'single':
                            m = oldmed[('single', 1, owner, q)]
                            value = prior_dec.get(m['payload_key'], {}).get(field)
                        else:
                            value = mdec.get(newmed[key]['key'], {}).get(field)
                        by_m[owner].append(value)
                        by_b[owner].append(dec.get(v['key'], {}).get(field))
                for owner in range(147, 151):
                    items = [x for x in masks if x['owner'] == owner and x['role'] == role and (role in ('T0', 'FIT') or x['eligible'] is True)]
                    if role == 'T1G':
                        items = []
                    groups[owner] = str(owner)
                    keys = [(mode, prefix, owner if mode == 'single' else 0, x['query_id']) for x in items]
                    by_m[owner] = [mdec.get(newmed[k]['key'], {}).get(field) for k in keys]
                    by_b[owner] = [dec.get(memit[k]['key'], {}).get(field) for k in keys]
                sm, sb = summary(by_m), summary(by_b)
                diffs, clusters = [], []
                for owner, values in by_m.items():
                    other = by_b[owner]
                    assert len(values) == len(other)
                    if values:
                        diffs.append(None if any(v is None for v in values + other) else statistics.mean(values) - statistics.mean(other))
                        clusters.append(groups[owner])
                panels[mode][field][role] = dict(MED=sm, BALANCEDIT=sb,
                    MED_minus_BALANCEDIT_pp=100 * statistics.mean(diffs) if diffs and all(x is not None for x in diffs) else None,
                    edit_bootstrap95_pp=interval(diffs), source_bootstrap95_pp=interval(diffs, clusters))
    result = dict(status='COMPLETE' if not read(scored / 'DECISIONS.json')['unknown'] else 'INCOMPLETE_SCORING',
                  cohort='exposed 146+4 development', independent_confirmation=False, panels=panels,
                  quality={mode: dict(n=sum(x['mode'] == mode for x in rows), cap=sum(x['mode'] == mode and x['at_cap'] for x in rows)) for mode in ('single', 'insertion', 'bank')},
                  scoring=read(scored / 'COMPLETE.json'), limitations=['BALANCEDIT original Stage17 FP16 adaptation; reconstructed independent expert bank; old 146 singles not included',
                    'new T1G semantics unknown', 'new T2L has no eligible support', 'source agreement is not clinical truth'])
    target = Path(os.environ['REPORT_OUTPUT'])
    assert not target.exists()
    target.mkdir()
    (target / 'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'panels': panels['bank'], 'quality': result['quality']}))


if __name__ == '__main__':
    main()
