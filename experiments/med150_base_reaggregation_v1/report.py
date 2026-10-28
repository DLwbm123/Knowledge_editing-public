"""Reaggregate accepted Base and MED judgments on the frozen MED150 panels."""
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/medical_proposition_v133'))
from med133_report import summary


def read(path):
    return json.loads(Path(path).read_text())


def main():
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU deadline')))
    signal.alarm(1800)
    start = time.time()
    out = Path(os.environ['BASE150_OUTPUT'])
    assert not out.exists() and os.statvfs(out.parent).f_bavail * os.statvfs(out.parent).f_frsize > 8 * 1024**3
    oldroot = Path(os.environ['MED146_SCORE_ROOT'])
    oldcons = read(oldroot / 'CONSUMERS_B_FROZEN.json')['consumers']
    olddec = read(oldroot / 'MERGED_DECISIONS.json')['decisions']
    meta = read(oldroot / 'PACKET_METADATA_B.json')
    newroot = Path(os.environ['MED150_SCORE_ROOT'])
    baseroot = Path(os.environ['MED150_BASE_SCORE_ROOT'])
    assert read(newroot / 'COMPLETE.json')['generation_and_scoring_complete']
    assert read(newroot / 'GUARD_EXIT.json')['code'] == 0
    assert read(baseroot / 'BASE_MASK_FROZEN.json')['status'] == 'FROZEN'
    newcons = read(newroot / 'PACKET_PRIVATE.json')['consumers']
    newdec = read(newroot / 'DECISIONS.json')['decisions']
    basemask = read(baseroot / 'BASE_MASK_FROZEN.json')['consumers']
    basedec = read(baseroot / 'PARTIAL_DECISIONS.json')['decisions']
    oldbase = {x['query_id']: x for x in oldcons if x['arm'] == 'BASE' and x['mode'] == 'bank' and x['prefix'] == 146}
    oldmed = {x['query_id']: x for x in oldcons if x['arm'] == 'MED' and x['mode'] == 'bank' and x['prefix'] == 146}
    newmed = {x['query_id']: x for x in newcons if x['mode'] == 'bank' and x['prefix'] == 150}
    newbase = {x['query_id']: x for x in basemask}
    assert len(oldbase) == len(oldmed) == 2093 and len(newbase) == 55 and len(newmed) == 2148
    assert not set(oldbase) & set(newbase) and set(oldbase) | set(newbase) == set(newmed)
    assert all(oldmed[q]['payload_key'] == newmed[q]['key'] for q in oldmed)
    assert all(x['payload_key'] in olddec for x in oldbase.values())
    assert all(x['key'] in basedec for x in newbase.values())
    candidates = read(os.environ['MED150_CANDIDATES'])['items']
    active = {(x['task']['native']['image_sha256'], x['task']['native']['question']) for x in candidates}
    newly_active = {q for q, x in oldmed.items() if (x['image_sha256'], x['question']) in active}
    oldmask = meta['base_mask']

    def roleids(task, role):
        if role == 'FIT':
            return [f"edit_{task['order']}_{i}" for i in range(1, 5)]
        ids = list(dict.fromkeys(q for e in task['events'] if e['task'] == role for q in e['all_probe_query_ids']))
        return ids if role == 'T0' else [q for q in ids if oldmask[q] == role.endswith('L')]

    panels = {}
    for field in ('core_correct', 'full_supported'):
        panels[field] = {}
        for role in ('T0', 'FIT', 'T1G', 'T2G', 'T1L', 'T2L'):
            base, med = {}, {}
            for task in meta['tasks']:
                ids = [q for q in roleids(task, role) if not (role.endswith('L') and (oldmed[q]['active_target'] or q in newly_active))]
                base[task['order']] = [olddec[oldbase[q]['payload_key']].get(field) for q in ids]
                med[task['order']] = [newdec.get(newmed[q]['key'], {}).get(field) for q in ids]
            for owner in range(147, 151):
                ids = [x['query_id'] for x in basemask if x['owner'] == owner and x['role'] == role and (role in ('T0', 'FIT') or x['eligible'] is True)]
                if role == 'T1G':
                    ids = []
                base[owner] = [basedec[newbase[q]['key']].get(field) for q in ids]
                med[owner] = [newdec.get(newmed[q]['key'], {}).get(field) for q in ids]
            before, after = summary(base), summary(med)
            assert before['n'] == after['n'] and before['support_edits'] == after['support_edits']
            paired = [statistics.mean(med[o]) - statistics.mean(base[o]) for o in base if base[o] and all(v is not None for v in base[o] + med[o])]
            panels[field][role] = {'base': before, 'MED': after, 'paired_complete_edits': len(paired),
                'paired_delta_macro_pp': 100 * statistics.mean(paired) if len(paired) == before['support_edits'] and paired else None}
    out.mkdir()
    result = {'status': 'COMPLETE' if all(v[a]['unknown'] == 0 for p in panels.values() for v in p.values() for a in ('base', 'MED')) else 'INCOMPLETE',
        'cohort': 'original146 plus four exposed development edits', 'old_queries': len(oldbase), 'new_queries': len(newbase),
        'generation': 0, 'new_judge': 0, 'input_contract': 'accepted original Base packet matched to MED; accepted new Base packet',
        'limitations': ['Base is unedited model, not a knowledge-editing method', 'new T1G semantics unknown', 'new T2L has no eligible support', 'not independent clinical validation'],
        'panels': panels, 'elapsed_seconds': time.time() - start}
    (out / 'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'old_queries': len(oldbase), 'new_queries': len(newbase), 'elapsed_seconds': result['elapsed_seconds']}))


if __name__ == '__main__':
    main()
