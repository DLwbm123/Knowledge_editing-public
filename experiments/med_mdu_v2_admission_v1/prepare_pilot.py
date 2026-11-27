"""Bind the frozen Stage-1 schedule and historical T2L references before training."""
import csv
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def main(root, worktree):
    panel = read(root / 'pilot_inputs.private.json')
    entries = panel['entries']
    plans = read(root / 'training_schedule.private.json')
    assert len(entries) == 30 and plans['steps'] == 160 and plans['planned_updates'] == 19200
    sample = list(csv.DictReader((root / 'blind_view_review.tsv').open(), delimiter='\t'))
    reviewed = list(csv.DictReader((root / 'astra62_blind_view_verdicts.private.tsv').open(), delimiter='\t'))
    assert len(sample) == len(reviewed) == 176
    assert {r['number'] for r in sample} == {r['number'] for r in reviewed}
    assert all(r['verdict'] == 'ACCEPT' for r in reviewed)

    history = worktree / '.private/med_full_d150_v1_20261102'
    packet = read(history / 'scoring/PACKET_PRIVATE.json')
    roles = read(history / 'EVALUATION_ROLES_PRIVATE.json')['roles']
    support = {value for e in entries for value in e['frozen_support']['T2L']}
    rows = {r['query_id']: r for r in packet['consumers'] if r['arm'] == 'MED' and r['query_id'] in support}
    originals = {r['query']: r for r in read(root / 'inventory.private.json')['frozen'] if r['query'] in support}
    assert len(support) == len(rows) == len(originals) == 18
    schedules = {p['owner']: p for p in plans['entries']}
    for item in entries:
        schedule = schedules.pop(item['owner'])
        assert len(schedule['steps']) == 160 and schedule['seed'] == item['source']['seed']
        item['steps'] = schedule['steps']
        item['historical_t2l'] = []
        for q in item['frozen_support']['T2L']:
            r = rows[q]
            record = packet['payloads'][r['key']]['record']
            assert originals[q]['sha'] == item['source']['source_rows'][0]['image_sha256']
            assert q in roles[str(item['owner'])]['T2L'] and record['question'] == originals[q]['question']
            item['historical_t2l'].append(dict(question=record['question'], reference=record['gold_answer']))
    assert not schedules
    output = dict(panel, status='ASTRA62_REVIEW_PASSED', arms=plans['arms'],
                  review=dict(model='gpt-6.2-astra', accepted=len(reviewed), human=0))
    with (root / 'stage1_panel.private.json').open('x') as out:
        json.dump(output, out, ensure_ascii=False)
    print('images', len(entries), 'historical_T2L', len(support), 'planned_updates', plans['planned_updates'])


if __name__ == '__main__':
    main(*(Path(arg) for arg in sys.argv[1:]))
