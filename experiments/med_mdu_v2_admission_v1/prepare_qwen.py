"""Score Qwen Base eligibility and bind additional, frozen T2G families for replication."""
from collections import Counter
import json
from pathlib import Path
import sys

from reference_scores import prepare_rows


def read(path):
    return json.loads(path.read_text())


def prepare(run, score, panel):
    complete = read(run / 'COMPLETE.json')
    assert complete['counts']['generations'] == complete['counts']['vision'] == 30
    cases = read(panel)['entries']
    responses = {v['qid']: v for p in (run / 'outputs').glob('*.json') if (v := read(p))}
    assert len(cases) == len(responses) == 30
    rows = []
    for item in cases:
        owner = item['owner']
        gold = item['source']['source_rows'][0]
        answers = responses[owner]
        assert answers['query_id'] == gold['query_id']
        rows.append(dict(owner=owner, role='native', question=gold['question'],
                         reference=gold['reference'], answers={'Q': {'answer': answers['raw_answer']}}))
    prepare_rows(rows, score, ('Q',))


def qualify(run, score, worktree, destination):
    stage0 = worktree / '.private/med_mdu_v2_stage0_20261127'
    cases = read(stage0 / 'prepared_r2/stage1_panel.private.json')['entries']
    data = {v['qid']: v for p in (run / 'outputs').glob('*.json') if (v := read(p))}
    for row in data.values():
        assert row['qid'] in {x['owner'] for x in cases}
    marks = read(score / 'DECISIONS.json')
    maps = read(score / 'consumers.private.json')
    assert len(maps) == len(cases) == len(data) == 30
    old = worktree / '.private/med_full_d150_v1_20261102/scoring/PACKET_PRIVATE.json'
    packet = read(old)
    references = {r['query_id']: packet['payloads'][r['key']]['record']
                  for r in packet['consumers'] if r['arm'] == 'MED'}
    diagnostics = worktree / '.private/med_mdu_family_diagnostic_20261127/metrics.jsonl'
    old_rows = [json.loads(line) for line in diagnostics.open()]
    counts = Counter()
    official = Counter()
    admitted = []
    for mark, row in zip(maps, cases, strict=True):
        owner = row['owner']
        assert mark['owner'] == owner
        prediction = data[owner]
        status = marks['decisions'].get(mark['key'])
        if prediction['at_cap'] or not prediction['EOS'] or not prediction['raw_answer'].strip():
            counts['invalid_generation'] += 1
        elif status is None:
            counts['unknown'] += 1
        elif status['core_correct']:
            counts['core_correct'] += 1
        else:
            rows = []
            for old in old_rows:
                if old['owner'] != owner or old['role'] != 'T2G':
                    continue
                ref = references[old['query']]
                role = 'official_' + old['variant']
                rows.append(dict(role=role, question=ref['question'], reference=ref['gold_answer']))
                official[role] += 1
            admitted.append(dict(row, base_core=False, base_eos=True, official_t2g=rows))
            counts['core_incorrect_admitted'] += 1
    with (destination / 'mechanism_panel.private.json').open('x') as out:
        json.dump(admitted, out, ensure_ascii=False)
    return dict(groups=dict(counts), official_families=dict(official),
                source_images=30, updates=0, immutable_judge_snapshot=None)


if __name__ == '__main__':
    mode, *args = sys.argv[1:]
    if mode == 'prepare':
        prepare(*(Path(v) for v in args))
    else:
        print(json.dumps(qualify(*(Path(v) for v in args)), ensure_ascii=False))
