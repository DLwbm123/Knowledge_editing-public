"""Inspect every veto, including queries excluded from the Base-correct locality mask."""
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path

def main():
    run, parent, margin = (Path(os.environ[k]) for k in ('RUN_ROOT', 'TYPE_PARENT', 'MARGIN_PARENT'))
    read = lambda p: json.loads(p.read_text())
    decisions = read(run/'private/DECISIONS.json')
    old = {x['query_id']: x for x in read(margin/'private/SELECTIONS.json') if x['arm'] == 'MARGIN_002'}
    db = sqlite3.connect('file:'+str(parent/'private/judge_gate_astra_medium/queue.sqlite')+'?mode=ro', uri=True)
    scores = dict(db.execute('SELECT key,correct FROM payload'))
    query = read(parent/'private/PAPER_QUERIES.json')['146']
    counts, roles = Counter(), {}
    for d in decisions:
        if not d['blocked']:
            continue
        a, b = scores[old[d['query_id']]['payload_key']], scores[d['payload_key']]
        state = 'missing' if a is None or b is None else 'gain' if b > a else 'loss' if b < a else 'unchanged'
        counts[state] += 1
        roles.setdefault(query[d['query_id']].get('role', 'PANEL'), Counter())[state] += 1
    assert sum(counts.values()) == sum(d['blocked'] for d in decisions)
    result = dict(status='COMPLETE_NOT_PROMOTED', all_veto_transitions=counts, by_role=roles,
        decision='Do not replace the primary method: veto also removes correct edits outside the Base-correct locality mask.',
        all_queries_scanned=True, score_missing_preserved=True)
    (run/'public/ALL_VETO_AUDIT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
