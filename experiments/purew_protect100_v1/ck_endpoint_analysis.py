"""Read terminal C/K scores; report paired counts without new generation or Judge."""
import json
import os
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0, os.environ['RUN_ROOT'] + '/private/tools')
from logic import transitions


def main():
    root = Path(os.environ['RUN_ROOT'])
    result = json.loads((root/'public/RESULTS.json').read_text())
    assert result['status'] == 'COMPLETE_WITH_MISSING'
    assert (root/'private/judge_purew146/ALL_WORKERS_COMPLETE.json').exists()
    ledger = json.loads((root/'private/EVAL_LEDGER.json').read_text())
    tasks = json.loads((root/'private/BENCHMARK146_QUEUE.json').read_text())['tasks'][:100]
    active = {(t['native']['image_sha256'], t['native']['question']) for t in tasks}
    db = sqlite3.connect('file:'+str(root/'private/judge_purew146/queue.sqlite')+'?mode=ro', uri=True)
    rows = list(db.execute("SELECT c.method,c.query_id,c.payload_key,p.correct,p.status FROM consumer c JOIN payload p ON c.payload_key=p.key WHERE c.method IN ('C','K') AND c.mode='bank_R0' AND c.prefix=100"))
    assert len(rows) == 2148 and all(row[4] in ('FORMAT_VALID', 'MISSING') for row in rows)
    lookup = {(a, q): k for a, q, k, _, _ in rows}
    assert len(lookup) == len(rows)
    scores = {k: v for _, _, k, v, _ in rows}
    panels = {(x['arm'], x['task']): x for x in result['panels'] if x['mode']=='bank_R0' and x['prefix']==100 and x['arm'] in ('C','K')}
    paired = {}
    for task in ('T0','T1G','T2G','T1L','T2L'):
        ids = [t['edit_id'] for t in tasks] if task=='T0' else [q for t in tasks for e in t['events'] if e['task']==task for q in e['all_probe_query_ids']]
        if task != 'T0':
            ids = [q for q in ids if ledger['Base_correctness'][q] == task.endswith('L')]
        if task.endswith('L'):
            ids = [q for q in ids if (ledger['queries'][q]['image_sha256'],ledger['queries'][q]['question']) not in active]
        assert len(ids) == len(set(ids)) == panels['C',task]['observations'] == panels['K',task]['observations']
        for arm in ('C','K'):
            keys = [lookup[arm,q] for q in ids]
            assert sum(scores[k]==1 for k in keys) == panels[arm,task]['known_correct']
            assert sum(scores[k] is None for k in keys) == panels[arm,task]['missing']
        paired[task] = transitions([(lookup['C',q],lookup['K',q]) for q in ids],scores)
    db.close()
    data = dict(status='TERMINAL_CK_SAME_PREFIX100_PAIRED_ANALYSIS', comparison='C versus K on identical frozen eligible inputs; cross-method endpoints, not temporal forgetting', paired=paired,
        panel_missing={a:{t:panels[a,t]['missing'] for t in ('T0','T1G','T2G','T1L','T2L')} for a in ('C','K')},
        new_generation=0,new_Judge=0, masks_and_active_target_rule_unchanged=True,
        primary_result_unchanged=True, scientific_success=result['scientific_success'])
    path=root/'public/CK_FINAL_ANALYSIS.json';temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
    print(json.dumps(data,ensure_ascii=False))


if __name__=='__main__':main()
