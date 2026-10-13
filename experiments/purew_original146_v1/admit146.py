"""Revalidate retained outputs without training, generation or new Judge requests."""
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
sys.path.insert(0, os.environ['RUN_ROOT'] + '/private/tools')
import common as c
from contract import source_audit

R = c.RUN
P = Path(c.read(R/'private/FROZEN_PROTOCOL.json')['common_parent'])


def connect(path):
    db = sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    return db


def main():
    started = time.time()
    tasks = c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks']
    ledger = c.read(R/'private/EVAL_LEDGER.json')
    audit = source_audit(tasks, ledger)
    bindings = c.read(R/'private/EVAL_BINDINGS.json')
    original = c.read(R/'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json')
    assert [t['edit_id'] for t in tasks] == original['main_T0']
    assert all(ledger['Base_correctness'][q] == original['Base_correctness'][q] for q in ledger['queries'])
    assert c.read(P/'public/ADMISSION.json')['status'] == 'PASS'
    assert c.read(P/'public/RECOVERY_AUDIT.json')['status'] == 'PASS_RECOVERY200_COMPLETE'
    db = connect(P/'private/judge_common146_medium/queue.sqlite')
    recovery = connect(P/'private/judge_common146_medium_recovery1/queue.sqlite')
    payloads = {r['key']: dict(r) for r in db.execute('SELECT * FROM payload')}
    for row in recovery.execute('SELECT * FROM payload'):
        assert payloads[row['key']]['status'] == 'MISSING'
        assert (row['record'], row['binding']) == (payloads[row['key']]['record'], payloads[row['key']]['binding'])
        assert row['status'] == 'FORMAT_VALID'
        payloads[row['key']] = dict(row)
    assert len(payloads) == 3146 and all(x['status'] == 'FORMAT_VALID' for x in payloads.values())
    checks = {}
    for method in ('TT88_W0_RETRO146', 'balancedit'):
        rows = list(db.execute('SELECT * FROM consumer WHERE method=?', (method,)))
        assert len(rows) == 5722
        slots = set()
        for row in rows:
            d = c.read(row['path'])
            assert c.digest(d) == row['output_binding'], 'Historical output changed'
            q = ledger['queries'][row['query_id']]
            b = bindings[q['opaque_Base_id']]
            assert d['binding']['input'] == q
            output = d['R0'] if method.startswith('TT') else d.get('modes', d)['R0']
            full = json.loads(payloads[row['payload_key']]['binding'])
            assert all(full[k] == b[k] for k in ('question', 'reference', 'image_sha256', 'image_path', 'prompt_ids', 'attention_mask', 'runtime', 'generation'))
            assert full['query_id'] == q['query_id'] and full['output'] == output
            assert c.digest(full) == row['payload_key']
            assert (full['judge']['model'], full['judge']['reasoning_effort']) == ('gpt-6-astra', 'medium')
            slots.add((row['mode'], row['prefix'], row['edit_order'], row['query_id']))
        assert len(slots) == 5722
        checks[method] = dict(raw_consumers=5722, complete_input_output_binding=True, accepted_scores_reusable=True)
    result = dict(status='PASS', methods=checks, data=audit, baseline_missing=0,
                  original_masks=True, precision='NATIVE_FP16', new_training=0, new_generation=0,
                  new_Judge=0, prior_common_attempts=3346, seconds=time.time()-started)
    c.write(R/'private/BASELINE_REUSE_ADMISSION.json', result)
    c.write(R/'public/BASELINE_REUSE_ADMISSION.json', result)
    c.write(R/'private/CPU_ADMISSION.json', dict(status='PASS', data=audit))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
