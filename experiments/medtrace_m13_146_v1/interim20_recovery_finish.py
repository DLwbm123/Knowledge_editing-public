"""Resolve the authorized overlay, account once, and recalculate unchanged metrics."""
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import time
import traceback

RUN = Path(os.environ['RUN_ROOT'])
ORIGINAL = Path(os.environ['INTERIM20_ORIGINAL'])
PARENT = Path(os.environ['INTERIM20_PARENT'])
import astra_queue as a
q = a.q


def resolved_snapshot(old_path, recovery_path, destination):
    old = sqlite3.connect('file:'+str(old_path)+'?mode=ro', uri=True); old.row_factory = sqlite3.Row
    new = sqlite3.connect('file:'+str(recovery_path)+'?mode=ro', uri=True); new.row_factory = sqlite3.Row
    assert not new.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
    rows = new.execute('SELECT * FROM payload').fetchall(); assert len(rows) == 200
    assert not destination.exists()
    out = sqlite3.connect(destination); old.backup(out)
    for row in rows:
        before = old.execute('SELECT * FROM payload WHERE key=?', (row['key'],)).fetchone()
        assert before and before['status'] == 'MISSING' and before['correct'] is None
        assert (row['record'], row['binding']) == (before['record'], before['binding'])
        assert row['status'] in ('FORMAT_VALID', 'MISSING')
        assert (row['status'] == 'MISSING' and row['correct'] is None) or (
            row['status'] == 'FORMAT_VALID' and row['correct'] in (0, 1))
        out.execute('UPDATE payload SET status=?,correct=?,batch=? WHERE key=?',
                    (row['status'], row['correct'], row['batch'], row['key']))
    out.commit()
    assert out.execute('SELECT count(*) FROM payload').fetchone()[0] == 717
    assert out.execute('SELECT count(*) FROM consumer').fetchone()[0] == 1077
    assert all(
        tuple(x) == tuple(y) for x,y in zip(old.execute('SELECT * FROM consumer ORDER BY id'), out.execute('SELECT * FROM consumer ORDER BY id')))
    for before in old.execute("SELECT * FROM payload WHERE status='FORMAT_VALID'"):
        assert tuple(before) == tuple(out.execute('SELECT * FROM payload WHERE key=?', (before['key'],)).fetchone())
    state = dict(out.execute('SELECT status,count(*) FROM payload GROUP BY status'))
    out.close(); new.close(); old.close()
    return state


def main():
    repair = os.environ.get('RECOVERY_ZERO_WORK_REPAIR') == '1'
    if repair:
        assert q.read(RUN/'private/RECOVERY_ZERO_WORK_REPAIR1.json')['status'] == 'PASS_ZERO_ATTEMPTS'
    with (RUN/'private'/('FINISH_ONCE2' if repair else 'FINISH_ONCE1')).open('x'): pass
    queue = RUN/'private/judge_interim20_recovery1'
    while not ((queue/'ALL_WORKERS_COMPLETE.json').exists() and (queue/'workers/0/SCORER_DONE.json').exists()):
        assert time.time() < q.read(RUN/'RUN_MANIFEST.json')['deadline_epoch']-600
        assert not (queue/'workers/0/SCORER_FAILURE.json').exists()
        assert not (queue/'workers/0/SCORER_STOP.json').exists()
        time.sleep(10)
    snapshot = RUN/'private/INTERIM20_RESOLVED_QUEUE.sqlite'
    state = resolved_snapshot(ORIGINAL/'private/judge_interim20_astra_medium/queue.sqlite', queue/'queue.sqlite', snapshot)
    own = q.read(RUN/'RESOURCE_LEDGER.json'); inherited = q.read(RUN/'private/INHERITED_COST.json')
    delta = own['Judge_attempts']-inherited['Judge_attempts']; assert delta == 200
    assert not own['gpu_sessions'] and own['gpu_seconds_used'] == inherited['gpu_seconds_used']
    with (PARENT/'RESOURCE_LEDGER.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX); ledger = q.read(PARENT/'RESOURCE_LEDGER.json')
        assert 'interim20_recovery1_Judge_merged' not in ledger
        ledger['Judge_attempts'] += delta
        assert ledger['Judge_attempts'] <= q.read(PARENT/'RUN_MANIFEST.json')['Judge_limit']
        ledger['interim20_recovery1_Judge_merged'] = dict(increment=delta,run=str(RUN),epoch=time.time())
        q.write(PARENT/'RESOURCE_LEDGER.json', ledger)
        q.write(RUN/'private/COST_TRANSFER_Judge1.json', ledger['interim20_recovery1_Judge_merged'])
    with (PARENT/'private/INTERIM20_REUSE.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX); path = PARENT/'private/M13_146_REUSE_QUEUES.json'
        paths = q.read(path); original = str(ORIGINAL/'private/judge_interim20_astra_medium/queue.sqlite')
        assert paths.count(original) == 1 and str(snapshot) not in paths
        q.write(path, [str(snapshot) if p == original else p for p in paths])
    records,scores = {},{}
    with sqlite3.connect('file:'+str(snapshot)+'?mode=ro', uri=True) as db:
        for path,binding,status,score in db.execute('SELECT c.path,c.output_binding,p.status,p.correct FROM consumer c JOIN payload p ON p.key=c.payload_key'):
            data = q.read(path); assert q.digest(data) == binding
            key = tuple(data[k] for k in ('method','mode','prefix','owner'))+(data['binding']['input']['query_id'],)
            assert key not in records; records[key] = data; scores[key] = score if status == 'FORMAT_VALID' else None
    from selected_report146 import calculate
    result, owners = calculate(q.read(ORIGINAL/'private/ORIGINAL146_TASKS.json')[:20],
        q.read(ORIGINAL/'private/ORIGINAL_BASE_MASK146.json'),records,scores,interim20=True)
    result.update(status='INTERIM20_AUTHORIZED_RECOVERY_RESOLVED',consumers=1077,
        original_missing_payloads_preserved=200,recovery_attempts=delta,full146_complete=False,
        interim_descriptive_only=True,selection_or_stopping_authorized=False,bank146_evaluated=False)
    q.write(RUN/'public/INTERIM20_RECOVERED_RESULTS.json', result)
    q.write(RUN/'private/INTERIM20_RECOVERED_OWNER_AUDIT.json',owners)
    q.write(RUN/'private/RECOVERY_COMPLETE1.json',dict(status='COMPLETE_PUBLICATION_PENDING',payload_status=state,
        new_Judge=delta,new_GPU_seconds=0,new_generation=0,old_valid_untouched=517,
        original_missing_preserved=200,merged_missing_consumers=result['missing_consumers'],epoch=time.time()))


if __name__ == '__main__':
    code = 0
    suffix = '2' if os.environ.get('RECOVERY_ZERO_WORK_REPAIR') == '1' else '1'
    try: main()
    except BaseException as error:
        code = 1; q.write(RUN/'private'/('FAILURE_recovery_finish'+suffix+'.json'),dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,epoch=time.time()))
        raise
    finally: q.write(RUN/'private'/('RECOVERY_FINISH_EXIT'+suffix+'.json'),dict(exit_code=code,epoch=time.time()))
