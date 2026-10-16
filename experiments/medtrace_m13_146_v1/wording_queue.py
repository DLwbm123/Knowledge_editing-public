"""Original sealed binary Judge, one serial worker; no original rejudging."""
import collections
import fcntl
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).parent))
import contract
import wording_pair as protocol
import queue146 as base

base.MAX_JUDGE = protocol.CONSUMERS
base.contract = SimpleNamespace(original=contract.original, METHOD_CONSUMERS=dict.fromkeys(protocol.METHODS, protocol.N))


def main():
    import astra_queue as a
    from fine_queue import full_identity, selfcheck
    q, run = a.q, a.RUN
    q.ROOT = run/'private'/protocol.QUEUE
    q.BATCH_LEDGER = 'Astra_wording_pair_batches'
    q.EFFORT = 'medium'
    operation = json.loads(os.environ.get('QUEUE_REQUEST', '{"action":"status"}'))
    q.ROOT.mkdir(parents=True, exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = q.connect(); q.ingest = lambda _: None
        if list((run/'private').glob('FAILURE_wording*.json')) and operation['action'] in ('next', 'reserve'):
            assert operation['action'] != 'reserve', 'Own stage failed'
            print(json.dumps(dict(state=dict(status='STOPPED', reason='OWN_STAGE_FAILED'), rows=[]))); return
        if (run/'private/WORDING_GENERATION_COMPLETE1.json').exists():
            gate = protocol.check(run)
            judge = dict(model='gpt-6-astra', reasoning_effort='medium', protocol=q.PROTOCOL, prompt=q.PROMPT)
            epoch = dict(epoch=protocol.EPOCH, **judge, workers=1, batch_size=50, attempts_per_payload=1,
                         maximum_new_Judge=protocol.CONSUMERS, scientific_lock=q.digest(q.read(run/'private/WORDING_SCOPE1.json')), selfcheck=selfcheck())
            ep = q.ROOT/'EPOCH_MANIFEST.json'
            if ep.exists(): assert q.read(ep) == epoch
            else: q.write(ep, epoch)
            if not db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone():
                outputs = ((p, q.read(p)) for p in sorted((run/'private/outputs').glob('*/*/*.json')))
                with db: base.ingest_generated(q, db, q.read(run/'private/EVAL_PLAN146.json'), outputs,
                    full_identity, judge, epoch['scientific_lock'], {})
        else:
            assert operation['action'] in ('status', 'next', 'reserved')
        if operation['action'] == 'reserve':
            keys = operation['keys']; assert 0 < len(keys) <= 50 and len(keys) == len(set(keys))
            prior = q.read(run/'private/INHERITED_COST.json')['Judge_attempts']
            own = q.read(run/'RESOURCE_LEDGER.json')['Judge_attempts']-prior
            assert own+len(keys) <= protocol.CONSUMERS
            parent = Path(q.read(run/'private/WORDING_ADMISSION1.json')['parent'])
            assert q.read(parent/'RESOURCE_LEDGER.json')['Judge_attempts']+own+len(keys) <= q.read(parent/'RUN_MANIFEST.json')['Judge_limit']
        answer = base.coverage(db) if operation['action'] == 'ingest' else q.request(db, operation)
        coverage = base.coverage(db)
        if coverage['all_attempts_resolved']:
            q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json', dict(coverage, epoch=time.time()))
    print(json.dumps(answer, ensure_ascii=False))


if __name__ == '__main__': main()
