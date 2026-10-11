"""Four disjoint scorers; independent 3146-attempt ledger, no inherited scores."""
import fcntl
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')
import astra_queue as a
q,RUN=a.q,a.RUN
q.ROOT=RUN/'private/judge_common146_medium'
q.BATCH_LEDGER='Astra_common146_batches'
q.ingest=lambda db:None


if __name__=='__main__':
    op=json.loads(os.environ['QUEUE_REQUEST'])
    assert q.read(RUN/'public/ADMISSION.json')['status']=='PASS'
    assert q.read(q.ROOT/'EPOCH_MANIFEST.json')['maximum_new_Judge']==3146
    assert q.read(RUN/'RUN_MANIFEST.json')['Judge_limit']==3146
    with (q.ROOT/'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=q.connect()
        answer=a.request(db,op)
        if not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone():
            q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(status='SCORING_TERMINAL',epoch=time.time(),state=q.status(db)))
    print(json.dumps(answer,ensure_ascii=False))
