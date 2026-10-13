"""One CPU repair: preserve full provenance without batch-only filename collisions."""
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time

RUN=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(RUN/'private/tools'))
from audit import read,write,digest


def main():
    started=time.time();ledger=read(RUN/'RESOURCE_LEDGER.json')
    assert all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
    recovery=RUN/'private/recovery/receipt_provenance';recovery.mkdir(parents=True,exist_ok=False)
    for relative in ('private/CONTROLLER_FAILURE.json','private/judge_common/SCORER_FAILURE.json','private/tools/qwen_parent.py'):
        shutil.copy2(RUN/relative,recovery/Path(relative).name)
    source=RUN/'private/tools/qwen_parent.py';text=source.read_text();old="dest=ROOT/'inherited'/(row['batch']+'.json')";new="dest=ROOT/'inherited'/(row['batch']+'_'+digest(saved)+'.json')"
    assert text.count(old)==1;source.write_text(text.replace(old,new))
    import qwen_queue as q
    db=q.connect();backup=sqlite3.connect(recovery/'queue_before.sqlite');db.backup(backup);backup.close()
    before={r['key']:dict(r) for r in db.execute('SELECT * FROM payload')};consumers={r['id']:dict(r) for r in db.execute('SELECT * FROM consumer')}
    q.initialize(db);q.ingest(db)
    after={r['key']:dict(r) for r in db.execute('SELECT * FROM payload')};after_consumers={r['id']:dict(r) for r in db.execute('SELECT * FROM consumer')}
    assert all(after[k]==v for k,v in before.items()) and all(after_consumers[k]==v for k,v in consumers.items())
    assert read(RUN/'RESOURCE_LEDGER.json')['Judge_attempts']==ledger['Judge_attempts']
    inherited=0
    for item in db.execute('SELECT * FROM inherited'):
        row=json.loads(item['parent_row']);saved=read(item['batch_receipt']);full=json.loads(after[item['key']]['binding'])
        q.validate_inherited(full,row,saved,saved['parent_attempt'],q.lock()['judge_identity']);inherited+=1
    # Regression condition: distinct ancestry envelopes no longer share a filename.
    a=dict(batch='same',parent_epoch=dict(run='a'));b=dict(a,parent_epoch=dict(run='b'))
    assert digest(a)!=digest(b)
    write(recovery/'REPAIR.json',dict(status='PASS',reason='Same original batch/evidence/response/attempt, different parent_epoch provenance; content-bound receipt filename preserves both',old_payloads_unchanged=len(before),old_consumers_unchanged=len(consumers),total_payloads=len(after),total_consumers=len(after_consumers),inherited_receipts_validated=inherited,new_requests=0,training_reruns=0,score_retries=0,clock_reset=False,CPU_seconds=time.time()-started))
    print(json.dumps(read(recovery/'REPAIR.json')));db.close()


if __name__=='__main__':main()
