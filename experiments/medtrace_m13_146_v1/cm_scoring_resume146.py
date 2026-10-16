"""One authorized continuation of unattempted scoring; original failures stay intact."""
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import time
import traceback
import cm_controller146 as controller
import cm_queue146 as queue

RUN=controller.RUN
RECOVERY_ID=int(os.environ.get("CM_SCORING_RESUME_ID","1"))
assert RECOVERY_ID in (1,2)


def merge_judge():
    parent=Path(controller.read(RUN/'private/CM_ADMISSION1.json')['parent'])
    with (parent/'RESOURCE_LEDGER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ledger=controller.read(parent/'RESOURCE_LEDGER.json')
        own=controller.read(RUN/'RESOURCE_LEDGER.json')['Judge_attempts']
        prior=controller.read(RUN/'private/INHERITED_COST.json')['Judge_attempts']
        merged=ledger['cm_eval146_v1_Judge_merged']
        assert merged['run']==str(RUN)
        delta=own-prior-merged['increment'];assert delta>=0
        ledger['Judge_attempts']+=delta
        assert ledger['Judge_attempts']<=controller.read(parent/'RUN_MANIFEST.json')['Judge_limit']
        merged.update(increment=own-prior,epoch=time.time())
        controller.write(parent/'RESOURCE_LEDGER.json',ledger)
        controller.write(RUN/f'private/CM_RESUME_COST_TRANSFER_Judge{RECOVERY_ID}.json',dict(increment=own-(19152 if RECOVERY_ID==1 else 20002),cumulative_increment=own-prior,epoch=time.time()))


def verify_preserved(db,resume):
    for key,status,batch,correct in resume['preserved']:
        assert tuple(db.execute('SELECT status,batch,correct FROM payload WHERE key=?',(key,)).fetchone())==(status,batch,correct)
    assert db.execute('SELECT count(*) FROM payload').fetchone()[0]==5002
    assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==17081


def main():
    resume=queue.recovery(RUN);assert resume is not None and resume.get('recovery_id',1)==RECOVERY_ID
    assert controller.read(RUN/'private/CM_RESUME_CPU_CHECK1.json')['status']=='PASS'
    controller.protocol.admit(RUN);controller.protocol.project_budget(RUN)
    with (RUN/f'private/CM_SCORING_RESUME_ONCE{RECOVERY_ID}').open('x'):pass
    controller.execute('cm_queue146.py',f'cm_resume_ingest{RECOVERY_ID}',operation=dict(action='ingest'))
    root=RUN/'private'/controller.protocol.QUEUE
    assert queue.all_workers_done(queue.completion_root(root,resume),4)
    with sqlite3.connect('file:'+str(root/'queue.sqlite')+'?mode=ro',uri=True) as db:
        verify_preserved(db,resume)
        assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
    merge_judge()
    controller.execute('cm_report146.py',f'cm_resume_report{RECOVERY_ID}')
    assert (RUN/'private/CM_REPORT_COMPLETE1.json').exists()
    controller.write(RUN/'private/CM_COMPLETE1.json',dict(status='COMPLETE_PUBLICATION_PENDING',recovery=RECOVERY_ID,epoch=time.time()))


if __name__=='__main__':
    code=0
    try:main()
    except BaseException as error:
        code=1
        controller.write(RUN/f'private/FAILURE_cm_scoring_resume{RECOVERY_ID}.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,epoch=time.time()))
        raise
    finally:
        try:merge_judge()
        except BaseException as error:
            code=1;controller.write(RUN/f'private/CM_RESUME_COST_TRANSFER_ERROR{RECOVERY_ID}.json',dict(error=repr(error)))
        controller.write(RUN/f'private/CM_SCORING_RESUME_EXIT{RECOVERY_ID}.json',dict(exit_code=code,epoch=time.time()))
