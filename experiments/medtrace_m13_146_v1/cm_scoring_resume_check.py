"""Small CPU checks for recovery authorization, preservation, and failure boundaries."""
import json
from pathlib import Path
import sqlite3
import tempfile
import cm_queue146 as q
import cm_scoring_resume146 as c


def main():
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE payload(key TEXT,status TEXT,batch TEXT,correct INTEGER)')
    db.executemany('INSERT INTO payload VALUES(?,?,?,?)',[('a','PENDING',None,None),('b','FORMAT_VALID','old',1),('c','RESERVED','new',None)])
    resume={'pending_keys':['a','c'],'prior_failures':{'FAILURE_cm_old.json':{'error':'old'}}}
    q.check_resume_reservation(db,['a'],resume)
    for keys in [['b'],['c']]:
        try:q.check_resume_reservation(db,keys,resume)
        except AssertionError:pass
        else:raise AssertionError('Attempted or unauthorized key accepted')
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'private').mkdir()
        p=root/'private/FAILURE_cm_old.json';p.write_text(json.dumps({'error':'old'}))
        assert not q.new_failures(root,resume)
        p.write_text(json.dumps({'error':'changed'}));assert q.new_failures(root,resume)==[p]
        p.write_text(json.dumps({'error':'old'}));n=root/'private/FAILURE_cm_new.json';n.write_text('{}');assert q.new_failures(root,resume)==[n]
        assert q.completion_root(root,resume)==root/'resume1'
        assert q.completion_root(root,None)==root
    # Exact persisted verdicts must survive the recovery, including batch identities.
    db.execute('CREATE TABLE consumer(id INTEGER)')
    db.executemany('INSERT INTO payload VALUES(?,?,?,?)',[(str(i),'PENDING',None,None) for i in range(4999)])
    db.executemany('INSERT INTO consumer VALUES(?)',[(i,) for i in range(17081)])
    preserved={'preserved':[['b','FORMAT_VALID','old',1]]};c.verify_preserved(db,preserved)
    db.execute("UPDATE payload SET correct=0 WHERE key='b'")
    try:c.verify_preserved(db,preserved)
    except AssertionError:pass
    else:raise AssertionError('Changed historical score accepted')
    original=c.RUN
    try:
        with tempfile.TemporaryDirectory() as tmp:
            c.RUN=Path(tmp)/'run';parent=Path(tmp)/'parent'
            (c.RUN/'private').mkdir(parents=True);parent.mkdir()
            for path,data in [
                (c.RUN/'private/CM_ADMISSION1.json',{'parent':str(parent)}),
                (c.RUN/'private/INHERITED_COST.json',{'Judge_attempts':18452}),
                (c.RUN/'RESOURCE_LEDGER.json',{'Judge_attempts':19252}),
                (parent/'RUN_MANIFEST.json',{'Judge_limit':34439}),
                (parent/'RESOURCE_LEDGER.json',{'Judge_attempts':19152,'cm_eval146_v1_Judge_merged':{'run':str(c.RUN),'increment':700}})]:
                path.write_text(json.dumps(data))
            c.merge_judge();c.merge_judge()
            assert json.loads((parent/'RESOURCE_LEDGER.json').read_text())['Judge_attempts']==19252
    finally:c.RUN=original
    result=dict(status='PASS',incremental_cost_idempotent=True,unattempted_only=True,valid_reserved_rejected=True,prior_failure_exact_match=True,new_failure_stops=True,preserved_verdict_checked=True,production_Judge_attempts=0)
    c.controller.write(c.RUN/'private/CM_RESUME_CPU_CHECK1.json',result);print(json.dumps(result))


if __name__=='__main__':main()
