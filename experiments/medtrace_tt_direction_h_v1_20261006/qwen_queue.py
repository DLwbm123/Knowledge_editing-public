"""Incremental phase queue using PR27's exact full-key, one-attempt scorer."""
import random
from qwen_parent import *
import qwen_parent as parent


def ingest(db):
    phase=read(RUN/'private/SCORING_PHASE.json')['phase'];marker='generation:'+phase
    if db.execute('SELECT 1 FROM done WHERE name=?',(marker,)).fetchone():return
    # The finite controller sets this only after every paired block has ended.
    assert read(RUN/'private/SCORING_PHASE.json')['training_complete']
    paths=sorted((RUN/'private/outputs').glob('*/*/e*/*/*.json'))
    random.Random(20260912).shuffle(paths)
    for p in paths:
        d=read(p);bind=d['binding'];row=bind['input'];b=bind['judge_input'];arm,mode,order,folder=p.parts[-5:-1]
        assert bind['arm']==arm and bind['owner_order']==int(order[1:])
        key=payload(db,row,b,d['R0']);cid=digest([arm,mode,int(order[1:]),folder,row['query_id'],digest(d)])
        existing=db.execute('SELECT * FROM consumer WHERE method=? AND mode=? AND edit_order=? AND query_id=?',(arm,mode,int(order[1:]),row['query_id'])).fetchone()
        if existing:assert existing['payload_key']==key and existing['output_binding']==digest(d)
        else:db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',(cid,arm,mode,1,int(order[1:]),folder,row['query_id'],str(p),digest(d),key))
    pending=db.execute("SELECT count(*) FROM payload WHERE status='PENDING'").fetchone()[0]
    remaining=read(RUN/'RUN_MANIFEST.json')['Judge_limit']-read(RUN/'RESOURCE_LEDGER.json')['Judge_attempts']
    assert pending<=remaining,'Complete phase exceeds fixed Judge cap; no favorable subset'
    db.execute('INSERT INTO done VALUES (?)',(marker,));db.execute("INSERT OR IGNORE INTO done VALUES ('generation')");db.commit()
    write(ROOT/('QUEUE_ADMISSION_'+phase+'.json'),dict(status='PASS',phase=phase,new_pending=pending,registered_outputs=len(paths),full_key_only=True))


parent.ingest=ingest
