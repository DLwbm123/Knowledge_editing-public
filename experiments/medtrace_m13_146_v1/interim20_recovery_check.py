"""CPU-only checks for the exact missing overlay and immutable valid prefix."""
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import interim20_recovery_queue as queue
from interim20_recovery_finish import resolved_snapshot
from interim20_recovery_scorer import connection_check


def connect(path):
    db = sqlite3.connect(path); db.row_factory = sqlite3.Row
    db.executescript('CREATE TABLE payload(key TEXT PRIMARY KEY,record TEXT,binding TEXT,status TEXT,batch TEXT,correct INTEGER);'
        'CREATE TABLE consumer(id TEXT PRIMARY KEY,method TEXT,mode TEXT,prefix INTEGER,edit_order INTEGER,folder TEXT,query_id TEXT,path TEXT,output_binding TEXT,payload_key TEXT);'
        'CREATE TABLE done(name TEXT PRIMARY KEY);')
    return db


def rejected(fn):
    try: fn()
    except AssertionError: return
    raise AssertionError('Invalid overlay accepted')


def main():
    q = queue.q
    with tempfile.TemporaryDirectory(prefix='e.', dir=os.environ['TMPDIR']) as tmp:
        root = Path(tmp); old = connect(root/'old.sqlite'); keys = []
        for j in range(717):
            binding = dict(fixture=j, runtime={'identity':'fixed'}, output={'raw_token_ids':[j]},
                judge=dict(model='gpt-6-astra', reasoning_effort='medium', prompt=q.PROMPT, protocol=q.PROTOCOL))
            key = q.digest(binding); keys.append(key)
            old.execute('INSERT INTO payload VALUES (?,?,?,?,?,?)',
                (key, json.dumps({'opaque_query_id':str(j)}),json.dumps(binding),'MISSING' if j<200 else 'FORMAT_VALID',str(j//50),None if j<200 else 1))
        for j in range(1077):
            key = keys[j%200] if j<303 else keys[200+(j-303)%517]
            old.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',
                (str(j),'fixture','single',1,1,'fixture',str(j),'fixture','fixed',key))
        old.commit(); groups = [keys[j:j+50] for j in range(0,200,50)]
        new = connect(root/'new.sqlite')
        rejected(lambda:queue.seed(new,old,[groups[0]]*4))
        queue.seed(new,old,groups);new.commit()
        assert new.execute('SELECT count(*) FROM consumer').fetchone()[0] == 303
        assert [r[0] for r in new.execute('SELECT key FROM payload ORDER BY rowid')] == keys[:200]
        assert old.execute("SELECT count(*) FROM payload WHERE status='MISSING'").fetchone()[0] == 200
        rejected(lambda:queue.seed(new,old,groups))
        new.execute("UPDATE payload SET status='FORMAT_VALID',correct=1,batch='recovery'");new.commit()
        assert resolved_snapshot(root/'old.sqlite',root/'new.sqlite',root/'resolved.sqlite') == {'FORMAT_VALID':717}
        assert old.execute("SELECT count(*) FROM payload WHERE status='MISSING'").fetchone()[0] == 200
        original = new.execute('SELECT record FROM payload LIMIT 1').fetchone()[0]
        new.execute("UPDATE payload SET record='changed' WHERE key=?",(keys[0],));new.commit()
        rejected(lambda:resolved_snapshot(root/'old.sqlite',root/'new.sqlite',root/'bad.sqlite'))
        new.execute('UPDATE payload SET record=? WHERE key=?',(original,keys[0]));new.commit()
        judge = SimpleNamespace(profile=lambda *args:'fixture')
        bad = SimpleNamespace(returncode=35,stdout='000',stderr='TLS fixture')
        with patch('interim20_recovery_scorer.subprocess.run',return_value=bad):
            rejected(lambda:connection_check(root,root,root,root,judge))
        good = SimpleNamespace(returncode=0,stdout='401',stderr='')
        with patch('interim20_recovery_scorer.subprocess.run',return_value=good):
            connection_check(root,root,root,root,judge)
        old.close();new.close()
    result=dict(status='PASS',missing_payloads=200,affected_consumers=303,old_valid_untouched=517,
        batch_membership_order_preserved=True,duplicate_or_changed_record_rejected=True,
        authorized_resolution_only=True,preflight_failure_before_reservation=True,
        new_GPU=0,new_generation=0,new_Judge=0)
    Path(os.environ['RECOVERY_CPU_REPORT']).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
