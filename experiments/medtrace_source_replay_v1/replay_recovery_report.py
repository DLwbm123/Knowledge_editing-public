"""Merge only authorized same-input valid retries; preserve both original queues."""
import json
import os
import sqlite3
import sys
from pathlib import Path

RUN=Path(os.environ['RUN_ROOT'])
sys.path.insert(0,str(RUN/'private/tools'))
import replay_report as report


def main():
    q=report.queue.q
    root=RUN/'private/judge_replay_astra_medium_recovery1'
    assert (root/'ALL_WORKERS_COMPLETE.json').exists()
    def rows(folder):
        db=sqlite3.connect('file:'+str(folder/'queue.sqlite')+'?mode=ro',uri=True)
        db.row_factory=sqlite3.Row
        result={x['key']:dict(x) for x in db.execute('SELECT * FROM payload')}
        db.close()
        return result
    old=rows(q.ROOT);extra=rows(root)
    expected={k for k,v in old.items() if v['status']=='MISSING'}
    assert set(extra)==expected==set(q.read(root/'AUTHORIZATION.json')['keys'])
    assert len(expected)==302 and len(old)==2358
    scores={k:v['correct'] for k,v in old.items()};valid=0;evidence={}
    for k,row in extra.items():
        assert row['status'] in ('FORMAT_VALID','MISSING')
        assert row['binding']==old[k]['binding'] and row['record']==old[k]['record']
        if row['status']=='FORMAT_VALID':
            bid=row['batch']
            if bid not in evidence:
                saved=q.read(root/'evidence'/(bid+'.json'));ev=saved['evidence']
                assert ev['status']=='FORMAT_VALID' and ev['actual_model']=='gpt-6-astra'
                assert ev['reasoning_effort']=='medium' and ev['exit_code']==0
                assert not ev.get('errors') and not ev['tool_event_types']
                assert all(ev['isolation_checks'].values())
                assert ev['input_binding']==q.digest(saved['batch'])
                evidence[bid]=(saved['batch']['records'],{x['opaque_query_id']:x['is_correct'] for x in q.validate(saved['batch'],saved['response'])})
            records,decisions=evidence[bid];record=json.loads(row['record'])
            assert record in records and int(decisions[record['opaque_query_id']])==row['correct']
            assert row['correct'] in (0,1)
            scores[k]=row['correct'];valid+=1
    assert sum(old[k]['correct'] is not None for k in old)==2056
    assert all(scores[k]==v['correct'] for k,v in old.items() if v['status']=='FORMAT_VALID')
    ledger=q.read(RUN/'RESOURCE_LEDGER.json')
    batches=ledger['Astra_replay_recovery_batches']
    charged=[k for b in batches for k in b['keys']]
    assert len(charged)==len(set(charged))==302 and set(charged)==expected
    assert ledger['Judge_attempts']==11421+302
    recovery=dict(attempted=302,valid=valid,missing=302-valid,original_valid_unchanged=2056,
        original_missing_preserved=302,additional_attempts_per_payload=1,
        same_binding_and_evidence_validation='PASS',Judge_before=11421,Judge_after=ledger['Judge_attempts'])
    report.main(scores,recovery)


if __name__=='__main__':main()
