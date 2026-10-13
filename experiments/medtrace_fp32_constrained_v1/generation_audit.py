"""Read-only completion audit; retain missing ratings and count exact output sharing."""
import json
import os
import sqlite3
from pathlib import Path
import common as c

RUN=Path(os.environ['RUN_ROOT'])


def main():
    result=c.read(RUN/'public/GEN_RESULTS.json');lock=c.read(RUN/'private/GEN_LOCK.json')
    ledger=c.read(RUN/'RESOURCE_LEDGER.json');before=c.read(RUN/'private/INHERITED_COST.json')
    complete=c.read(RUN/'private/GENERATION_COMPLETE.json');parent=Path(os.environ['CONSTRAINED_PARENT'])
    starts=[c.read(p) for p in (RUN/'private').glob('START_CHAIN_*.json')]
    for s in starts:
        path=Path('/proc')/str(s['pid'])/'stat'
        assert not path.exists() or path.read_text().split()[21]!=s['start_ticks'] or path.read_text().split()[2]=='Z'
    sessions=ledger['gpu_sessions'];assert len(sessions)==6 and all(s.get('ended_epoch') for s in sessions)
    assert lock['parent_freeze']['epoch']<lock['epoch']<min(s['started_epoch'] for s in sessions)
    assert max(s['ended_epoch'] for s in sessions)<=complete['epoch']
    assert not list((RUN/'private').glob('FAILURE*'))
    values={str(p):c.read(p) for p in (RUN/'private/outputs').glob('*/*/*.json')}
    assert len(values)==2464 and sum(v['new_generation'] for v in values.values())==1656
    for v in values.values():
        assert v['lock']==c.digest(lock)
        if not v['new_generation']:
            source=values[v['alias_of']]
            assert v['arm']=='MATCHED_RAW' and source['arm']=='RAW' and source['R0']==v['R0']
            assert source['binding']==v['binding']
    counts=[c.read(RUN/'private'/f'GEN_COUNTS_{i}.json') for i in range(6)]
    assert sum(x['forwards'] for x in counts)==sum(v['forwards'] for v in values.values() if v['new_generation'])==complete['forwards']==41697
    assert all(v['EOS'] and not v['at_cap'] for v in values.values())
    queue=RUN/'private/judge_generation_astra_medium';db=sqlite3.connect('file:'+str(queue/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    rows=list(db.execute('SELECT c.*,p.status,p.correct FROM consumer c JOIN payload p ON c.payload_key=p.key'))
    assert len(rows)==2464 and {r['path'] for r in rows}==set(values)
    assert dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status'))=={'FORMAT_VALID':96,'MISSING':40}
    assert all(r['status']==('FORMAT_VALID' if r['folder']=='HELDOUT' else 'MISSING') for r in rows)
    inherited=c.read(queue/'TEXT_INHERITANCE.json');assert len(inherited)==2304
    parents={x['parent']:c.read(x['parent']) for x in inherited}
    held_exact_tokens=0
    for x in inherited:
        d=values[x['path']];old=parents[x['parent']]
        for k in ('query_id','question','reference','image_sha256'):assert d['binding']['input'][k]==old['binding']['input'][k]
        assert d['R0']['raw_answer']==old['R0']['raw_answer']
        held_exact_tokens+=d['R0']['raw_token_ids']==old['R0']['raw_token_ids']
    edits=[d for d in values.values() if d['role']=='EDIT'];bases={(d['expert_order'],d['index']):d for d in edits if d['arm']=='BASE'}
    edit_exact=sum(d['R0']==bases[d['expert_order'],d['index']]['R0'] for d in edits if d['arm']!='BASE')
    assert edit_exact==120
    batches=ledger['Astra_constrained_generation_batches'];assert len(batches)==2
    assert sorted(len(b['keys']) for b in batches)==[5,35] and len({k for b in batches for k in b['keys']})==40
    assert ledger['Judge_attempts']-before['Judge_attempts']==40
    for b in batches:
        assert b['status']=='FAILED_NO_RETRY' and b['transport_failure']
        evidence=c.read(queue/'evidence'/(b['id']+'.json'))['evidence']
        assert evidence['actual_model']=='gpt-6-astra' and evidence['reasoning_effort']=='medium'
        assert not evidence['tool_event_types'] and all(evidence['isolation_checks'].values())
        assert 'Connection failed: error sending request' in json.dumps(evidence['errors'])
    assert len(list((queue/'workers').glob('*/SCORER_DONE.json')))==4
    deletion=c.read(RUN/'private/DELETION_PLAN.json');assert len(deletion['paths'])==8
    assert all(Path(p).parent==parent/'private/candidates' and not Path(p).exists() for p in deletion['paths'])
    assert result['decision']=='INCOMPLETE_SEMANTIC_EVIDENCE' and result['edit_missing']==dict.fromkeys(('BASE','RAW','BOUNDED','MATCHED_RAW'),40)
    audit=dict(status='PASS',consumers=2464,new_generations=1656,exact_parameter_aliases=808,LLM_forwards=41697,
        GPU_sessions_ended=6,recorded_remote_processes_ended=len(starts),source_freeze_before_generation=True,
        all_outputs_EOS=True,at_cap=0,heldout_inherited_consumers=2304,heldout_exact_parent_tokens=held_exact_tokens,
        edit_candidates_exact_FP32_Base_text_and_tokens=edit_exact,edit_candidate_observations=120,
        payloads_valid=96,payloads_missing=40,payloads_pending_or_reserved=0,missing_consumers=160,
        failed_transport_batches=2,failed_batch_sizes=[5,35],failure='Connection failed: error sending request',
        new_Judge_attempts=40,retries=0,scorer_done_receipts=4,packages_deleted=8,
        deletion_after_all_GPU_consumers=True,missing_is_not_incorrect=True)
    c.write(RUN/'public/GEN_REVIEW_AUDIT.json',audit)
    print(json.dumps(audit))


if __name__=='__main__':main()
