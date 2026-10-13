"""Audit the source-only RAW32 endpoint without adding model or scoring calls."""
import os
import json
import sqlite3
from pathlib import Path
import common as c
RUN=Path(os.environ['RUN_ROOT'])


def main():
    result=c.read(RUN/'public/RAW32_RESULTS.json');lock=c.read(RUN/'private/SHORT32_LOCK.json')
    assert result['decision']=='STOP_RAW32_NO_NATIVE_REPAIR'
    assert not result['heldout_evaluated'] and not result['protected_training_started']
    starts=[c.read(p) for p in (RUN/'private').glob('START_CHAIN_*.json')]
    for s in starts:
        p=Path('/proc')/str(s['pid'])/'stat'
        assert not p.exists() or p.read_text().split()[21]!=s['start_ticks'] or p.read_text().split()[2]=='Z'
    assert not list((RUN/'private').glob('FAILURE*'))
    source=[c.read(RUN/'private/raw_results'/f'{i}.json') for i in range(1,9)]
    assert all(s['lock']==c.digest(lock) and s['updates']==32 and s['backwards']==64 and s['source_forwards']==77 for s in source)
    assert all(s['mechanical']['status']=='PASS' and s['mechanical']['logits_max_abs']==0 and s['all_four_cores_updated'] for s in source)
    outputs={str(p):c.read(p) for p in (RUN/'private/outputs/RAW32').glob('*/*.json')};assert len(outputs)==40
    assert all(d['lock']==c.digest(lock) and d['EOS'] and not d['at_cap'] and any(x['active_residual_norm']>0 for x in d['generation_trace']) for d in outputs.values())
    counts=[c.read(RUN/'private'/f'RAW32_COUNTS_{i}.json') for i in range(6)]
    forwards=sum(x['forwards'] for x in counts);assert forwards==616+sum(d['forwards'] for d in outputs.values())
    assert sum(x['backwards'] for x in counts)==512 and sum(x['updates'] for x in counts)==256 and sum(x['generations'] for x in counts)==40
    q=RUN/'private/judge_raw32_astra_medium';db=sqlite3.connect('file:'+str(q/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    rows=list(db.execute('SELECT c.path,p.binding,p.status,p.correct FROM consumer c JOIN payload p ON c.payload_key=p.key'))
    assert len(rows)==40 and all(r['status']=='FORMAT_VALID' for r in rows) and {r['path'] for r in rows}==set(outputs)
    old={r['path']:r['correct'] for r in c.read(Path(os.environ['GENERATION_PARENT'])/'private/EDIT_BASELINE_QUALIFICATION.json')['rows'] if r['arm']=='BASE'}
    transitions={name:dict(same_text=0,changed_text=0,retained=0,new_damage=0,repaired=0,still_wrong=0) for name in ('NATIVE','FIT')}
    for r in rows:
        d=outputs[r['path']];b=c.read(d['baseline_path']);name='NATIVE' if d['index']==0 else 'FIT';v=transitions[name]
        for k in ('query_id','question','reference','image_sha256'):assert d['binding']['input'][k]==b['binding']['input'][k]
        assert d['binding']['judge_input']==b['binding']['judge_input']
        same=d['R0']['raw_answer']==b['R0']['raw_answer'];v['same_text' if same else 'changed_text']+=1
        before,after=old[d['baseline_path']],r['correct'];v[('retained' if after else 'new_damage') if before else ('repaired' if after else 'still_wrong')]+=1
        if same:assert after==before
    ledger=c.read(RUN/'RESOURCE_LEDGER.json');before=c.read(RUN/'private/INHERITED_COST.json')
    assert len(ledger['gpu_sessions'])==6 and all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
    attempts=ledger['Judge_attempts']-before['Judge_attempts'];assert attempts==24
    batches=ledger['Astra_RAW32_batches'];assert len(batches)==1 and len(batches[0]['keys'])==24 and batches[0]['status']=='FORMAT_VALID'
    for p in (q/'evidence').glob('*.json'):
        e=c.read(p)['evidence'];assert e['status']=='FORMAT_VALID' and e['actual_model']=='gpt-6-astra' and e['reasoning_effort']=='medium'
        assert not e['tool_event_types'] and not e.get('errors') and all(e['isolation_checks'].values())
    plan=c.read(RUN/'private/RAW32_DELETION_PLAN.json');assert len(plan['paths'])==8
    assert all(Path(p).parent==RUN/'private/raw_candidates' and not Path(p).exists() for p in plan['paths'])
    value=dict(status='PASS',experts=8,updates=256,backwards=512,forwards=forwards,source_forwards=616,new_generations=40,
        source_free_generation_forwards=forwards-616,all_four_cores_updated=True,mechanical_passed=8,teacher_generation_prefix_max_abs=0.,
        nonzero_generation_residual_all40=True,all_EOS=True,at_cap=0,GPU_sessions_ended=6,recorded_remote_processes_ended=len(starts),
        transitions=transitions,actual_edit_gain_mean=sum(s['edit_gain'] for s in source)/8,
        actual_edit_gain_range=[min(s['edit_gain'] for s in source),max(s['edit_gain'] for s in source)],
        new_Judge=24,inherited_Judge=16,valid_payloads=40,missing=0,pending_or_reserved=0,retries=0,
        raw_packages_deleted=8,bytes_deleted=plan['bytes'],historical_assets_deleted=0,
        heldout_access=False,protected_training_started=False,step_extension=False)
    c.write(RUN/'public/RAW32_REVIEW_AUDIT.json',value);print(json.dumps(value))


if __name__=='__main__':main()
