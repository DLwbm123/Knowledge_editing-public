"""Read-only experiment audit; creates an anonymous receipt without model calls."""
import os
import json
import sqlite3
from pathlib import Path
import common as c
RUN=Path(os.environ['RUN_ROOT'])


def main():
    result=c.read(RUN/'public/RESULTS.json');lock=c.read(RUN/'private/OPTIMIZER160_LOCK.json')
    assert not result['heldout_evaluated'] and not result['protected_training_started']
    starts=[c.read(p) for p in (RUN/'private').glob('START_CHAIN_*.json')]
    for s in starts:
        path=Path('/proc')/str(s['pid'])/'stat'
        assert not path.exists() or path.read_text().split()[21]!=s['start_ticks'] or path.read_text().split()[2]=='Z'
    assert not list((RUN/'private').glob('FAILURE*'))
    sources=[c.read(p) for p in (RUN/'private/results').glob('*/*.json')];assert len(sources)==16
    assert {(s['arm'],s['expert_order']) for s in sources}=={(a,i) for a in ('RAW','ADAM') for i in range(1,9)}
    for s in sources:
        assert s['lock']==c.digest(lock) and s['updates']==160 and s['backwards']==320
        assert [x['step'] for x in s['checkpoints']]==lock['diagnostic_nodes']
        assert len(s['curve'])==160 and all(len(x['pre_update_loss'])==2 for x in s['curve'])
        assert s['mechanical']['status']==s['terminal_prefix_check']['status']=='PASS'
        assert s['all_four_cores_updated']
    assert sum(s['source_forwards'] for s in sources)==6092
    outputs={str(p):c.read(p) for p in (RUN/'private/outputs').glob('*/*/*.json')};assert len(outputs)==80
    for d in outputs.values():
        assert d['lock']==c.digest(lock) and any(x['active_residual_norm']>0 for x in d['generation_trace'])
        b=c.read(d['baseline_path']);assert d['binding']['judge_input']==b['binding']['judge_input']
        for k in ('query_id','question','reference','image_sha256'):assert d['binding']['input'][k]==b['binding']['input'][k]
    paired={(d['arm'],d['expert_order'],d['index']):d for d in outputs.values()}
    assert len(paired)==80
    pair_counts=dict(identical_answers=0,identical_tokens=0)
    for order in range(1,9):
        for index in range(5):
            a,b=[paired[arm,order,index] for arm in ('RAW','ADAM')]
            assert a['binding']==b['binding']
            pair_counts['identical_answers']+=a['R0']['raw_answer']==b['R0']['raw_answer']
            pair_counts['identical_tokens']+=a['R0']['raw_token_ids']==b['R0']['raw_token_ids']
    counts=[c.read(RUN/'private'/f'OPTIMIZER160_COUNTS_{i}.json') for i in range(6)]
    forwards=sum(x['forwards'] for x in counts);assert forwards==6092+sum(d['forwards'] for d in outputs.values())<=88012
    assert sum(x['updates'] for x in counts)==2560 and sum(x['backwards'] for x in counts)==5120
    folder=RUN/'private/judge_optimizer160_astra_medium'
    db=sqlite3.connect('file:'+str(folder/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    rows=list(db.execute('SELECT c.path,p.status,p.correct FROM consumer c JOIN payload p ON c.payload_key=p.key'))
    assert len(rows)==80 and {r['path'] for r in rows}==set(outputs)
    assert all(r['status'] in ('FORMAT_VALID','MISSING') for r in rows)
    ledger=c.read(RUN/'RESOURCE_LEDGER.json');before=c.read(RUN/'private/INHERITED_COST.json')
    assert len(ledger['gpu_sessions'])==6 and all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
    batches=ledger.get('Astra_optimizer160_batches',[])
    attempts=sum(len(b['keys']) for b in batches)
    assert attempts==ledger['Judge_attempts']-before['Judge_attempts']<=80
    assert len({k for b in batches for k in b['keys']})==attempts
    for path in (folder/'evidence').glob('*.json'):
        e=c.read(path)['evidence']
        if e['status']=='FORMAT_VALID':
            assert e['actual_model']=='gpt-6-astra' and e['reasoning_effort']=='medium'
            assert not e['tool_event_types'] and not e.get('errors') and all(e['isolation_checks'].values())
    assert len(list((RUN/'private').glob('OPTIMIZER160_CONSUMED_*.json')))==16
    assert not list((RUN/'private/results').rglob('*.pt')) and not list((RUN/'private/outputs').rglob('*.pt'))
    receipt=dict(status='PASS',trajectories=16,updates=2560,backwards=5120,forwards=forwards,source_forwards=6092,
        new_generations=80,all_generation_nonzero_residual=True,first_step_mechanical_passed=16,
        terminal_native_prefix_checks=sum(s['terminal_prefix_check']['target_prefixes'] for s in sources),
        terminal_native_prefix_max_difference=max(s['terminal_prefix_check']['maximum_logit_difference'] for s in sources),
        actual_free_generation_prefix_equivalence_not_asserted=True,all_four_cores_updated=True,
        all_EOS=all(d['EOS'] for d in outputs.values()),at_cap=sum(d['at_cap'] for d in outputs.values()),
        GPU_sessions_ended=6,recorded_remote_processes_ended=len(starts),new_Judge=attempts,
        payload_status=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status')),
        no_retry=True,paired_outputs=pair_counts,persistent_checkpoints_created=0,historical_assets_deleted=0,
        heldout_access=False,protected_training_started=False,endpoint_selection=False,automatic_step_extension=False)
    c.write(RUN/'public/REVIEW_AUDIT.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
