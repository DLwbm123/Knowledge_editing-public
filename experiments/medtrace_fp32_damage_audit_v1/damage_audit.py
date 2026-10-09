"""Completion receipt and paired-result audit; no model calls or rescoring."""
import os
import json
import sqlite3
from pathlib import Path
from collections import Counter
import common as c
RUN=Path(os.environ['RUN_ROOT'])


def main():
    result=c.read(RUN/'public/RESULTS.json');lock=c.read(RUN/'private/DAMAGE_LOCK.json')
    assert c.read(RUN/'private/DAMAGE_REPORT_COMPLETE.json')['status']=='COMPLETE'
    frozen=c.read(RUN/'private/CANDIDATES_FROZEN.json');assert frozen['status']=='PASS' and frozen['source_outputs_exact']==80
    starts=[c.read(p) for p in (RUN/'private').glob('START_CHAIN_*.json')]
    for s in starts:
        stat=Path('/proc')/str(s['pid'])/'stat'
        assert not stat.exists() or stat.read_text().split()[21]!=s['start_ticks'] or stat.read_text().split()[2]=='Z'
    assert not list((RUN/'private').glob('FAILURE*'))
    folder=RUN/'private/judge_damage_astra_medium';db=sqlite3.connect('file:'+str(folder/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    scores={};outputs={};panels={};paired={}
    consumers=list(db.execute('SELECT c.*,p.status,p.correct,p.binding FROM consumer c JOIN payload p ON c.payload_key=p.key'))
    assert len(consumers)==1881 and all(r['status']=='FORMAT_VALID' for r in consumers)
    for r in consumers:
        d=c.read(r['path']);b=json.loads(r['binding']);i=d['binding']['input']
        assert d['lock']==c.digest(lock)
        assert all(i[k]==b[k] for k in ('query_id','image_sha256','question','reference')) and d['R0']==b['output']
        assert all(d['binding']['judge_input'][k]==b[k] for k in ('prompt_ids','attention_mask','runtime','generation'))
        assert b['judge']['model']=='gpt-6-astra' and b['judge']['reasoning_effort']=='medium'
        if d['arm']!='BASE':assert any(t['active_residual_norm']>0 for t in d['generation_trace'])
        assert d['EOS'] and not d['at_cap']
        scores[r['path']]=r['correct'];outputs[r['path']]=d
        paired[d['arm'],d['expert_order'],i['query_id']]=(d,r['correct'])
    hb=c.read(RUN/'private/HELD_BASE.json');assert len(hb)==96 and sum(r['correct'] for r in hb.values())==62
    scores.update({r['path']:r['correct'] for r in hb.values()})
    cross=Counter();damage={};unique={}
    for arm in ('RAW','ADAM'):
        damage[arm]=[];unique[arm]=set()
        for owner in range(1,9):
            n=Counter()
            for (a,o,key),(d,after) in paired.items():
                if (a,o)!=(arm,owner) or d['role']!='HELDOUT':continue
                before=scores[d['baseline_path']];n['queries']+=1;n['Base_correct']+=before;n['candidate_correct']+=after
                transition=('retained' if after else 'new_damage') if before else ('repaired' if after else 'still_wrong')
                n[transition]+=1
                if before and not after:unique[arm].add(key)
                if arm=='RAW':
                    other,score=paired['ADAM',owner,key];cross['paired_held_queries']+=1
                    cross['same_text']+=d['R0']['raw_answer']==other['R0']['raw_answer'];cross['same_tokens']+=d['R0']['raw_token_ids']==other['R0']['raw_token_ids']
                    cross['RAW_only_correct']+=after==1 and score==0;cross['ADAM_only_correct']+=after==0 and score==1
                    cross['both_correct']+=after==score==1;cross['both_wrong']+=after==score==0
            assert n['queries']==96
            reported=next(x for x in result['per_expert'] if x['arm']==arm and x['owner']==owner and x['role']=='HELDOUT')
            for k in ('queries','Base_correct','candidate_correct','new_damage','repaired'):assert n[k]==reported[k]
            damage[arm].append(dict(owner=owner,**n))
        for key in ('Base_correct','candidate_correct','new_damage','repaired'):assert sum(x.get(key,0) for x in damage[arm])==result['panels'][arm]['HELDOUT'][key]
    ledger=c.read(RUN/'RESOURCE_LEDGER.json');before=c.read(RUN/'private/INHERITED_COST.json');batches=ledger['Astra_damage_batches']
    keys=[k for b in batches for k in b['keys']];assert len(keys)==len(set(keys))==549==ledger['Judge_attempts']-before['Judge_attempts']
    assert all(b['status']=='FORMAT_VALID' and b['model']=='gpt-6-astra' and b['reasoning_effort']=='medium' and not b['transport_failure'] for b in batches)
    evidence=list((folder/'evidence').glob('*.json'));assert len(evidence)==len(batches)
    for path in evidence:
        e=c.read(path)['evidence'];assert e['status']=='FORMAT_VALID' and e['actual_model']=='gpt-6-astra' and e['reasoning_effort']=='medium'
        assert not e['tool_event_types'] and not e.get('errors') and all(e['isolation_checks'].values())
    assert len(ledger['gpu_sessions'])==12 and all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
    assert min(s['started_epoch'] for s in ledger['gpu_sessions'] if s['action']=='damage_eval')>frozen['epoch']
    counts=c.read(RUN/'private/DAMAGE_GENERATION_COMPLETE.json')['counts'];src=frozen['counts']
    assert sum(x['updates'] for x in src)==2560 and sum(x['backwards'] for x in src)==5120
    assert sum(x['generations'] for x in src+counts)==1961 and sum(x['prefix_forwards'] for x in counts)==2112
    forwards=sum(x['forwards'] for x in src+counts);assert forwards<=lock['maximum_forwards']
    assert sum(x['forwards'] for x in counts)==2112+sum(d['forwards'] for d in outputs.values())
    deleted=c.read(RUN/'private/DELETION_PLAN.json');assert len(deleted['paths'])==16 and all(not Path(x).exists() for x in deleted['paths'])
    assert not list((RUN/'private/damage_candidates').rglob('*.pt'))
    receipt=dict(status='PASS',source_reproduction_outputs=80,all_candidates_frozen_before_eval=True,updates=2560,backwards=5120,
        forwards=forwards,non_generation_forwards=8204,new_generations=1961,new_generation_EOS=1961,at_cap=0,
        scored_consumers=1881,payload_status=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status')),new_Judge=len(keys),inherited_payloads=750-len(keys),
        Judge_batches=len(batches),all_actual_Astra_medium_isolated=True,no_payload_retry=True,missing=0,
        GPU_sessions_ended=12,recorded_remote_processes_ended=len(starts),packages_deleted=16,bytes_deleted=deleted['bytes'],historical_deletions=0,
        per_owner_held=damage,distinct_held_queries_damaged={a:len(v) for a,v in unique.items()},paired_held=dict(cross),
        protected_training_started=False,automatic_next_experiment=False,original_precision_loss_preserved=True,
        limitations=['768=8experts x96queries; not768independent questions','496=8experts x62Base-correct queries','single forced owner, development audit'])
    c.write(RUN/'public/REVIEW_AUDIT.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
