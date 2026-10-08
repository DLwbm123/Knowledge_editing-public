"""Posthoc aggregate evidence audit; no new generation, scoring or selection."""
from collections import Counter
import sqlite3
import statistics
import direction24_report as report

c,p,RUN,r=report.c,report.p,report.RUN,report.r


def main():
    assert (RUN/'private/REPORT_COMPLETE.json').exists()
    result=c.read(RUN/'public/RESULTS.json');lock=c.read(RUN/'private/DIRECTION_LOCK.json')
    db=sqlite3.connect('file:'+str(report.queue.q.ROOT/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    scores={x['key']:x['correct'] for x in db.execute('SELECT * FROM payload')}
    assert all(v in (0,1) for v in scores.values())
    records=[];mapping={}
    for row in db.execute("SELECT * FROM consumer WHERE mode='bank_R0'"):
        d=c.read(row['path']);assert c.digest(d)==row['output_binding']
        d=dict(d,binding=dict(d['binding'],arm=row['method'],mode=row['mode'],phase=dict(d['binding']['phase'],arm=row['method'],node=0,prefix=146)))
        records.append((d,row['payload_key']));mapping[row['method'],row['query_id']]=d,row['payload_key']
    routes=c.read(RUN/'private/ROUTES.json');cohorts=[]
    for scope,key in [('affected','selected_edits'),('new16','new_edits'),('pilot8','pilot_edits')]:
        chosen=set(lock[key]);subset=[(d,k) for d,k in records if routes[d['binding']['input']['query_id']]['effective_expert'] in chosen]
        a,ca,groups=r.panel(subset,scores,scope,'RAW_DIR_540',0,'bank_R0','T2G',p.tasks(),146)
        b,cb,_=r.panel(subset,scores,scope,'W0',0,'bank_R0','T2G',p.tasks(),146)
        diff={e:r.score_bounds(r.combine([ca[e],{k:-v for k,v in cb[e].items()}]),scores)[0] for e in ca}
        gain=sum(v>1e-12 for v in diff.values());loss=sum(v< -1e-12 for v in diff.values());n=len(diff)
        assert abs(100*sum(diff.values())/n-(a['macro']-b['macro']))<1e-10
        positive_sources=len({groups[e] for e,v in diff.items() if v>1e-12})
        cohorts.append(dict(scope=scope,editing_experts=len(chosen),evaluation_edit_units=n,positive_edit_units=gain,negative_edit_units=loss,zero_edit_units=n-gain-loss,positive_source_clusters=positive_sources,
            exact_bootstrap_probability_of_no_positive_unit=((n-gain)/n)**n,
            micro_correct_before=b['known_correct'],micro_correct_after=a['known_correct'],observations=a['observations']))
    # Query-level score transitions and termination, without publishing query IDs.
    ledger=c.read(RUN/'private/EVAL_LEDGER.json');qids=set()
    for t in p.tasks():
        qids.update(q for event in t['events'] if event['task']=='T2G' for q in event['all_probe_query_ids'] if ledger['Base_correctness'][q] is False)
    transitions=Counter()
    for q in qids:
        if routes[q]['effective_expert'] not in set(lock['selected_edits']):continue
        old,ok=mapping['W0',q];new,nk=mapping['RAW_DIR_540',q]
        delta=scores[nk]-scores[ok]
        transitions['gain' if delta>0 else 'loss' if delta<0 else 'unchanged']+=1
        if delta>0:
            transitions['gain_old_at_cap' if len(old['R0']['raw_token_ids'])==1024 else 'gain_old_below_cap']+=1
            transitions['gain_new_at_cap' if len(new['R0']['raw_token_ids'])==1024 else 'gain_new_below_cap']+=1
    gs=[g for t in result['training'] for step in t['curve'] for g in step['groups']]
    assert len(gs)==10240 and all(abs(g['actual_norm']-g['Adam_candidate_norm'])<=g['roundoff_bound'] for g in gs)
    audit=dict(status='PASS',posthoc=True,no_new_training_generation_or_Judge=True,cohorts=cohorts,T2G_unique_query_transitions=dict(transitions),
        groups=len(gs),zero_gradient_fallback=sum(g['zero_gradient_Adam_fallback'] for g in gs),all_group_norms_within_bound=True,
        median_Adam_negative_gradient_cosine=statistics.median(g['Adam_negative_gradient_cosine'] for g in gs),
        minimum_actual_negative_gradient_cosine=min(g['actual_negative_gradient_cosine'] for g in gs),
        native_FIT_targets=80,all_native_FIT_argmax_correct=all(x['all_argmax_correct'] for t in result['training'] for x in t['final']),
        maximum_final_CE=max(x['nll'] for t in result['training'] for x in t['final']),
        interpretation='Descriptive audit only. Evaluation edit units differ from edited experts under frozen routing. No revised interval or promotion rule; termination association is not isolated causality.')
    assert audit['minimum_actual_negative_gradient_cosine']>.99999
    c.write(RUN/'public/POSTHOC_EVIDENCE_AUDIT.json',audit)
    print(audit)


if __name__=='__main__':main()
