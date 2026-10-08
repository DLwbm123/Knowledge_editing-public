"""Frozen main masks, all-query regressions, and held-out source stress tests."""
import random
import sqlite3
from collections import Counter, defaultdict
import astra_report as ar
import replay as exp
import replay_queue as queue

c,RUN,r=exp.c,exp.RUN,ar.r
LABELS=('W0',)+exp.ARMS


def transitions(pairs):
    return dict(Counter('missing' if a is None or b is None else
        'gain' if b>a else 'loss' if b<a else 'unchanged' for a,b in pairs))


def summary(rows):
    # Equal three-query source groups; preserve missing scores as bounds.
    groups=defaultdict(list)
    for group,value in rows:groups[group].append(value)
    assert groups
    bounds=[[sum(v or 0 for v in values)/len(values),
        sum(1 if v is None else v for v in values)/len(values)] for values in groups.values()]
    means=[sum(x[k] for x in bounds)/len(bounds)*100 for k in (0,1)]
    rng=random.Random(20261008);samples=[]
    for _ in range(2000):
        draw=rng.choices(bounds,k=len(bounds))
        samples.append([sum(x[k] for x in draw)/len(draw)*100 for k in (0,1)])
    return dict(observations=len(rows),source_groups=len(groups),known_correct=sum(v==1 for _,v in rows),
        missing=sum(v is None for _,v in rows),macro_bounds=means,
        source_ci=[sorted(x[0] for x in samples)[49],sorted(x[1] for x in samples)[1949]])


def main():
    root=queue.q.ROOT;assert (root/'ALL_WORKERS_COMPLETE.json').exists()
    db=sqlite3.connect('file:'+str(root/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    scores={x['key']:x['correct'] for x in db.execute('SELECT * FROM payload')}
    records=[];mapping={}
    for row in db.execute('SELECT * FROM consumer'):
        d=c.read(row['path']);assert c.digest(d)==row['output_binding']
        d=dict(d,binding=dict(d['binding'],arm=row['method'],mode=row['mode'],
            phase=dict(d['binding']['phase'],arm=row['method'])))
        records.append((d,row['payload_key']));mapping[row['method'],row['query_id']]=d,row['payload_key']
    assert len(records)==4923
    panels=[];coefs={};groups={};tasks=exp.p.tasks()
    for label in LABELS:
        for task in ('T0','T1G','T2G','T1L','T2L'):
            m,coef,group=r.panel(records,scores,'RETRO146',label,0,'bank_R0',task,tasks,146)
            m.update(ar.bootstrap(coef,scores,group));panels.append(m)
            coefs[label,task]=coef;groups.update(group)
    prior=c.read(exp.PARENT/'public/RESULTS.json')
    for panel in panels[:5]:
        old=next(x for x in prior['panels'] if x['arm']=='MARGIN_002' and x['task']==panel['task'])
        for key in ('macro_bounds','observations','missing','edit_units','known_correct'):assert panel[key]==old[key]
    contrasts=[]
    for a,b in [(exp.ARMS[1],'W0'),(exp.ARMS[1],exp.ARMS[0])]:
        for task in ('T0','T1G','T2G','T1L','T2L'):
            aa,bb=coefs[a,task],coefs[b,task];assert set(aa)==set(bb)
            coef={e:r.combine([aa[e],{k:-v for k,v in bb[e].items()}]) for e in aa}
            bounds=r.score_bounds(r.combine([{k:v/len(coef) for k,v in x.items()} for x in coef.values()]),scores) if coef else [None,None]
            contrasts.append(dict(a=a,b=b,task=task,paired_edits=len(coef),
                delta_bounds_pp=[None if v is None else v*100 for v in bounds],**ar.bootstrap(coef,scores,groups)))
    changes={arm:transitions([(scores[mapping['W0',qid][1]],scores[mapping[arm,qid][1]])
        for qid in exp.p.queries(146)]) for arm in exp.ARMS}
    check_rows=c.read(RUN/'private/REPLAY_CHECK.json');checks=[]
    for arm in ('BASE',)+LABELS:
        values={row['query_id']:scores[mapping[arm,row['query_id']][1]] for row in check_rows}
        reference={row['query_id']:scores[mapping['BASE',row['query_id']][1]] for row in check_rows}
        retained=[row for row in check_rows if reference[row['query_id']]==1]
        exact=sum(mapping[arm,row['query_id']][0]['Base_token_consistency'] for row in check_rows)/len(check_rows)
        checks.append(dict(arm=arm,all_accuracy=summary([(row['source_group'],values[row['query_id']]) for row in check_rows]),
            by_answer_type={kind:summary([(row['source_group'],values[row['query_id']]) for row in check_rows if row['answer_kind']==kind]) for kind in ('yes','no','open')},
            Base_correct_retention=summary([(row['source_group'],values[row['query_id']]) for row in retained]) if retained else None,
            Base_missing=sum(v is None for v in reference.values()),
            transitions_from_Base=transitions([(reference[q],values[q]) for q in values]),
            exact_Base_tokens=exact,expert_assignments=32,forced_expert=True,natural_bank_route=False))
    natural=[]
    for label in LABELS:
        ds=[d for d,_ in records if d['binding']['arm']==label and d['binding']['input'].get('role')=='CHECK']
        assert len(ds)==4
        natural.append(dict(arm=label,observations=4,Base_token_consistency=sum(d['Base_token_consistency'] for d in ds)/4,
            mean_KL=sum(d['U_KL'] for d in ds)/4,route_ON=sum(d['effective_expert'] is not None for d in ds)))
    resource=c.read(RUN/'RESOURCE_LEDGER.json');inherited=c.read(RUN/'private/INHERITED_COST.json')
    training=[c.read(RUN/'private/weights'/arm/t['anonymous_edit']/'TRAINING.json') for t in tasks for arm in exp.ARMS]
    assert len(training)==292 and all(x['updates']==192 and not x['Base_gradient'] for x in training)
    result=dict(status='TERMINAL_WITH_MISSING' if any(v is None for v in scores.values()) else 'COMPLETE',
        panels=panels,contrasts=contrasts,all_main_query_transitions=changes,source_CHECK=checks,natural_CHECK=natural,
        baseline_reproduction='PASS',training_experts=292,training_updates=sum(x['updates'] for x in training),
        new_outputs=3410,scoring=c.read(root/'READY.json'),missing_payloads=sum(v is None for v in scores.values()),
        Judge_attempts=resource['Judge_attempts'],new_Judge_attempts=resource['Judge_attempts']-inherited['Judge_attempts'],
        GPU_process_hours=c.used()/3600,new_GPU_process_hours=(c.used()-inherited['gpu_seconds_used'])/3600,
        deletion=c.read(RUN/'private/DELETION.json')['all_consumers_complete'],
        independent_confirmation=False,medical_scope_qualification=False,
        interpretation='Source-annotated development experiment; new source CHECK forces one of first32 experts, not natural bank routing. No automatic promotion.')
    c.write(RUN/'public/RESULTS.json',result);exp.p.done('REPORT_COMPLETE');exp.p.progress('RESULTS_COMPLETE_REVIEW_PUBLICATION_PENDING')


if __name__=='__main__':
    main()
