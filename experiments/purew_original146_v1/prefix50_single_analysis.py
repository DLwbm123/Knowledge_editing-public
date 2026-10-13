"""Read-only paired diagnostics: independent first50 singles versus prefix50."""
from collections import Counter
from itertools import product
import json
import math
import os
import sqlite3
import statistics as st
import sys
import time

sys.path.insert(0, os.environ['RUN_ROOT']+'/private/tools')
import astra_report as ar
from contract import ARM, METHODS

R,c,r=ar.RUN,ar.common,ar.r


def transitions(pairs,scores):
    """Enumerate shared unknown keys jointly, including repeated consumers."""
    unknown=sorted({k for a,b in pairs for k in (a,b) if scores[k] is None})
    assert len(unknown)<=11, 'Outside the frozen missing budget for this diagnostic'
    counts=[]
    for values in product((0,1),repeat=len(unknown)):
        assigned={**scores,**dict(zip(unknown,values))}
        counts.append(Counter(str(assigned[a])+str(assigned[b]) for a,b in pairs))
    return dict(pairs=len(pairs),shared_unknown_payloads=len(unknown),
                known_transitions=dict(Counter(str(scores[a])+str(scores[b]) for a,b in pairs if scores[a] is not None and scores[b] is not None)),
                unresolved_pairs=sum(scores[a] is None or scores[b] is None for a,b in pairs),
                count_bounds={key:[min(x[key] for x in counts),max(x[key] for x in counts)] for key in ('11','10','01','00')})


def summary(xs):
    return dict(n=len(xs),mean=st.mean(xs),median=st.median(xs),minimum=min(xs),maximum=max(xs)) if xs else None


def selfcheck():
    out=transitions([('m','m'),('m','z'),('o','m')],dict(m=None,z=0,o=1))
    assert out['count_bounds']=={'11':[0,2],'10':[1,1],'01':[0,0],'00':[0,2]}
    assert transitions([('a','b')],dict(a=1,b=0))['count_bounds']['10']==[1,1]


def main():
    selfcheck()
    assert (R/'private/outputs/bank_R0/0/50/COMPLETE.json').exists()
    assert (R/'private/single_R0_COMPLETE.json').exists()
    db=sqlite3.connect('file:'+str(R/'private/judge_purew146/queue.sqlite')+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row;db.execute('BEGIN')
    rows=list(db.execute("SELECT * FROM consumer WHERE (mode='single_R0' AND edit_order<=50) OR (mode='bank_R0' AND prefix=50)"))
    payloads={x['key']:x for x in db.execute('SELECT * FROM payload')}
    assert all(payloads[x['payload_key']]['status'] in ('FORMAT_VALID','MISSING') for x in rows)
    scores={k:x['correct'] for k,x in payloads.items()}
    tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'][:50]
    ledger=c.read(R/'private/EVAL_LEDGER.json')
    active={(t['native']['image_sha256'],t['native']['question']) for t in tasks}
    records=[];lookup={}
    for x in rows:
        if x['method']==ARM:
            d=c.read(x['path']);assert c.digest(d)==x['output_binding']
            assert json.loads(payloads[x['payload_key']]['binding'])['output']==d['R0']
        query=ledger['queries'][x['query_id']]
        records.append((dict(binding=dict(phase=dict(arm=x['method'],node=0,prefix=x['prefix']),input=query,mode=x['mode'],owner_order=x['edit_order']),active_target=(query['image_sha256'],query['question']) in active,effective_expert=None),x['payload_key']))
        key=(x['method'],x['mode'],x['edit_order'],x['query_id']);assert key not in lookup
        lookup[key]=x['payload_key']
    panels=[];contrasts=[];allcoef={};groups={};per_edit={}
    for method in METHODS:
        for task in ('T0','T1G','T2G','T1L','T2L'):
            for mode,prefix in [('single_R0',1),('bank_R0',50)]:
                m,coef,g=r.panel(records,scores,'RETRO146',method,0,mode,task,tasks,prefix)
                m.pop('route_ON');n=m['observations'];a=m['known_correct'];u=m['missing']
                assert n==dict(T0=50,T1G=196,T2G=195,T1L=2,T2L=17)[task]
                m['micro_bounds']=[100*a/n,100*(a+u)/n]
                if task.endswith('L'):m['Base_correct_to_wrong_count_bounds']=[n-a-u,n-a]
                panels.append(m);allcoef[method,mode,task]=coef;groups.update(g)
            a,b=allcoef[method,'bank_R0',task],allcoef[method,'single_R0',task]
            assert set(a)==set(b)
            delta={e:r.combine([a[e],{k:-v for k,v in b[e].items()}]) for e in a}
            per_edit[method,task]=delta
            bounds=r.score_bounds(r.combine([{k:v/len(delta) for k,v in unit.items()} for unit in delta.values()]),scores)
            pairs=[]
            for t in tasks:
                qids=list(dict.fromkeys(q for e in t['events'] if e['task']==task for q in e['all_probe_query_ids']))
                if task!='T0':qids=[q for q in qids if ledger['Base_correctness'][q]==task.endswith('L')]
                for q in qids:
                    query=ledger['queries'][q]
                    assert not task.endswith('L') or (query['image_sha256'],query['question']) not in active
                    pairs.append((lookup[method,'single_R0',t['order'],q],lookup[method,'bank_R0',0,q]))
            assert len(pairs)==n
            item=dict(method=method,task=task,bank_minus_single_macro_pp=[100*x for x in bounds],
                      edit_direction_counts=dict(definitely_worse=sum(r.score_bounds(v,scores)[1]<-1e-12 for v in delta.values()),definitely_better=sum(r.score_bounds(v,scores)[0]>1e-12 for v in delta.values()),definitely_equal=sum(max(abs(x) for x in r.score_bounds(v,scores))<1e-12 for v in delta.values())),transitions=transitions(pairs,scores))
            item['edit_direction_counts']['uncertain']=len(delta)-sum(item['edit_direction_counts'].values())
            if task=='T2G':item.update(ar.bootstrap(delta,scores,groups))
            contrasts.append(item)
    windows=[]
    for lo in range(1,51,10):
        ts=tasks[lo-1:lo+9];delta={t['edit_id']:per_edit[ARM,'T2G'][t['edit_id']] for t in ts}
        bound=r.score_bounds(r.combine([{k:v/10 for k,v in unit.items()} for unit in delta.values()]),scores)
        windows.append(dict(order_group=[lo,lo+9],T2G_bank_minus_single_macro_pp=[100*x for x in bound]))
    events={}
    for mode in ('single_R0','bank_R0'):
        events[mode]={i:[c.read(R/'private/steps'/mode/str(i)/(str(s)+'.json')) for s in range(1,161)] for i in range(1,51)}
        for i,es in events[mode].items():
            for s,e in enumerate(es,1):
                assert (e['owner'],e['step'],e['mode'])==(i,s,mode)
                assert len(e['source_losses'])==(3 if mode=='bank_R0' and i>1 else 2)
                assert all(math.isfinite(x) for x in e['source_losses']+[e['proposal_norm'],e['gradient_norm'],e['actual_native_norm']])
    diagnostics=[]
    for mode,owners in events.items():
        es=[e for unit in owners.values() for e in unit]
        out=dict(mode=mode,steps=len(es),gradient_norm=summary([e['gradient_norm'] for e in es]),
                 proposal_norm=summary([e['proposal_norm'] for e in es]),actual_native_norm=summary([e['actual_native_norm'] for e in es]),
                 sum_proposal_path_length=sum(e['proposal_norm'] for e in es),sum_native_path_length=sum(e['actual_native_norm'] for e in es),
                 clipped_steps=sum(e['gradient_norm']>5 for e in es),zero_native_writes=sum(e['changed_elements']==0 for e in es))
        for label,index in [('native',0),('FIT',1)]:
            out[label+'_first4_loss']=summary([st.mean(e['source_losses'][index] for e in unit[:4]) for unit in owners.values()])
            out[label+'_last4_loss']=summary([st.mean(e['source_losses'][index] for e in unit[-4:]) for unit in owners.values()])
        out['last4_both_current_losses_below_001_edits']=sum(all(st.mean(e['source_losses'][index] for e in unit[-4:])<.01 for index in (0,1)) for unit in owners.values())
        diagnostics.append(out)
    replay=[e for unit in events['bank_R0'].values() for e in unit if len(e['source_losses'])==3]
    last_replay={}
    for e in events['bank_R0'][50]:last_replay[(e['step']-1)%49+1]=e['source_losses'][2]
    assert len(last_replay)==49
    history=dict(replay_events=len(replay),all_history_loss=summary([e['source_losses'][2] for e in replay]),
                 owner50_last_visit_loss=summary(list(last_replay.values())),owner50_last_visit_ge_001=sum(x>=.01 for x in last_replay.values()),
                 history_loss_ge_01=sum(e['source_losses'][2]>=.1 for e in replay),
                 history_loss_ge_1=sum(e['source_losses'][2]>=1 for e in replay))
    lost=[t for t in tasks if r.score_bounds(per_edit[ARM,'T2G'][t['edit_id']],scores)[1]<-1e-12]
    lost_diagnostics=dict(definitely_worse_edits=len(lost),
        own_bank_last4_native_loss=summary([st.mean(e['source_losses'][0] for e in events['bank_R0'][t['order']][-4:]) for t in lost]),
        own_bank_last4_FIT_loss=summary([st.mean(e['source_losses'][1] for e in events['bank_R0'][t['order']][-4:]) for t in lost]))
    result=dict(status='PAIRED_FIRST50_SINGLE_VS_PREFIX50',checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),panels=panels,contrasts=contrasts,order_groups=windows,training_diagnostics=diagnostics,history_diagnostics=history,T2G_worse_edit_training=lost_diagnostics,
                same_first50_and_original_masks=True,bootstrap_replicates=10000,bootstrap_seed=20260912,
                interpretation='Single is an independently edited Base comparator, never insertion-time performance. Contrasts combine inherited W, history-native rehearsal and sequential exposure; no causal gradient-conflict attribution.',
                limits=['Losses are pre-update source-only CE, not evaluation accuracy or fixed prefix50 losses','Sum of update norms is path length, not net displacement; single sums combine 50 separate trajectories','No gradient directions, T2G insertion measurements or saved historical full W available; no new forwards or Judge calls','Historical locality is small-support development evidence; patient independence unknown'])
    c.write(R/'public/PREFIX50_VS_SINGLE.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
