"""Read-only three-way diagnosis using fixed masks and existing accepted decisions."""
import itertools
import json
import os
from pathlib import Path
import re
import sqlite3
from collections import Counter,defaultdict

PARENT=Path(os.environ['REPLAY_PARENT']);RUN=Path(os.environ['RUN_ROOT'])
ARMS=('W0','CE192','SOURCE_REPLAY192')
PATTERNS=[''.join(map(str,x)) for x in itertools.product((0,1),repeat=3)]
read=lambda p:json.loads(Path(p).read_text())

def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def normalize(s):return ' '.join(re.findall(r'\w+',s.casefold()))
def kind(s):return normalize(s) if normalize(s) in ('yes','no') else 'open'

def possibilities(keys,scores):
    unknown=sorted({k for k in keys if scores[k] is None});out=set()
    for values in itertools.product((0,1),repeat=len(unknown)):
        filled={**scores,**dict(zip(unknown,values))}
        out.add(''.join(str(filled[k]) for k in keys))
    return sorted(out)

def table(rows):
    return dict(observations=len(rows),unique_queries=len({x['query_id'] for x in rows}),
        complete=sum(len(x['possible_patterns'])==1 for x in rows),
        unresolved=sum(len(x['possible_patterns'])>1 for x in rows),
        patterns={p:dict(lower=sum(x['possible_patterns']==[p] for x in rows),
            upper=sum(p in x['possible_patterns'] for x in rows)) for p in PATTERNS})

def main():
    assert possibilities(['x','x','z'],{'x':None,'z':1})==['001','111']
    assert possibilities(['a','b','c'],{'a':1,'b':0,'c':1})==['101']
    assert kind(' Yes! ' )=='yes' and kind('left lung')=='open'
    root=PARENT/'private/judge_replay_astra_medium'
    db=sqlite3.connect('file:'+str(root/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    payloads={x['key']:dict(x) for x in db.execute('SELECT * FROM payload')}
    retry=sqlite3.connect('file:'+str(PARENT/'private/judge_replay_astra_medium_recovery1/queue.sqlite')+'?mode=ro',uri=True);retry.row_factory=sqlite3.Row
    for x in retry.execute("SELECT * FROM payload WHERE status='FORMAT_VALID'"):
        assert payloads[x['key']]['status']=='MISSING'
        assert all(payloads[x['key']][k]==x[k] for k in ('record','binding'))
        payloads[x['key']]=dict(x)
    scores={k:x['correct'] for k,x in payloads.items()}
    assert sum(v is None for v in scores.values())==102
    consumers={(x['method'],x['query_id']):dict(x) for x in db.execute('SELECT * FROM consumer')}
    outputs={k:read(x['path']) for k,x in consumers.items()}
    tasks=read(PARENT/'private/BENCHMARK146_QUEUE.json')['tasks'];ledger=read(PARENT/'private/EVAL_LEDGER.json')
    expert_ids={t['edit_id']:t['anonymous_edit'] for t in tasks}
    sources=sorted({x['binding']['input']['source_group'] for x in outputs.values()})
    source_ids={s:f'S{i:03d}' for i,s in enumerate(sources)}
    rows=[]
    for t in tasks:
        qids=list(dict.fromkeys(q for e in t['events'] if e['task']=='T2G' for q in e['all_probe_query_ids']))
        qids=[q for q in qids if ledger['Base_correctness'][q] is False]
        for qid in qids:
            ds={a:outputs[a,qid] for a in ARMS};q=ds['W0']['binding']['input']
            keys=[consumers[a,qid]['payload_key'] for a in ARMS]
            experts={ds[a]['effective_expert'] for a in ARMS};assert len(experts)==1
            raw={a:ds[a]['R0']['raw_answer'] for a in ARMS}
            rows.append(dict(query_id=qid,owner=t['anonymous_edit'],expert=expert_ids[next(iter(experts))],
                source=source_ids[q['source_group']],dataset=q['dataset'],answer_type=kind(q['reference']),
                question=q['question'],reference=q['reference'],answers=raw,
                scores={a:scores[k] for a,k in zip(ARMS,keys)},possible_patterns=possibilities(keys,scores),
                image_path=q['image_path'],source_group=q['source_group'],
                output_type={a:kind(s) for a,s in raw.items()},
                lexical_equal={a:normalize(s)==normalize(q['reference']) for a,s in raw.items()}))
    assert len(rows)==557
    result=read(PARENT/'public/RECOVERED_RESULTS.json')
    for arm in ARMS:
        groups=defaultdict(list)
        for x in rows:groups[x['owner']].append(x['scores'][arm])
        bounds=[sum(sum((v if v is not None else missing) for v in values)/len(values) for values in groups.values())/len(groups)*100 for missing in (0,1)]
        prior=next(x for x in result['panels'] if x['arm']==arm and x['task']=='T2G')
        assert all(abs(a-b)<1e-8 for a,b in zip(bounds,prior['macro_bounds']))
    checks=[]
    for q in read(PARENT/'private/REPLAY_CHECK.json'):
        arms=('BASE',)+ARMS;qid=q['query_id']
        d=dict(q,answers={a:outputs[a,qid]['R0']['raw_answer'] for a in arms},
            scores={a:scores[consumers[a,qid]['payload_key']] for a in arms})
        assert all(v in (0,1) for v in d['scores'].values())
        d['Base_loss']=d['scores']['BASE']==1 and d['scores']['SOURCE_REPLAY192']==0
        d['open']=q['answer_kind']=='open';checks.append(d)
    assert sum(x['Base_loss'] for x in checks)==16 and sum(x['open'] for x in checks)==32
    strata={}
    for key in ('answer_type','dataset','source','expert','owner'):
        grouped=defaultdict(list)
        for row in rows:grouped[row[key]].append(row)
        strata[key]={k:table(v) for k,v in sorted(grouped.items())}
    public=dict(status='COMPLETE_EXISTING_OUTPUT_ANALYSIS',T2G=table(rows),strata=strata,
        score_order=list(ARMS),denominator='557 fixed-mask edit-query observations; unique queries reported separately; pattern bounds retain shared missing keys',
        source_CHECK=dict(observations=96,open=32,Base_correct_replay_wrong=16,
            Base_losses_by_W0_CE=dict(Counter(str(x['scores']['W0'])+str(x['scores']['CE192']) for x in checks if x['Base_loss'])),
            open_scores={a:sum(x['scores'][a] for x in checks if x['open']) for a in ('BASE',)+ARMS}),
        prior_panel_reproduction='PASS',new_training=0,new_generation=0,new_Judge_attempts=0,
        semantic_review='SEPARATE_PRIVATE_ANNOTATIONS_NOT_FORMAL_RESCORE')
    write(RUN/'private/T2G_CASES.json',rows);write(RUN/'private/CHECK_CASES.json',checks)
    write(RUN/'public/TRANSITIONS.json',public)
    print(json.dumps({k:v for k,v in public.items() if k!='strata'},ensure_ascii=False))

if __name__=='__main__':main()
