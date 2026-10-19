"""Complete routed comparisons and blinded exact-identity verdict reuse."""
import json,collections,random
from pathlib import Path
from route_math_v19 import ARMS,validate,same_input
from audit_v19 import read,write

def collect(root,v18):
    rows=[]
    for g in (0,1):
        assert read(root/str(g)/'EXIT.json')['code']==0 and read(root/str(g)/'COMPLETE.json')['status']=='COMPLETE'
        rows+=read(root/str(g)/'CONSUMERS.json')
    tasks=read(v18/'DATA_ADMISSION.json')['tasks'];qs=read(v18/'EVAL_ROWS.json')+[t[k] for t in tasks for k in ('source','cross','rephrase')];validate(rows,[(a,q['query_id']) for a in ARMS for q in qs]);return rows,tasks,qs

def blind(root,v18):
    rows,tasks,qs=collect(root,v18);known={};base=read(v18/'BASE_REUSE.json');base_labels={x['query_id']:x['label'] for x in read(v18/'BASE_VERDICTS.json')['rows']}
    def identity(answer):
        b=answer['binding'];return (b['image_sha256'],b['question'],b['reference'],answer['raw_answer'])
    def add(answer,label):
        key=identity(answer)
        if key in known:assert known[key]==label,'Conflicting same-input same-answer frozen labels'
        known[key]=label
    for q,p in base.items():add(read(p),base_labels[q])
    assert (v18/'BLIND_VERDICTS.json').exists(),'V18 scores required before inheriting ratings'
    aliases=read(v18/'BLIND_ALIASES.json');verdicts=read(v18/'BLIND_VERDICTS.json');assert set(aliases)==set(verdicts)
    for code,consumers in aliases.items():
        for task,arm,qid in consumers:
            g=next(i for i,t in enumerate(tasks) if t['key']==task);add(read(v18/str(g)/task/'outputs'/arm/(qid+'.json')),verdicts[code]['label'])
    groups={}
    for row in rows:groups.setdefault(identity(read(row['output_path'])),[]).append([row['arm'],row['query_id']])
    keys=list(groups);random.Random(20261019).shuffle(keys);materials=[];mapping={};reuse={}
    for i,key in enumerate(keys):
        code='R%04d'%(i+1);materials.append(dict(code=code,question=key[1],reference=key[2],answer=key[3]));mapping[code]=groups[key]
        if key in known:reuse[code]=dict(label=known[key],reason='Exact input/reference/raw-answer matches frozen Base or V18 judgement')
    write(root/'BLIND_MATERIALS.json',materials);write(root/'BLIND_ALIASES.json',mapping);write(root/'EXACT_VERDICT_REUSE.json',reuse);return dict(consumers=len(rows),unique_answers=len(keys),ratings_reused=len(reuse),new_review=len(keys)-len(reuse))

def summarize(root,v18):
    rows,tasks,qs=collect(root,v18);aliases=read(root/'BLIND_ALIASES.json');verdicts=read(root/'BLIND_VERDICTS.json');assert set(verdicts)==set(aliases);labels={tuple(key):verdicts[code]['label'] for code,keys in aliases.items() for key in keys};assert all(v in ('CORRECT','INCORRECT','UNCERTAIN','ABSTAIN') for v in labels.values())
    vlabels={tuple(key):read(v18/'BLIND_VERDICTS.json')[code]['label'] for code,keys in read(v18/'BLIND_ALIASES.json').items() for key in keys};lookup={(r['arm'],r['query_id']):r for r in rows};base_labels={r['query_id']:r['label'] for r in read(v18/'BASE_VERDICTS.json')['rows']};metrics={}
    for arm in ARMS:
        m={}
        for kind in ('source','cross','rephrase'):
            events=[]
            for task in tasks:
                q=task[kind]['query_id'];forced=vlabels[task['key'],arm,q];natural=labels[arm,q];selected=lookup[arm,q]['route']['logical_edit_id'];events.append(dict(forced=forced,natural=natural,base=base_labels[q],selected_own=selected==task['key'],OFF=selected is None,forced_correct_natural_not_correct=forced=='CORRECT' and natural!='CORRECT',forced_not_correct=forced!='CORRECT'))
            m[kind]=dict(denominator=2,forced_correct=sum(e['forced']=='CORRECT' for e in events),natural_correct=sum(e['natural']=='CORRECT' for e in events),forced_correct_natural_not_correct=sum(e['forced_correct_natural_not_correct'] for e in events),forced_not_correct=sum(e['forced_not_correct'] for e in events),own_selection=sum(e['selected_own'] for e in events),OFF=sum(e['OFF'] for e in events))
        ids=[q['query_id'] for q in read(v18/'EVAL_ROWS.json')];correct=[q for q in ids if base_labels[q]=='CORRECT'];m['retention']=dict(unique_inputs=22,Base_correct_denominator=len(correct),post_labels=dict(collections.Counter(labels[arm,q] for q in correct)),activation=sum(lookup[arm,q]['route']['activated'] for q in ids),clinical_scope_negative_verified=False)
        metrics[arm]=m
    counts=collections.Counter()
    for g in (0,1):counts.update(read(root/str(g)/'EXIT.json')['counts'])
    out=dict(status='COMPLETE',natural_consumers=140,unique_queries=28,independent_edit_sources=2,arms=metrics,counts=dict(counts),evidence_scope='TWO_EXPERT_DEVELOPMENT_DIAGNOSIS_NOT_146_OR_PATIENT_INDEPENDENT',scores_missing=0);write(root/'RESULTS_PRIVATE.json',out);return out

def retire(root):
    import time
    v18=Path(read(root/'RUN_CONFIG.json')['V18_RUN']);collect(root,v18)
    assert len(read(root/'BE_SINGLE_REUSE_INDEX.json'))==100
    paths=[]
    for a in read(root/'START.json'):
        p=Path('/proc')/str(a['pid']);assert not p.exists() or (p/'stat').read_text().split()[21]!=str(a['start_ticks']);assert read(root/str(a['gpu'])/'EXIT.json')['code']==0;paths.append(root/str(a['gpu'])/'QUERY_KEYS.pt')
    assert all(p.is_file() and not p.is_symlink() and root.resolve() in p.resolve().parents for p in paths)
    receipt=dict(count=len(paths),bytes=sum(p.stat().st_size for p in paths),reason='All routing outputs and bindings persisted; no model-state copies created',time=time.time())
    for p in paths:p.unlink()
    write(root/'RETIREMENT.json',receipt);return receipt
