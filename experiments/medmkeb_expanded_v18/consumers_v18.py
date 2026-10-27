"""Strict finite output completeness and explicit owned-state lifecycle."""
import json,collections
from pathlib import Path
ARMS=('SOURCE','CE','KL','CE_PROJ','RESERVE','BALANCEDIT','BALANCEDIT_CE')
LABELS={'CORRECT','INCORRECT','UNCERTAIN','ABSTAIN'}
def validate(records,expected,verdicts=None):
    keys=[(r['task'],r['arm'],r['query_id']) for r in records];assert len(keys)==len(set(keys)) and set(keys)==set(expected),'Missing/extra/duplicate output'
    assert all(isinstance(r['answer']['raw_token_ids'],list) and isinstance(r['answer']['raw_answer'],str) and 'binding' in r['answer'] for r in records)
    if verdicts is not None:
        assert set(verdicts)==set(keys) and all(v in LABELS for v in verdicts.values()),'Missing/invalid semantic verdict'
    return dict(outputs=len(records),EOS=sum(r['answer']['EOS'] for r in records),cap=sum(r['answer']['at_cap'] for r in records))
def training(result):
    accepted=result['accepted_steps'];remaining=result['remaining_NA']
    assert 0<=accepted<=160 and remaining==160-accepted and len(result['curve'])==accepted
    assert result['status'] in ('COMPLETE','NO_FEASIBLE_LEARNING_STEP')
    if result['status']=='COMPLETE':assert accepted==160
    return True

def retire(root):
    import os,time
    records,expected=collect(root);assert validate(records,expected)['outputs']==350
    launch=json.loads((root/'FORMAL_START.json').read_text());paths=[]
    for rec in launch:
        gpu=rec['gpu'];out=root/str(gpu);assert json.loads((out/'FORMAL_EXIT.json').read_text())['code']==0
        proc=Path('/proc')/str(rec['pid']);assert not proc.exists() or (proc/'stat').read_text().split()[21]!=rec['start_ticks']
        assert json.loads((out/'STUDY_COMPLETE.json').read_text())['status']=='COMPLETE'
        task=json.loads((root/'DATA_ADMISSION.json').read_text())['tasks'][gpu]['key'];folder=out/task
        assert json.loads((folder/'MODEL_CONSUMERS_COMPLETE.json').read_text())['status']=='PASS'
        for arm in ARMS[:5]:
            training(json.loads((folder/arm/'TRAIN_COMPLETE.json').read_text()));paths.extend([folder/arm/'active.pt',folder/arm/'final.pt'])
        paths.extend(folder/(arm+'.pt') for arm in ARMS[5:])
    paths.append(root/'TEACHER_CACHE.pt');assert len(paths)==25 and not (root/'RETIREMENT.json').exists()
    assert all(p.is_file() and not p.is_symlink() and root.resolve() in p.resolve().parents for p in paths)
    receipt=dict(status='RETIRED',count=len(paths),bytes=sum(p.stat().st_size for p in paths),paths=[str(p) for p in paths],reason='all frozen model consumers and outputs complete; semantic review independent of weights',recoverable_copy=False,reconstruction='requires new authorized training',time=time.time())
    for p in paths:p.unlink()
    (root/'RETIREMENT.json').write_text(json.dumps(receipt,indent=2)+'\n');return {k:receipt[k] for k in ['status','count','bytes']}

def selfcheck():
    key=('E','SOURCE','Q');r=dict(task='E',arm='SOURCE',query_id='Q',answer=dict(raw_token_ids=[2],raw_answer='x',binding={},EOS=True,at_cap=False));assert validate([r],[key],{key:'CORRECT'})['EOS']==1
    for records,verdicts in [([],{}),([r],{}),([r,r],{key:'CORRECT'})]:
        try:validate(records,[key],verdicts)
        except AssertionError:pass
        else:raise AssertionError('Missingness/duplication accepted')
    assert training(dict(status='NO_FEASIBLE_LEARNING_STEP',accepted_steps=2,remaining_NA=158,curve=[{},{}]))
    try:training(dict(status='COMPLETE',accepted_steps=2,remaining_NA=158,curve=[{},{}]))
    except AssertionError:pass
    else:raise AssertionError('Early stop disguised as full trajectory')
    return dict(status='PASS',missing_outputs_rejected=True,missing_scores_rejected=True,duplicates_rejected=True)
def collect(root):
    read=lambda p:json.loads(Path(p).read_text());tasks=read(root/'DATA_ADMISSION.json')['tasks'];evaluation=read(root/'EVAL_ROWS.json');expected=[];records=[]
    for gpu,task in enumerate(tasks):
        folder=root/str(gpu)/task['key'];assert read(folder/'FROZEN.json')['all_terminals_frozen']
        ids=[r['query_id'] for r in evaluation]+[task[k]['query_id'] for k in ['source','cross','rephrase']]
        for arm in ARMS:
            for q in ids:
                expected.append((task['key'],arm,q));p=folder/'outputs'/arm/(q+'.json')
                if p.exists():records.append(dict(task=task['key'],arm=arm,query_id=q,answer=read(p)))
    return records,expected
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['selfcheck','collect','retire']);p.add_argument('--root',type=Path);a=p.parse_args()
    if a.mode=='selfcheck':print(selfcheck())
    elif a.mode=='retire':print(retire(a.root))
    else:
        rows,expected=collect(a.root);print(validate(rows,expected))

# These consumers run after model work; raw answers, aliases and labels stay private.
def blind(root,base_verdict_path):
    import random
    records,expected=collect(root);validate(records,expected);labels={r['query_id']:r['label'] for r in json.loads(Path(base_verdict_path).read_text())['rows']};reuse=json.loads((root/'BASE_REUSE.json').read_text());groups={}
    for r in records:
        answer=r['answer'];b=answer['binding'];key=(b['image_sha256'],b['question'],b['reference'],answer['raw_answer']);groups.setdefault(key,[]).append((r['task'],r['arm'],r['query_id']))
    keys=list(groups);random.Random(20261019).shuffle(keys);materials=[];aliases={};reused={}
    for i,key in enumerate(keys):
        code='B%04d'%(i+1);materials.append(dict(code=code,question=key[1],reference=key[2],answer=key[3]));aliases[code]=groups[key]
        q=groups[key][0][2];base=json.loads(Path(reuse[q]).read_text());b=base['binding']
        if (b['image_sha256'],b['question'],b['reference'],base['raw_answer'])==key:reused[code]=dict(label=labels[q],reason='Exact same input/reference/raw answer as frozen Base verdict')
    (root/'BLIND_MATERIALS.json').write_text(json.dumps(materials,indent=2)+'\n');(root/'BLIND_ALIASES.json').write_text(json.dumps(aliases,indent=2)+'\n');(root/'EXACT_BASE_VERDICT_REUSE.json').write_text(json.dumps(reused,indent=2)+'\n')
    return dict(all_outputs=len(records),unique_answers=len(materials),frozen_Base_reused=len(reused),new_review_required=len(materials)-len(reused))

def summarize(root):
    records,expected=collect(root);aliases=json.loads((root/'BLIND_ALIASES.json').read_text());verdict=json.loads((root/'BLIND_VERDICTS.json').read_text());assert set(verdict)==set(aliases)
    labels={tuple(key):verdict[code]['label'] for code,keys in aliases.items() for key in keys};out=validate(records,expected,labels)
    base={q:json.loads(Path(p).read_text()) for q,p in json.loads((root/'BASE_REUSE.json').read_text()).items()};base_labels=json.loads((root/'BASE_VERDICTS.json').read_text());base_labels={x['query_id']:x['label'] for x in base_labels['rows']};eval_ids=[x['query_id'] for x in json.loads((root/'EVAL_ROWS.json').read_text())];tasks=json.loads((root/'DATA_ADMISSION.json').read_text())['tasks'];metrics={}
    for arm in ARMS:
        m={kind:sum(labels[(t['key'],arm,t[kind]['query_id'])]=='CORRECT' for t in tasks) for kind in ['source','cross','rephrase']};m['edit_denominator']=len(tasks);pairs=[(t['key'],q) for t in tasks for q in eval_ids if base_labels[q]=='CORRECT'];m['Base_correct_pairs']=len(pairs);m['post_labels_on_Base_correct']=dict(collections.Counter(labels[(t,arm,q)] for t,q in pairs));m['full_eval_labels']=dict(collections.Counter(labels[(t['key'],arm,q)] for t in tasks for q in eval_ids));m['full_eval_pairs']=len(tasks)*len(eval_ids)
        control={tq:labels[(tq[0],'SOURCE',tq[1])] for tq in pairs};current={tq:labels[(tq[0],arm,tq[1])] for tq in pairs};m['avoided_clear_damage']=sum(control[p]=='INCORRECT' and current[p]=='CORRECT' for p in pairs);m['additional_clear_damage']=sum(control[p]=='CORRECT' and current[p]=='INCORRECT' for p in pairs);m['protection_identifiable']=any(x=='INCORRECT' for x in control.values());metrics[arm]=m
    counts=collections.Counter()
    for gpu in [0,1]:
        for name in ['EXIT.json','FORMAL_EXIT.json']:
            result=json.loads((root/str(gpu)/name).read_text());assert result['code']==0;counts.update(result['counts'])
    out.update(arms=metrics,counts=dict(counts),evidence_scope='IMAGE_SOURCE_ISOLATED_FORCED_SINGLE_EDIT_BENCHMARK_NOT_PATIENT_INDEPENDENT',support_composition='4 modality, 2 color, 2 plane questions',TT_natural_routing_efficacy='NOT_EVALUATED')
    (root/'RESULTS_PRIVATE.json').write_text(json.dumps(out,indent=2)+'\n');return out
