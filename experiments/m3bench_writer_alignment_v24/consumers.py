"""Finite coverage, exact frozen-verdict reuse and paired three-arm summaries."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import re
ARMS=('A','B','C')
CLASSES=('CORRECT','INCORRECT','UNCERTAIN','ABSTAIN')
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def validate(root):
    config=read(root/'DATA_ADMISSION.json');queries=read(root/'QUERY_ROWS.json')
    expected={f"{(t['order']-1)%5}/{t['key']}/outputs/{a}/{r['query_id']}.json" for t in config['tasks'] for a in ARMS for r in t['eval_rows']}
    actual={str(p.relative_to(root)) for p in root.glob('[0-4]/E*/outputs/*/*.json')}
    assert len(expected)==648 and actual==expected,'Missing/extra method outputs'
    actualbase=[p.stem for p in (root/'base').glob('*/*.json')]
    assert len(actualbase)==len(set(actualbase))==180 and set(actualbase)==set(queries)
    return dict(status='PASS',method_outputs=648,base_outputs=180)

def blind(root,prior):
    validate(root)
    oldpacket={x['code']:x for x in read(prior/'BLIND_PACKET.json')['items']}
    oldverdicts=read(prior/'BLIND_REUSE.json');oldverdicts.update(read(prior/'BLIND_VERDICTS.json'))
    oldqueries=read(prior/'QUERY_ROWS.json');old={}
    for path,code in read(prior/'BLIND_MAPPING.json').items():
        item=oldpacket[code];qid=Path(path).stem
        key=(oldqueries[qid]['image_sha256'],item['question'],item['reference'],item['answer'])
        if key in old:assert old[key]['label']==oldverdicts[code]['label']
        old[key]=oldverdicts[code]
    paths=sorted((root/'base').glob('*/*.json'))+sorted(root.glob('[0-4]/E*/outputs/*/*.json'))
    unique={};items=[];mapping={};reuse={}
    for path in paths:
        x=read(path);b=x['binding'];key=(b['image_sha256'],b['question'],b['reference'],x['raw_answer'])
        if key not in unique:
            code='V%06d'%(len(unique)+1);unique[key]=code
            items.append(dict(code=code,question=b['question'],reference=b['reference'],answer=x['raw_answer']))
            if key in old:reuse[code]=dict(old[key],reason='Exact frozen V22 input/reference/answer reuse: '+old[key]['reason'])
        mapping[str(path.relative_to(root))]=unique[key]
    write(root/'BLIND_PACKET.json',dict(status='ALL_OUTPUTS_HIDDEN',classes=CLASSES,items=items,
          instruction='Fixed-reference four-class review only; no inferred method, no reference repair.'))
    write(root/'BLIND_MAPPING.json',mapping);write(root/'BLIND_REUSE.json',reuse)
    return dict(status='READY_FOR_HIDDEN_REVIEW',unique=len(items),frozen_reuse=len(reuse),new_required=len(items)-len(reuse))

def summarize(root):
    validate(root);mapping=read(root/'BLIND_MAPPING.json');judged=read(root/'BLIND_REUSE.json')
    judged.update(read(root/'BLIND_VERDICTS.json'))
    assert set(judged)==set(mapping.values()),'Every unique answer requires a verdict'
    assert all(v.get('label') in CLASSES and v.get('reason') for v in judged.values())
    def label(p):return judged[mapping[str(p.relative_to(root))]]['label']
    config=read(root/'DATA_ADMISSION.json');base={p.stem:p for p in (root/'base').glob('*/*.json')}
    table={};edit_scores={};norm=lambda s:re.sub(r'\s+',' ',s.strip()).rstrip('.').casefold()
    for arm in ARMS:
        roles=defaultdict(Counter);damage=Counter();copy=Counter();peredit={}
        for task in config['tasks']:
            folder=root/str((task['order']-1)%5)/task['key'];scores=defaultdict(Counter)
            tr=read(folder/arm/'TRAIN_COMPLETE.json');assert tr['accepted_steps']==160 and tr['save_load_exact']
            for row in task['eval_rows']:
                path=folder/'outputs'/arm/(row['query_id']+'.json');v=label(path);b=label(base[row['query_id']])
                for role in row['roles']:
                    roles[role][v]+=1;scores[role][v]+=1
                    if role in ('T1L','T2L') and b=='CORRECT':
                        damage['Base_correct_denominator']+=1;damage['lost_correct']+=v!='CORRECT';damage['explicit_incorrect']+=v=='INCORRECT'
                    if role=='T1L':
                        copy['denominator']+=1;copy['source_target_copy']+=norm(read(path)['raw_answer'])==norm(task['source']['reference'])
            peredit[task['key']]={k:dict(v) for k,v in scores.items()}
        table[arm]=dict(roles={k:dict(v) for k,v in roles.items()},damage=dict(damage),T1L_source_copy=dict(copy),training_updates=1600)
        edit_scores[arm]=peredit
    def c(arm,role):return table[arm]['roles'][role].get('CORRECT',0)
    paired={}
    for left,right in [('B','A'),('C','B')]:
        paired[left+'-'+right]={role:c(left,role)-c(right,role) for role in ('T0','T1G','T2G','T1L','T2L','REPHRASE_DIAGNOSTIC')}
    result=dict(status='COMPLETE_SCORED_DEVELOPMENT',table=table,paired_correct_differences=paired,per_edit=edit_scores,
                independent_confirmation=False,natural_routing_evaluated=False,review='Assistant fixed-reference review, not independent clinical adjudication')
    write(root/'RESULTS.json',result);return result

def selfcheck():
    import tempfile
    with tempfile.TemporaryDirectory() as name:
        p=Path(name);write(p/'DATA_ADMISSION.json',dict(tasks=[]));write(p/'QUERY_ROWS.json',{})
        try:validate(p)
        except AssertionError:pass
        else:raise AssertionError('Missing data accepted')
    return dict(status='PASS',missing_output_rejected=True)
