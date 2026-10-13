"""Complete DEV router block, followed only by a frozen passing REG candidate."""
import copy,time,fcntl
from pathlib import Path
from resources import ROOT,read,write
from judge_protocol import packet,NAMESPACE,read_scores
def normalize(r,cohort):
    r=copy.deepcopy(r);r['arm']='R0';r['cohort']=cohort
    if r['mode']=='holdout':r.update(mode='EXPOSED_REGRESSION',edit='EXPOSED_REGRESSION',task='EXPOSED_REGRESSION')
    return r
def references(phase):
    dst=ROOT/f'private/references/{phase}.json'
    if dst.exists():return read(dst)
    assert read(ROOT/'public/INSERTION_HAZARD_AUDIT.json')['status']=='COMPLETE','Do not unblind before latent freeze'
    pred=read(ROOT/'PREDECESSOR.json');rows=[]
    if phase=='DEV':
        rows=[normalize(r,phase) for r in read(Path(pred['pr9_root'])/'private/references/DEV.json') if r['arm']=='H']
    else:
        assert read(ROOT/'public/JOINT_ROUTER_DECISION.json')['REG_allowed']
        old=Path(pred['root']);files=[]
        for j in read(old/'QUEUE.json'):
            if j['mode']=='train' and j['order'] in range(25,49):files.extend((old/'jobs'/j['id']/'AH/single').glob('CONSUMERS.json'))
            if j['mode']=='bank' and j['method']=='AH' and j['id'].startswith('P4-'):files.extend((old/'jobs'/j['id']).rglob('CONSUMERS.json'))
        files.extend((old/'auxiliary/holdout/REG24-AH').rglob('CONSUMERS.json'))
        oldscores={p.stem:read(p)['is_correct'] for p in (old/'private/judge/scores').glob('*.json')}
        for p in files:
            for original in read(p):
                r=normalize(original,phase);r['frozen_base_correct']=oldscores.get(original['base_judge_key'])
                for field in ['judge_key','base_judge_key']:
                    rec=read(old/'private/judge/pending'/f'{original[field]}.json')['record'];pk=packet(rec['question'],rec['gold_answer'],rec['raw_base_answer']);path=ROOT/NAMESPACE/'pending'/f'{pk["key"]}.json'
                    if not path.exists():write(path,pk)
                    r[field]=pk['key']
                rows.append(r)
    unique={}
    for r in rows:
        k=tuple(r[f] for f in ['mode','prefix','edit','task','input_id'])
        if k in unique:assert unique[k]['judge_key']==r['judge_key']
        unique[k]=r
    rows=list(unique.values());assert rows
    frozen={r['input_id']:r['frozen_base_correct'] for r in rows if r['mode']=='EXPOSED_REGRESSION'}
    assert len(frozen)==47 and sum(v is True for v in frozen.values())==35
    write(dst,rows);return rows
def candidates(phase):
    rows=[]
    for j in read(ROOT/'QUEUE.json'):
        if j['phase']==phase:
            for p in (ROOT/'jobs'/j['id']).rglob('CONSUMERS.json'):rows.extend(read(p))
    frozen={r['input_id']:r['frozen_base_correct'] for r in references(phase) if r['mode']=='EXPOSED_REGRESSION'}
    for r in rows:
        if r['mode']=='EXPOSED_REGRESSION':r['frozen_base_correct']=frozen[r['input_id']]
    return rows
def methods(phase):
    return ['R0','RCAP','NEG0','SAFE'] if phase=='DEV' else ['R0',read(ROOT/'public/JOINT_ROUTER_DECISION.json')['selected']]
def enqueue(phase):
    queue=[];orders=list(range(1,25)) if phase=='DEV' else list(range(25,49))
    def add(j):queue.append(dict(j,id=phase+'-'+str(len(queue)),phase=phase,status='PENDING'))
    for method in methods(phase):
        for start in [0,8,16]:add(dict(mode='single',method=method,orders=orders[start:start+8]))
        add(dict(mode='bank',method=method,orders=orders,prefixes=[4,8,12,24]))
        add(dict(mode='bank',exposed=True,method=method,orders=orders,prefixes=[4,8,12,24]))
    references(phase)
    with (ROOT/'QUEUE.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);q=read(ROOT/'QUEUE.json');assert not any(j['phase']==phase for j in q);write(ROOT/'QUEUE.json',q+queue)
def score_closed(phase):
    rows=references(phase)+candidates(phase);keys={r[k] for r in rows for k in ['judge_key','base_judge_key']};missing=set(read(ROOT/NAMESPACE/'JUDGE_MISSING_LOCK.json')['keys'])
    return bool(rows) and not keys-set(read_scores(ROOT))-missing and all(a['status']!='RESERVED' for a in read(ROOT/'RESOURCE_LEDGER.json')['judge_attempts'])
def decide():
    dst=ROOT/'public/JOINT_ROUTER_DECISION.json'
    if dst.exists():return read(dst)['REG_allowed']
    from decision import joint
    import torch
    features=torch.load(ROOT/'private/ROUTER_FEATURES.pt',map_location='cpu',weights_only=False)
    result=joint(candidates('DEV'),references('DEV'),read_scores(ROOT),read(ROOT/'private/BASE_MASKS.json'),features['banks']['DEV'])
    write(dst,result);return result['REG_allowed']
