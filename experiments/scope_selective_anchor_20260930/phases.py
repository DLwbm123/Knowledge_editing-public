"""Finite complete blocks, model-uniform references, CHECK-only selection."""
import os,json,time,copy
from pathlib import Path
from resources import read,write,ROOT
from judge_protocol import packet,NAMESPACE,read_scores,digest
from policy import select_grid,joint
PILOT=[4,5,6,11,12,15,16,21]
REF={'E_orig':'E_orig','A0':'A0','AH':'H','AHS_01':'S'}

def normalize(r):
    r=copy.deepcopy(r)
    if r['mode']=='holdout':r.update(edit='EXPOSED_HOLDOUT',task='EXPOSED_HOLDOUT')
    return r

def references(phase):
    dst=ROOT/f'private/references/{phase}.json'
    if dst.exists():return read(dst)
    old=Path(read(ROOT/'PREDECESSOR.json')['root']);tasks=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'];orders=PILOT if phase=='PILOT' else list(range(1,25)) if phase=='DEV' else list(range(25,49))
    files=[]
    for j in read(old/'QUEUE.json'):
        if j['mode']=='train' and j['order'] in orders:
            for m in REF:
                files.extend((old/'jobs'/j['id']/m/'single').glob('CONSUMERS.json'))
                if phase=='PILOT':files.extend((old/'jobs'/j['id']/m/'check').glob('CONSUMERS.json'))
        if j['mode']=='bank' and j['method'] in REF and j['id'].startswith({'PILOT':'P1-','DEV':'P2-','REG':'P4-'}[phase]):files.extend((old/'jobs'/j['id']).rglob('CONSUMERS.json'))
    if phase!='PILOT':
        for m in REF:files.extend((old/'auxiliary/holdout'/f'{phase}24-{m}').rglob('CONSUMERS.json'))
    scores={p.stem:read(p)['is_correct'] for p in (old/'private/judge/scores').glob('*.json')};rows={}
    for file in files:
        for original in read(file):
            r=normalize(original);r['arm']=REF[r['arm']];r['frozen_base_correct']=scores.get(original['base_judge_key'])
            for field in ['judge_key','base_judge_key']:
                record=read(old/'private/judge/pending'/f'{original[field]}.json')['record'];p=packet(record['question'],record['gold_answer'],record['raw_base_answer']);path=ROOT/NAMESPACE/'pending'/f'{p["key"]}.json'
                if not path.exists():write(path,p)
                r[field]=p['key']
            identity=tuple(r[k] for k in ['arm','mode','prefix','edit','task','input_id'])
            if identity in rows:assert rows[identity]['judge_key']==r['judge_key']
            rows[identity]=r
    assert rows
    result=list(rows.values());write(dst,result);return result

def arms():
    s=read(ROOT/'public/BETA_GAMMA_SELECTION.json');b=s['beta']['selected'];g=s['gamma']['selected'];assert b is not None and g is not None
    suffix='01' if b==.1 else '1'
    return ['P_'+suffix,'SP_'+suffix,'L','G_'+str(g)]

def candidates(phase):
    methods=['P_01','SP_01','P_1','SP_1','L','G_0.25','G_0.5','G_0.75'] if phase=='PILOT' else arms()
    rows=[]
    for j in read(ROOT/'QUEUE.json'):
        eligible=j['phase']==phase or (phase=='DEV' and j['phase']=='PILOT' and j['mode']=='train')
        if not eligible:continue
        for p in (ROOT/'jobs'/j['id']).rglob('CONSUMERS.json'):
            rows += [normalize(r) for r in read(p) if r['arm'] in methods]
    ref=references(phase);frozen={r['input_id']:r.get('frozen_base_correct') for r in ref if r['mode']=='holdout'}
    for r in rows:
        if r['mode']=='holdout':r['frozen_base_correct']=frozen[r['input_id']]
    return rows

def score_closed(phase):
    rows=references(phase)+candidates(phase);keys={r[k] for r in rows for k in ['judge_key','base_judge_key']}
    scored=read_scores(ROOT);missing=set(read(ROOT/NAMESPACE/'JUDGE_MISSING_LOCK.json')['keys'])
    return not keys-set(scored)-missing

def enqueue(phase):
    tasks=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'];orders=PILOT if phase=='PILOT' else list(range(1,25)) if phase=='DEV' else list(range(25,49))
    methods=['P_01','SP_01','P_1','SP_1','L'] if phase=='PILOT' else arms()[:3];queue=[];upper=0
    def add(j,n):
        nonlocal upper
        queue.append(dict(j,id=phase+'-'+str(len(queue)),phase=phase,status='PENDING'));upper+=n
    for o in orders:
        if phase=='DEV' and o in PILOT:continue
        t=next(t for t in tasks if t['order']==o);extra=19 if phase=='PILOT' else 0
        add(dict(mode='train',order=o,methods=methods,check=phase=='PILOT'),len(methods)*(len(t['evaluation'])+extra))
    prefixes=[4,8] if phase=='PILOT' else [4,8,12,24]
    for m in methods+([] if phase=='PILOT' else [arms()[3]]):
        requires=[f'adapters/s20260929/{m}/e{o:03d}.pt' for o in orders] if not m.startswith('G_') else []
        add(dict(mode='bank',method=m,orders=orders,prefixes=prefixes,requires=requires),sum(sum(len(t['evaluation']) for t in tasks if t['order'] in orders[:p]) for p in prefixes))
        if phase!='PILOT':add(dict(mode='bank',holdout=True,method=m,orders=orders,prefixes=prefixes,requires=requires),47*4)
    if phase=='PILOT':
        for gamma in [.25,.5,.75]:add(dict(mode='check',method='G_'+str(gamma),orders=orders),8*19)
    else:add(dict(mode='single',method=arms()[3],orders=orders),sum(len(t['evaluation']) for t in tasks if t['order'] in orders))
    per_single=sum(len(t['evaluation']) for t in tasks if t['order'] in orders)
    per_bank=sum(sum(len(t['evaluation']) for t in tasks if t['order'] in orders[:p]) for p in prefixes)
    reference_upper=8*(per_single+per_bank+(152 if phase=='PILOT' else 188))
    write(ROOT/f'private/reservations/{phase}.json',dict(status='RESERVED_AUTHORIZED_NO_INTERNAL_CAP',total_attempt_items_upper_bound=upper+reference_upper,unknown_dedup_discount=0,before_any_new_pending=True))
    refs=references(phase);keys={r[k] for r in refs for k in ['judge_key','base_judge_key']};success=read_scores(ROOT)
    reserve=upper+len(keys-set(success));assert read(ROOT/'RESOURCE_LEDGER.json')['judge_submission_attempt_items_limit'] is None
    write(ROOT/f'private/reservations/{phase}.json',dict(status='RESERVED_AUTHORIZED_NO_INTERNAL_CAP',new_generation_rows_upper_bound=upper,known_reference_keys_without_confirmed_success=len(keys-set(success)),total_attempt_items_upper_bound=reserve,unknown_dedup_discount=0,started_epoch=time.time()))
    q=read(ROOT/'QUEUE.json');assert not any(j['phase']==phase for j in q);write(ROOT/'QUEUE.json',q+queue)
    write(ROOT/'RUN_STATUS.json',dict(status='READY',phase=phase,epoch=time.time(),judge_model='gpt-6.1-sol'))

def select():
    rows=candidates('PILOT');refs=references('PILOT');scores=read_scores(ROOT)
    baseline=[r for r in refs if r['arm']=='H' and r['mode']=='CHECK_FORCED_ON']
    beta=select_grid({b:[r for r in rows if r['arm']==m and r['mode']=='CHECK_FORCED_ON'] for b,m in [(.1,'SP_01'),(1.,'SP_1')]},baseline,scores)
    grid={g:[r for r in rows if r['arm']=='G_'+str(g) and r['mode']=='CHECK_FORCED_ON'] for g in [.25,.5,.75]};grid[1.]=baseline
    gamma=select_grid(grid,baseline,scores)
    s=dict(beta=beta,gamma=gamma,epoch=time.time(),shared_beta_for_P_SP=True,formal_full_DEV_unblinded=False,selection_only_CHECK=True);s['id']=digest(s);write(ROOT/'public/BETA_GAMMA_SELECTION.json',s)
    return beta['selected'] is not None and gamma['selected'] is not None

def decide():
    rows=candidates('DEV');refs=references('DEV');masks={(x['edit'],x['task'],x['query_id']):x['base_correct'] for x in read(ROOT/'private/BASE_MASKS.json')['rows']}
    def qualified(rs):return [r for r in rs if r['task'] not in ['T1L','T2L'] or masks.get((r['edit'],r['task'],r['query_id'])) is True]
    result=joint({m:qualified([r for r in rows if r['arm']==m]) for m in arms()},{m:qualified([r for r in refs if r['arm']==m]) for m in ['A0','E_orig','H','S']},read_scores(ROOT))
    result['Judge']='gpt-6.1-sol/high';result['holdout_qualification']='original frozen 35 Base-correct inputs; Sol disagreement reported separately'
    write(ROOT/'public/JOINT_GAIN_DECISION.json',result);return result['REG_allowed']
