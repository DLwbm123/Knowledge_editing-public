"""P1 -> pilot CHECK -> frozen rho -> DEV -> conditional REG, finite blocks."""
import copy,time,fcntl
from pathlib import Path
from resources import ROOT,read,write
from judge_protocol import packet,NAMESPACE,read_scores,digest
from policy import select_shared,joint,mechanism
PILOT=[4,5,6,11,12,15,16,21]
def normalize(r):
    r=copy.deepcopy(r)
    if r['mode']=='holdout':r.update(mode='EXPOSED_REGRESSION',edit='EXPOSED_REGRESSION',task='EXPOSED_REGRESSION')
    return r
def references(phase):
    dst=ROOT/f'private/references/{phase}.json'
    if dst.exists():return read(dst)
    old=Path(read(ROOT/'PREDECESSOR.json')['pr8_root']);rows=[]
    if phase=='PILOT':
        rows=[normalize(r) for r in read(old/'private/references/PILOT.json') if r['mode']=='CHECK_FORCED_ON']
        for j in read(old/'QUEUE.json'):
            if j['phase']!='PILOT' or j['mode']!='train':continue
            for p in (old/'jobs'/j['id']/'SP_01'/'check').glob('CONSUMERS.json'):
                for r in read(p):r=normalize(r);r['arm']='SP';rows.append(r)
    elif phase=='DEV':
        rows=[normalize(r) for r in read(old/'private/references/DEV.json')]
        for j in read(old/'QUEUE.json'):
            if j['phase'] not in ['PILOT','DEV'] or (j['phase']=='PILOT' and j['mode']!='train'):continue
            for p in (old/'jobs'/j['id']).rglob('CONSUMERS.json'):
                for r in read(p):
                    if r['arm']=='SP_01' and r['mode']!='CHECK_FORCED_ON':r=normalize(r);r['arm']='SP';rows.append(r)
    elif phase=='REG':
        old=Path(read(ROOT/'PREDECESSOR.json')['root']);names={'E_orig':'E_orig','A0':'A0','AH':'H','AHS_01':'S'};files=[]
        for j in read(old/'QUEUE.json'):
            if j['mode']=='train' and j['order'] in range(25,49):
                for m in names:files.extend((old/'jobs'/j['id']/m/'single').glob('CONSUMERS.json'))
            if j['mode']=='bank' and j['method'] in names and j['id'].startswith('P4-'):files.extend((old/'jobs'/j['id']).rglob('CONSUMERS.json'))
        for m in names:files.extend((old/'auxiliary/holdout'/f'REG24-{m}').rglob('CONSUMERS.json'))
        oldscores={p.stem:read(p)['is_correct'] for p in (old/'private/judge/scores').glob('*.json')}
        for p in files:
            for original in read(p):
                r=normalize(original);r['arm']=names[r['arm']];r['frozen_base_correct']=oldscores.get(original['base_judge_key'])
                for field in ['judge_key','base_judge_key']:
                    rec=read(old/'private/judge/pending'/f'{original[field]}.json')['record'];pk=packet(rec['question'],rec['gold_answer'],rec['raw_base_answer']);path=ROOT/NAMESPACE/'pending'/f'{pk["key"]}.json'
                    if not path.exists():write(path,pk)
                    r[field]=pk['key']
                rows.append(r)
    else:return []
    assert rows
    unique={}
    for r in rows:
        k=tuple(r[f] for f in ['arm','mode','prefix','edit','task','input_id'])
        if k in unique:assert unique[k]['judge_key']==r['judge_key']
        unique[k]=r
    rows=list(unique.values());write(dst,rows);return rows
def arms(phase):
    if phase=='PILOT':return [m+'_'+str(rho) for m in ['CAP','EGP','EGP_A'] for rho in [1,2]]
    selected=read(ROOT/'public/RHO_SELECTION.json')['selected'];assert selected in [1,2]
    if phase=='REG':return read(ROOT/'public/JOINT_GAIN_DECISION.json')['REG_methods']
    return [m+'_'+str(selected) for m in ['CAP','EGP','EGP_A']]
def candidates(phase):
    rows=[]
    for j in read(ROOT/'QUEUE.json'):
        if j['phase']!=phase:continue
        for p in (ROOT/'jobs'/j['id']).rglob('CONSUMERS.json'):rows.extend(normalize(r) for r in read(p))
    if phase in ['DEV','REG']:
        frozen={r['input_id']:r['frozen_base_correct'] for r in references(phase) if r['mode']=='EXPOSED_REGRESSION' and r['arm']=='H'}
        assert len(frozen)==47 and sum(v is True for v in frozen.values())==35,'Frozen historical qualification changed'
        for r in rows:
            if r['mode']=='EXPOSED_REGRESSION':r['frozen_base_correct']=frozen[r['input_id']]
    return rows
def score_closed(phase):
    rows=references(phase)+candidates(phase);keys={r[k] for r in rows for k in ['judge_key','base_judge_key']};scores=read_scores(ROOT);missing=set(read(ROOT/NAMESPACE/'JUDGE_MISSING_LOCK.json')['keys'])
    return bool(rows) and not keys-set(scores)-missing
def curves(phase):
    tasks=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'];orders=PILOT if phase=='PILOT' else list(range(1,25)) if phase=='DEV' else list(range(25,49));result={}
    from training import derive_seed
    for m in arms(phase):
        byedit={}
        for o in orders:
            t=next(t for t in tasks if t['order']==o);seed=derive_seed(t['canonical_edit_id'],20260929)+1
            byedit[str(o)]=read(ROOT/f'private/curves/s{seed}/{m}/e{o}/continuation.json')['curve']
        result[m]=byedit
    return result
def enqueue(phase):
    queue=[];tasks=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'];upper=0
    def add(j,n):
        nonlocal upper
        queue.append(dict(j,id=phase+'-'+str(len(queue)),phase=phase,status='PENDING'));upper+=n
    if phase=='ROUTE':
        from route_removal import early_panel
        selected=[t for t in tasks if t['order'] in range(25,37)];n=47+sum(len(t['evaluation']) for t in early_panel(selected))
        for condition in ['FULL','REMOVE_9','REMOVE_10','REMOVE_11','REMOVE_12']:add(dict(mode='route',condition=condition),n)
    else:
        methods=arms(phase);orders=PILOT if phase=='PILOT' else list(range(1,25)) if phase=='DEV' else list(range(25,49))
        for o in orders:
            if phase=='DEV' and o in PILOT:continue
            t=next(t for t in tasks if t['order']==o);n=19 if phase=='PILOT' else len(t['evaluation'])
            add(dict(mode='train',order=o,methods=methods,check=phase=='PILOT',formal=phase!='PILOT'),len(methods)*n)
        if phase!='PILOT':
            if phase=='DEV':
                for m in methods:add(dict(mode='single',method=m,orders=PILOT),sum(len(t['evaluation']) for t in tasks if t['order'] in PILOT))
            for m in methods:
                requires=[f'adapters/s20260929/{m}/e{o:03d}.pt' for o in orders]
                add(dict(mode='bank',method=m,orders=orders,prefixes=[4,8,12,24],requires=requires),sum(len(t['evaluation']) for p in [4,8,12,24] for t in tasks if t['order'] in orders[:p]))
                add(dict(mode='bank',holdout=True,method=m,orders=orders,prefixes=[4,8,12,24],requires=requires),47*4)
    assert read(ROOT/'RESOURCE_LEDGER.json')['judge_submission_attempt_items_limit'] is None
    refs=references(phase);known=read_scores(ROOT);blocked=set(read(ROOT/NAMESPACE/'JUDGE_MISSING_LOCK.json')['keys']);keys={r[k] for r in refs for k in ['judge_key','base_judge_key']}
    write(ROOT/f'private/reservations/{phase}.json',dict(status='AUTHORIZED_NO_INTERNAL_CAP',new_generation_rows_upper_bound=upper,known_reference_keys_to_submit=len(keys-set(known)-blocked),unknown_dedup_discount=0,epoch=time.time()))
    with (ROOT/'QUEUE.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);q=read(ROOT/'QUEUE.json');assert not any(j['phase']==phase for j in q);write(ROOT/'QUEUE.json',q+queue)
def select():
    if (ROOT/'public/RHO_SELECTION.json').exists():return read(ROOT/'public/RHO_SELECTION.json')['selected'] is not None
    mm=mechanism(curves('PILOT'));selection=select_shared(candidates('PILOT'),references('PILOT'),read_scores(ROOT),mm);selection['epoch']=time.time();selection['id']=digest(selection)
    write(ROOT/'public/RHO_SELECTION.json',selection)
    if selection['selected'] is not None:
        selected=set(arms('DEV'));delete_adapters(lambda rel:rel.startswith('adapters/') and rel.split('/')[2] not in selected)
    return selection['selected'] is not None
def decide():
    if (ROOT/'public/JOINT_GAIN_DECISION.json').exists():return read(ROOT/'public/JOINT_GAIN_DECISION.json')['REG_allowed']
    phase='DEV';rows=candidates(phase);refs=references(phase);masks={(r['edit'],r['task'],r['query_id']):r['base_correct'] for r in read(ROOT/'private/BASE_MASKS.json')['rows']}
    qualified=lambda rs:[r for r in rs if r['task'] not in ['T1L','T2L'] or masks.get((r['edit'],r['task'],r['query_id'])) is True]
    mm=mechanism(curves(phase));result=joint({m:qualified([r for r in rows if r['arm']==m]) for m in arms(phase)},{m:qualified([r for r in refs if r['arm']==m]) for m in ['A0','E_orig','H','S','SP']},read_scores(ROOT),mm)
    write(ROOT/'public/JOINT_GAIN_DECISION.json',result);return result['REG_allowed']
def delete_adapters(predicate):
    from storage import Store
    store=Store(ROOT)
    with store.lock() as d:rels=[r for r,a in d['artifacts'].items() if a['status']=='READY' and predicate(r) and not a['readers'] and not a['consumers']]
    for rel in rels:
        with store.lock() as d:d['artifacts'][rel]['pin']=False
        store.delete(rel)
