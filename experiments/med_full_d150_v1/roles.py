"""Build same-support role identities from pre-existing task and Base masks only."""
import json
from pathlib import Path
import index
import identity_lock
import prior_binding
import stream_extract as stream

ROLES=('T0','FIT','T1G','T2G','T1L','T2L')


def additions(path):
    rows=[]
    with Path(path).open('rb') as f:
        if stream.nonspace(f)!=b'{':raise ValueError('mask root expected')
        while True:
            t=stream.nonspace(f)
            if t==b'}':break
            if t!=b'"':raise ValueError('mask key expected')
            k=stream.string(f,True)
            if stream.nonspace(f)!=b':':raise ValueError('mask colon expected')
            t=stream.nonspace(f)
            if k=='consumers':
                if t!=b'[':raise ValueError('mask array expected')
                while True:
                    t=stream.nonspace(f)
                    if t==b']':break
                    v=prior_binding.select(f,t,{'owner':True,'role':True,'query_id':True,'eligible':True})
                    if set(v)!={'owner','role','query_id','eligible'}:raise ValueError('missing mask identity')
                    rows.append(v);t=stream.nonspace(f)
                    if t==b']':break
                    if t!=b',':raise ValueError('mask array separator expected')
            else:index.skip(f,t)
            t=stream.nonspace(f)
            if t==b'}':break
            if t!=b',':raise ValueError('mask root separator expected')
    if len(rows)!=55:raise ValueError('additional mask count mismatch')
    return rows


def original_active(path, bank_ids):
    result={}
    with Path(path).open('rb') as f:
        if stream.nonspace(f)!=b'{':raise ValueError('consumer root expected')
        while True:
            t=stream.nonspace(f)
            if t==b'}':break
            if t!=b'"':raise ValueError('consumer key expected')
            k=stream.string(f,True)
            if stream.nonspace(f)!=b':':raise ValueError('consumer colon expected')
            t=stream.nonspace(f)
            if k=='consumers':
                if t!=b'[':raise ValueError('consumer array expected')
                while True:
                    t=stream.nonspace(f)
                    if t==b']':break
                    row=prior_binding.select(f,t,{'arm':True,'mode':True,'prefix':True,'query_id':True,'active_target':True})
                    if row.get('arm')=='MED' and row.get('mode')=='bank' and row.get('prefix')==146:
                        q=row.get('query_id');v=row.get('active_target')
                        if q in result or q not in bank_ids or type(v) is not bool:raise ValueError('invalid original bank identity')
                        result[q]=v
                    t=stream.nonspace(f)
                    if t==b']':break
                    if t!=b',':raise ValueError('consumer delimiter expected')
            else:index.skip(f,t)
            t=stream.nonspace(f)
            if t==b'}':break
            if t!=b',':raise ValueError('consumer root delimiter expected')
    if len(result)!=2093:raise ValueError('original bank identities incomplete')
    return result


def remote_metadata(original, root, extra):
    tasks=identity_lock.tasks(Path(original)/'INPUTS_FROZEN.json')
    bank=json.loads((Path(root)/'BANK_INPUTS_PRIVATE.json').read_text())['records']
    byid={x['query']['query_id']:x for x in bank}
    if len(byid)!=2148:raise ValueError('bank query identities incomplete')
    needed=set()
    for task in tasks:
        for e in task.get('events',[]):
            if e.get('task') in ('T1G','T2G','T1L','T2L'):needed.update(e.get('all_probe_query_ids',[]))
    mask=index.read_mask(Path(original)/'INPUTS_FROZEN.json',needed)
    extra_source=json.loads((Path(root)/'TRAIN_SOURCE_BOUND_PRIVATE.json').read_text())
    new_native={(x['task']['native']['image_sha256'],x['task']['native']['question'])
                for x in extra_source if x['task']['order']>=147}
    if len(new_native)!=4:raise ValueError('additional source identity missing')
    newly_active=[q for q,r in byid.items() if (r['query']['image_sha256'],r['query']['question']) in new_native]
    rolemap={}
    for task in tasks:
        owner=task['order'];rr={}
        for role in ROLES:
            ids=([f'edit_{owner}_{i}' for i in range(1,5)] if role=='FIT' else
                 list(dict.fromkeys(q for e in task.get('events',[]) if e.get('task')==role for q in e.get('all_probe_query_ids',[]))))
            if role not in ('T0','FIT'):ids=[q for q in ids if mask[q]==role.endswith('L')]
            if any(q not in byid for q in ids):raise ValueError('role query missing from bank')
            rr[role]=ids
        rolemap[str(owner)]=rr
    records=additions(Path(extra)/'BASE_MASK_FROZEN.json')
    for owner in range(147,151):
        rolemap[str(owner)]={role:([] if role=='T1G' else [x['query_id'] for x in records if x['owner']==owner and x['role']==role and (role in ('T0','FIT') or x['eligible'] is True)]) for role in ROLES}
        if any(q not in byid for ids in rolemap[str(owner)].values() for q in ids):raise ValueError('addition role missing from bank')
    return {'status':'ROLE_IDENTITIES_BEFORE_ACTIVE_FILTER','roles':rolemap,'newly_active':newly_active,'bank_ids':list(byid),
            'qa_or_answers_decoded_from_mixed_files':0}


def finalize(metadata, old_consumers):
    if metadata.get('status')!='ROLE_IDENTITIES_BEFORE_ACTIVE_FILTER':raise ValueError('role identity stage missing')
    active=original_active(old_consumers,set(metadata['bank_ids'])); newly=set(metadata['newly_active'])
    rolemap=metadata['roles']
    for owner in range(1,147):
        for role in ('T1L','T2L'):
            rolemap[str(owner)][role]=[q for q in rolemap[str(owner)][role] if not active[q] and q not in newly]
    return {'status':'EVALUATION_ROLES_FROZEN','roles':rolemap,'owners':150}
