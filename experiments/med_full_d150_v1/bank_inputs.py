"""Freeze the original bank inputs and R0 decisions without references or answers."""
import json
from pathlib import Path
import re
import index
import prior_binding
import stream_extract as stream
import identity_lock


def plan_identities(path):
    plan=None
    with Path(path).open('rb') as f:
        if stream.nonspace(f)!=b'{':raise ValueError('root expected')
        while True:
            t=stream.nonspace(f)
            if t==b'}':break
            if t!=b'"':raise ValueError('root key expected')
            k=stream.string(f,True)
            if stream.nonspace(f)!=b':':raise ValueError('root colon expected')
            t=stream.nonspace(f)
            if k=='plan':
                if plan is not None or t!=b'[':raise ValueError('invalid plan')
                plan=index.array(f,'plan')
            else:index.skip(f,t)
            t=stream.nonspace(f)
            if t==b'}':break
            if t!=b',':raise ValueError('root delimiter expected')
    if plan is None:raise ValueError('missing plan')
    result=[]
    for row in plan:
        if row.get('mode')=='bank' and row.get('prefix')==150:
            q=row.get('query',{})
            result.append((row.get('method'),row.get('mode'),row.get('prefix'),row.get('owner'),q.get('query_id')))
    if len(result)!=2148 or len(set(result))!=2148:raise ValueError('bank plan identities differ')
    return set(result)


def build(old150, root, reserved):
    old150,root=Path(old150),Path(root)
    expected=plan_identities(old150/'EVAL_PLAN_FROZEN.json')
    reference=json.loads((old150/'EVAL_LOCK.json').read_text())
    admission=json.loads((root/'ROLE_ADMISSION.json').read_text())
    entries=json.loads((root/'TRAIN_SOURCE_BOUND_PRIVATE.json').read_text())
    spans=json.loads((root/'SPAN_LOCK.private.json').read_text())
    if admission.get('status')!='TRAINING_SOURCE_ROLE_ADMITTED' or spans.get('status')!='WEAK_TOKEN_PROXY_ONLY':
        raise ValueError('training source or span admission missing')
    owners={x['order'] for x in spans['records'] if x['status']=='WEAK_TOKEN_MATCH'}
    ids={x['task']['edit_id'] for x in entries if x['task']['order'] in owners}
    if len(ids)!=25:raise ValueError('matched expert identity coverage mismatch')
    fields={'consumer_id':True,'method':True,'mode':True,'prefix':True,'owner':True,'route':True,
            'effective_expert':True,'effective_state_hash':True,
            'binding':{'input':{'query_id':True,'question':True,'image_path':True,'image_sha256':True,'audit_role':True,'source_group':True,'dataset':True},
                       'judge_input':{'question':True,'image_sha256':True,'prompt_ids':True,'attention_mask':True,'generation':True}},
            'prediction_binding':{'source_commit':True,'runtime_reference':True,'rng_reference':True,'deterministic_greedy':True,'effective_state':True}}
    rows=[];found=set();off=0;candidate_count=0;unresolved_vqa=0
    for path in sorted((old150/'eval_outputs').glob('*.json')):
        with path.open('rb') as f:
            meta=prior_binding.select(f,stream.nonspace(f),{'mode':True,'prefix':True})
        if meta.get('mode')!='bank' or meta.get('prefix')!=150:continue
        with path.open('rb') as f:row=prior_binding.select(f,stream.nonspace(f),fields)
        q=row.get('binding',{}).get('input',{});b=row.get('binding',{}).get('judge_input',{})
        if not {'query_id','question','image_path','image_sha256','audit_role'} <= set(q) or set(b)!={'question','image_sha256','prompt_ids','attention_mask','generation'}:
            raise ValueError('incomplete query/input binding')
        ident=(row.get('method'),row.get('mode'),row.get('prefix'),row.get('owner'),q['query_id'])
        if ident not in expected or ident in found:raise ValueError('unexpected original consumer')
        found.add(ident)
        if q['question']!=b['question'] or q['image_sha256']!=b['image_sha256']:
            raise ValueError('original source/input mismatch')
        source=q.get('source_group') or q['image_path']
        group=identity_lock.group(source)
        if group and group in reserved:raise ValueError('protected evaluation source group')
        if not group:
            if not re.search(r'vqa[-_]?rad',str(source),re.I):raise ValueError('unknown evaluation dataset')
            unresolved_vqa+=1
        prediction=row.get('prediction_binding',{})
        if any(prediction.get(k)!=reference.get(k) for k in ('source_commit','runtime_reference','rng_reference')) or prediction.get('deterministic_greedy') is not True:
            raise ValueError('original execution binding mismatch')
        if prediction.get('effective_state')!=row.get('effective_state_hash'):
            raise ValueError('original state hash mismatch')
        route=row.get('route',{});selected=row.get('effective_expert')
        if type(route.get('activated')) is not bool or (route.get('logical_edit_id') if route['activated'] else None)!=selected:
            raise ValueError('original routing binding mismatch')
        off+=not route['activated'];candidate_count+=selected in ids
        rows.append({'consumer_id':row['consumer_id'],'method':row['method'],'mode':row['mode'],'prefix':row['prefix'],
                     'owner':row['owner'],'query':q,'judge_input':b,'route':route,'effective_expert':selected,
                     'original_state_hash':row['effective_state_hash'],'execution_binding':prediction,'original_filename':path.name})
    if found!=expected or len(rows)!=2148 or candidate_count!=392 or off!=0:
        raise ValueError('bank coverage or original routing counts differ')
    return {'status':'BANK_INPUTS_BOUND','records':rows,'candidate_count':candidate_count,'off':off,
            'vqa_group_unknown':unresolved_vqa,'reference_or_answer_decoded':0,
            'route_provenance':'existing actual MED150 R0; no R0 parameters changed'}
