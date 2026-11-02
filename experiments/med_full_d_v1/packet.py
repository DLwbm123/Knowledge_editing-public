"""Freeze 9-owner predictions, then read only their admitted references."""
import hashlib,json,time
from pathlib import Path
import stream_extract as s
import prior_binding
import extract_reference

CONTRACT='MEDPROP_SMALL4_CORE_FULL_SOURCE_V1'

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def select_rows(path,owners):
    out=[];count=0
    with Path(path).open('rb') as f:
        if s.nonspace(f)!=b'[':raise ValueError('Expected array')
        while True:
            token=s.nonspace(f)
            if token==b']':break
            start=f.tell()-1;ident=prior_binding.select(f,token,{'owner':True});end=f.tell();count+=1
            if ident.get('owner') in owners:
                f.seek(start);out.append(json.loads(f.read(end-start)));f.seek(end)
            token=s.nonspace(f)
            if token==b']':break
            if token!=b',':raise ValueError('Bad array delimiter')
    if count!=508 or len(out)!=371:raise ValueError('Input selection coverage')
    return out

def build(root,fixed,source,span_lock):
    root,fixed=Path(root),Path(fixed)
    if any(json.loads((root/n).read_text())['code']!=0 for n in ('WORKER_EXIT.json','GUARD_EXIT.json')):
        raise ValueError('Execution did not finish successfully')
    complete=json.loads((root/'EVAL_COMPLETE.json').read_text())
    if complete.get('predictions')!=371 or complete.get('status')!='GENERATION_COMPLETE':
        raise ValueError('Generation incomplete')
    lock=json.loads(Path(span_lock).read_text());owners={r['order'] for r in lock['records'] if r['status']=='WEAK_TOKEN_MATCH'}
    if lock['status']!='WEAK_TOKEN_PROXY_ONLY' or len(owners)!=9:raise ValueError('Invalid span lock')
    inputs=select_rows(fixed/'EVAL_INPUTS_PRIVATE.json',owners)
    old_rows=select_rows(fixed/'PRIOR_INPUT_BINDINGS_PRIVATE.json',owners)
    expected={(x['owner'],x['query']['query_id']):x['query'] for x in inputs}
    prior={(x['owner'],x['query_id']):x for x in old_rows}
    if len(expected)!=371 or set(expected)!=set(prior):raise ValueError('Input binding coverage')
    files=list((root/'predictions').glob('*.json'))
    if len(files)!=371:raise ValueError('Prediction count mismatch')
    predictions={}
    for p in files:
        row=json.loads(p.read_text());key=(row['owner'],row['query_id'],row['arm'])
        if key in predictions or key[:2] not in expected or key[2] not in ('D',):raise ValueError('Unexpected prediction')
        q,old=expected[key[:2]],prior[key[:2]];route=old['route']
        norm=dict(activated=route['activated'],logical_edit_id=route['logical_edit_id'] if route['activated'] else None)
        if (row['route']!=norm or row['original_route']!=route or row['original_consumer_id']!=old['consumer_id']
            or row['original_execution_binding']!=old['prediction_binding'] or not row['raw_token_ids']
            or row['question']!=q['question'] or row['image_sha256']!=q['image_sha256']
            or any(row[k]!=old['input'][k] for k in ('question','image_sha256','prompt_ids','attention_mask','generation'))):
            raise ValueError('Prediction input or execution mismatch')
        predictions[key]=row
    if set(predictions)!={(o,q,a) for o,q in expected for a in ('D',)}:raise ValueError('Panel incomplete')
    frozen=dict(status='ALL_PREDICTIONS_FROZEN',events=371,predictions=371,time=time.time())
    with (root/'PREDICTIONS_FROZEN.json').open('x') as f:json.dump(frozen,f)
    admission=json.loads((fixed/'ROLE_ADMISSION.json').read_text())
    plan=json.loads((fixed/'PANEL_IDENTITY_LOCK.json').read_text())
    if admission['status']!='SOURCE_AND_EVALUATION_ROLE_ADMITTED' or admission['owners']!=plan['index']['owners']:
        raise ValueError('Original role admission mismatch')
    selected=[x for x in plan['index']['plan'] if x['owner'] in owners]
    if len(selected)!=371:raise ValueError('Reference identity coverage')
    with Path(source).open('rb') as f: refs=[extract_reference.plan_ref(f,x) for x in selected]
    references={(x['owner'],x['query_id']):x['reference'] for x in refs}
    if set(references)!=set(expected):raise ValueError('Reference coverage mismatch')
    payloads={};consumers=[]
    for (owner,qid,arm),row in sorted(predictions.items()):
        ref=references[(owner,qid)]
        semantic=dict(protocol=CONTRACT,model='gpt-6-astra',reasoning='medium',question=row['question'],gold_answer=ref,raw_base_answer=row['raw_answer'])
        key=digest(semantic)
        legacy=dict(query_id=qid,question=row['question'],reference=ref,image_sha256=row['image_sha256'],prompt_ids=row['prompt_ids'],answer=dict(raw_answer=row['raw_answer'],raw_token_ids=row['raw_token_ids']))
        record=dict(opaque_query_id=key,question=row['question'],gold_answer=ref,raw_base_answer=row['raw_answer'])
        if key not in payloads:payloads[key]=dict(record=record,legacy_keys=[])
        if payloads[key]['record']!=record:raise ValueError('Payload collision')
        payloads[key]['legacy_keys'].append(digest(legacy))
        consumers.append(dict(owner=owner,query_id=qid,arm=arm,key=key,eos=row['eos'],at_cap=row['at_cap'],empty=not row['raw_answer'].strip(),route=row['route']))
    packet=dict(status='SCORE_PACKET_FROZEN',consumers=consumers,payloads=payloads,source_events=371,predictions=371,method_blind=True)
    with (root/'SCORE_PACKET_PRIVATE.json').open('x') as f:json.dump(packet,f,ensure_ascii=False)
    (root/'SCORE_PACKET_PRIVATE.json').chmod(0o600)
    return dict(status=packet['status'],consumers=371,unique_payloads=len(payloads))
