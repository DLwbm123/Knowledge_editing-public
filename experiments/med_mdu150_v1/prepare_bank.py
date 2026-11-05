"""Bind the 2148 fixed bank events to old predictions and new expert routes."""
import hashlib,json,os
from pathlib import Path
import prior_binding as pb
import stream_extract as s
root=Path(os.environ['EXPERIMENT_ROOT']);prior=Path(os.environ['PRIOR_101_ROOT']);old=Path(os.environ['ADDITIONS_ROOT'])
entries=json.loads((root/'TRAIN_SOURCE_BOUND_PRIVATE.json').read_text()); oldentries=json.loads((prior/'TRAIN_SOURCE_BOUND_PRIVATE.json').read_text())
new={x['task']['edit_id'] for x in entries};previous={x['task']['edit_id'] for x in oldentries}
if len(new)!=49 or len(previous)!=101 or new & previous:raise ValueError('expert identities differ')
oldbank=json.loads((prior/'BANK_INPUTS_PRIVATE.json').read_text())
if oldbank.get('status')!='BANK_INPUTS_BOUND' or len(oldbank['records'])!=2148:raise ValueError('prior bank incomplete')
fields={'consumer_id':True,'route':True,'effective_expert':True,'effective_state_hash':True,
        'binding':{'input':{'query_id':True,'question':True,'image_path':True,'image_sha256':True,'audit_role':True},
                   'judge_input':{'question':True,'image_sha256':True,'prompt_ids':True,'attention_mask':True,'generation':True}}}
result=[];gen=0;reuse=0
for row in oldbank['records']:
 cid=row['consumer_id'];key=hashlib.sha256(cid.encode()).hexdigest();selected=row['effective_expert']
 if selected in new:
  name=row['original_filename']
  if Path(name).name!=name:raise ValueError('unsafe filename')
  with (old/'eval_outputs'/name).open('rb') as f:data=pb.select(f,s.nonspace(f),fields)
  q=data['binding']['input'];j=data['binding']['judge_input']
  if data['consumer_id']!=cid or q['query_id']!=row['query']['query_id'] or data['route']!=row['route'] or data['effective_expert']!=selected or data['effective_state_hash']!=row['original_state_hash']:
   raise ValueError('new route input identity mismatch')
  if q['question']!=j['question'] or q['image_sha256']!=j['image_sha256']:raise ValueError('prompt image mismatch')
  row=dict(row,query=q,judge_input=j);gen+=1
 elif selected in previous:
  pred=json.loads((prior/'predictions'/f'{key}.json').read_text());q=row['query'];j=row['judge_input']
  if not pred.get('candidate') or pred.get('original_route')!=row['route'] or pred.get('consumer_id')!=cid or pred.get('query_id')!=q['query_id']:
   raise ValueError('previous prediction identity mismatch')
  if any(pred.get(k)!=j[k] for k in ('question','image_sha256','prompt_ids','attention_mask','generation')):raise ValueError('previous prediction input mismatch')
  reuse+=1
 else:raise ValueError('unexpected effective expert')
 result.append(row)
if len({x['consumer_id'] for x in result})!=2148 or (gen,reuse)!=(635,1513):raise ValueError('route coverage mismatch')
p=root/'BANK_INPUTS_PRIVATE.json'
with p.open('x') as f:json.dump({'status':'BANK_INPUTS_BOUND','records':result,'candidate_count':gen},f,ensure_ascii=False,separators=(',',':'))
p.chmod(0o600)
print(json.dumps({'status':'BANK_INPUTS_BOUND','events':2148,'new':gen,'reused':reuse,'references_decoded':0}))
