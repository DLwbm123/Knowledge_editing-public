"""Bind the fixed 101 sources and 25 completed states before new work."""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time
import torch
import selected_records

new=Path(os.environ['EXPERIMENT_ROOT']);du=Path(os.environ['DU_ROOT']);prior=Path(os.environ['PRIOR_MEDINIT_ROOT'])
med=Path(os.environ['ORIGINAL_ROOT']);extra=Path(os.environ['ADDITIONS_ROOT']);deadline=float(os.environ['EXPERIMENT_DEADLINE'])

def digest(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def write(p,obj):
 fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
 with os.fdopen(fd,'w') as f:json.dump(obj,f,ensure_ascii=False,separators=(',',':'))

def main():
 if time.time()>=deadline-600:raise TimeoutError('original deadline')
 resource.setrlimit(resource.RLIMIT_FSIZE,(256_000_000,256_000_000))
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('preparation budget')))
 signal.alarm(min(600,max(1,int(deadline-time.time()-120))))
 admission=json.loads((du/'ROLE_ADMISSION.json').read_text());token=json.loads((du/'TOKEN_LOCK.private.json').read_text());weak=json.loads((du/'SPAN_LOCK.private.json').read_text())
 source=json.loads((du/'TRAIN_SOURCE_BOUND_PRIVATE.json').read_text());owners=set(admission['owners'])
 if admission.get('status')!='TRAINING_SOURCE_ROLE_ADMITTED' or len(owners)!=101 or len(source)!=101 or [x['task']['order'] for x in source]!=admission['owners']:
  raise ValueError('role/source identity differs')
 if {x['order'] for x in token['records'] if x['status']=='TOKEN_BOUND'}!=owners or [x['order'] for x in token['records']]!=admission['owners']:
  raise ValueError('token identities differ')
 prior_owners={x['order'] for x in weak['records'] if x['status']=='WEAK_TOKEN_MATCH'}
 if len(prior_owners)!=25 or not prior_owners.issubset(owners):raise ValueError('prior owner lock differs')
 tasks={x['task']['edit_id']:x['task']['order'] for x in source}
 if len(tasks)!=101:raise ValueError('expert identity collision')
 fields={'consumer_id':True,'owner':True,'query':{'query_id':True},'route':True,'effective_expert':True,'original_filename':True,'original_state_hash':True}
 meta=selected_records.identities(du/'BANK_INPUTS_PRIVATE.json','records',fields)
 if len(meta)!=2148 or len({x['consumer_id'] for x in meta})!=2148:raise ValueError('bank identity coverage differs')
 wanted={x['consumer_id'] for x in meta if x['effective_expert'] in tasks}
 old_ids={x['consumer_id'] for x in meta if tasks.get(x['effective_expert']) in prior_owners}
 if len(wanted)!=1513 or len(old_ids)!=392:raise ValueError('frozen routed cohorts differ')
 detail_fields=dict(fields,query={'query_id':True,'question':True,'image_path':True,'image_sha256':True,'audit_role':True},judge_input={'question':True,'image_sha256':True,'prompt_ids':True,'attention_mask':True,'generation':True})
 details=selected_records.read_selected(du/'BANK_INPUTS_PRIVATE.json','records',{'consumer_id':True},lambda r:r['consumer_id'],wanted,detail_fields)
 for item in meta:
  if item['consumer_id'] in wanted:
   detail=details[item['consumer_id']]
   if any(detail[k]!=item[k] for k in ('consumer_id','owner','route','effective_expert','original_filename','original_state_hash')) or detail['query']['query_id']!=item['query']['query_id']:raise ValueError('selected bank identity changed')
 records=[details.get(x['consumer_id'],x) for x in meta]
 library=json.loads((med/'EXPERT_LIBRARY_FROZEN.json').read_text());states={}
 if library.get('experts')!=146:raise ValueError('original library receipt missing')
 for entry in source:
  task=entry['task'];owner=task['order'];eid=task['edit_id'];state=torch.load((med if owner<=146 else extra)/'candidates'/f'MED_{owner}.pt',map_location='cpu',weights_only=True);h=digest(state)
  routed={r['original_state_hash'] for r in meta if r['effective_expert']==eid}
  if routed!={h} or (owner<=146 and library['states'].get(eid)!=h) or sum(v.numel() for v in state.values())!=7168 or any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('original state binding differs')
  states[owner]=h
 for arm in ('MCE','MDU'):
  old=prior/arm
  if (old/'tools/matched_train.py').read_bytes()!=Path(os.environ['TRAIN_SOURCE']).read_bytes():raise ValueError('prior training code differs')
  past=json.loads((old/'TRAIN_SOURCE_BOUND_PRIVATE.json').read_text());past_by={x['task']['order']:x for x in past}
  if set(past_by)!=prior_owners or any(past_by[x['task']['order']]!=x for x in source if x['task']['order'] in prior_owners):raise ValueError('prior training input differs')
  past_token=json.loads((old/'TOKEN_LOCK.private.json').read_text())
  if [x for x in past_token['records'] if x['order'] in prior_owners] != [x for x in token['records'] if x['order'] in prior_owners]:raise ValueError('prior token binding differs')
  if json.loads((old/'TRAIN_GUARD_EXIT.json').read_text())['code']!=0 or json.loads((old/'EVAL_GUARD_EXIT.json').read_text())['code']!=0:raise ValueError('prior completion missing')
  binding=json.loads((old/'STATE_BINDING.private.json').read_text())['initial_digests']
  if any(binding[str(k)]!=states[k] for k in prior_owners):raise ValueError('prior initial state differs')
  previous=sum((json.loads((old/f'TRAIN_COMPLETE_{i}.json').read_text())['results'] for i in range(2)),[])
  if {x['owner'] for x in previous}!=prior_owners:raise ValueError('prior training receipts differ')
  for row in previous:
   o=row['owner'];v=torch.load(old/'weights'/f'{o}_{arm}.pt',map_location='cpu',weights_only=True)
   if digest(v['state'])!=row['final_digest'] or v['initial_digest']!=states[o] or v['arm']!=arm or v['steps']!=160:raise ValueError('prior weight binding differs')
  for cid in old_ids:
   r=details[cid];key=hashlib.sha256(cid.encode()).hexdigest();pred=json.loads((old/'predictions'/f'{key}.json').read_text())
   if not pred.get('candidate') or pred.get('consumer_id')!=cid or pred.get('query_id')!=r['query']['query_id'] or pred.get('original_route')!=r['route'] or any(pred.get(k)!=r['judge_input'][k] for k in ('question','image_sha256','prompt_ids','attention_mask','generation')):raise ValueError('prior prediction binding differs')
  target=new/arm;target.mkdir(mode=0o700);(target/'tools').mkdir(mode=0o700)
  for name,value in (('ROLE_ADMISSION.json',admission),('SPAN_LOCK.private.json',weak),('TOKEN_LOCK.private.json',token),('TRAIN_SOURCE_BOUND_PRIVATE.json',source),('BANK_INPUTS_PRIVATE.json',{'status':'BANK_INPUTS_BOUND','records':records,'candidate_count':1513}),('STATE_BINDING.private.json',{'status':'STATE_BINDING_PASS','initial_digests':states}),('REUSE_LOCK.private.json',{'prior_owners':sorted(prior_owners),'prior_event_count':392,'new_event_count':1121})):
   write(target/name,value)
 write(new/'ROUTING_LOCK.private.json',{'status':'ROUTING_FROZEN','records':[{'consumer_id':r['consumer_id'],'query_id':r['query']['query_id'],'owner':r['owner'],'route':r['route'],'group':('original25' if r['consumer_id'] in old_ids else ('new76' if r['consumer_id'] in wanted else 'med49'))} for r in meta],'admitted_owners':sorted(owners)})
 write(new/'INPUT_BINDING.private.json',{'status':'BOUND_101','experts':101,'previous':25,'additional':76,'events':2148,'old_candidate_reuse':392,'new_generations':1121,'med_reuse':635})
 print(json.dumps({'status':'BOUND_101','experts':101,'previous':25,'additional':76,'events':2148,'new_generations':1121}))
if __name__=='__main__':main()
