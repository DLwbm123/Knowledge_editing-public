"""Bind existing admitted source and bank metadata without reading additional sources."""
import json
import os
from pathlib import Path
import resource
import signal
import time

parent=Path(os.environ['BASE_ROOT'])
old=Path(os.environ['PRIOR_RUN'])
new=Path(os.environ['EXPERIMENT_ROOT'])
deadline=float(os.environ['EXPERIMENT_DEADLINE'])
resource.setrlimit(resource.RLIMIT_FSIZE,(256_000_000,256_000_000))
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('preparation timed out')))
if deadline-time.time()<=120:raise TimeoutError('original deadline expired')
signal.alarm(min(120,max(1,int(deadline-time.time()-120))))
def read(p):
 with p.open() as f:return json.load(f)
def write(p,v):
 fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
 with os.fdopen(fd,'w') as f:json.dump(v,f,ensure_ascii=False,separators=(',',':'))
a=read(old/'ROLE_ADMISSION.json'); entries=read(old/'TRAIN_SOURCE_BOUND_PRIVATE.json')
weak=read(old/'SPAN_LOCK.private.json'); token=read(parent/'TOKEN_LOCK.private.json')
bank=read(old/'BANK_INPUTS_PRIVATE.json')
owners=a['owners']
if a.get('status')!='TRAINING_SOURCE_ROLE_ADMITTED' or len(owners)!=101 or len(set(owners))!=101 or [e['task']['order'] for e in entries]!=owners:raise ValueError('original admission does not bind 101 owners')
if token.get('status')!='SOURCE_TOKEN_CHECK_ONLY' or [v['order'] for v in token['records']]!=owners or any(v['status']!='TOKEN_BOUND' for v in token['records']):raise ValueError('token lock incomplete')
if weak.get('status')!='WEAK_TOKEN_PROXY_ONLY' or [v['order'] for v in weak['records']]!=owners:raise ValueError('weak lock identity mismatch')
weakowners={v['order'] for v in weak['records'] if v['status']=='WEAK_TOKEN_MATCH'}
if len(weakowners)!=25 or not weakowners.issubset(owners):raise ValueError('weak coverage mismatch')
if bank.get('status')!='BANK_INPUTS_BOUND' or len(bank['records'])!=2148:raise ValueError('bank input count mismatch')
if len({e['task']['edit_id'] for e in entries})!=101:raise ValueError('duplicate expert ID')
for arm in ('DU','D'):
 ids={e['task']['edit_id'] for e in entries if arm=='DU' or e['task']['order'] in weakowners}
 count=sum(v['effective_expert'] in ids for v in bank['records'])
 if count!=(1513 if arm=='DU' else 392):raise ValueError('routing coverage changed')
 root=new/arm
 root.mkdir(mode=0o700,parents=False,exist_ok=False)
 (root/'tools').mkdir(mode=0o700)
 for name,data in [('ROLE_ADMISSION.json',a),('TRAIN_SOURCE_BOUND_PRIVATE.json',entries),('SPAN_LOCK.private.json',weak),('TOKEN_LOCK.private.json',token)]:write(root/name,data)
 arm_bank=dict(bank,candidate_count=count)
 write(root/'BANK_INPUTS_PRIVATE.json',arm_bank)
 write(root/'BINDING_RECEIPT.json',{'status':'BOUND_NOT_YET_EXECUTION_ADMISSION','owners':len(ids),'events':len(bank['records']),'generated_plan':count,'fallback_plan':2148-count,'clock_end':deadline})
print(json.dumps({'status':'PREPARATION_BOUND','DU_candidates':1513,'D_candidates':392,'admitted':101,'unresolved_not_loaded':49,'new_updates':0,'judge_attempts':0}))
