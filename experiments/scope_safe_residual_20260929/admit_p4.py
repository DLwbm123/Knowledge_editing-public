import os,sys,json,time,hashlib,shutil,fcntl
from pathlib import Path
import torch
r=Path('/data/bmw/Knowledge_editing/outputs/scope-safe-residual-20260929/run');old=r.parents[1]/'appendix-20260929/run'
os.environ['RUN_ROOT']=str(r);sys.path.insert(0,str(r))
from resources import read,write
assert not (r/'P4_PROTOCOL.json').exists() and not (r/'STOP').exists()
q=read(r/'QUEUE.json');assert all(x['status']=='COMPLETE' for x in q)
assert all(x['status'] in ['FORMAT_VALID','FAILED_NO_RETRY'] for x in read(r/'RESOURCE_LEDGER.json')['judge_attempts'])
fix=r/'fix/p2_before_p4';fix.mkdir(parents=True,exist_ok=True)
for n in ['QUEUE.json','RESOURCE_LEDGER.json','private/INITIALIZERS.json','private/CHECK_POS.json']: (fix/n).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(r/n,fix/n)
protocol=dict(phase='P4_FROZEN_REG24',methods=['A0','AH','AHS_01'],lambda_res=.1,orders=list(range(25,49)),seed=20260929,steps=80,selection='DEV only: A0 baseline; AH simple control with better generalization than AU and better pressure than AK; AHS fixed CHECK lambda candidate. No REG retuning.',candidate_gain_confirmed=False,reg_CHECK_positive_unavailable=True,original_science_id=read(r/'SCIENCE_LOCK.json')['id'])
protocol['id']=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest();write(r/'P4_PROTOCOL.json',protocol)
write(r/'P3_FEASIBILITY.json',dict(status='UNSUPPORTED_CAL_SCOPE',Rneg='No certified cross-image CAL positives',R1='No legal CAL_SCOPE positive set; do not fit thresholds on CHECK/formal data',R0_comparisons='Existing A0/AHS DEV outputs retained',not_performance_failure=True))
tasks=read(r/'private/TASKS_R2_LOCKED.json')['tasks'];init=read(r/'private/INITIALIZERS.json');checks=read(r/'private/CHECK_POS.json');storage=read(old/'STORAGE_LEDGER.json')['artifacts'];oldsci=read(old/'SCIENCE_LOCK.json')['id']
for t in tasks[24:48]:
 o=t['order'];src=old/f'adapters/s20260927/M1/e{o:03d}.pt';x=torch.load(src,map_location='cpu',weights_only=False);b=x['binding']
 assert x['step']==80 and x['kind']=='LR' and b['method']=='M1' and b['science_id']==oldsci and b['edit']==t['canonical_edit_id']
 assert b['initialization_W0_hash']==storage[f'checkpoints/W0/s20260927/CP/e{o:03d}.pt']['hash']
 dst=r/f'initializers/e{o:03d}.pt';assert not dst.exists();shutil.copy2(src,dst)
 init.append(dict(order=o,sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),binding=b,relative_path=str(dst.relative_to(r)),extension_protocol=protocol['id']));checks[str(o)]=[]
 q.append(dict(id=f'P4-{o}',mode='train',order=o,methods=['E_orig']+protocol['methods'],status='PENDING',requires=[]))
for m in ['E_orig']+protocol['methods']:q.append(dict(id=f'P4-bank-{m}',mode='bank',method=m,orders=list(range(25,49)),prefixes=[4,8,12,24],status='PENDING',requires=[f'adapters/s20260929/{m}/e{o:03d}.pt' for o in range(25,49)]))
write(r/'private/INITIALIZERS.json',init);write(r/'private/CHECK_POS.json',checks)
write(r/'P4_DATA_AMENDMENT.json',dict(unchanged_DEV_initializers=24,unchanged_DEV_CHECK=48,REG_initializers_imported=24,REG_positive_CHECK=0,REG_negative_CHECK='same fixed pool; diagnostic only; no parameter selection',original_science_lock_unchanged=True,added_binding_field='extension_protocol'))
with (r/'QUEUE.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);write(r/'QUEUE.json',q)
write(r/'RUN_STATUS.json',dict(status='RUNNING',phase='P4_FROZEN_REG24',epoch=time.time(),continuations=72))
if (r/'SCORER_DONE').exists():(r/'SCORER_DONE').rename(fix/'SCORER_DONE')
print('P4 admitted: 24 paired edit jobs, 4 banks, 72 continuations; no old training repeated')
