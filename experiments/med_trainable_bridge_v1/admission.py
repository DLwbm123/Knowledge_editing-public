"""Fail-closed binding of the previously admitted source panel and saved expert states."""
import hashlib,json,os,time
from pathlib import Path
import torch

root=Path(os.environ['EXPERIMENT_ROOT']);old=Path(os.environ['DU_ROOT'])
med=Path(os.environ['ORIGINAL_ROOT']);extra=Path(os.environ['ADDITIONS_ROOT'])
deadline=float(os.environ['EXPERIMENT_DEADLINE'])

def digest(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def main():
 if time.time()>=deadline-120:raise TimeoutError('original deadline')
 admission=json.loads((old/'ROLE_ADMISSION.json').read_text())
 sources=json.loads((old/'TRAIN_SOURCE_BOUND_PRIVATE.json').read_text())
 bank=json.loads((old/'BANK_INPUTS_PRIVATE.json').read_text())
 library=json.loads((med/'EXPERT_LIBRARY_FROZEN.json').read_text())
 if admission.get('status')!='TRAINING_SOURCE_ROLE_ADMITTED' or len(admission['owners'])!=101 or [e['task']['order'] for e in sources]!=admission['owners']:
  raise ValueError('source role or owner mismatch')
 if bank.get('status')!='BANK_INPUTS_BOUND' or len(bank['records'])!=2148 or library.get('experts')!=146:
  raise ValueError('input or library evidence missing')
 by_id={}
 for row in bank['records']:
  k=row.get('effective_expert');v=row.get('original_state_hash')
  if k:
   if k in by_id and by_id[k]!=v:raise ValueError('inconsistent MED state binding')
   by_id[k]=v
 receipt={}
 for i in range(4):
  if json.loads((old/f'TRAIN_EXIT_{i}.json').read_text())['code']!=0:raise ValueError('DU training failure')
  for row in json.loads((old/f'TRAIN_COMPLETE_{i}.json').read_text())['results']:
   if row['owner'] in receipt:raise ValueError('duplicate DU owner')
   receipt[row['owner']]=row['final_digest']
 out=[]
 for entry in sources:
  task=entry['task'];owner=task['order'];identity=task['edit_id']
  state=torch.load((med if owner<=146 else extra)/'candidates'/f'MED_{owner}.pt',map_location='cpu',weights_only=True)
  candidate=torch.load(old/'weights'/f'{owner}_DU.pt',map_location='cpu',weights_only=True)
  mh=digest(state);dh=digest(candidate['state'])
  if owner not in receipt or dh!=receipt[owner] or candidate['module']!='model.layers.21.mlp.down_proj' or candidate['steps']!=160 or candidate['arm']!='DU':raise ValueError('DU state evidence mismatch')
  if mh!=by_id.get(identity) or (owner<=146 and mh!=library['states'].get(identity)):raise ValueError('MED owner state evidence mismatch')
  if set(state)!=set(candidate['state']) or any(state[k].shape!=candidate['state'][k].shape for k in state):raise ValueError('expert shapes differ')
  if any(not torch.isfinite(v).all() for v in list(state.values())+list(candidate['state'].values())):raise ValueError('nonfinite expert')
  out.append({'owner':owner,'med_digest':mh,'du_digest':dh,'seed':task['seed'],'edit_id':identity})
 if len(out)!=101 or len({x['owner'] for x in out})!=101:raise ValueError('owner coverage differs')
 p=root/'BRIDGE_ADMISSION.private.json'
 with p.open('x') as f:json.dump({'status':'STATE_IDENTITIES_BOUND','owners':out,'same_support_events':2148},f)
 p.chmod(0o600)
 print(json.dumps({'status':'STATE_IDENTITIES_BOUND','experts':len(out)}))

if __name__=='__main__':main()
