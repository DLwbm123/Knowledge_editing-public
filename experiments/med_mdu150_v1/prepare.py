"""Bind the 49 fixed additional sources and their original MED states."""
import hashlib,json,os,signal,time
from pathlib import Path
import torch
import identity_lock,extract_training,index
root=Path(os.environ['EXPERIMENT_ROOT']);original=Path(os.environ['ORIGINAL_ROOT']);prior=Path(os.environ['PRIOR_101_ROOT'])
def write(name,obj):
 p=root/name
 with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,separators=(',',':'))
 p.chmod(0o600)
def digest(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def main():
 if time.time()>float(os.environ['EXPERIMENT_DEADLINE'])-600:raise TimeoutError('deadline')
 if json.loads((prior/'EVAL_GUARD_EXIT.json').read_text())['code']!=0:raise ValueError('prior bank incomplete')
 priorowners=set(json.loads((prior/'ROLE_ADMISSION.json').read_text())['owners']);wanted=set(range(1,147))-priorowners
 if len(priorowners)!=101 or len(wanted)!=49:raise ValueError('owner coverage')
 tasks=[t for t in identity_lock.tasks(original/'INPUTS_FROZEN.json') if t['order'] in wanted]
 with (original/'INPUTS_FROZEN.json').open('rb') as f: entries=[extract_training.task_at(f,t) for t in tasks]
 rows=extract_training.source_rows(original/'INPUTS_FROZEN.json',{str(n) for n in wanted})
 for e in entries:
  task=e['task'];r=rows[str(task['order'])]
  if len(r)!=5 or [x['audit_role'] for x in r]!=['NATIVE','FIT','FIT','FIT','FIT'] or [x['question'] for x in r]!=[task['native']['question']]+task['fit_questions']:raise ValueError('source binding')
  e['source_rows']=r
 bank=json.loads((prior/'BANK_INPUTS_PRIVATE.json').read_text())['records'];lib=json.loads((original/'EXPERT_LIBRARY_FROZEN.json').read_text());states={}
 for e in entries:
  n=e['task']['order'];eid=e['task']['edit_id'];v=torch.load(original/'candidates'/f'MED_{n}.pt',map_location='cpu',weights_only=True);h=digest(v)
  if len(v)!=4 or sum(x.numel() for x in v.values())!=7168 or any(not torch.isfinite(x).all() for x in v.values()):raise ValueError('invalid original state')
  if lib['states'].get(eid)!=h or {x['original_state_hash'] for x in bank if x['effective_expert']==eid}!={h}:raise ValueError('original state binding')
  states[n]=h
 if len(states)!=49:raise ValueError('state count')
 write('ROLE_ADMISSION.json',{'status':'SOURCE_49_INPUTS_CHECKED','owners':[e['task']['order'] for e in entries],'basis':'user-authorized sources; test-question overlap check recorded in public preparation report'})
 write('TRAIN_SOURCE_BOUND_PRIVATE.json',entries)
 write('STATE_BINDING.private.json',{'status':'STATE_BINDING_PASS','initial_digests':states})
 write('SPAN_LOCK.private.json',{'status':'WEAK_TOKEN_PROXY_ONLY','records':[]})
 print(json.dumps({'status':'SOURCES_AND_STATES_BOUND','owners':49,'rows':245,'model_calls':0}))
if __name__=='__main__':main()
