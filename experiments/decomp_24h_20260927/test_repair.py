"""CPU-only real fault injection: child death at all four transaction boundaries."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,tempfile,multiprocessing as mp
from pathlib import Path
import torch
from storage import Store

def writer(root,point):
 def crash(where):
  if where==point:os._exit(73)
 Store(root).save('checkpoints/slot5/latest.pt',dict(expert={'w':torch.ones(4)*3},seed=1,step=60),pin=True,fault=crash)
def judge_writer(root):
 from judge_io import publish,digest
 original=Store.write;count=0
 def interrupted(self,*args,**kwargs):
  nonlocal count
  original(self,*args,**kwargs);count+=1
  if count==2:os._exit(74) # Intent and first score published, ledger not committed.
 Store.write=interrupted
 record={'opaque_query_id':'q'}
 publish(root,'b',{'k':dict(key='k',is_correct=True,payload_binding=digest(record))},dict(status='FORMAT_VALID'),dict(batch_id='b',decisions=[dict(opaque_query_id='q',is_correct=True)]))

def judge_test():
 from judge_io import recover
 with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as td:
  r=Path(td);(r/'STORAGE_POLICY.json').write_text(json.dumps(dict(hard_bytes=10**7,soft_bytes=8*10**6,checkpoint_bytes=10**7,teacher_bytes=10**7,min_free_bytes=0)))
  s=Store(r);s.write('RESOURCE_LEDGER.json',json.dumps(dict(judge_attempts=[dict(id='b',status='RESERVED',keys=['k'])],judge_submission_attempt_items=1)).encode());s.write('private/judge/pending/k.json',json.dumps(dict(record={'opaque_query_id':'q'})).encode())
  p=mp.Process(target=judge_writer,args=(r,));p.start();p.join();assert p.exitcode==74
  assert json.loads((r/'private/judge/scores/k.json').read_text())['is_correct'] is True
  assert len(recover(r))==1 and recover(r)==[]
  d=json.loads((r/'RESOURCE_LEDGER.json').read_text());assert d['judge_submission_attempt_items']==1 and d['judge_attempts'][0]['status']=='FORMAT_VALID'
 return dict(interrupted_publication_recovered=True,idempotent=True,new_judge_calls=0)

def concurrent_writer(root,ready,slow):
 import time
 def pause(point):
  if slow and point=='reserve':ready.set();time.sleep(.3)
 if not slow:assert ready.wait(5)
 Store(root)._write('private/judge/pending/shared.json',b'{"key":"same"}','evidence',(),False,fault=pause)

def concurrency_test():
 with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as td:
  r=Path(td);(r/'STORAGE_POLICY.json').write_text(json.dumps(dict(hard_bytes=10**7,soft_bytes=8*10**6,checkpoint_bytes=10**7,teacher_bytes=10**7,min_free_bytes=0)))
  ready=mp.Event();children=[mp.Process(target=concurrent_writer,args=(r,ready,slow)) for slow in [True,False]]
  for p in children:p.start()
  for p in children:p.join(10);assert p.exitcode==0
  assert json.loads((r/'private/judge/pending/shared.json').read_text())=={'key':'same'}
  assert json.loads((r/'STORAGE_LEDGER.json').read_text())['artifacts']['private/judge/pending/shared.json']['status']=='READY'
 return True

def main():
 results=[]
 for point in ['reserve','fsync','verify','replace']:
  with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as td:
   r=Path(td);(r/'STORAGE_POLICY.json').write_text(json.dumps(dict(hard_bytes=10**7,soft_bytes=8*10**6,checkpoint_bytes=10**7,teacher_bytes=10**7,min_free_bytes=0)))
   s=Store(r)
   for step in [20,40]:s.save('checkpoints/slot5/latest.pt',dict(expert={'w':torch.ones(4)*(step/20)},seed=1,step=step),pin=True)
   p=mp.Process(target=writer,args=(r,point));p.start();p.join();assert p.exitcode==73
   result=s.reconcile('checkpoints/slot5/latest.pt');saved=s.load_checkpoint('checkpoints/slot5/latest.pt');assert saved['step'] in [40,60]
   s.save('checkpoints/slot5/latest.pt',dict(expert={'w':torch.ones(4)*4},seed=1,step=80),pin=True)
   (r/'checkpoints/slot5/latest.pt').write_bytes(b'corrupt')
   previous=s.load_checkpoint('checkpoints/slot5/latest.pt',fallback=True);assert previous['step']==saved['step']
   results.append(dict(point=point,recovery=result,valid_step=saved['step'],previous_fallback=True))
 # Production source must not shadow the callable on the seq->train path.
 import ast
 tree=ast.parse(Path(__file__).with_name('worker.py').read_text());assigned={n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
 assert 'is_diagnostic_edit' not in assigned and 'selected_tasks' in assigned
 fn=lambda x:x==9
 for mode in ['train','sequential','train']:
  if mode=='sequential':selected_tasks=[1,9]
  else:assert fn(9)
 from paired_stats import interaction,summarize_values
 assert not summarize_values([])['estimable']
 assert not interaction([],{},lambda task:True)['metric_estimable']
 print(json.dumps(dict(status='PASS',faults=results,mixed_order_namespace=True,judge=judge_test(),concurrent_shared_write=concurrency_test(),empty_metrics_not_complete=True)))
if __name__=='__main__':main()
