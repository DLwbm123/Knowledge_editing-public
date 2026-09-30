"""Evaluate every pre-frozen holdout candidate on the already trained banks."""
import os,sys,time,json,io,fcntl,traceback,hashlib
from pathlib import Path
ROOT=Path(os.environ['RUN_ROOT']);sys.path[:0]=[str(ROOT),str(ROOT/'source_patch'),str(ROOT/'source')]
from resources import read,write,check,session

def evaluation_task(rows):
 assert rows and all(x['purpose']=='LOCALITY_STRESS_HOLDOUT' and x['scope']=='negative' for x in rows)
 assert len({(x['image_sha256'],x['question']) for x in rows})==len(rows)
 return dict(canonical_edit_id='AUX_HOLDOUT_NO_ASSOCIATED_EDIT',order=0,native=rows[0],fit_questions=[rows[0]['question']],evaluation=[dict(x,task='NEW_STRESS_HOLDOUT',query_id='holdout-'+hashlib.sha256((x['image_sha256']+'\0'+x['question']).encode()).hexdigest()) for x in rows])

def main():
 import torch
 from freshstart import runtime as rt
 import worker_v3 as old
 from scripts.medtrace import stage15,run_selective_write
 from storage import Store,digest_file
 from training import load
 from bindings import expected
 from router_r3 import RejectRouter
 import evaluation
 store=Store(ROOT)
 def guarded(p,d):store.write(str(Path(p).relative_to(ROOT)),json.dumps(d,ensure_ascii=False).encode())
 def tensor_save(p,x):
  assert torch.is_tensor(x) and x.numel()<1000000
  b=io.BytesIO();torch.save(x,b);store.write(str(Path(p).relative_to(ROOT)),b.getvalue())
 old.write=stage15.write=evaluation.write=guarded;run_selective_write.save=tensor_save
 rt.ROOT=ROOT;rt.GPU=os.environ['CUDA_VISIBLE_DEVICES'];rt.check_budget=lambda **kw:check(kw.get('training',False));rt.gpu_session=session;evaluation.SEED=20260929
 slot=int(os.environ['PHYSICAL_GPU']);tasks=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'];h=evaluation_task(read(ROOT/'private/LOCALITY_STRESS_HOLDOUT.json')['rows'])
 with session('HOLDOUT_EVALUATION_'+str(slot),7200):
  runtime=rt.load_runtime(ROOT/'auxiliary'/('worker'+str(slot)),20260929);calls=[0]
  def count(*args):check();calls[0]+=1
  runtime.model.register_forward_pre_hook(count)
  protocol=dict(model=str((ROOT/'models/llava-med-v1.5-mistral-7b').resolve()),precision='float16',backend=dict(torch=str(torch.__version__),cuda=torch.version.cuda))
  while True:
   with (ROOT/'HOLDOUT_QUEUE.lock').open('a') as f:
    fcntl.flock(f,fcntl.LOCK_EX);q=read(ROOT/'HOLDOUT_QUEUE.json');j=next((x for x in q if x['status']=='PENDING'),None)
    if j is None:break
    j.update(status='RUNNING',pid=os.getpid(),gpu=slot,started=time.time());write(ROOT/'HOLDOUT_QUEUE.json',q)
   before=calls[0];started=time.time();bank={};router=RejectRouter();bindings=[];method=j['method'];selected=[t for t in tasks if t['order'] in j['orders']]
   for i,t in enumerate(selected,1):
    check();dst=f'adapters/s20260929/{method}/e{t["order"]:03d}.pt';expert,info=load(dst,expected(t,20260929,'LR',method,'continuation',0 if method=='E_orig' else 80));expert.requires_grad_(False);bank[t['canonical_edit_id']]=expert
    key,radius=old.router_entry(runtime,t);router.add(t['canonical_edit_id'],key,radius);bindings.append(dict(adapter=dst,sha256=digest_file(ROOT/dst),actual_steps=info['step'],origin='INIT_POST80'))
    if i in [4,8,12,24]:
     folder=ROOT/'auxiliary/holdout'/j['id']/f'p{i}'
     if not (folder/'CONSUMERS.json').exists():
      torch.cuda.synchronize();start=time.time();n=calls[0]
      result=evaluation.evaluate(runtime,[h],bank,router,method,'holdout',i,folder,bindings,protocol)
      assert len(result)==len(h['evaluation']) and runtime.base_guard.verify()['unchanged']
      torch.cuda.synchronize();write(folder/'COST.json',dict(seconds=time.time()-start,model_calls=calls[0]-n,candidates=len(result),includes_preparation_base_feature_and_cache=True))
   del bank,expert;torch.cuda.empty_cache()
   write(ROOT/'auxiliary/holdout'/j['id']/'COMPUTE_COUNTS.json',dict(seconds=time.time()-started,model_calls=calls[0]-before))
   with (ROOT/'HOLDOUT_QUEUE.lock').open('a') as f:
    fcntl.flock(f,fcntl.LOCK_EX);q=read(ROOT/'HOLDOUT_QUEUE.json');next(x for x in q if x['id']==j['id']).update(status='COMPLETE',ended=time.time());write(ROOT/'HOLDOUT_QUEUE.json',q)
 write(ROOT/'auxiliary'/f'WORKER_{slot}.json',dict(status='IDLE',pid=os.getpid()))
 if all(x['status']=='COMPLETE' for x in read(ROOT/'HOLDOUT_QUEUE.json')):write(ROOT/'RUN_STATUS.json',dict(status='GPU_QUEUE_FINISHED',phase='FROZEN_HOLDOUT_EVALUATION',original_training_jobs=65,original_continuations=200))
if __name__=='__main__':
 try:main()
 except Exception as e:
  write(ROOT/'auxiliary'/f'WORKER_{os.environ["PHYSICAL_GPU"]}.json',dict(status='FAILED',pid=os.getpid(),error=str(e),traceback=traceback.format_exc()));write(ROOT/'RUN_STATUS.json',dict(status='BLOCKED',reason='HOLDOUT_WORKER_FAILURE'));raise
