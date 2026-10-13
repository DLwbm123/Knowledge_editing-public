"""Resident pilot worker reusing frozen M1 CE, optimizer, routing and generation."""
import os,sys,json,time,hashlib,io,copy,fcntl,traceback
from pathlib import Path
ROOT=Path(os.environ['RUN_ROOT']);sys.path[:0]=[str(ROOT),str(ROOT/'source_patch'),str(ROOT/'source')]
from resources import read,write,check,session
from storage import Store

def main():
 import torch
 from freshstart import runtime as rt
 import worker_v3 as old
 from scripts.medtrace import stage15,run_selective_write
 from scripts.medtrace.run_selective_write import teacher_batch
 from scripts.medtrace.stage18_cfact import assert_base_off
 from methods.medtrace.selective_write import full_vocab_kl
 from structures import LR4
 from training import stage,save_final,load,u_teachers,derive_seed
 import diagnostics,evaluation
 from router_r3 import RejectRouter
 from bindings import expected
 from protection import protector
 from mechanisms import residual_energy
 from hard_pool import select
 slot=int(os.environ['PHYSICAL_GPU']);store=Store(ROOT);rt.ROOT=ROOT;rt.GPU=os.environ['CUDA_VISIBLE_DEVICES'];rt.check_budget=lambda **kw:check(kw.get('training',False));rt.gpu_session=session
 # Historical forced-support diagnostics are not part of this new finite pilot.
 diagnostics.selected=lambda t:False;diagnostics.diagnose=lambda *a,**k:None
 def guarded(p,d):store.write(str(Path(p).relative_to(ROOT)),json.dumps(d,ensure_ascii=False).encode())
 def tensor_save(p,x):
  assert torch.is_tensor(x) and x.numel()<1000000
  b=io.BytesIO();torch.save(x,b);store.write(str(Path(p).relative_to(ROOT)),b.getvalue())
 old.write=stage15.write=evaluation.write=guarded;run_selective_write.save=tensor_save
 tasks=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'];initializers=read(ROOT/'private/INITIALIZERS.json');began=time.time()
 def rel(m,o):return f'adapters/s20260929/{m}/e{o:03d}.pt'
 try:
  with session('SCOPE_WORKER_'+str(slot),7200):
   runtime=rt.load_runtime(ROOT/'jobs'/('worker'+str(slot)),20260929);runtime.model.register_forward_pre_hook(lambda m,a:check())
   counters={'all_model_calls':0}
   def count(*a):counters['all_model_calls']+=1
   runtime.model.register_forward_pre_hook(count)
   protocol=dict(model=str((ROOT/'models/llava-med-v1.5-mistral-7b').resolve()),precision='float16',backend=dict(torch=str(torch.__version__),cuda=torch.version.cuda))
   evaluation.SEED=20260929
   while time.time()-began<6600 and not (ROOT/'ADMISSION_STOP').exists():
    with (ROOT/'QUEUE.lock').open('a') as f:
     fcntl.flock(f,fcntl.LOCK_EX);q=read(ROOT/'QUEUE.json');j=next((j for j in q if j['status']=='PENDING' and j.get('resume_gpu',slot)==slot and all((ROOT/x).exists() for x in j['requires'])),None)
     if j is None:break
     j.update(status='RUNNING',pid=os.getpid(),gpu=slot,started_epoch=time.time());write(ROOT/'QUEUE.json',q)
    before=counters.copy();jid=j['id'];started=time.time()
    if j['mode']=='train':
     t=next(t for t in tasks if t['order']==j['order']);o=t['order'];ss=derive_seed(t['canonical_edit_id'],20260929)+1;rec=old.record(t)
     meta=next(x for x in initializers if x['order']==o);p=ROOT/meta['relative_path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256'];initial=torch.load(p,map_location='cpu',weights_only=False)
     ex=LR4().cuda();ex.load_state_dict(initial['expert']);ex.requires_grad_(False)
     key,radius=old.router_entry(runtime,t);router=RejectRouter();router.add(t['canonical_edit_id'],key,radius)
     bg=read(ROOT/'private/U_bg.json')['rows']
     for row in bg:row['input_hash']=old.input_id(row)
     features=[old.key(runtime,row,rec) for row in bg];hardmeta=select(bg,features,key,radius);hardids={r['input_hash'] for r in hardmeta};hard=[next(x for x in bg if x['input_hash']==m['input_hash']) for m in hardmeta];assert hard
     write(ROOT/'private/hard'/f'{o}.json',dict(rows=hardmeta,universe=len(bg),source_cap=2,frozen_before_new_outputs=True));del features
     groups,size=u_teachers(runtime,t)
     # Bounded lazy teacher cache: retain at most two background examples, not a full vocabulary dataset.
     from collections import OrderedDict
     cache=OrderedDict()
     def teacher(row,hook=None):
      k=old.input_id(row)
      if k in cache:cache.move_to_end(k);return cache[k]
      if hook:hook.detach()
      try:
       assert_base_off(runtime);out,_=old.base(runtime,row,rec,score=False);kw,labels,mask,b=teacher_batch(runtime,dict(row,eqkey=k),out['raw_token_ids'])
       with torch.no_grad():lp=runtime.model(**kw).logits[mask].float().log_softmax(-1).cpu()
       value=(kw,labels,mask,lp)
      finally:
       if hook:hook.attach()
      cache[k]=value
      while len(cache)>2:cache.popitem(last=False)
      return value
     class LazyRows:
      def __init__(self,rows):self.rows=rows;self.hook=None
      def __len__(self):return len(self.rows)
      def __getitem__(self,i):return teacher(self.rows[i],self.hook)
     background=LazyRows(bg);hardrows=LazyRows(hard)
     # Same Q, same common start, fixed mean scale for both lambda variants.
     scale_path=ROOT/'private/scales'/f'{o}.json'
     if scale_path.exists():scale=torch.tensor(read(scale_path)['c_i'],device='cuda')
     else:
      energies=[];baseenergies=[]
      for row in [groups[1][0],teacher(hard[0])]:
       kw,labels,mask,lp=row
       def capture(module,args,out):
        energies.append(float(residual_energy(ex.A,ex.B,args[0],mask)));baseenergies.append(float(out[mask].float().square().sum(-1).mean()))
       handle=runtime.get_module(rt.LAYER).register_forward_hook(capture)
       try:
        with torch.no_grad():runtime.model(**kw)
       finally:handle.remove()
      scale=torch.tensor(max(sum(energies)/len(energies),1e-6*sum(baseenergies)/len(baseenergies),1e-12),device='cuda')
      write(scale_path,dict(c_i=float(scale),initial_sha256=meta['sha256'],Q=['first_original_U_new','first_hard_by_frozen_distance'],Q_weights=[.5,.5],layer_output_not_logits=True))
     for method in j['methods']:
      dst=rel(method,o)
      if not (ROOT/dst).exists():
       ex.load_state_dict(initial['expert']);ex.requires_grad_(False)
       if method!='E_orig':
        branch='AHS' if method.startswith('AHS') else method;lam=.01 if method=='AHS_001' else .1 if method=='AHS_01' else 0
        apply=protector(runtime,runtime.get_module(rt.LAYER),ex,*groups,background,hardrows,branch,ss,full_vocab_kl,scale,lam)
        def protect(hook,step):
         background.hook=hardrows.hook=hook
         return apply(hook,step)
        stage(runtime,t,ex,'LR','continuation',ss,method,protect,run_seed=20260929,structure='LR')
       save_final(store,dst,ex,20260929,t['canonical_edit_id'],'LR',80 if method!='E_orig' else 0,expected(t,20260929,'LR',method,'continuation',80 if method!='E_orig' else 0))
      expert,info=load(dst,expected(t,20260929,'LR',method,'continuation',80 if method!='E_orig' else 0));expert.requires_grad_(False)
      binding=[dict(adapter=dst,sha256=hashlib.sha256((ROOT/dst).read_bytes()).hexdigest(),actual_steps=info['step'],origin='INIT_POST80')]
      evaluation.evaluate(runtime,[t],{t['canonical_edit_id']:expert},router,method,'single',1,ROOT/'jobs'/jid/method/'single',binding,protocol)
      chk=copy.deepcopy(t);chk['evaluation']=read(ROOT/'private/CHECK_POS.json')[str(o)]+[dict(r,task='CHECK_NEG',query_id='check-'+old.input_id(r)) for r in read(ROOT/'private/CHECK_NEG.json')['rows']]
      evaluation.evaluate(runtime,[chk],{t['canonical_edit_id']:expert},router,method,'CHECK_FORCED_ON',1,ROOT/'jobs'/jid/method/'check',binding,protocol,forced=True)
      del expert
     del ex,groups,cache,background,hardrows
    else:
     bank={};router=RejectRouter();bs=[];selected=[next(t for t in tasks if t['order']==o) for o in j['orders']];method=j['method']
     for i,t in enumerate(selected,1):
      dst=rel(method,t['order']);expert,info=load(dst,expected(t,20260929,'LR',method,'continuation',0 if method=='E_orig' else 80));expert.requires_grad_(False);bank[t['canonical_edit_id']]=expert;key,radius=old.router_entry(runtime,t);router.add(t['canonical_edit_id'],key,radius);bs.append(dict(adapter=dst,sha256=hashlib.sha256((ROOT/dst).read_bytes()).hexdigest(),actual_steps=info['step'],origin='INIT_POST80'))
      if i in j['prefixes']:evaluation.evaluate(runtime,selected[:i],bank,router,method,'sequential',i,ROOT/'jobs'/jid/f'p{i}',bs,protocol)
     del bank
    assert runtime.base_guard.verify()['unchanged'];torch.cuda.empty_cache();write(ROOT/'jobs'/jid/'COMPUTE_COUNTS.json',dict(model_calls=counters['all_model_calls']-before['all_model_calls'],seconds=time.time()-started))
    with (ROOT/'QUEUE.lock').open('a') as f:
     fcntl.flock(f,fcntl.LOCK_EX);q=read(ROOT/'QUEUE.json');next(x for x in q if x['id']==jid).update(status='COMPLETE',ended_epoch=time.time());write(ROOT/'QUEUE.json',q)
   write(ROOT/'jobs'/f'WORKER_{slot}.json',dict(status='IDLE',pid=os.getpid()))
 except Exception as e:
  write(ROOT/'jobs'/f'WORKER_{slot}.json',dict(status='FAILED',pid=os.getpid(),error=str(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
