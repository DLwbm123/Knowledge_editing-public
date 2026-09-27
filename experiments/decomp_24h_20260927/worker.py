"""One resident backbone; dependency-ordered stage branches and real bank evaluation."""
import os,sys,json,time,hashlib,copy,traceback,io
from pathlib import Path
ROOT=Path(os.environ['RUN_ROOT']);sys.path[:0]=[str(ROOT),str(ROOT/'source_patch'),str(ROOT/'source')]
from resources import read,write,session,check
from storage import Store

def main():
 import torch
 from freshstart import runtime as rt
 from m3bench_repro.editors.llava_runtime import seed_everything
 import worker_v3 as old
 from scripts.medtrace import stage15,run_selective_write
 from structures import convert
 from training import new,stage,load,save_final,u_teachers,p_teacher,protector,derive_seed
 from diagnostics import selected
 from router_r3 import RejectRouter
 import evaluation
 slot=int(os.environ['PHYSICAL_GPU']);store=Store(ROOT);rt.ROOT=ROOT;rt.GPU=os.environ['CUDA_VISIBLE_DEVICES'];rt.check_budget=lambda **kw:check(kw.get('training',False));rt.gpu_session=session
 def guarded_write(p,d):
  rel=str(Path(p).relative_to(ROOT));store.write(rel,json.dumps(d,ensure_ascii=False).encode())
 def key_save(p,x):
  assert torch.is_tensor(x) and x.numel()<1000000,'Only small router key saves permitted';b=io.BytesIO();torch.save(x,b);store.write(str(Path(p).relative_to(ROOT)),b.getvalue())
 old.write=guarded_write;stage15.write=guarded_write;evaluation.write=guarded_write;run_selective_write.save=key_save
 started=time.time();tasks=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'];gs={x['order']:x for x in read(ROOT/'private/G_SUPPORTS.json')};methods={'CP':['M0','M1','M2','M3'],'TK':['M4','M5'],'LR':['M6']}
 def finalrel(seed,m,o):return f'adapters/s{seed}/{m}/e{o:03d}.pt'
 try:
  with session('WORKER_'+str(slot),float(os.environ.get('LEASE_SECONDS','7200'))):
   runtime=rt.load_runtime(ROOT/'jobs'/('worker'+str(slot)),20260927);runtime.model.register_forward_pre_hook(lambda m,a:check())
   counters=dict(model_calls=0,input_positions=0)
   def count_forward(module,args,kwargs):
    counters['model_calls']+=1;x=kwargs.get('inputs_embeds',kwargs.get('input_ids'))
    if x is not None:counters['input_positions']+=x.shape[0]*x.shape[1]
   runtime.model.register_forward_pre_hook(count_forward,with_kwargs=True)
   protocol=dict(model=str((ROOT/'models/llava-med-v1.5-mistral-7b').resolve()),precision='float16',backend=dict(torch=str(torch.__version__),cuda=torch.version.cuda),code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT).glob('*.py')})
   while True:
    if time.time()-started>float(os.environ.get('LEASE_SECONDS','7200'))-600:break
    import fcntl
    with (ROOT/'QUEUE.lock').open('a') as f:
     fcntl.flock(f,fcntl.LOCK_EX);queue=read(ROOT/'QUEUE.json');job=None
     for j in queue:
      if j['status']!='PENDING' or j.get('resume_gpu',slot)!=slot:continue
      if any(not (ROOT/p).exists() for p in j.get('requires',[])):continue
      job=j;break
     if job is None:break
     job.update(status='RUNNING',pid=os.getpid(),gpu=slot,started_epoch=time.time());write(ROOT/'QUEUE.json',queue)
    jid=job['id'];seed=job['seed'];evaluation.SEED=seed;counts_before=counters.copy();job_began=time.time()
    if job['mode']=='train':
     t=next(t for t in tasks if t['order']==job['order']);kind=job['kind'];ss=derive_seed(t['canonical_edit_id'],seed);binding=dict(edit=t['canonical_edit_id'],seed=seed,structure=kind,backend=protocol['backend'],code=protocol['code'],support_digest=old.digest(dict(P=t['semantic_fit_questions'],Uold=t['U_fit'],Unew=t['U_new'],G=gs[t['order']])))
     wrel=f'checkpoints/W0/s{seed}/{kind}/e{t["order"]:03d}.pt';arms=job['methods']
     if (ROOT/wrel).exists():ex,_=load(wrel)
     else:
      seed_everything(ss);ex=new(kind);resume_stage='native';rolling=ROOT/f'checkpoints/slot{slot}/latest.pt'
      if rolling.exists():
       saved=torch.load(rolling,map_location='cuda',weights_only=True);ident=saved['binding']
       if ident.get('edit')==t['canonical_edit_id'] and ident.get('kind')==kind and ident.get('method')==kind and ident.get('seed') in [ss,ss+1]:resume_stage=ident['stage'];ex.load_state_dict(saved['expert'])
      stages=['native','A2','W0'];assert resume_stage in stages
      for phase in stages[stages.index(resume_stage):]:stage(runtime,t,ex,kind,phase,ss+1 if phase=='W0' else ss,kind)
      future=([('M1' if kind=='CP' else 'M4')+suffix for suffix in ['_STRUCT','_R8']] if seed==20260927 and kind in ['CP','TK'] and selected(t) else [])
      save_final(store,wrel,ex,seed,t['canonical_edit_id'],kind,320,binding,consumers=arms+future)
     initial={k:v.detach().clone() for k,v in ex.state_dict().items()};groups,cachebytes=u_teachers(runtime,t)
     for method in arms:
      rel=finalrel(seed,method,t['order'])
      if (ROOT/rel).exists():expert,_=load(rel)
      else:
       ex.load_state_dict(initial);expert=copy.deepcopy(ex) if method.endswith('_STRUCT') else convert(ex,ss+1,rank=8 if method.endswith('_R8') else 4);deployment_kind=kind if method.endswith('_STRUCT') else 'LR8' if method.endswith('_R8') else 'LR';g=gs[t['order']]['G_fit'];kd=None
       if method in ['M3','M5']:
        teacher=finalrel(seed,'M0',t['order']);kd=p_teacher(runtime,t,teacher,g);guarded_write(ROOT/'private/kd'/f'{seed}-{t["order"]}-{method}.json',dict(quality=[x[3] is not None for x in kd],rule='literal normalized free generation equals native target; unreliable G keeps CE and zero KD',teacher=teacher,temperature=1,normalization='uniform qualified G_fit mean; CE retains all G_fit; lambda_D=0.10',teacher_working_set_bytes=cachebytes))
       stage(runtime,t,expert,deployment_kind,'continuation',ss+1,method,protector(runtime,groups,method,ss+1,kd),g)
       save_final(store,rel,expert,seed,t['canonical_edit_id'],deployment_kind,80,dict(binding,method=method),consumers=())
      key,radius=old.router_entry(runtime,t);router=RejectRouter();router.add(t['canonical_edit_id'],key,radius)
      evaluation.evaluate(runtime,[t],{t['canonical_edit_id']:expert},router,method,'single',1,ROOT/'jobs'/jid/method,[dict(adapter=rel,sha256=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest(),actual_steps=80)],protocol)
      if selected(t):evaluation.evaluate(runtime,[t],{t['canonical_edit_id']:expert},router,method,'diagnostic_FORCED_ON',1,ROOT/'jobs'/jid/(method+'-forced'),[dict(adapter=rel,sha256=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest(),actual_steps=80)],protocol,forced=True)
      if selected(t):
       support=copy.deepcopy(t);support['evaluation']=[]
       for role,questions in [('G_FIT',gs[t['order']]['G_fit']),('G_CHECK',gs[t['order']]['G_check'])]:
        for question in questions:support['evaluation'].append(dict(t['native'],task=role,question=question,reference=old.record(t).target,query_id='support-'+old.digest([role,t['canonical_edit_id'],question])))
       for role,rows in [('U_FIT_OLD',t['U_fit']),('U_FIT_NEW',t['U_new'])]:
        for row in rows:support['evaluation'].append(dict(row,task=role,query_id='support-'+old.digest([role,t['canonical_edit_id'],old.input_id(row)])))
       evaluation.evaluate(runtime,[support],{t['canonical_edit_id']:expert},router,method,'support_FORCED_ON',1,ROOT/'jobs'/jid/(method+'-supports'),[dict(adapter=rel,sha256=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest(),actual_steps=80)],protocol,forced=True)
      with store.lock() as d:remaining=method in d['artifacts'][wrel]['consumers']
      if remaining:store.consumed(wrel,method)
      del expert
     with store.lock() as d:finished=not d['artifacts'][wrel]['consumers']
     if finished:store.delete(wrel)
     del groups,ex,initial
    else:
     bank={};router=RejectRouter();bindings=[];selected=[t for t in tasks if t['order'] in job['orders']]
     for i,t in enumerate(selected,1):
      rel=finalrel(seed,job['method'],t['order']);expert,_=load(rel);expert.requires_grad_(False);bank[t['canonical_edit_id']]=expert;key,radius=old.router_entry(runtime,t);router.add(t['canonical_edit_id'],key,radius);bindings.append(dict(adapter=rel,sha256=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest(),actual_steps=80))
      if i in job['prefixes']:evaluation.evaluate(runtime,selected[:i],bank,router,job['method'],'sequential',i,ROOT/'jobs'/jid/f'p{i}',bindings,protocol)
     del bank
    guarded_write(ROOT/'jobs'/jid/'COMPUTE_COUNTS.json',dict(model_calls=counters['model_calls']-counts_before['model_calls'],input_positions=counters['input_positions']-counts_before['input_positions'],seconds=time.time()-job_began,scope='Includes Base, teacher, CE, KL and generation model calls for this completed worker attempt',seed=seed,mode=job['mode'],methods=job.get('methods',[job.get('method')]),physical_gpu=slot))
    assert runtime.base_guard.verify()['unchanged'];torch.cuda.empty_cache()
    with (ROOT/'QUEUE.lock').open('a') as f:
     fcntl.flock(f,fcntl.LOCK_EX);queue=read(ROOT/'QUEUE.json');j=next(j for j in queue if j['id']==jid);j.update(status='COMPLETE',ended_epoch=time.time());write(ROOT/'QUEUE.json',queue)
   write(ROOT/'jobs'/f'WORKER_{slot}.json',dict(status='IDLE_QUEUE_DRAINED',pid=os.getpid(),seconds=time.time()-started))
 except Exception as e:
  write(ROOT/'jobs'/f'WORKER_{slot}.json',dict(status='FAILED',pid=os.getpid(),error=str(e),traceback=traceback.format_exc(),seconds=time.time()-started));raise
if __name__=='__main__':main()
