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
 from e2 import structure as method_structure,deploy,parity
 from training import new,stage,load,save_final,u_teachers,p_teacher,protector,derive_seed,teacher_bytes
 from diagnostics import selected as is_diagnostic_edit
 from bindings import expected,rolling,verify_science_files,g_supports,initialization,state_digest
 from router_r3 import RejectRouter
 import evaluation
 slot=int(os.environ['PHYSICAL_GPU']);verify_science_files();store=Store(ROOT)
 modules={name:dict(path=sys.modules[name].__file__,sha256=hashlib.sha256(Path(sys.modules[name].__file__).read_bytes()).hexdigest()) for name in ['training','structures','diagnostics','bindings','evaluation']}
 write(ROOT/'fix/pr5_v1'/f'WORKER_MODULES_{os.getpid()}.json',modules)
 rt.ROOT=ROOT;rt.GPU=os.environ['CUDA_VISIBLE_DEVICES'];rt.check_budget=lambda **kw:check(kw.get('training',False));rt.gpu_session=session
 def guarded_write(p,d):
  rel=str(Path(p).relative_to(ROOT));store.write(rel,json.dumps(d,ensure_ascii=False).encode())
 def key_save(p,x):
  assert torch.is_tensor(x) and x.numel()<1000000,'Only small router key saves permitted';b=io.BytesIO();torch.save(x,b);store.write(str(Path(p).relative_to(ROOT)),b.getvalue())
 old.write=guarded_write;stage15.write=guarded_write;evaluation.write=guarded_write;run_selective_write.save=key_save
 started=time.time();tasks=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'];gs=g_supports();methods={'CP':['M0','M1','M2','M3'],'TK':['M4','M5'],'LR':['M6']}
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
    if (ROOT/'ADMISSION_STOP').exists():break
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
     if (ROOT/wrel).exists():ex,_=load(wrel,expected(t,seed,kind,kind,'W0',320))
     else:
      seed_everything(ss);ex=new(kind);assert state_digest(ex)==initialization(t['canonical_edit_id'],seed,kind)['state_sha256'],'Actual initializer mismatch';resume_stage='native';rolling=ROOT/f'checkpoints/slot{slot}/latest.pt'
      saved=__import__('bindings').rolling(str(rolling.relative_to(ROOT)),t,seed,kind)
      if saved:resume_stage=saved['binding']['stage'];ex.load_state_dict(saved['expert'])
      stages=['native','A2','W0'];assert resume_stage in stages
      for phase in stages[stages.index(resume_stage):]:stage(runtime,t,ex,kind,phase,ss+1 if phase=='W0' else ss,kind,run_seed=seed,structure=kind)
      future=[] # Only consumers admitted in this appendix may retain W0.
      save_final(store,wrel,ex,seed,t['canonical_edit_id'],kind,320,expected(t,seed,kind,kind,'W0',320),consumers=arms+future+job.get('future_methods',[]))
     if job.get('block')=='E2_REG24':parity(runtime,t,ex,ss+1,kind+'-W0')
     initial={k:v.detach().clone() for k,v in ex.state_dict().items()};groups,cachebytes=u_teachers(runtime,t)
     for method in arms:
      rel=finalrel(seed,method,t['order'])
      if (ROOT/rel).exists():expert,_=load(rel,expected(t,seed,kind,method,'continuation',80))
      else:
       ex.load_state_dict(initial);expert=copy.deepcopy(ex) if method.endswith('_STRUCT') else convert(ex,ss+1,rank=8 if method.endswith('_R8') else 4);deployment_kind=kind if method.endswith('_STRUCT') else 'LR8' if method.endswith('_R8') else 'LR';g=gs[t['order']]['G_fit'];kd=None
       if method in ['M3','M5','M7']:
        teacher=finalrel(seed,'M0',t['order']);kd=p_teacher(runtime,t,teacher,g,expected(t,seed,'CP','M0','continuation',80));guarded_write(ROOT/'private/kd'/f'{seed}-{t["order"]}-{method}.json',dict(quality=[x[3] is not None for x in kd],rule='literal normalized free generation equals native target; unreliable G keeps CE and zero KD',teacher=teacher,temperature=1,normalization='uniform qualified G_fit mean; CE retains all G_fit; lambda_D=0.10',teacher_working_set_bytes=cachebytes))
       cachebytes=teacher_bytes([groups,kd]);stage(runtime,t,expert,deployment_kind,'continuation',ss+1,method,protector(runtime,groups,method,ss+1,kd),g,run_seed=seed,structure=kind)
       save_final(store,rel,expert,seed,t['canonical_edit_id'],deployment_kind,80,expected(t,seed,kind,method,'continuation',80),consumers=())
      expert,_=load(rel,expected(t,seed,kind,method,'continuation',80));expert.requires_grad_(False)
      if method.endswith('_STRUCT'):expert=parity(runtime,t,expert,ss+1,method+'-final')
      key,radius=old.router_entry(runtime,t);router=RejectRouter();router.add(t['canonical_edit_id'],key,radius)
      evaluation.evaluate(runtime,[t],{t['canonical_edit_id']:expert},router,method,'single',1,ROOT/'jobs'/jid/method,[dict(adapter=rel,sha256=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest(),actual_steps=80,deployment='expanded_free_rank4' if method.endswith('_STRUCT') else 'native_free_rank4')],protocol)
      if is_diagnostic_edit(t) and job.get('block')!='E2_REG24':evaluation.evaluate(runtime,[t],{t['canonical_edit_id']:expert},router,method,'diagnostic_FORCED_ON',1,ROOT/'jobs'/jid/(method+'-forced'),[dict(adapter=rel,sha256=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest(),actual_steps=80,deployment='expanded_free_rank4' if method.endswith('_STRUCT') else 'native_free_rank4')],protocol,forced=True)
      if is_diagnostic_edit(t) and job.get('block')!='E2_REG24':
       support=copy.deepcopy(t);support['evaluation']=[]
       for role,questions in [('G_FIT',gs[t['order']]['G_fit']),('G_CHECK',gs[t['order']]['G_check'])]:
        for question in questions:support['evaluation'].append(dict(t['native'],task=role,question=question,reference=old.record(t).target,query_id='support-'+old.digest([role,t['canonical_edit_id'],question])))
       for role,rows in [('U_FIT_OLD',t['U_fit']),('U_FIT_NEW',t['U_new'])]:
        for row in rows:support['evaluation'].append(dict(row,task=role,query_id='support-'+old.digest([role,t['canonical_edit_id'],old.input_id(row)])))
       evaluation.evaluate(runtime,[support],{t['canonical_edit_id']:expert},router,method,'support_FORCED_ON',1,ROOT/'jobs'/jid/(method+'-supports'),[dict(adapter=rel,sha256=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest(),actual_steps=80,deployment='expanded_free_rank4' if method.endswith('_STRUCT') else 'native_free_rank4')],protocol,forced=True)
      with store.lock() as d:remaining=method in d['artifacts'][wrel]['consumers']
      if remaining:store.consumed(wrel,method)
      del expert
      if 'kd' in locals():del kd
     with store.lock() as d:finished=not d['artifacts'][wrel]['consumers']
     if finished:store.delete(wrel)
     del groups,ex,initial
    else:
     bank={};router=RejectRouter();bindings=[];selected_tasks=[t for t in tasks if t['order'] in job['orders']]
     for i,t in enumerate(selected_tasks,1):
      rel=finalrel(seed,job['method'],t['order']);structure=method_structure(job['method']);expert,_=load(rel,expected(t,seed,structure,job['method'],'continuation',80));expert=deploy(expert,job['method'],derive_seed(t['canonical_edit_id'],seed)+1);expert.requires_grad_(False);bank[t['canonical_edit_id']]=expert;key,radius=old.router_entry(runtime,t);router.add(t['canonical_edit_id'],key,radius);bindings.append(dict(adapter=rel,sha256=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest(),actual_steps=80,deployment='expanded_free_rank4' if job['method'].endswith('_STRUCT') else 'native_free_rank4'))
      if i in job['prefixes']:evaluation.evaluate(runtime,selected_tasks[:i],bank,router,job['method'],'sequential',i,ROOT/'jobs'/jid/f'p{i}',bindings,protocol)
     del bank
    guarded_write(ROOT/'jobs'/jid/'COMPUTE_COUNTS.json',dict(model_calls=counters['model_calls']-counts_before['model_calls'],input_positions=counters['input_positions']-counts_before['input_positions'],seconds=time.time()-job_began,scope='Includes Base, teacher, CE, KL and generation model calls for this completed worker attempt',seed=seed,mode=job['mode'],methods=job.get('methods',[job.get('method')]),physical_gpu=slot))
    assert runtime.base_guard.verify()['unchanged'];torch.cuda.empty_cache()
    with (ROOT/'QUEUE.lock').open('a') as f:
     fcntl.flock(f,fcntl.LOCK_EX);queue=read(ROOT/'QUEUE.json');j=next(j for j in queue if j['id']==jid);j.update(status='COMPLETE',ended_epoch=time.time());write(ROOT/'QUEUE.json',queue)
   write(ROOT/'jobs'/f'WORKER_{slot}.json',dict(status='IDLE_QUEUE_DRAINED',pid=os.getpid(),seconds=time.time()-started))
 except Exception as e:
  write(ROOT/'jobs'/f'WORKER_{slot}.json',dict(status='FAILED',pid=os.getpid(),error=str(e),traceback=traceback.format_exc(),seconds=time.time()-started));raise
if __name__=='__main__':main()
