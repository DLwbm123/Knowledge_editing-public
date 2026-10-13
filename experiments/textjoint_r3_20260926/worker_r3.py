"""R3 frozen 80-step writers; full execution-bound generation, sparse real banks."""
import os,sys,json,time,copy,hashlib,inspect,importlib
from pathlib import Path
from dataclasses import asdict
ROOT=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(ROOT))
import worker_v3 as old
import worker_r2 as r2
from freshstart import runtime as rt
from budget import read,write,install
install(rt)
OLD=ROOT.parents[1]/'textjoint-r2-20260925/run';SEED=20260925
TRAINING=False

def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def checkpoint(t,arm):
 p=ROOT/'runs/s20260925'/arm/f'e{t["order"]:03d}'/'private/edits'/f'e{t["order"]:03d}'/'C_NO_H/step-80.pt'
 return p if p.exists() else OLD/p.relative_to(ROOT)
def load(runtime,t,arm):
 import torch
 from methods.medtrace import AsymmetricCPExpert
 from methods.medtrace.selective_write import LowRankExpert
 p=checkpoint(t,arm);s=torch.load(p,map_location=runtime.device,weights_only=True)
 assert s['step']==80 and s['canonical_edit_id']==t['canonical_edit_id'] and s['seed']==t['seed']+1
 assert max(int(v['step']) for v in s['optimizer']['state'].values())==80
 expert=LowRankExpert(AsymmetricCPExpert(14336,4096,4).to(runtime.device),t['seed']+1,rank=4).to(runtime.device)
 expert.load_state_dict(s['expert']);expert.requires_grad_(False)
 return expert,dict(sha256=sha(p),actual_optimizer_steps=80,edit=t['canonical_edit_id'],seed=s['seed'],path=str(p))
def initial(runtime,t):
 import torch
 from methods.medtrace import AsymmetricCPExpert
 from methods.medtrace.selective_write import LowRankExpert
 folder=OLD/'runs/s20260925/P'/f'e{t["order"]:03d}'
 p=folder/'private/edits'/f'e{t["order"]:03d}'/'initial/W0_COMPLETE.pt'
 s=torch.load(p,map_location=runtime.device,weights_only=True);assert s['canonical_edit_id']==t['canonical_edit_id']
 cp=AsymmetricCPExpert(14336,4096,4).to(runtime.device);cp.load_state_dict(s['expert'])
 ex=LowRankExpert(cp,t['seed']+1,rank=4).to(runtime.device)
 saved=torch.load(folder/'STEP0.pt',map_location=runtime.device,weights_only=True)
 assert all(torch.equal(v,saved['expert'][k]) for k,v in ex.state_dict().items())
 with torch.no_grad():
  x=torch.linspace(-1,1,14336,device=runtime.device).reshape(1,-1)
  assert torch.allclose(cp.residual(x),ex.residual(x),rtol=2e-4,atol=2e-5)
 return ex,dict(P_W0_sha256=sha(p),P_STEP0_sha256=sha(folder/'STEP0.pt'),bitwise_shared_step0=True,CP_conversion='PASS')
def train(runtime,t):
 import torch
 from scripts.medtrace import stage15
 from scripts.medtrace.run_selective_write import save
 global TRAINING
 p=checkpoint(t,'P+S')
 if p.exists():return load(runtime,t,'P+S')
 TRAINING=True;rt.check_budget(training=True)
 folder=ROOT/'runs/s20260925/P+S'/f'e{t["order"]:03d}'
 task=copy.deepcopy(t);task.update(seed=t['seed']+1,fit_questions=t['semantic_fit_questions'],probes=[t['native']],U=t['U_fit'],continuation_steps=80)
 ex,binding=initial(runtime,t);write(folder/'SHARED_P_W0.json',binding)
 save(folder/'STEP0.pt',dict(expert=ex.state_dict(),actual_optimizer_steps=0,seed=task['seed'],edit=t['canonical_edit_id']))
 protection=r2.make_protection(runtime,task,folder,ex)
 m=read(ROOT/'RUN_MANIFEST.json');cfg=dict(campaign_epoch=rt.epoch(m['first_started_at']),train_seconds=rt.epoch(m['no_new_training_after'])-rt.epoch(m['first_started_at']))
 stage15.train_steps(runtime,folder,cfg,task,ex,'C_NO_H',record=old.record(task),layer_path=rt.LAYER,protection=protection)
 restored,meta=load(runtime,t,'P+S')
 assert all(torch.equal(a,b) for a,b in zip(ex.state_dict().values(),restored.state_dict().values()))
 save(folder/'FINAL.pt',dict(expert=ex.state_dict(),actual_optimizer_steps=80,seed=task['seed'],edit=t['canonical_edit_id'],layer=rt.LAYER))
 write(folder/'TRAINING_COMPLETE.json',dict(status='PASS',actual_optimizer_steps=80,shared_initialization=binding,reload_equal=True))
 assert runtime.base_guard.verify()['unchanged'];TRAINING=False
 return restored,meta

def imports():
 from methods.medtrace import MedTraceLayerHook,AsymmetricCPExpert
 from methods.medtrace.selective_write import LowRankExpert,full_vocab_kl
 from scripts.medtrace.run_selective_write import teacher_batch
 names=['freshstart.runtime','scripts.medtrace.stage15','methods.medtrace.selective_write','m3bench_repro.editors.routing','m3bench_repro.editors.llava_runtime','worker_v3','worker_r2','worker_r3','router_r3','budget','metrics_r2']
 out={}
 for name in names:
  p=Path(importlib.import_module(name).__file__).resolve();out[name]=dict(path=str(p),sha256=sha(p))
 for obj in [MedTraceLayerHook,AsymmetricCPExpert,LowRankExpert,full_vocab_kl,teacher_batch]:
  p=Path(inspect.getfile(obj)).resolve();out[obj.__name__]=dict(path=str(p),sha256=sha(p))
 return out

def evaluate(runtime,tasks,bank,router,label,mode,prefix,folder,bindings,protocol,forced=False):
 from scripts.medtrace import stage15
 from methods.medtrace import MedTraceLayerHook
 consumers=[]
 for t in tasks:
  for row in t['evaluation']:
   rt.check_budget()
   raw,_,ib=stage15.prepared(runtime,row,old.record(t))
   # The router only receives the Base feature. Associated IDs are diagnostics below.
   decision,extra=router.diagnostic(old.key(runtime,row,old.record(t)));decision=asdict(decision)
   expert=bank.get(decision['logical_edit_id'])
   if forced:
    assert len(bank)==1;expert=next(iter(bank.values()))
   binding=dict(input=ib,model=protocol['model'],precision=protocol['precision'],backend=protocol['backend'],code=protocol['code'],bank=bindings,router=dict(kappa=router.kappa,mu=router.mu),prefix=prefix,forced_diagnostic=forced)
   ident=old.digest(binding);path=ROOT/'private/generations'/f'{ident}.json'
   if path.exists():
    saved=read(path);assert saved['execution_binding']==binding;out=saved['output']
   else:
    if time.time()>=rt.epoch(read(ROOT/'RUN_MANIFEST.json')['no_new_generation_after']):raise TimeoutError('Hour 11: no new generation expansion')
    hook=MedTraceLayerHook(runtime.get_module(rt.LAYER),expert) if expert is not None else None
    if hook:hook.attach()
    try:out=stage15.generate(runtime,raw,ib,hook)
    finally:
     if hook:hook.detach()
    write(path,dict(execution_binding=binding,output=out))
   base,bj=old.base(runtime,row,old.record(t));assert base['binding']==ib,'Base cache input/protocol mismatch'
   consumers.append(dict(arm=label,seed=SEED,mode=mode,prefix=prefix,edit=t['canonical_edit_id'],order=t['order'],task=row['task'],query_id=row['query_id'],source_group=row['source_group'],input_id=old.input_id(row),execution_id=ident,judge_key=old.request(row,out),base_judge_key=bj,route=decision,route_diagnostics=extra,output=out,exact_Base_token_consistency=out['raw_token_ids']==base['raw_token_ids']))
 write(folder/'CONSUMERS.json',consumers)
 return consumers

def canary(runtime,tasks,folder,protocol):
 import torch
 from router_r3 import RejectRouter
 from scripts.medtrace import stage15
 for t in tasks[:2]:
  ex,proof=initial(runtime,t)
  task=dict(t,seed=t['seed']+1)
  r2.make_protection(runtime,task,folder/f'e{t["order"]:03d}',ex)
  assert read(folder/f'e{t["order"]:03d}'/'OLD_CALLBACK_EQUIVALENCE.json')['status']=='PASS'
  write(folder/f'e{t["order"]:03d}'/'INITIALIZATION.json',proof)
  raw,_,ib=stage15.prepared(runtime,t['native'],old.record(t));fresh=stage15.generate(runtime,raw,ib)
  prior,_=old.base(runtime,t['native'],old.record(t));assert fresh['raw_token_ids']==prior['raw_token_ids']
  probe=dict(t,evaluation=[dict(t['native'],task='T0')])
  for arm in ['P','P+S']:
   expert,b=load(runtime,t,arm);r=RejectRouter();k,rad=old.router_entry(runtime,t);r.add(t['canonical_edit_id'],k,rad)
   evaluate(runtime,[probe],{t['canonical_edit_id']:expert},r,arm+'@80/R0','canary',1,folder/arm/f'p{t["order"]:03d}',[b],protocol)
  assert runtime.base_guard.verify()['unchanged']
 write(folder/'CANARY.json',dict(status='PASS',edits=2,Base_OFF_matches_prior=True,Base_frozen=True,shared_P_step0=True,actual_optimizer_steps=80,conversion_and_reload=True,teacher_prefix_cap=runtime.generation_config['max_new_tokens'],scoring='PENDING_OR_REUSED',cross_GPU='R2 numerical-floor PASS retained; arm allocation alternated in R3'))

def calibration(runtime,tasks,folder):
 import torch
 from router_r3 import calibrate
 from scripts.medtrace.run_selective_write import save
 cal=read(ROOT/'private/CAL_ROUTE.json');entries=[];positives=[];negatives=[]
 for t in tasks[:24]:
  k,rad=old.router_entry(runtime,t);entries.append(dict(edit=t['canonical_edit_id'],key=k.cpu(),radius=float(rad)))
 for p in cal['positives']:positives.append(dict(p,key=old.key(runtime,p,old.record(tasks[p['order']-1])).cpu()))
 for p in cal['negatives']:negatives.append(dict(p,key=old.key(runtime,p,old.record(tasks[0])).cpu()))
 save(ROOT/'private/CAL_FEATURES.pt',dict(entries=entries,positives=positives,negatives=negatives))
 if cal['support_status']=='SUPPORTED':lock=calibrate(entries,positives,negatives)
 else:lock=dict(status='UNSUPPORTED',kappa=1.,mu=0.)
 lock.update(calibration_sha256=sha(ROOT/'private/CAL_ROUTE.json'),frozen_before_formal_generation=True,writer_independent=True,seed=SEED,steps=80)
 assert not (ROOT/'ROUTER_LOCK.json').exists();write(ROOT/'ROUTER_LOCK.json',lock);write(ROOT/'public/ROUTER_LOCK.json',lock)

def main():
 import torch
 from router_r3 import RejectRouter,selfcheck
 selfcheck();job=read(Path(os.environ['RUN_JOB_JSON']));folder=ROOT/'jobs'/job['id'];started=time.time()
 write(folder/'STATUS.json',dict(status='RUNNING',job=job,pid=os.getpid(),started_epoch=started))
 try:
  with rt.gpu_session(job['id']):
   runtime=rt.load_runtime(folder/'runtime',SEED);modules=imports();write(folder/'MODULE_PATHS.json',modules)
   protocol=dict(model=str((ROOT/'models/llava-med-v1.5-mistral-7b').resolve()),precision=str(next(runtime.model.parameters()).dtype),backend=dict(torch=torch.__version__,cuda=torch.version.cuda,adapter=type(runtime.adapter).__name__),code={k:v['sha256'] for k,v in modules.items()})
   assert runtime.generation_config['max_new_tokens']==1024
   handle=runtime.model.register_forward_pre_hook(lambda m,a:rt.check_budget(training=TRAINING))
   tasks=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks']
   chosen=[t for t in tasks if t['order'] in job['orders']]
   if job['mode']=='canary':canary(runtime,chosen,folder,protocol)
   elif job['mode']=='calibration':calibration(runtime,tasks,folder)
   else:
    lock=read(ROOT/'ROUTER_LOCK.json');bank={};bindings=[]
    for i,t in enumerate(chosen,1):
     write(folder/'PROGRESS.json',dict(order=t['order'],phase='TRAIN_OR_LOAD',epoch=time.time()))
     expert,b= train(runtime,t) if job['writer']=='P+S' else load(runtime,t,'P')
     if job['mode']=='single':bank={};bindings=[]
     bank[t['canonical_edit_id']]=expert;bindings.append(b)
     for route in job['routers']:
      rr=RejectRouter(*((1.,0.) if route=='R0' else (lock['kappa'],lock['mu'])))
      scope=[t] if job['mode']=='single' else chosen[:i]
      for item in scope:
       k,rad=old.router_entry(runtime,item);rr.add(item['canonical_edit_id'],k,rad)
      if job['mode']=='single' or i in job['prefixes']:
       label=job['writer']+'@80_'+route
       evaluation_scope=[x for x in scope if x['order'] in job.get('eval_orders',[v['order'] for v in scope])]
       evaluate(runtime,evaluation_scope,bank,rr,label,job['mode'],1 if job['mode']=='single' else i,folder/label/f'p{i:03d}',bindings,protocol)
     if job.get('diagnose'):
      rr=RejectRouter();k,rad=old.router_entry(runtime,t);rr.add(t['canonical_edit_id'],k,rad)
      evaluate(runtime,[t],{t['canonical_edit_id']:expert},rr,job['writer']+'@80_FORCED_ON','diagnostic',1,folder/'FORCED_ON'/f'p{i:03d}',[b],protocol,forced=True)
     write(folder/'PROGRESS.json',dict(order=t['order'],phase='GENERATED',epoch=time.time()))
    assert all(not p.requires_grad for e in bank.values() for p in e.parameters())
   handle.remove();assert runtime.base_guard.verify()['unchanged']
  write(folder/'STATUS.json',dict(status='GPU_COMPLETE',job=job,seconds=time.time()-started))
 except Exception as e:
  import traceback
  write(folder/'STATUS.json',dict(status='FAILED',job=job,error=str(e),traceback=traceback.format_exc(),seconds=time.time()-started));raise
if __name__=='__main__':main()
