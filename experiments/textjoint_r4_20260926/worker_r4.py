"""Two dose arms from identical P-W0; reuse the R3 generation implementation."""
import os,sys,time,copy,random,importlib
from pathlib import Path
ROOT=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(ROOT))
import worker_r3 as shared
import worker_v3 as old
import worker_r2 as dose
from freshstart import runtime as rt
from budget import read,write
R3=ROOT.parents[1]/'textjoint-r3-20260926/run';R2=ROOT.parents[1]/'textjoint-r2-20260925/run'
BETA={'P':0.,'B125':.125,'B25':.25,'P+S':.5};SEED=20260925

def checkpoint(t,arm):
 rel=Path('runs/s20260925')/arm/f'e{t["order"]:03d}'/'private/edits'/f'e{t["order"]:03d}'/'C_NO_H/step-80.pt'
 candidates=[ROOT/rel,R3/rel,R2/rel] if arm in ['P','P+S'] else [ROOT/rel]
 return next((p for p in candidates if p.exists()),candidates[0])
shared.checkpoint=checkpoint

def train(runtime,t,arm):
 import torch
 from scripts.medtrace import stage15
 from scripts.medtrace.run_selective_write import save
 if checkpoint(t,arm).exists():return shared.load(runtime,t,arm)
 assert arm in ['B125','B25'];shared.TRAINING=True;rt.check_budget(training=True);began=time.time()
 folder=ROOT/'runs/s20260925'/arm/f'e{t["order"]:03d}';task=copy.deepcopy(t)
 task.update(seed=t['seed']+1,fit_questions=t['semantic_fit_questions'],probes=[t['native']],U=t['U_fit'],continuation_steps=80)
 ex,binding=shared.initial(runtime,t)
 write(folder/'SHARED_P_W0.json',binding);save(folder/'STEP0.pt',dict(expert=ex.state_dict(),actual_optimizer_steps=0,edit=t['canonical_edit_id'],seed=task['seed']))
 callback=dose.make_protection(runtime,task,folder,ex,beta=BETA[arm])
 m=read(ROOT/'RUN_MANIFEST.json');cfg=dict(campaign_epoch=rt.epoch(m['first_started_at']),train_seconds=rt.epoch(m['no_new_training_after'])-rt.epoch(m['first_started_at']))
 stage15.train_steps(runtime,folder,cfg,task,ex,'C_NO_H',record=old.record(task),layer_path=rt.LAYER,protection=callback)
 restored,meta=shared.load(runtime,t,arm)
 assert all(torch.equal(v,restored.state_dict()[k]) for k,v in ex.state_dict().items())
 save(folder/'FINAL.pt',dict(expert=ex.state_dict(),actual_optimizer_steps=80,edit=t['canonical_edit_id'],seed=task['seed'],beta=BETA[arm],layer=rt.LAYER))
 write(folder/'TRAINING_COMPLETE.json',dict(status='PASS',actual_optimizer_steps=80,beta=BETA[arm],seconds=time.time()-began,shared_initialization=binding,reload_equal=True,support_binding=old.digest({k:t[k] for k in ['native','semantic_fit_questions','U_fit','U_new']})))
 assert runtime.base_guard.verify()['unchanged'];shared.TRAINING=False
 return restored,meta

def canary(runtime,tasks,folder,protocol):
 import torch
 from dataclasses import replace
 from methods.medtrace import MedTraceLayerHook
 from methods.medtrace.selective_write import optimizer_for
 from scripts.medtrace import stage15
 import baseline_r2
 results=[]
 for t in tasks:
  task=dict(t,seed=t['seed']+1,fit_questions=t['semantic_fit_questions'])
  for beta in [0.,.5]:
   ex,_=shared.initial(runtime,t);a=copy.deepcopy(ex);b=copy.deepcopy(ex)
   prior=old.protection(runtime,task,folder/'old0'/f'e{t["order"]}',task['U_fit']) if beta==0 else baseline_r2.make_protection(runtime,task,folder/'old5'/f'e{t["order"]}',a)
   rng=random.getstate();cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
   new=dose.make_protection(runtime,task,folder/str(beta)/f'e{t["order"]}',b,beta=beta)
   assert rng==random.getstate() and torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state()),'teacher/callback construction altered training RNG'
   fit=[1,2,3,4];random.Random(task['seed']).shuffle(fit)
   rec=old.record(task);batches=[runtime.build_edit_batch(rec),runtime.build_edit_batch(replace(rec,question=task['fit_questions'][fit[0]-1]))]
   saved=[]
   for expert,callback in [(a,prior),(b,new)]:
    expert.requires_grad_(True);opt=optimizer_for(expert,runtime.model);opt.zero_grad(set_to_none=True);hook=MedTraceLayerHook(runtime.get_module(rt.LAYER),expert);hook.attach()
    try:
     ce=0.
     for batch in batches:
      hook.set_teacher_routing(batch.labels);value=runtime.compute_loss(batch);(.5*value).backward();ce+=.5*float(value.detach())
     before=random.getstate();kl=callback(runtime,hook,expert,1,'C_NO_H');assert before==random.getstate()
     gradients=[p.grad.detach().clone() for p in expert.parameters()];torch.nn.utils.clip_grad_norm_(expert.parameters(),1.);opt.step();expert.normalize_factors_(verify_dense=False)
     saved.append((ce+.01*kl,gradients,{k:v.detach().clone() for k,v in expert.state_dict().items()}))
    finally:hook.detach()
   x,y=saved;grad_diff=max(float((p-q).abs().max()) for p,q in zip(x[1],y[1]));update_diff=max(float((x[2][k]-y[2][k]).abs().max()) for k in x[2])
   assert abs(x[0]-y[0])<1e-6 and all(torch.allclose(p,q,rtol=1e-4,atol=1e-7) for p,q in zip(x[1],y[1]))
   assert all(torch.allclose(x[2][k],y[2][k],rtol=1e-5,atol=1e-7) for k in x[2])
   results.append(dict(order=t['order'],beta=beta,loss_abs_diff=abs(x[0]-y[0]),gradient_max_abs_diff=grad_diff,first_update_max_abs_diff=update_diff,RNG_unchanged=True,status='PASS'))
   del a,b,ex,prior,new,saved,batches
  # Full 80-step training -> reload -> real generation -> Judge closure. These
  # are the first formal DEV edits, so completed work is reused by the queue.
  from router_r3 import RejectRouter
  for arm in ['B125','B25']:
   ex,meta=train(runtime,t,arm);router=RejectRouter();k,rad=old.router_entry(runtime,t);router.add(t['canonical_edit_id'],k,rad)
   shared.evaluate(runtime,[t],{t['canonical_edit_id']:ex},router,arm+'@80_R0','single',1,folder/arm/f'p{t["order"]:03d}',[meta],protocol)
 write(folder/'CANARY.json',dict(status='PASS',equivalence=results,actual_steps=80,teacher_cap=1024,Base_frozen=runtime.base_guard.verify()['unchanged'],Judge='await closure',edits=len(tasks)))

def main():
 import torch
 from router_r3 import RejectRouter
 job=read(Path(os.environ['RUN_JOB_JSON']));folder=ROOT/'jobs'/job['id'];began=time.time();write(folder/'STATUS.json',dict(status='RUNNING',job=job,pid=os.getpid(),started_epoch=began))
 try:
  with rt.gpu_session(job['id']):
   runtime=rt.load_runtime(folder/'runtime',SEED);modules=shared.imports()
   for name in ['worker_r4','worker_r2','baseline_r2']:
    p=Path(importlib.import_module(name).__file__).resolve();modules[name]=dict(path=str(p),sha256=shared.sha(p))
   write(folder/'MODULE_PATHS.json',modules)
   protocol=dict(model=str((ROOT/'models/llava-med-v1.5-mistral-7b').resolve()),precision=str(next(runtime.model.parameters()).dtype),backend=dict(torch=torch.__version__,cuda=torch.version.cuda,adapter=type(runtime.adapter).__name__),code={k:v['sha256'] for k,v in modules.items()})
   assert runtime.generation_config['max_new_tokens']==1024
   handle=runtime.model.register_forward_pre_hook(lambda m,a:rt.check_budget(training=shared.TRAINING))
   tasks=[t for t in read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'] if t['order'] in job['orders']]
   if job['mode']=='canary':canary(runtime,tasks,folder,protocol)
   else:
    assert read(ROOT/'ROUTER_LOCK.json')['kappa']==1. and read(ROOT/'ROUTER_LOCK.json')['mu']==0.
    bank={};bindings=[];router=RejectRouter();arm=job['writer']
    for i,t in enumerate(tasks,1):
     write(folder/'PROGRESS.json',dict(order=t['order'],arm=arm,phase='TRAIN_OR_LOAD',epoch=time.time()))
     ex,meta=train(runtime,t,arm)
     if job['mode']=='single':bank={};bindings=[];router=RejectRouter()
     bank[t['canonical_edit_id']]=ex;bindings.append(meta);k,rad=old.router_entry(runtime,t);router.add(t['canonical_edit_id'],k,rad)
     if job['mode']=='single' or i in job['prefixes']:
      chosen=[t] if job['mode']=='single' else tasks[:i]
      shared.evaluate(runtime,chosen,bank,router,arm+'@80_R0',job['mode'],1 if job['mode']=='single' else i,folder/arm/f'p{i:03d}',bindings,protocol)
     write(folder/'PROGRESS.json',dict(order=t['order'],arm=arm,phase='GENERATED',epoch=time.time()))
   handle.remove();assert runtime.base_guard.verify()['unchanged']
  write(folder/'STATUS.json',dict(status='GPU_COMPLETE',job=job,seconds=time.time()-began))
 except Exception as e:
  import traceback
  write(folder/'STATUS.json',dict(status='FAILED',job=job,error=str(e),traceback=traceback.format_exc(),seconds=time.time()-began));raise
if __name__=='__main__':main()
