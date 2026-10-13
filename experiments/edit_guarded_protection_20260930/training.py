"""Same frozen stage objectives for CP/Tucker/Direct; bounded per-edit teachers."""
import os,sys,time,random,copy,hashlib,json
from pathlib import Path
from dataclasses import replace
import torch
ROOT=Path(os.environ['RUN_ROOT']);sys.path[:0]=[str(ROOT),str(ROOT/'source_patch'),str(ROOT/'source')]
from resources import check,write,read
from storage import Store
from structures import TuckerC4,LR4,convert,optimizer
from methods.medtrace import AsymmetricCPExpert,MedTraceLayerHook
from methods.medtrace.selective_write import full_vocab_kl
from scripts.medtrace import stage15
from scripts.medtrace.run_dev16 import derive_seed,generate,TRAIN_CAP
from scripts.engram.stage0_generation_audit_utils import normalize_medical_answer
from m3bench_repro.editors.llava_runtime import seed_everything
import worker_v3 as old
LAYER='model.layers.30.mlp.down_proj'

def new(kind):return {'CP':lambda:AsymmetricCPExpert(14336,4096,4),'TK':TuckerC4,'LR':LR4,'LR8':lambda:LR4(rank=8)}[kind]().cuda()
def save_final(store,rel,ex,seed,edit,kind,step,binding,consumers=()):
 payload=dict(expert={k:v.detach().cpu() for k,v in ex.state_dict().items()},seed=seed,edit=edit,kind=kind,step=step,binding=binding)
 store.save(rel,payload,consumers=consumers,pin=not consumers)
def load(rel,want):
 from bindings import load_expected
 d=load_expected(rel,want);ex=new(d['kind']);ex.load_state_dict(d['expert']);return ex,d

def stage(runtime,t,ex,kind,stage_name,seed,method,protect=None,g=None,run_seed=None,structure=None,anchor=None,diagnostic=None,stop_after=None):
 from bindings import expected,rolling
 assert run_seed is not None and structure is not None
 store=Store(ROOT);slot=os.environ['PHYSICAL_GPU'];rel=f'checkpoints/slot{slot}/latest.pt';identity=expected(t,run_seed,structure,method,stage_name,0)
 from diagnostics import selected,diagnose
 diag=selected(t);hdiag=torch.randn(4,14336,device='cuda',generator=torch.Generator(device='cuda').manual_seed(20260927)) if diag else None
 seed_everything(seed);opt=optimizer(ex,runtime.model,stage_name);rec=old.record(t);batch=[runtime.build_edit_batch(rec)]+[runtime.build_edit_batch(replace(rec,question=q)) for q in t['semantic_fit_questions']]
 gb=[runtime.build_edit_batch(replace(rec,question=q)) for q in (g or [])];curve=[];start=0;steps={'native':200,'A2':80,'W0':320,'continuation':80}[stage_name];forwards=tokens=0;began=time.time();p=ROOT/rel
 origin_parameters=[p.detach().clone() for p in ex.parameters()]
 d=rolling(rel,t,run_seed,structure,method)
 if d and d['binding']['stage']==stage_name:
  ex.load_state_dict(d['expert']);opt.load_state_dict(d['optimizer']);start=d['step'];curve=d['curve'];forwards=d['forwards'];tokens=d['tokens'];torch.set_rng_state(d['torch_rng'].cpu());torch.cuda.set_rng_state(d['cuda_rng'].cpu());random.setstate(d['python_rng'])
 order=list(range(1,5))
 if stage_name=='continuation':random.Random(seed).shuffle(order)
 hook=MedTraceLayerHook(runtime.get_module(LAYER),ex);hook.attach();ex.requires_grad_(True)
 from anchor import PositiveCapture,predictor_mask
 ref,cplus,beta=anchor if anchor is not None else (None,None,0.)
 anchor_backwards=0;diagnostic_records=[]
 from gradient_guard import aggregate,config
 guard_config=config(method)
 limit=stop_after if stop_after is not None else steps
 assert stop_after is None or (method.startswith("SMOKE_") and stop_after in [40,80])
 try:
  for step in range(start,limit+1):
   check(training=True)
   if step>start:
    opt.zero_grad(set_to_none=True);values=[];positive_losses=[];selected=[(batch[0],1.)] if stage_name=='native' else [(batch[0],.5),(batch[order[(step-1)%4]],.5)]
    if stage_name=='continuation' and method in ['M2','M3','M5','M7']:
     assert gb;selected=[(batch[0],.5),(batch[order[(step-1)%4]],.25),(gb[(step-1)%len(gb)],.25)]
    for b,weight in selected:
     hook.set_teacher_routing(b.labels)
     cap=PositiveCapture(runtime.get_module(LAYER),ex,ref,predictor_mask(b.labels,b.attention_mask),cplus) if beta else None
     try:
      loss=runtime.compute_loss(b);assert torch.isfinite(loss);(weight*loss).backward();values.append(float(loss.detach()));forwards+=1;tokens+=len(b.target_token_ids)
      if cap:
       assert cap.loss is not None;positive_losses.append(weight*cap.loss)
     finally:
      if cap:cap.close()
    ce=[p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in ex.parameters()]
    plus_value=sum(float(x.detach()) for x in positive_losses)
    if positive_losses:(beta*sum(positive_losses)).backward();anchor_backwards+=1
    after_plus=[p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in ex.parameters()]
    pg=[v-c for v,c in zip(after_plus,ce)]
    kl=protect(hook,step) if protect else {}
    u=[(p.grad if p.grad is not None else torch.zeros_like(p))-v for p,v in zip(ex.parameters(),after_plus)];cn=float(sum(v.float().square().sum() for v in ce).sqrt());un=float(sum(v.float().square().sum() for v in u).sqrt());cos=float(sum(a.float().mul(b.float()).sum() for a,b in zip(ce,u)))/(cn*un) if cn and un else None
    surgery_forwards=forwards
    import random as _random
    rng_before=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone(),_random.getstate())
    final,guard_record=aggregate(after_plus,u,guard_config['rho'],guard_config['projection'],guard_config['cap'])
    if guard_config['projection'] or guard_config['cap']:
     for p,v in zip(ex.parameters(),final,strict=True):p.grad.copy_(v)
    assert forwards==surgery_forwards
    assert torch.equal(torch.get_rng_state(),rng_before[0]) and torch.equal(torch.cuda.get_rng_state(),rng_before[1]) and _random.getstate()==rng_before[2]
    assert all(p.grad is None for p in runtime.model.parameters()) and all(not p.requires_grad and p.grad is None for p in (ref or []))
    norm=float(torch.nn.utils.clip_grad_norm_(ex.parameters(),1));postclip=float(sum(p.grad.float().square().sum() for p in ex.parameters() if p.grad is not None).sqrt());
    with torch.no_grad():before_action=ex.residual(hdiag).clone() if diag else None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ex.parameters());before=[p.detach().clone() for p in ex.parameters()];opt.step();ex.normalize_factors_(verify_dense=False)
    assert all(torch.isfinite(p).all() for p in ex.parameters());update=float(sum((p-v).float().square().sum() for p,v in zip(ex.parameters(),before)).sqrt())
    if stage_name=='native' and isinstance(ex,AsymmetricCPExpert):assert float(torch.linalg.cond(ex.input_basis().float()))<=1e4,'CP native condition hard stop'
    with torch.no_grad():action_change=float((ex.residual(hdiag)-before_action).norm()) if diag else None
    curve.append(dict(step=step,function_space_update_norm=action_change,postclip_norm=postclip,CE=values,CE_gradient_norm=cn,protection_gradient_norm=un,CE_U_cosine=cos,preclip_norm=norm,update_norm=update,L_plus=plus_value,positive_gradient_norm=float(sum(v.float().square().sum() for v in pg).sqrt()),anchor_beta=beta,gradient_guard=guard_record,surgery_backbone_forwards=0,**kl))
   if diagnostic and step in [0,20,40,60,80]:
    diagnostic_records.append(dict(step=step,**diagnostic(hook,batch)))
   stop=False
   if stage_name=='native' and step%20==0:
    hook.set_teacher_routing(batch[0].labels)
    with torch.no_grad():score=runtime.score_target(batch[0])
    out=generate(runtime,dict(query_id=rec.record_id,question=rec.question,image_path=str(rec.image_path)),hook,TRAIN_CAP)
    literal=normalize_medical_answer(out['decoded_text'])==normalize_medical_answer(rec.target)
    stop=bool(step and literal and score['first_target_token_rank']==1 and not out['cap_hit'])
    if isinstance(ex,AsymmetricCPExpert):
     condition=float(torch.linalg.cond(ex.input_basis().float()));assert condition<=1e4,'CP native condition hard stop'
   if step and (step%20==0 or stop):
    store.save(rel,dict(expert=ex.state_dict(),optimizer=opt.state_dict(),step=step,seed=seed,stage=stage_name,edit=rec.record_id,binding=expected(t,run_seed,structure,method,stage_name,step),curve=curve,forwards=forwards,tokens=tokens,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),python_rng=random.getstate()),pin=True)
    print(method,t['order'],stage_name,step,flush=True)
   if stop:break
 finally:hook.detach()
 diagnose(runtime,t,ex,stage_name,seed,method)
 net_update_norm=float(sum((p-v).double().square().sum() for p,v in zip(ex.parameters(),origin_parameters,strict=True)).sqrt())
 if curve:curve[-1]['parameter_net_update_norm']=net_update_norm
 report=dict(parameter_net_update_norm=net_update_norm,identity=dict(edit=t['canonical_edit_id'],kind=kind,stage=stage_name,seed=seed,method=method),steps=step,forwards=forwards,target_tokens=tokens,seconds=time.time()-began,curve=curve,parameters=sum(p.numel() for p in ex.parameters()))
 store.write(f'private/gradients/{method}/e{t["order"]}.json',json.dumps(dict(records=diagnostic_records,anchor_backward_calls=anchor_backwards)).encode())
 store.write(f'private/curves/s{seed}/{method}/e{t["order"]}/{stage_name}.json',json.dumps(report).encode());ex.requires_grad_(False);return ex

def teacher_bytes(value):
 # Count shared storage once, including cached GPU embeddings and labels.
 seen=set();total=0
 def visit(x):
  nonlocal total
  if torch.is_tensor(x):
   storage=x.untyped_storage();key=(str(x.device),storage.data_ptr())
   if key not in seen:seen.add(key);total+=storage.nbytes()
  elif isinstance(x,dict):
   for v in x.values():visit(v)
  elif isinstance(x,(list,tuple)):
   for v in x:visit(v)
 visit(value);assert total<=8*1024**3,'Total teacher tensor working set exceeds 8 GiB';return total

def u_teachers(runtime,t):
 from scripts.medtrace.run_selective_write import teacher_batch
 from scripts.medtrace.stage18_cfact import assert_base_off
 assert_base_off(runtime);groups=[];size=0
 for rows in [t['U_fit'],t['U_new']]:
  ts=[]
  for row in rows:
   output,_=old.base(runtime,row,old.record(t),score=False)
   kw,labels,mask,binding=teacher_batch(runtime,dict(row,eqkey=old.input_id(row)),output['raw_token_ids'])
   with torch.no_grad():logp=runtime.model(**kw).logits[mask].float().log_softmax(-1).cpu()
   ts.append((kw,labels,mask,logp));size=teacher_bytes([groups,ts])
  assert ts;groups.append(ts)
 return groups,size

def p_teacher(runtime,t,teacher_rel,questions,teacher_expected):
 import fcntl
 quality_id=old.digest(dict(teacher=teacher_rel,weight_sha256=hashlib.sha256((ROOT/teacher_rel).read_bytes()).hexdigest(),questions=questions));quality_path=ROOT/'private/teacher_quality'/f'{quality_id}.json'
 ex,meta=load(teacher_rel,teacher_expected);ex.requires_grad_(False);hook=MedTraceLayerHook(runtime.get_module(LAYER),ex);hook.attach();rows=[]
 try:
  with (ROOT/'TEACHER_QUALITY.lock').open('a') as lockfile:
   fcntl.flock(lockfile,fcntl.LOCK_EX)
   if quality_path.exists():qualities=read(quality_path)['quality']
   else:
    qualities=[]
    for q in questions:
     rec=replace(old.record(t),question=q);out=generate(runtime,dict(query_id=rec.record_id,question=q,image_path=str(rec.image_path)),hook,1024);qualities.append(normalize_medical_answer(out['decoded_text'])==normalize_medical_answer(rec.target))
    Store(ROOT).write(str(quality_path.relative_to(ROOT)),json.dumps(dict(quality=qualities,teacher=teacher_rel,rule='literal normalized native agreement; one shared teacher quality mask for both structures')).encode())
  for q,quality in zip(questions,qualities,strict=True):
   rec=replace(old.record(t),question=q);b=runtime.build_edit_batch(rec)
   hook.set_teacher_routing(b.labels);mask=torch.zeros_like(b.labels,dtype=torch.bool);mask[:,:-1]=b.labels[:,1:]!=-100
   kw=b.forward_kwargs()
   with torch.no_grad():lp=runtime.model(**kw).logits[mask].float().log_softmax(-1).cpu() if quality else None
   rows.append((kw,b.labels,mask,lp))
 finally:hook.detach()
 del hook,ex;return rows

def protector(runtime,groups,method,seed,kd):
 old_u,new_u=groups
 qualified=[r for r in (kd or []) if r[3] is not None]
 def apply(hook,step):
  losses=[]
  def term(row,weight):
   kw,labels,mask,lp=row;hook.set_teacher_routing(labels);v=full_vocab_kl(runtime.model(**kw).logits[mask],lp);(weight*v).backward();return float(v.detach())
  for row in old_u:losses.append(term(row,(.01 if method=='M0' else .005)/len(old_u)))
  if method!='M0':losses.append(term(new_u[random.Random(seed*1000003+step).randrange(len(new_u))],.005))
  kdvalue=None
  if qualified:
   # The mean is over qualified teachers; coverage cannot dilute lambda_D.
   kdvalue=term(qualified[random.Random(seed*1000033+step).randrange(len(qualified))],.10)
  return dict(U_KL=sum(losses)/len(losses) if method=='M0' else .5*sum(losses[:-1])/len(old_u)+.5*losses[-1],D_KL=kdvalue)
 return apply
