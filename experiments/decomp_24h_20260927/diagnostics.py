"""Instantaneous four-edit diagnostics; no dense residual or logits on disk."""
import torch,math
from dataclasses import replace
from resources import ROOT,read,write
from structures import convert
from methods.medtrace import AsymmetricCPExpert,MedTraceLayerHook
from scripts.medtrace.run_dev16 import generate
from scripts.medtrace.run_selective_write import teacher_batch
import worker_v3 as old
LAYER='model.layers.30.mlp.down_proj'

def generated_ids(result):
 assert 'generated_token_ids' in result and 'raw_token_ids' not in result, 'run_dev16 generation schema mismatch'
 assert isinstance(result['generated_token_ids'],list) and all(type(x) is int for x in result['generated_token_ids'])
 return result['generated_token_ids']

def selected(t):return t['order'] in read(ROOT/'EXPERIMENT_LOCK.json')['stage_diagnostics_orders']
def spectrum(s):
 s=s.detach().float();nz=s[s>s.max().clamp_min(1e-20)*1e-6];return dict(effective_rank=len(nz),condition=float(nz.max()/nz.min()) if len(nz) else None,energy=float(s.square().sum()),singular_values=s.cpu().tolist())
def diagnose(runtime,t,ex,stage,seed,method):
 if not selected(t):return
 rec=old.record(t);lr=convert(ex,seed);b,a=lr.factors()
 with torch.no_grad():
  rb=torch.linalg.qr(b,mode='reduced')[1];ra=torch.linalg.qr(a.T,mode='reduced')[1];matrix=spectrum(torch.linalg.svdvals(rb@ra.T))
  def root(g):v,q=torch.linalg.eigh(g);return q*v.clamp_min(0).sqrt()[None,:]
  out=(b@root(a@a.T)).reshape(64,64,b.shape[1]);inn=(a.T@root(b.T@b)).reshape(112,128,a.shape[0])
  modes=[spectrum(torch.linalg.svdvals(x.movedim(dim,0).reshape(x.shape[dim],-1))) for x in [out,inn] for dim in [0,1]]
 rows=[]
 for role,r in [('native',rec),('P_fit',replace(rec,question=t['semantic_fit_questions'][0]))]:
  b0=runtime.build_edit_batch(r);rows.append((role,b0.forward_kwargs(),b0.labels,b0))
 for role,row in [('U_old',t['U_fit'][0]),('U_new',t['U_new'][0])]:
  output,_=old.base(runtime,row,rec,score=False);kw,labels,mask,_=teacher_batch(runtime,dict(row,eqkey=old.input_id(row)),output['raw_token_ids']);rows.append((role,kw,labels,None))
 energies=[];native=None
 for role,kwargs,labels,batch in rows:
  captures=[]
  def capture(module,args):
   h=args[0].detach().float().reshape(-1,14336);idx=torch.nonzero(labels.reshape(-1)!=-100).flatten()-1;captures.append(h[idx.clamp_min(0)[:8]])
  handle=runtime.get_module(LAYER).register_forward_pre_hook(capture)
  with torch.no_grad():base=runtime.model(**kwargs).logits.float()
  handle.remove();h=captures[0]
  with torch.no_grad():
   x=ex.residual(h);y=lr.residual(h);energies.append(dict(role=role,input_RMS=float(h.square().mean().sqrt()),residual_RMS=float(x.square().mean().sqrt()),conversion_relative=float((x-y).norm()/x.norm().clamp_min(1e-12))))
  if role=='native':native=(batch,base)
 batch,base=native;outputs=[];logits=[]
 for expert in [ex,lr]:
  hook=MedTraceLayerHook(runtime.get_module(LAYER),expert);hook.attach();hook.set_teacher_routing(batch.labels)
  try:
   with torch.no_grad():logits.append(runtime.model(**batch.forward_kwargs()).logits.float())
   outputs.append(generate(runtime,dict(query_id=rec.record_id,question=rec.question,image_path=str(rec.image_path)),hook,1024))
  finally:hook.detach()
 mask=batch.labels[:,1:]!=-100;student=logits[0][:,:-1][mask];teacher=base[:,:-1][mask];kl=float((teacher.softmax(-1)*(teacher.log_softmax(-1)-student.log_softmax(-1))).sum(-1).mean())
 write(ROOT/'private/diagnostics'/f'{seed}-{method}-{t["order"]}-{stage}.json',dict(stage=stage,method=method,seed=seed,parameters=sum(p.numel() for p in ex.parameters()),matrix=matrix,modes=modes,residuals=energies,conversion_logit_max_abs=float((logits[0]-logits[1]).abs().max()),conversion_generation_equal=generated_ids(outputs[0])==generated_ids(outputs[1]),Base_to_student_native_answer_KL=kl,outputs=outputs))
 del lr
