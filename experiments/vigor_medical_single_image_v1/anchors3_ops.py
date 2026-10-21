"""Frozen per-record parameter response intersection; no old records in online API."""
import time
import torch
from .editor import editable_fp32,nll,finite
from .equiv3_ops import relnorm
from .scalar3_ops import AUTH as PREVIOUS
AUTH=dict(PREVIOUS,seconds=1800,calibration_records=32,calibration_seconds=240)

def memory_admission(host_available,gpu_free,loaded=False):
 if host_available<32*1024**3 or gpu_free<(24 if loaded else 48)*1024**3:raise MemoryError('Frozen memory reserve')
 return dict(host_available_bytes=host_available,gpu_free_bytes=gpu_free,host_extra_bound=20*1024**3,GPU_own_bound=44*1024**3)

def calibrate(rt,layer,path,ids,emit):
 # This function owns and releases raw training records/batches before returning statistics.
 import json
 from pathlib import Path
 from m3bench_repro.editors.llava_runtime import EditorRecord
 records=json.loads(Path(path).read_text())
 if len(records)!=32 or [r['record_id'] for r in records]!=ids:raise ValueError('Frozen training order')
 started=time.time();stats=[];rows=[]
 for i,row in enumerate(records):
  if time.time()-started>=240:raise TimeoutError('Calibration work ceiling')
  record=EditorRecord(**dict(row,image_path=Path(row['image_path'])))
  batch=rt.build_edit_batch(record)
  with editable_fp32(rt.model,layer,readonly=True):
   try:
    loss=nll(rt.model,batch);loss.backward();finite(layer.weight.grad)
    g=layer.weight.grad.detach().float().cpu().clone()
   finally:layer.weight.grad=None
  norm=float(torch.linalg.vector_norm(g.double()));finite(g)
  stats.append(g);rows.append(dict(index=i+1,reference_NLL=float(loss.detach()),gradient_norm=norm,inactive=norm==0));emit(rows[-1])
  del batch,record,loss
 del records
 return stats,dict(records=32,forwards=32,backwards=32,seconds=time.time()-started,raw_records_released=True,rows=rows)

class Intersection:
 def __init__(self,teacher,q,statistics):
  self.q=q;self.I=(teacher@q)@q.T;self.active=[];self.norms=[];self.H=torch.empty((len(statistics),teacher.numel()),device=teacher.device,dtype=torch.float64);b=[];self.fixed=[]
  for j,stat in enumerate(statistics):
   if stat.device.type!='cpu' or stat.dtype!=torch.float32 or tuple(stat.shape)!=tuple(teacher.shape):raise ValueError('Parameter-statistics-only interface')
   g=stat.to(device=teacher.device,dtype=torch.float64);finite(g);norm=float(g.norm());self.norms.append(norm);self.active.append(norm>0)
   if norm:g.div_(norm)
   b.append(-(g*self.I).sum());self.fixed.append(float((g*self.I).sum()))
   h=g-(g@q)@q.T;self.H[j].copy_(h.flatten());del g,h
  self.b=torch.stack(b);self.M=self.H@self.H.T;finite(self.M)
  eig,v=torch.linalg.eigh(self.M);threshold=float(eig.max())*1e-10
  if float(eig.min()) < -max(float(eig.max()),1e-30)*1e-10:raise ValueError('Non-PSD Gram')
  retained=eig>threshold;self.eig=eig[retained];self.v=v[:,retained]
  c=self.solve(self.b);res=float((self.M@c-self.b).norm()/self.b.norm().clamp_min(1e-30))
  if res>1e-8:raise ValueError('Response/source intersection inconsistent')
  correction=(c@self.H).reshape_as(teacher);self.anchor=self.I+correction
  predicted=float(self.b@c);actual=float(correction.square().sum())
  self.meta=dict(active_anchors=sum(self.active),anchor_denominator=len(statistics),gram_rank=int(retained.sum()),gram_dimension=len(statistics),gram_eigenvalues=eig.tolist(),gram_min_eigenvalue=float(eig.min()),gram_max_eigenvalue=float(eig.max()),retained_min_eigenvalue=float(self.eig.min()) if self.eig.numel() else None,nonzero_spectral_condition=float(self.eig.max()/self.eig.min()) if self.eig.numel() else None,discarded_eigenvalues=int((~retained).sum()),gram_rtol=1e-10,linear_residual=res,minimum_compensation_squared=predicted,actual_compensation_squared=actual,compensation_norm=float(correction.norm()),source_minimum_norm=float(self.I.norm()),compensation_identity_relative_error=abs(actual-predicted)/max(abs(predicted),1e-30))
  if self.meta['compensation_identity_relative_error']>1e-8:raise ValueError('Compensation norm identity')
  if self.response_error(self.anchor)>1e-8:raise ValueError('Initial response residual')
 def solve(self,b):
  return self.v@((self.v.T@b)/self.eig) if self.eig.numel() else torch.zeros_like(b)
 def project(self,candidate):
  d=candidate-self.anchor;d-=(d@self.q)@self.q.T
  a=self.solve(self.H@d.flatten());return self.anchor+d-(a@self.H).reshape_as(d)
 def responses(self,d):
  # <g,d> = <g,I> + <H,d-I> when source affine constraint holds.
  source=(d@self.q)@self.q.T
  return torch.tensor(self.fixed,device=d.device,dtype=d.dtype)+self.H@(d-source).flatten()
 def response_error(self,d):return float(self.responses(d).abs().max()/d.norm().clamp_min(1e-12))
 def native_responses(self,d,statistics):
  values=[]
  for stat,norm in zip(statistics,self.norms):
   if norm:
    g=stat.to(d.device,dtype=torch.float64);values.append(float((g*d).sum()/norm));del g
   else:values.append(0.)
  return dict(per_record=values,max_normalized=max(map(abs,values),default=0.)/max(float(d.norm()),1e-12))

def repair(model,layer,base,teacher_delta,keys,q,null_batch,base_null,read_endpoint,emit,statistics):
 space=Intersection(teacher_delta,q,statistics);current=space.anchor;reference=teacher_delta@keys;initial=layer.weight.detach().clone();rows=[]
 opt=torch.optim.Adam([layer.weight],lr=.0005,betas=(.9,.999),eps=1e-8,weight_decay=0.)
 def evidence(d,native):
  response=space.responses(d);error=relnorm((d-teacher_delta)@keys,reference)
  if error>1e-8 or space.response_error(d)>1e-8 or not torch.isfinite(d).all():raise ValueError('Source/response residual')
  return dict(source_action_error=error,response_error=space.response_error(d),per_record_response=response.tolist(),native_response=space.native_responses(native,statistics))
 try:
  with torch.no_grad():layer.weight.copy_((base.double()+current).to(base.dtype))
  initial_evidence=evidence(current,layer.weight.detach().double()-base.double());endpoint=read_endpoint();initial_endpoint=endpoint
  status='INITIAL_NULL_PASSED' if abs(endpoint['cumulative_null_change'])<=.01 else 'STEP_CAP_REACHED'
  if not endpoint['gates']['trust']:status='INITIAL_TRUST_FAILED'
  else:
   for step in range(25):
    if abs(endpoint['cumulative_null_change'])<=.01:break
    with editable_fp32(model,layer):
     try:
      with torch.no_grad():layer.weight.copy_((base.double()+current).float())
      opt.zero_grad(set_to_none=True);loss=nll(model,null_batch);h=.5*(loss-base_null).square();before=float(loss.detach());objective=float(h.detach());h.backward();finite(layer.weight.grad);opt.step();finite(layer.weight)
      state=opt.state[layer.weight]
      if int(state['step'])!=step+1 or state['exp_avg'].dtype!=torch.float32 or state['exp_avg_sq'].dtype!=torch.float32:raise ValueError('Adam continuity/dtype')
      if not all(torch.isfinite(x).all() for x in state.values() if isinstance(x,torch.Tensor)):raise ValueError('Nonfinite optimizer state')
      proposed=space.project(layer.weight.detach().double()-base.double())
     finally:opt.zero_grad(set_to_none=True)
    with torch.no_grad():layer.weight.copy_((base.double()+proposed).to(base.dtype))
    native=layer.weight.detach().double()-base.double();ev=evidence(proposed,native);measured=read_endpoint();trust=measured['gates']['trust']
    row=dict(step=step+1,FP32_null_before=before,FP32_objective_before=objective,optimizer_step=int(state['step']),optimizer_state_FP32=True,native_action_error=relnorm((native-teacher_delta)@keys,reference),endpoint=measured,accepted_trust=trust,**ev);rows.append(row);emit(row)
    if not trust:
     with torch.no_grad():layer.weight.copy_((base.double()+current).to(base.dtype))
     status='TRUST_REJECTED_STOP';break
    current=proposed;endpoint=measured
    if abs(endpoint['cumulative_null_change'])<=.01:status='NULL_TARGET_REACHED';break
  final=evidence(current,layer.weight.detach().double()-base.double())
  return dict(status=status,geometry=space.meta,initial_evidence=initial_evidence,final_evidence=final,initial_endpoint=initial_endpoint,endpoint=endpoint,steps=rows,repair_backwards=len(rows),accepted_updates=sum(x['accepted_trust'] for x in rows),final_action_error=final['source_action_error'],final_native_action_error=relnorm((layer.weight.detach().double()-base.double()-teacher_delta)@keys,reference))
 except BaseException:
  with torch.no_grad():layer.weight.data=initial.clone()
  raise
 finally:opt.zero_grad(set_to_none=True)

def run_checks():
 from .scalar3_ops import run_checks as prior
 prior();torch.manual_seed(27)
 q=torch.eye(4,dtype=torch.float64)[:,:1];t=torch.randn(2,4,dtype=torch.float64);stats=[torch.randn(2,4).float() for _ in range(2)]
 sp=Intersection(t,q,stats);assert sp.response_error(sp.anchor)<1e-8 and relnorm((sp.anchor-t)@q,t@q)<1e-8
 out=sp.project(torch.randn_like(t));assert sp.response_error(out)<1e-8 and relnorm((out-t)@q,t@q)<1e-8
 assert sp.meta['compensation_identity_relative_error']<1e-8 and torch.allclose(sp.responses(out),torch.tensor([float((g.double()/g.double().norm()*out).sum()) for g in stats],dtype=torch.float64),atol=1e-12)
 # A response entirely fixed by source action cannot be canceled.
 try:Intersection(torch.ones(2,4,dtype=torch.float64),q,[torch.tensor([[1.,0,0,0],[0,0,0,0]])])
 except ValueError:pass
 else:raise AssertionError('Expected inconsistent intersection')
 for stat in [torch.full((2,4),float('nan')),torch.ones(2,3)]:
  try:Intersection(t,q,[stat])
  except ValueError:pass
  else:raise AssertionError('Invalid statistics')
 for h,g in [(31*1024**3,48*1024**3),(32*1024**3,47*1024**3)]:
  try:memory_admission(h,g)
  except MemoryError:pass
  else:raise AssertionError('Memory admission')
 from types import SimpleNamespace
 class Toy(torch.nn.Module):
  def __init__(self):super().__init__();self.layer=torch.nn.Linear(3,3,bias=False).half()
  def forward(self,inputs_embeds):return SimpleNamespace(logits=self.layer(inputs_embeds))
 class Batch:
  labels=torch.tensor([[-100,1,2]])
  def forward_kwargs(self):return dict(inputs_embeds=torch.eye(3).half().unsqueeze(0))
 m=Toy();layer=m.layer;base=layer.weight.detach().clone();q=torch.eye(3,dtype=torch.float64)[:,:1];t=torch.zeros_like(base,dtype=torch.float64);t[:,0]=.1;stat=torch.zeros_like(base,dtype=torch.float32);stat[:,1]=1
 def endpoint(drift=-1.,trust=True):return dict(gates=dict(null=abs(drift)<=.01,visual=True,trust=trust),cumulative_null_change=drift)
 rows=[];r=repair(m,layer,base,t,q,q,Batch(),float(nll(m,Batch()).detach())+1,lambda:endpoint(),rows.append,[stat]);assert len(rows)==25 and all(x['response_error']<1e-8 and x['optimizer_state_FP32'] for x in rows) and rows[-1]['optimizer_step']==25
 with torch.no_grad():layer.weight.copy_(base)
 count=[0]
 def reject():count[0]+=1;return endpoint(trust=count[0]==1)
 r=repair(m,layer,base,t,q,q,Batch(),4.,reject,lambda x:None,[stat]);assert r['status']=='TRUST_REJECTED_STOP' and r['accepted_updates']==0
 with torch.no_grad():layer.weight.copy_(base)
 r=repair(m,layer,base,t,q,q,Batch(),4.,lambda:endpoint(trust=False),lambda x:None,[stat]);assert r['status']=='INITIAL_TRUST_FAILED' and r['repair_backwards']==0
 original=layer.weight.detach().clone()
 def fail(row):raise RuntimeError('injected')
 try:repair(m,layer,base,t,q,q,Batch(),4.,lambda:endpoint(),fail,[stat])
 except RuntimeError:pass
 else:raise AssertionError('Expected failure')
 assert torch.equal(original,layer.weight) and layer.weight.grad is None and layer.weight.dtype==torch.float16 and not layer._forward_hooks and not layer._forward_pre_hooks
 return dict(status='PASS',checks=['new_repair_25_step_response_and_Adam_continuity','new_repair_initial_trust_stop','new_repair_trust_rollback_exception_restore','inherited_scalar_objective_Adam_dtype_stop_rollback_restore','source_response_intersection','full_proposal_projection','direct_response_identity','compensation_minimum_norm_identity','inconsistent_intersection_rejected','nonfinite_shape_rejected','memory_reserve_rejected'])
