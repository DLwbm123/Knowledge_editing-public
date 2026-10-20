"""Fixed projected Adam repair of the original scalar null loss in a source-action class."""
import torch
from .editor import editable_fp32,nll,finite
from .equiv3_ops import relnorm
MODULE='model.layers.6.mlp.down_proj'
AUTH=dict(samples=['E01','E02','E03'],layer=MODULE,forward_max=4500,work_forward_max=4300,fit_forward_max=26,fit_forward_total_max=78,per_generation_forward_max=128,seconds=900,disk_max=251658240,generation_max=24,judge_payload_max=21,checkpoint=0,retries=0,native_null_gate=.01,trust=.001,rank_rtol=1e-10,residual_max=1e-8,repair_steps=25,lr=.0005)

def affine(candidate,anchor,q):
 diff=candidate-anchor;return anchor+diff-(diff@q)@q.T

def repair(model,layer,base,teacher_delta,keys,q,null_batch,base_null,read_endpoint,emit):
 anchor=(teacher_delta@q)@q.T;current=anchor;reference=teacher_delta@keys;initial=layer.weight.detach().clone();rows=[]
 opt=torch.optim.Adam([layer.weight],lr=.0005,betas=(.9,.999),eps=1e-8,weight_decay=0.)
 try:
  with torch.no_grad():layer.weight.copy_((base.double()+current).to(base.dtype))
  endpoint=read_endpoint();initial_endpoint=endpoint
  if not endpoint['gates']['trust']:raise ValueError('Initial I trust binding')
  status='INITIAL_NULL_PASSED' if abs(endpoint['cumulative_null_change'])<=.01 else 'STEP_CAP_REACHED'
  for step in range(25):
   if abs(endpoint['cumulative_null_change'])<=.01:break
   with editable_fp32(model,layer):
    try:
     with torch.no_grad():layer.weight.copy_((base.double()+current).float())
     opt.zero_grad(set_to_none=True);loss=nll(model,null_batch);h=.5*(loss-base_null).square();before=float(loss.detach());objective=float(h.detach());h.backward();finite(layer.weight.grad);opt.step();finite(layer.weight)
     state=opt.state[layer.weight]
     if int(state['step'])!=step+1 or state['exp_avg'].dtype!=torch.float32 or state['exp_avg_sq'].dtype!=torch.float32:raise ValueError('Adam state continuity/dtype')
     if not all(torch.isfinite(x).all() for x in state.values() if isinstance(x,torch.Tensor)):raise ValueError('Nonfinite optimizer state')
     proposed=affine(layer.weight.detach().double()-base.double(),anchor,q)
    finally:opt.zero_grad(set_to_none=True)
   error=relnorm((proposed-teacher_delta)@keys,reference)
   if error>1e-8 or not torch.isfinite(proposed).all():raise ValueError('Source action residual')
   with torch.no_grad():layer.weight.copy_((base.double()+proposed).to(base.dtype))
   native_delta=layer.weight.detach().double()-base.double();native_error=relnorm((native_delta-teacher_delta)@keys,reference)
   measured=read_endpoint();trust=measured['gates']['trust'];row=dict(step=step+1,FP32_null_before=before,FP32_objective_before=objective,optimizer_step=int(state['step']),optimizer_state_FP32=True,source_action_error=error,native_action_error=native_error,endpoint=measured,accepted_trust=trust)
   rows.append(row);emit(row)
   if not trust:
    with torch.no_grad():layer.weight.copy_((base.double()+current).to(base.dtype))
    status='TRUST_REJECTED_STOP';break
   current=proposed;endpoint=measured
   if abs(endpoint['cumulative_null_change'])<=.01:status='NULL_TARGET_REACHED';break
  return dict(status=status,initial_endpoint=initial_endpoint,endpoint=endpoint,steps=rows,repair_backwards=len(rows),accepted_updates=sum(x['accepted_trust'] for x in rows),final_action_error=relnorm((current-teacher_delta)@keys,reference),final_native_action_error=relnorm((layer.weight.detach().double()-base.double()-teacher_delta)@keys,reference))
 except BaseException:
  with torch.no_grad():layer.weight.data=initial.clone()
  raise
 finally:opt.zero_grad(set_to_none=True)

def run_checks():
 from types import SimpleNamespace
 from .teacherL3_ops import run_checks as prior
 prior()
 class Toy(torch.nn.Module):
  def __init__(self):super().__init__();self.layer=torch.nn.Linear(3,3,bias=False).half()
  def forward(self,inputs_embeds):return SimpleNamespace(logits=self.layer(inputs_embeds))
 class Batch:
  labels=torch.tensor([[-100,1,2]])
  def forward_kwargs(self):return dict(inputs_embeds=torch.eye(3).half().unsqueeze(0))
 torch.manual_seed(9);m=Toy();layer=m.layer;base=layer.weight.detach().clone();q=torch.eye(3,dtype=torch.float64)[:,:1];keys=q;teacher=torch.zeros_like(base,dtype=torch.float64);teacher[:,0]=.1
 current=nll(m,Batch()).detach();x=current.clone().requires_grad_();h=.5*(x-(current+1)).square();assert torch.autograd.grad(h,x)[0]==-1
 def endpoint(drift=-1.,trust=True):return dict(gates=dict(null=abs(drift)<=.01,visual=True,trust=trust),cumulative_null_change=drift)
 rows=[];r=repair(m,layer,base,teacher,keys,q,Batch(),float(current)+1,lambda:endpoint(),rows.append);assert len(rows)==25 and rows[-1]['optimizer_step']==25 and r['accepted_updates']==25 and r['final_action_error']<1e-8 and all(x['optimizer_state_FP32'] for x in rows)
 with torch.no_grad():layer.weight.copy_(base)
 rows=[];r=repair(m,layer,base,teacher,keys,q,Batch(),float(current),lambda:endpoint(0.),rows.append);assert not rows and r['status']=='INITIAL_NULL_PASSED'
 with torch.no_grad():layer.weight.copy_(base)
 seen=[0]
 def reached():seen[0]+=1;return endpoint(0. if seen[0]>1 else -1.)
 r=repair(m,layer,base,teacher,keys,q,Batch(),float(current)+1,reached,lambda x:None);assert r['status']=='NULL_TARGET_REACHED' and r['repair_backwards']==1
 with torch.no_grad():layer.weight.copy_(base)
 count=[0]
 def reject():count[0]+=1;return endpoint(trust=count[0]==1)
 r=repair(m,layer,base,teacher,keys,q,Batch(),float(current)+1,reject,lambda x:None);expected=(base.double()+teacher@q@q.T).half();assert r['status']=='TRUST_REJECTED_STOP' and r['accepted_updates']==0 and torch.equal(layer.weight,expected)
 original=layer.weight.detach().clone()
 def fail(row):raise RuntimeError('injected')
 try:repair(m,layer,base,teacher,keys,q,Batch(),float(current)+1,lambda:endpoint(),fail)
 except RuntimeError:pass
 else:raise AssertionError('Expected injected exception')
 assert torch.equal(layer.weight,original) and layer.weight.dtype==torch.float16 and layer.weight.grad is None and not layer._forward_hooks and not layer._forward_pre_hooks
 return dict(status='PASS',checks=['inherited_fixed_teacher_endpoint_restore','scalar_squared_loss_gradient','affine_source_action_preserved','Adam_FP32_state_continuity_25_steps','initial_and_after_update_target_early_stop','trust_rejection_rollback','exception_native_restore'])
