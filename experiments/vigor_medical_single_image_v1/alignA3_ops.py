"""Full-response on-policy reverse KL with an isolated parameter teacher."""
import torch
from .hpse2_ops import raw_forward, response, next_radius

def causal_logits(rt,raw,tokens):
 if not tokens:raise ValueError('Empty complete trajectory')
 logits=raw_forward(rt,raw,tokens[:-1])[0,-len(tokens):]
 if logits.shape[0]!=len(tokens) or not torch.isfinite(logits).all():
  raise ValueError('Causal alignment or nonfinite logits')
 return logits

def reverse_kl(logits,logq):
 if logits.shape!=logq.shape or logq.requires_grad:raise ValueError('Teacher shape/stop-gradient binding')
 lp=logits.float().log_softmax(-1)
 value=(lp.exp()*(lp-logq)).sum(-1).mean()
 if not torch.isfinite(value):raise ValueError('Nonfinite reverse KL')
 return value

def kl_loss(rt,raw,tokens,logq):
 return reverse_kl(causal_logits(rt,raw,tokens),logq)

def teacher_logq(rt,student,base0,teacher_layer,base6,teacher6,raw,tokens):
 # Teacher calls precede the student autograd context; its casting hooks are invalid here.
 if any(m._forward_hooks or m._forward_pre_hooks for m in [student,teacher_layer]):
  raise ValueError('Teacher called with editable-layer hooks')
 if student.weight.dtype!=base0.dtype or teacher_layer.weight.dtype!=base6.dtype:
  raise ValueError('Teacher requires native matrices')
 if not torch.equal(teacher_layer.weight,base6):raise ValueError('Student contains teacher weights')
 current0=student.weight.detach().clone();current6=teacher_layer.weight.detach().clone()
 try:
  with torch.no_grad():
   student.weight.copy_(base0);teacher_layer.weight.copy_(teacher6)
   return causal_logits(rt,raw,tokens).log_softmax(-1).detach()
 finally:
  with torch.no_grad():student.weight.copy_(current0);teacher_layer.weight.copy_(current6)

def complete_rollout(rt,raw):
 with torch.inference_mode():value=rt.adapter.generate_prepared_with_result(raw,rt.generation_config)
 tokens=list(value.raw_token_ids)
 if not tokens or not value.decoded_text.strip() or len(tokens)>=rt.generation_config['max_new_tokens']:
  raise ValueError('Incomplete natural student trajectory')
 return tokens,dict(token_count=len(tokens),complete=True,full_vocabulary=True,
                    sampling='original_official_greedy',partial_trajectory_used=False)

def run_checks():
 from types import SimpleNamespace
 x=torch.tensor([[2.,1.,0.],[0.,1.,2.]],requires_grad=True)
 logq=torch.tensor([[0.,1.,2.],[1.,0.,2.]]).log_softmax(-1)
 loss=reverse_kl(x,logq);g=torch.autograd.grad(loss,x)[0]
 lp=x.detach().log_softmax(-1);p=lp.exp();d=lp-logq
 expected=p*(d-(p*d).sum(-1,keepdim=True))/2
 assert torch.allclose(g,expected,atol=1e-7)
 assert not torch.allclose(g,(p-logq.exp())/2,atol=1e-5)
 try:reverse_kl(x,logq.clone().requires_grad_())
 except ValueError:pass
 else:raise AssertionError('Teacher gradient must be rejected')
 s=torch.nn.Linear(1,1,bias=False).half();t=torch.nn.Linear(1,1,bias=False).half()
 s.weight.data.fill_(2);t.weight.data.zero_();base0=torch.zeros_like(s.weight);base6=t.weight.detach().clone();teacher6=torch.full_like(t.weight,3)
 seen=[]
 class Model:
  broken=False
  def __call__(self,**kw):
   a=float(s.weight.detach());b=float(t.weight.detach());seen.append((a,b))
   if self.broken:raise RuntimeError('injected')
   n=kw['input_ids'].shape[1]+3
   # Include a synthetic image expansion to exercise suffix indexing.
   logits=torch.arange(n).float()[None,:,None]*torch.tensor([1.,0.,-1.,2.])[None,None,:]
   return SimpleNamespace(logits=logits+(a+10*b)*torch.tensor([1.,0.,-1.,2.]))
 raw=dict(input_ids=torch.tensor([[1,2]]),images=torch.zeros(1));model=Model();rt=SimpleNamespace(model=model)
 q=teacher_logq(rt,s,base0,t,base6,teacher6,raw,[1,2,3,4])
 assert q.shape==(4,4) and not q.requires_grad and seen==[(0.,3.)]
 assert float(s.weight.detach())==2. and torch.equal(t.weight,base6)
 want=(torch.arange(4,8).float()[:,None]+30)*torch.tensor([1.,0.,-1.,2.])[None,:]
 assert torch.allclose(q,want.log_softmax(-1))
 model.broken=True
 try:teacher_logq(rt,s,base0,t,base6,teacher6,raw,[1])
 except RuntimeError:pass
 else:raise AssertionError('Teacher failure must propagate')
 assert float(s.weight.detach())==2. and torch.equal(t.weight,base6)
 h=s.register_forward_pre_hook(lambda m,args:None)
 try:
  try:teacher_logq(rt,s,base0,t,base6,teacher6,raw,[1])
  except ValueError:pass
  else:raise AssertionError('Casting hooks must be rejected')
 finally:h.remove()
 class Adapter:
  def generate_prepared_with_result(self,raw,config):
   assert config['max_new_tokens']==1024
   return SimpleNamespace(raw_token_ids=list(range(20)),decoded_text='complete')
 rt.adapter=Adapter();rt.generation_config=dict(max_new_tokens=1024)
 tokens,stats=complete_rollout(rt,raw);assert tokens==list(range(20)) and stats['token_count']==20
 assert response(2.,1.5,0.,-1.,[-1.,1.,0.],3.,2.,-2.)['passed']
 assert not response(2.,2.1,0.,1.,[-1.,1.,0.],3.,2.,-2.)['passed']
 assert not response(2.,1.5,0.,1.,[-1.,1.,0.],3.,3.1,-2.)['passed']
 return dict(status='PASS',checks=['full_reverse_KL_analytic_gradient','teacher_stop_gradient',
  'causal_suffix_with_image_expansion','double_matrix_native_swap',
  'teacher_exception_double_restore','cast_hook_rejection','complete_trajectory_no_truncation',
  'source_and_joint_native_guards'])
