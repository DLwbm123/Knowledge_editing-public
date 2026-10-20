"""Bounded HPSE-inspired gated prefixes with an immutable source-context teacher."""
import torch
from .nll_control import response as nll_response

def raw_forward(rt,raw,tokens):
 ids=raw['input_ids'];prefix=torch.tensor([tokens],device=ids.device,dtype=ids.dtype)
 ids=torch.cat([ids,prefix],dim=1)
 return rt.model(input_ids=ids,attention_mask=torch.ones_like(ids),images=raw['images'],use_cache=False).logits.float()

def trajectory(rt,layer,base,student,teacher,max_tokens=8):
 current=layer.weight.detach().clone();tokens=[];targets=[];events=[]
 try:
  for _ in range(max_tokens):
   with torch.no_grad():
    sl=raw_forward(rt,student,tokens)[0,-1];sp=sl.log_softmax(-1)
    layer.weight.copy_(base)
    try:tl=raw_forward(rt,teacher,tokens)[0,-1];tp=tl.log_softmax(-1)
    finally:layer.weight.copy_(current)
    if not torch.isfinite(sl).all() or not torch.isfinite(tl).all():raise ValueError('Nonfinite rollout logits')
    st=int(sl.argmax());tt=int(tl.argmax());confidence=float(tp[tt].exp());gap=float(tp[tt]-sp[tt]);gate=confidence>.3 and gap>2.
    values,ids=torch.topk(tl,16);q=values.softmax(-1);mass=float(tp[ids].exp().sum());token=tt if gate else st
    tokens.append(token);targets.append((ids.detach(),q.detach()));events.append(dict(teacher_confidence=confidence,teacher_student_log_gap=gap,gate=gate,teacher_top16_mass=mass,teacher_student_agree=tt==st))
    if token==rt.adapter.tokenizer.eos_token_id:break
 finally:
  with torch.no_grad():layer.weight.copy_(current)
 if not tokens:raise ValueError('Empty rollout')
 return tokens,targets,dict(token_count=len(tokens),teacher_gated=sum(x['gate'] for x in events),events=events,token_limit=8,complete_answer_not_claimed=True)

def kl_loss(rt,raw,tokens,targets):
 # Last T positions predict the generated T tokens: original-prompt last + T-1 prefix.
 logits=raw_forward(rt,raw,tokens[:-1])[0,-len(tokens):]
 if logits.shape[0]!=len(tokens):raise ValueError('Causal prefix alignment')
 lp=logits.log_softmax(-1);loss=[]
 for i,(ids,q) in enumerate(targets):loss.append((torch.xlogy(q,q)-q*lp[i,ids]).sum())
 return torch.stack(loss).mean()

def response(before,after,mb,ma,linear,jbefore,jafter,jlinear):
 r=nll_response(before,after,mb,ma,linear);gain=jbefore-jafter;rho=gain/(-jlinear) if jlinear<0 else None
 r.update(joint_gain=gain,rho_joint=rho);r['passed']=r['passed'] and rho is not None and torch.isfinite(torch.tensor(rho)).item() and gain>1e-5 and rho>=.1;return r

def next_radius(used,r):
 if not r['passed']:raise ValueError('Rejected radius update')
 rho=min(r['rho_NLL'],r['rho_joint']);return used*.5 if rho<.25 else min(1.,used*2) if rho>.75 else used

def run_checks():
 # Causal index math and exact KL gradient with truncated Q extended by zero elsewhere.
 x=torch.tensor([[2.,1.,0.],[0.,1.,2.]],requires_grad=True);ids=torch.tensor([0,2]);q=torch.tensor([.75,.25]);loss=(q*(q.log()-x.log_softmax(-1)[0,ids])).sum();g=torch.autograd.grad(loss,x)[0][0];expected=x[0].softmax(-1).detach()-torch.tensor([.75,0.,.25]);assert torch.allclose(g,expected,atol=1e-7)
 assert response(2.,1.5,0.,-1.,[-1.,1.,0.],3.,2.,-2.)['passed']
 assert not response(2.,2.1,0.,1.,[-1.,1.,0.],3.,2.,-2.)['passed']
 assert not response(2.,1.5,0.,1.,[-1.,1.,0.],3.,3.1,-2.)['passed']
 # Prompt suffix with T-1 appended tokens has exactly T teacher-forced read positions.
 for prompt_len in [1,5,577]:
  for t in [1,8]:assert list(range(prompt_len+t-1))[-t:]==list(range(prompt_len-1,prompt_len+t-1))
 from types import SimpleNamespace
 layer=torch.nn.Linear(1,1,bias=False);layer.weight.data.fill_(2.);base=torch.zeros_like(layer.weight);seen=[]
 class Model:
  def __call__(self,**kw):
   w=float(layer.weight);seen.append(w);v=torch.zeros(1,kw['input_ids'].shape[1],32);v[0,-1,0]=2. if w else 0.;v[0,-1,1]=5. if not w else 0.;return SimpleNamespace(logits=v)
 raw=dict(input_ids=torch.tensor([[1]]),images=torch.zeros(1));rt=SimpleNamespace(model=Model(),adapter=SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=-1)))
 tokens,targets,stats=trajectory(rt,layer,base,raw,raw,2)
 assert tokens==[1,1] and stats['teacher_gated']==2 and float(layer.weight)==2. and seen==[2.,0.,2.,0.]
 class Broken:
  def __call__(self,**kw):
   if float(layer.weight)==0.:raise RuntimeError('injected teacher failure')
   return Model()(**kw)
 rt.model=Broken()
 try:trajectory(rt,layer,base,raw,raw,2)
 except RuntimeError:pass
 else:raise AssertionError('failure not propagated')
 assert float(layer.weight)==2.
 return dict(status='PASS',checks=['KL_teacher_to_student_gradient','source_and_joint_guards','prefix_causal_indices','margin_diagnostic','immutable_teacher_swap','teacher_exception_restore'])
