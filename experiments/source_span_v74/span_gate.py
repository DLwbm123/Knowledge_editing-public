"""Fixed source-predictor input span; continuous projection-energy residual gate."""
import torch

def basis(rows):
 x=rows.detach().cpu().double();assert x.ndim==2 and torch.isfinite(x).all()
 _,s,vh=torch.linalg.svd(x,full_matrices=False)
 tolerance=torch.finfo(x.dtype).eps*max(x.shape)*float(s[0]);rank=int((s>tolerance).sum());assert rank>0
 u=vh[:rank].T.contiguous();torch.testing.assert_close(u.T@u,torch.eye(rank,dtype=torch.float64),rtol=1e-10,atol=1e-10)
 return u.float(),dict(rows=x.shape[0],dimension=x.shape[1],rank=rank,numerical_rank_tolerance=tolerance)

def gate(x,u):
 a=x.float();den=a.square().sum(-1,keepdim=True);num=(a@u).square().sum(-1,keepdim=True)
 g=torch.where(den>0,num/den.clamp_min(torch.finfo(a.dtype).tiny),torch.zeros_like(den)).clamp(0,1)
 assert torch.isfinite(g).all()
 return g

def selfcheck():
 x=torch.tensor([[1.,0.,0.],[0.,2.,0.],[1.,1.,0.]])
 u,stats=basis(x);assert stats['rank']==2
 torch.testing.assert_close(gate(x,u),torch.ones(3,1),rtol=1e-6,atol=1e-6)
 assert gate(torch.tensor([[0.,0.,1.]]),u).item()==0 and gate(torch.zeros(1,3),u).item()==0
 y=torch.tensor([[1.,0.,1.],[0.,2.,2.]])
 torch.testing.assert_close(gate(y,u),torch.full((2,1),.5))
 torch.testing.assert_close(gate(y,u),torch.cat([gate(v[None],u) for v in y]))
 torch.testing.assert_close(gate(3*y,u),gate(y,u))
 assert not u.requires_grad
 from token_hook import TokenHook
 from types import SimpleNamespace
 module=torch.nn.Linear(3,2,bias=False);expert=SimpleNamespace(residual=lambda z:z[...,:2])
 h=TokenHook(module,expert,'ALL_TOKENS');h.basis=u;h.set_layout(dict(prefix_length=1,visual_start=0,visual_count=0))
 xx=torch.cat([y,torch.zeros(1,3)])[None];base=module(xx).detach();labels=torch.tensor([[-100,1,1]])
 h.teacher(labels);h.attach()
 try:
  teacher=module(xx);expected=base+xx[...,:2]*gate(xx,u);torch.testing.assert_close(teacher,expected)
  with h.generation_request():
   full=module(xx[:,:2]);step=module(xx[:,2:])
  torch.testing.assert_close(torch.cat([full,step],dim=1),expected)
  assert len(h.last_generation_trace)==2
  expert.residual=lambda z:torch.zeros_like(z[...,:2]);h.teacher(labels);assert torch.equal(module(xx),base)
 finally:h.detach()
 return dict(status='PASS',source_span_unit=True,orthogonal_zero=True,zero_safe=True,continuous_half=True,batch_step_parity=True,scale_invariant=True,detached_basis=True,hook_teacher_generation_parity=True,zero_residual_exact=True,real_model_calls=0)
