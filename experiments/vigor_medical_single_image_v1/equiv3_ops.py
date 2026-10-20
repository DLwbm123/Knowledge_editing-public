"""Fixed source-action constrained minimum-cost representatives, not CanonEdit reproduction."""
import torch
MODULE='model.layers.6.mlp.down_proj'
AUTH=dict(samples=['E01','E02','E03'],layer=MODULE,forward_max=6800,work_forward_max=6600,fit_forward_max=26,fit_forward_total_max=78,per_generation_forward_max=128,seconds=1800,disk_max=251658240,generation_max=45,judge_payload_max=42,checkpoint=0,retries=0,native_null_gate=.01,trust=.001,rank_rtol=1e-10,residual_max=1e-8)

def relnorm(x,reference):return float(x.norm()/reference.norm().clamp_min(1e-12))
def basis(keys):
 if keys.dtype!=torch.float64 or keys.ndim!=2 or not torch.isfinite(keys).all():raise ValueError('Invalid source keys')
 q,s,_=torch.linalg.svd(keys,full_matrices=False);r=int((s>1e-10*s[0]).sum()) if s.numel() else 0
 if not r:raise ValueError('Empty source action space')
 q=q[:,:r];identity=torch.eye(r,device=q.device,dtype=q.dtype);orth=relnorm(q.T@q-identity,identity)
 if orth>1e-8:raise ValueError('Orthogonality residual')
 return q,dict(input_dimension=keys.shape[0],positions=keys.shape[1],rank=r,remaining_input_freedom=keys.shape[0]-r,singular_max=float(s[0]),singular_min=float(s[-1]),singular_min_retained=float(s[r-1]),discarded_nonzero=int((s[r:]>0).sum()),orthogonality_residual=orth)

def rewrite(delta,keys,q,metric):
 if delta.dtype!=torch.float64 or metric.dtype!=torch.float64 or not torch.isfinite(delta).all() or not torch.isfinite(metric).all() or not bool((metric>0).all()):raise ValueError('Nonfinite update or nonpositive metric')
 if delta.shape[1]!=keys.shape[0] or metric.shape!=(keys.shape[0],):raise ValueError('Metric shape')
 weighted=q.T/metric[None,:];gram=weighted@q;sol=torch.linalg.solve(gram,weighted);res=relnorm(gram@sol-weighted,weighted)
 candidate=(delta@q)@sol;action=delta@keys;effect=relnorm((candidate-delta)@keys,action)
 if not torch.isfinite(candidate).all() or res>1e-8 or effect>1e-8:raise ValueError('Solve or source action residual')
 # Diagonal C permits a block computation without storing another dense weight matrix.
 cost=lambda x:sum(float((x[:,j:j+256].square()*metric[None,j:j+256]).sum())/2 for j in range(0,x.shape[1],256))
 return candidate,dict(solve_residual=res,source_action_relative_error=effect,teacher_metric_cost=cost(delta),candidate_metric_cost=cost(candidate))

def run_checks():
 from .teacherL3_ops import run_checks as prior
 prior();torch.manual_seed(21);k=torch.randn(7,3,dtype=torch.float64);d=torch.randn(4,7,dtype=torch.float64);q,meta=basis(k);metric=torch.arange(1,8,dtype=torch.float64)
 i,si=rewrite(d,k,q,torch.ones_like(metric));u,su=rewrite(d,k,q,metric)
 assert torch.allclose(i,(d@q)@q.T,atol=1e-10) and not torch.allclose(i,u)
 assert su['candidate_metric_cost']<=su['teacher_metric_cost']+1e-9 and float((u.square()*metric).sum())<=float((i.square()*metric).sum())+1e-9
 assert torch.linalg.norm((u*metric)@(torch.eye(7,dtype=q.dtype)-q@q.T))<1e-9
 full=torch.eye(7,dtype=torch.float64);fq,_=basis(full);f,_=rewrite(d,full,fq,metric);assert torch.allclose(f,d,atol=1e-10)
 for bad in [metric*0,metric*float('nan')]:
  try:rewrite(d,k,q,bad)
  except ValueError:pass
  else:raise AssertionError('Invalid metric accepted')
 return dict(status='PASS',checks=['inherited_fixed_teacher_and_endpoint_restore','source_action_constraint','KKT_minimum_metric_cost','identity_control_and_U_difference','full_rank_degeneracy','invalid_metric_rejected'])
