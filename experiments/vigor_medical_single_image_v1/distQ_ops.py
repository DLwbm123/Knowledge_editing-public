"""Offline Gaussian full-softmax VJP measurement; fit directions never receive audit."""
import math
import torch
MODULE='model.layers.6.mlp.down_proj'
AUTH=dict(layer=MODULE,build=24,heldout=8,fit=16,audit=16,vocab=32000,fit_seed=314159001,audit_seed=271828001,random_seeds=[101,102,103,104],mis_seed=424242,scales=[.25,.5,1.],work_norm=17.58567032970479,forward_max=2600,work_forward_max=2500,backward_max=774,seconds=14400,work_seconds=13800,TERM_seconds=14100,disk_max=251658240,judge_max=0,checkpoint=0,retries=0,chunk=262144)

def finite(x):
 if not torch.isfinite(x).all():raise ValueError('Nonfinite measurement state')
 return x

def select(logits,labels):
 mask=labels[:,1:]!=-100
 if logits.ndim!=3 or logits.shape[0]!=1 or mask.shape!=logits[:,:-1].shape[:2] or not mask.any():raise ValueError('Causal answer mask')
 return logits[0,:-1][mask[0]].float()

def coefficients(p,noise):
 if p.shape!=noise.shape or not torch.isfinite(p).all() or not torch.isfinite(noise).all() or (p<0).any():raise ValueError('Invalid probabilities/noise')
 s=p.sqrt();return (s*noise-p*(s*noise).sum(-1,keepdim=True)).detach()

def probe(selected,parameter,seed,retain=False):
 p=selected.detach().softmax(-1);gen=torch.Generator(device=p.device).manual_seed(seed)
 noise=torch.randn(p.shape,device=p.device,dtype=p.dtype,generator=gen);r=coefficients(p,noise)
 g=torch.autograd.grad((selected*r).sum()/math.sqrt(24*selected.shape[0]),parameter,retain_graph=retain)[0]
 return finite(g.detach())

def log_distribution(selected):return finite(selected.double().log_softmax(-1).detach())
def KL(log0,selected):
 log1=log_distribution(selected);log0=log0.to(log1.device);return float(((log0.exp()*(log0-log1)).sum(-1)).mean())

def _normalize(d):
 n=float(d.double().norm())
 if not math.isfinite(n) or n<=0:raise ValueError('Zero/nonfinite direction')
 return d/n

def fit_directions(fit_statistics,reference_gradients,device):
 # Only fit data enter this function. Audit and held-out observations are absent.
 shape=fit_statistics[0].shape;D=fit_statistics[0].numel();chunk=AUTH['chunk'];m=16
 if len(fit_statistics)!=16 or len(reference_gradients)!=4:raise ValueError('Direction inputs')
 gram=torch.zeros((m,m),device=device,dtype=torch.float64)
 def block(start):return torch.stack([x.reshape(-1)[start:start+chunk] for x in fit_statistics]).to(device=device,dtype=torch.float64)
 for start in range(0,D,chunk):
  g=block(start);gram+=g@g.T
 eigen,v=torch.linalg.eigh(gram);keep=eigen>float(eigen.max())*1e-10
 if not keep.any() or float(eigen.min()) < -max(float(eigen.max()),1e-30)*1e-10:raise ValueError('Invalid fit Gram')
 v=v[:,keep];e=eigen[keep];directions=[];aliases=[]
 for i,seed in enumerate(AUTH['random_seeds']):
  generator=torch.Generator().manual_seed(seed);d=torch.randint(0,2,shape,generator=generator,dtype=torch.int8).float().mul_(2).sub_(1);directions.append(_normalize(d));aliases.append('R'+str(i+1))
 for i,g in enumerate(reference_gradients):directions.append(_normalize(g));aliases.append('G'+str(i+1))
 residuals=[]
 for i in range(4):
  d=directions[i].reshape(-1);rhs=torch.zeros(m,device=device,dtype=torch.float64)
  for start in range(0,D,chunk):rhs+=block(start)@d[start:start+chunk].to(device=device,dtype=torch.float64)
  c=v@((v.T@rhs)/e);out=torch.empty_like(d)
  for start in range(0,D,chunk):out[start:start+chunk]=(d[start:start+chunk].to(device=device,dtype=torch.float64)-c@block(start)).float().cpu()
  out=_normalize(out).reshape(shape);dots=torch.zeros(m,device=device,dtype=torch.float64)
  for start in range(0,D,chunk):dots+=block(start)@out.reshape(-1)[start:start+chunk].to(device=device,dtype=torch.float64)
  residual=float(dots.norm()/gram.trace().sqrt().clamp_min(1e-30));residuals.append(residual)
  if residual>1e-6:raise ValueError('Blind-direction fit projection')
  directions.append(out);aliases.append('B'+str(i+1))
 return directions,aliases,dict(rank=int(keep.sum()),eigenvalues=eigen.tolist(),condition=float(e.max()/e.min()),blind_residuals=residuals,only_fit_statistics_used=True,all_directions_frozen_before_audit=True)

def risks(statistics,delta,sign):
 D=delta.numel();chunk=AUTH['chunk'];flat=delta.reshape(-1);wrong=flat.flip(0)*sign.double();dots=torch.zeros(32,device=delta.device,dtype=torch.float64);wrongdots=torch.zeros_like(dots)
 for start in range(0,D,chunk):
  g=statistics[:,start:start+chunk].double();dots+=g@flat[start:start+chunk];wrongdots+=g@wrong[start:start+chunk]
 return dict(fit=float(dots[:16].square().mean()/2),audit=float(dots[16:].square().mean()/2),misaligned_audit=float(wrongdots[16:].square().mean()/2))

def summarize(rows):
 from scipy.stats import spearmanr
 work=[r for r in rows if r['scale']==1.]
 def corr(key,subset):
  if len(work)<3:return None
  x=[r[key] for r in work];y=[r[subset+'_KL'] for r in work]
  if len(set(x))<2 or len(set(y))<2 or not all(math.isfinite(t) for t in x+y):return None
  value=float(spearmanr(x,y).statistic);return value if math.isfinite(value) else None
 valid=[r for r in work if r['native_norm']>0 and r['C_KL']>1e-10];accurate=[r for r in valid if .5<=r['audit']/r['C_KL']<=2];severe=[r['point'] for r in valid if r['audit']<r['C_KL']/4]
 coverage=[r['point'] for r in work if r['direction'].startswith('B') and r['H_KL']>1e-10 and r['audit']<r['H_KL']/4]
 correlations={s:{k:corr(k,s) for k in ['audit','F0','norm_squared','misaligned_audit']} for s in ['C','H']}
 N=len(work)==24 and len(valid)>=20 and len(accurate)>=.8*len(valid) and not severe
 H=correlations['H']['audit'] is not None and correlations['H']['audit']>=.6 and not coverage
 I=all(v['audit'] is not None and all(v[k] is not None for k in ['F0','norm_squared','misaligned_audit']) and v['audit']>=max(v[k] for k in ['F0','norm_squared','misaligned_audit'])+.1 for v in correlations.values())
 complete=len(rows)==72 and all(len(r['record_KL'])==32 for r in rows)
 numeric=all(math.isfinite(x) and x>=-1e-12 for r in rows for x in r['record_KL'])
 return dict(status='PASS' if complete and numeric and N and H and I else ('INCOMPLETE' if not complete else 'NOT_QUALIFIED'),complete=complete,numerically_valid=numeric,work_points=len(work),discernible_work_points=len(valid),factor_two_points=len(accurate),serious_C_underestimates=severe,blind_H_underestimates=coverage,correlations=correlations,gates=dict(N=N,H=H,I=I),editor_allowed=False,conditional_prefix_distribution_only=True)

def run_checks():
 from scipy.stats import chi2
 from .editor import editable_fp32
 torch.manual_seed(20261020);dtype=torch.float64;D=6;V=5;T=5;J=torch.randn(T,V,D,dtype=dtype);w=torch.randn(D,dtype=dtype,requires_grad=True);z=torch.einsum('tvd,d->tv',J,w);p=z.detach().softmax(-1);weights=torch.tensor([1/4,1/4,1/6,1/6,1/6],dtype=dtype)
 A=torch.stack([torch.diag(x.sqrt())-x[:,None]*x.sqrt()[None,:] for x in p]);C=torch.stack([torch.diag(x)-x[:,None]*x[None,:] for x in p]);assert torch.allclose(A@A.transpose(-1,-2),C,atol=1e-12,rtol=1e-12)
 noise=torch.randn(T,V,dtype=dtype);r=coefficients(p,noise);vjp=torch.autograd.grad((z*r*weights.sqrt()[:,None]).sum(),w)[0];exact=torch.einsum('tvd,tv,t->d',J,r,weights.sqrt());assert torch.allclose(vjp,exact,atol=1e-10,rtol=1e-10) and not r.requires_grad
 F=torch.einsum('tvi,tvw,twj,t->ij',J,C,J,weights);n=8192;xi=torch.randn(n,T,V,dtype=dtype);rr=coefficients(p.expand_as(xi),xi);gs=torch.einsum('tvd,ntv,t->nd',J,rr,weights.sqrt());x=math.log(8/.001)
 for direction in torch.randn(4,D,dtype=dtype):
  q=float(direction@F@direction);ratio=float((gs@direction).square().mean()/q);assert 1-2*math.sqrt(x/n)<=ratio<=1+2*math.sqrt(x/n)+2*x/n
 fit=gs[:2];_,_,vh=torch.linalg.svd(fit,full_matrices=False);d=torch.randn(D,dtype=dtype);d-=vh.T@(vh@d);assert (fit@d).norm()<1e-10 and (gs[2:18]@d).square().mean()>0
 P=torch.eye(D,dtype=dtype).flip(0);P[:,::2]*=-1;assert torch.allclose((gs[:16]@P)@(gs[:16]@P).T,gs[:16]@gs[:16].T,atol=1e-10)
 layer=torch.nn.Linear(3,3,bias=False).half();model=torch.nn.Sequential(layer);before=layer.weight.detach().clone()
 try:
  with editable_fp32(model,layer):
   layer.weight.data.add_(1);raise RuntimeError('injected')
 except RuntimeError:pass
 assert torch.equal(before,layer.weight) and layer.weight.dtype==torch.float16 and not layer._forward_hooks and not layer._forward_pre_hooks
 for pbad in [torch.tensor([[float('nan')]]),torch.tensor([[-1.]])]:
  try:coefficients(pbad,torch.ones_like(pbad))
  except ValueError:pass
  else:raise AssertionError('Invalid probability accepted')
 try:select(torch.ones(1,2,3),torch.full((1,2),-100))
 except ValueError:pass
 else:raise AssertionError('Empty mask accepted')
 assert summarize([])['status']=='INCOMPLETE' and chi2.ppf(.05/144,16)>0
 return dict(status='PASS',checks=['exact_softmax_covariance','causal_VJP_sqrt_record_position_weights','8192_Gaussian_quadratic_samples_registered_bounds','fit_blind_independent_audit','signed_permutation_preserves_spectrum','native_exception_restore','nonfinite_negative_probability_empty_mask_rejected','missing_not_qualified','chi_square_dependency'],model_calls=0,judge_calls=0)
