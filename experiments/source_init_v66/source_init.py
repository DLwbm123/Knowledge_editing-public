"""Fixed TT-SVD initialization from four source-only unedited prefix features."""
import torch

def input_cores(features):
 assert features.shape==(4,14336) and torch.isfinite(features).all()
 x=features.detach().cpu().double()
 x=x/(x.square().mean(-1,keepdim=True).sqrt()+1e-6)
 q,r=torch.linalg.qr(x.T,mode='reduced')
 assert torch.diag(r).abs().min()>1e-8,'SOURCE_FEATURE_RANK_DEFICIENT'
 signs=torch.where(torch.diag(r)<0,-torch.ones(4,dtype=q.dtype),torch.ones(4,dtype=q.dtype));q=q*signs
 target=q.T.contiguous();u,s,vh=torch.linalg.svd(target.reshape(448,128),full_matrices=False)
 g3=(u[:,:8]*s[:8]).reshape(4,112,8).float();g4=vh[:8].reshape(8,128,1).float()
 approx=torch.einsum('aib,bj->aij',g3.double(),g4[:,:,0].double()).reshape(4,14336)
 assert torch.isfinite(approx).all() and approx.norm()>0
 return g3,g4,dict(retained_energy=float(s[:8].square().sum()/s.square().sum()),relative_projection_error=float((approx-target).norm()/target.norm()),feature_QR_min_diagonal=float(torch.diag(r).abs().min()))

def selfcheck():
 gen=torch.Generator().manual_seed(660023);x=torch.randn(4,14336,generator=gen)
 a,b,m=input_cores(x);c,d,n=input_cores(x)
 assert torch.equal(a,c) and torch.equal(b,d) and m==n
 assert a.shape==(4,112,8) and b.shape==(8,128,1) and 0<m['retained_energy']<=1
 try:input_cores(torch.zeros_like(x))
 except AssertionError:pass
 else:raise AssertionError('Zero/rank deficient features admitted')
 return dict(status='PASS',fixed_rank8_TT_SVD=True,deterministic=True,rank_deficient_rejected=True,real_model_calls=0)
