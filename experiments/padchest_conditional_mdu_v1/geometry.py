"""Conditional TT residuals and source-only contrast-calibrated routing."""
import math
import types
import torch

CONCEPTS=('pleural effusion','alveolar pattern')
def concept(question):
 found=[x for x in CONCEPTS if x in question.lower()]
 return found[0] if len(found)==1 else None

def evidence_key(last_hidden,vision):
 assert vision.ndim==3 and vision.shape[0]==2
 delta=vision[1].float().mean(0)-vision[0].float().mean(0)
 assert delta.norm()>1e-10
 h=last_hidden.float().reshape(-1)
 return torch.cat([h/h.norm().clamp_min(1e-12),delta/delta.norm()])/math.sqrt(2)

def basis(rows):
 h=torch.cat(rows).double();assert h.ndim==2 and h.shape[1]==14336 and h.shape[0]<=640
 h=h/h.norm(dim=1,keepdim=True).clamp_min(1e-12)
 _,s,v=torch.linalg.svd(h,full_matrices=False)
 keep=s>s[0]*1e-6;q=v[keep].T.float().contiguous()
 assert q.shape[1]>0 and float((q.T@q-torch.eye(q.shape[1])).abs().max())<1e-4
 return q,dict(predictors=len(h),rank=q.shape[1],dimension=q.shape[0],relative_cutoff=1e-6)

def project_normalized(x,q,epsilon=1e-6):
 # Preserve the amplitude of the remaining evidence. Renormalizing after
 # projection would amplify numerical/irrelevant remnants back to unit RMS.
 z=x/(x.square().mean(-1,keepdim=True).sqrt()+epsilon)
 out=z-(z@q)@q.T
 relative=out.norm(dim=-1,keepdim=True)/z.norm(dim=-1,keepdim=True).clamp_min(1e-12)
 return torch.where(relative<=1e-5,torch.zeros_like(out),out)

def bind_projection(expert,q):
 assert q.requires_grad is False
 def residual(self,activation):
  b,a=self.factors();z=project_normalized(activation.to(a.dtype),q.to(a.dtype),self.epsilon)
  return (z@a.T)@b.T
 expert.residual=types.MethodType(residual,expert)
 return expert

def calibrate(tasks,positive_keys,negative_keys):
 entries=[]
 for t in tasks:
  pos=torch.stack(positive_keys[t['edit_id']]).float();neg=torch.stack(negative_keys[t['edit_id']]).float();center=pos.mean(0)
  near=(pos-center).norm(dim=1).max();far=(neg-center).norm(dim=1).min()
  assert far>near+1e-6,'No source-only contrastive scope separation; stop rather than tune on probes'
  entries.append(dict(edit_id=t['edit_id'],concept=concept(t['native']['question']),center=center,radius=float((near+far)/2),positive_max=float(near),negative_min=float(far)))
 return entries

def route(entries,key,question):
 c=concept(question);eligible=[e for e in entries if e['concept']==c]
 if not eligible:return dict(activated=False,logical_edit_id=None,reason='SEMANTIC_SCOPE_MISMATCH')
 candidates=[(float((key-e['center']).norm())/max(e['radius'],1e-12),e) for e in eligible]
 score,e=min(candidates,key=lambda x:(x[0],x[1]['edit_id']))
 return dict(activated=score<=1,logical_edit_id=e['edit_id'] if score<=1 else None,nearest_logical_edit_id=e['edit_id'],relative_distance=score,radius=e['radius'],reason='CONTRAST_CALIBRATED_SCOPE')
