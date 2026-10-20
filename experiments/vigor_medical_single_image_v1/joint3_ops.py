"""Source/null strict layer-action consistency, not scalar null feasibility."""
import torch
from .equiv3_ops import basis,relnorm
MODULE='model.layers.6.mlp.down_proj'
AUTH=dict(samples=['E01','E02','E03'],layer=MODULE,forward_max=600,work_forward_max=560,fit_forward_max=26,fit_forward_total_max=78,per_generation_forward_max=128,seconds=900,disk_max=251658240,generation_max=3,judge_payload_max=0,checkpoint=0,retries=0,native_null_gate=.01,trust=.001,rank_rtol=1e-10,residual_max=1e-8)

def rowspace_residual(matrix,target):
 if matrix.dtype!=torch.float64 or not torch.isfinite(matrix).all() or not torch.isfinite(target).all():raise ValueError('Nonfinite consistency input')
 _,s,vh=torch.linalg.svd(matrix,full_matrices=False);r=int((s>1e-10*s[0]).sum()) if s.numel() and s[0]>0 else 0
 vr=vh[:r];res=relnorm(target-(target@vr.T)@vr,target)
 return dict(rank=r,singular_max=float(s[0]) if s.numel() else 0.,singular_min=float(s[-1]) if s.numel() else 0.,min_retained=float(s[r-1]) if r else None,condition_retained=float(s[0]/s[r-1]) if r else None,relative_residual=res,consistent=res<=1e-8)

def exact_dot(weight,base,key):
 if any(x.dtype!=torch.float16 or not torch.isfinite(x).all() for x in [weight,base,key]):raise ValueError('Exact witness requires native finite half')
 w=(weight.double()*2**24).tolist();b=(base.double()*2**24).tolist();k=(key.double()*2**24).tolist()
 if any(float(x)!=int(x) for seq in [w,b,k] for x in seq):raise ValueError('Non-dyadic half scaling')
 numerator=sum((int(x)-int(y))*int(z) for x,y,z in zip(w,b,k));return numerator,numerator/(2**48)

def shared_witness(source,null,teacher,base):
 if source.dtype!=torch.float16 or null.dtype!=torch.float16:return dict(status='NA_NON_HALF'),None
 for i in range(source.shape[1]):
  hits=torch.nonzero((null==source[:,i:i+1]).all(0)).flatten()
  if not hits.numel():continue
  j=int(hits[0]);effect=(teacher.double()-base.double())@source[:,i].double();coords=torch.nonzero(effect!=0).flatten()
  if not coords.numel():return dict(status='NA_FIRST_SHARED_ZERO_ACTION',shared_key_found=True),dict(source_column=i,null_column=j)
  coordinate=int(coords[0]);num,value=exact_dot(teacher[coordinate],base[coordinate],source[:,i])
  return dict(status='EXACT_STRICT_ACTION_CONTRADICTION' if num else 'NA_EXACT_ZERO',shared_key_found=True,exact_nonzero=num!=0,absolute_action=abs(value),dyadic_denominator_power=48),dict(source_column=i,null_column=j,coordinate=coordinate,exact_numerator=str(num))
 return dict(status='NA_NO_EXACT_SHARED_KEY',shared_key_found=False),None

def diagnose(source,null,teacher,base):
 ks=source.double();kn=null.double();delta=teacher.double()-base.double();q,meta=basis(ks);projected=q@(q.T@kn);perp=kn-projected;fixed=(delta@q)@(q.T@kn)
 norms=kn.norm(dim=0).clamp_min(1e-12);ratios=perp.norm(dim=0)/norms
 separated=rowspace_residual(perp,-fixed);joined=torch.cat([ks,kn],dim=1);target=torch.cat([delta@ks,torch.zeros((delta.shape[0],kn.shape[1]),device=ks.device,dtype=ks.dtype)],dim=1);joint=rowspace_residual(joined,target)
 numerical=('CONSISTENT_STRONG_SUBPROBLEM' if joint['consistent'] else 'INCONSISTENT_STRONG_SUBPROBLEM') if separated['consistent']==joint['consistent'] else 'UNRESOLVED_NUMERICAL_CLASSIFICATION'
 witness,private=shared_witness(source,null,teacher,base)
 return dict(source_space=meta,null_space=rowspace_residual(kn,torch.zeros((1,kn.shape[1]),device=kn.device,dtype=kn.dtype)),null_positions=kn.shape[1],projected_fraction=relnorm(projected,kn),perpendicular_fraction=relnorm(perp,kn),per_column_perpendicular_fraction=dict(min=float(ratios.min()),median=float(ratios.median()),max=float(ratios.max())),fixed_null_action_norm=float(fixed.norm()),perpendicular_input_norm=float(perp.norm()),separated_consistency=separated,joint_consistency=joint,numerical_status=numerical,exact_witness=witness,scope='Strict layer-action subproblem only; scalar null NLL feasibility untested'),private

def run_checks():
 from .teacherL3_ops import run_checks as inherited
 inherited();d=torch.tensor([[1.,2.,3.]],dtype=torch.float16);zero=torch.zeros_like(d);eye=torch.eye(3,dtype=torch.float16)
 feasible,_=diagnose(eye[:,:1],eye[:,1:2],d,zero);assert feasible['numerical_status']=='CONSISTENT_STRONG_SUBPROBLEM'
 conflict,priv=diagnose(eye[:,:1],eye[:,:1],d,zero);assert conflict['numerical_status']=='INCONSISTENT_STRONG_SUBPROBLEM' and conflict['exact_witness']['exact_nonzero'] and priv['exact_numerator']!=str(0)
 num,value=exact_dot(torch.tensor([.5,-.25],dtype=torch.float16),torch.zeros(2,dtype=torch.float16),torch.tensor([2.,4.],dtype=torch.float16));assert num==0 and value==0
 ks=torch.eye(3,dtype=torch.float64)[:,:1];kn=ks.clone();kn[1,0]=1e-12;qq,_=basis(ks);sep=rowspace_residual(kn-qq@(qq.T@kn),torch.ones(1,1,dtype=torch.float64));joint=rowspace_residual(torch.cat([ks,kn],1),torch.tensor([[1.,0.]],dtype=torch.float64));assert sep['consistent'] and not joint['consistent']
 try:rowspace_residual(torch.full((2,2),float('nan'),dtype=torch.float64),torch.ones(1,2,dtype=torch.float64))
 except ValueError:pass
 else:raise AssertionError('Nonfinite accepted')
 return dict(status='PASS',checks=['inherited_fixed_fit_endpoint_exception_restore','separable_strong_constraints','shared_key_conflict_both_residuals','exact_half_integer_nonzero_and_zero','near_overlap_classification_unresolved','nonfinite_rejected'])
