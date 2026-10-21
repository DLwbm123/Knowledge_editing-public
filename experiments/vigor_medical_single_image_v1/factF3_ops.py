"""Source-span token binding and fixed fact-balanced teacher-forced loss."""
import torch
import torch.nn.functional as F
from .t0probe import token_stat

def bind_groups(runtime,batch,annotation):
 labels=batch.labels[0,1:];positions=torch.where(labels!=-100)[0];ids=labels[positions].tolist();eos=runtime.adapter.tokenizer.eos_token_id
 content=[(i,t) for i,t in enumerate(ids) if t!=eos];text=annotation['source_target'];expected=text+' ';tokens=[];prefix=[];old=0
 for ordinal,tid in content:
  prefix.append(tid);decoded=runtime.adapter.tokenizer.decode(prefix,skip_special_tokens=False,clean_up_tokenization_spaces=False).lstrip()
  if not expected.startswith(decoded) or len(decoded)<old:raise ValueError('Source prefix/token alignment')
  tokens.append((ordinal,old,len(decoded)));old=len(decoded)
 if decoded!=expected:raise ValueError('Teacher labels differ from source plus exact Mistral template space')
 groups=[]
 for span in annotation['groups']:
  g=[i for i,start,end in tokens if max(start,span['start'])<min(end,span['end'])]
  if not g:raise ValueError('Empty source fact group')
  groups.append(g)
 if sum(map(len,groups))!=len(set(i for g in groups for i in g)):raise ValueError('Token straddles distinct facts')
 return groups,dict(group_count=len(groups),tokens_per_group=list(map(len,groups)),valid_tokens=len(ids),EOS_tokens=sum(t==eos for t in ids),content_tokens=sum(start<len(text) and end>start for _,start,end in tokens),template_space_tokens=sum(start>=len(text) for _,start,end in tokens),exact_template_suffix_binding=True,source_prefix_binding=True)

def objectives(logits,labels,groups):
 ce=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1),ignore_index=-100,reduction='none').reshape(labels.shape[0],-1)
 valid=labels[:,1:]!=-100;values=ce[valid]
 if not groups or not values.numel():raise ValueError('Missing facts or target labels')
 full=values.mean();facts=[]
 for group in groups:
  if not group or min(group)<0 or max(group)>=len(values):raise ValueError('Invalid fact token indices')
  facts.append(values[group].mean())
 fact=torch.stack(facts).mean();return full,fact,.5*full+.5*fact

def read(runtime,batch,groups,position=None,competitor=None):
 with torch.no_grad():
  out=runtime.model(**batch.forward_kwargs());labels=batch.labels[:,1:];pos=torch.where(labels[0]!=-100)[0];targets=labels[0,pos];vectors=out.logits[0,pos].float();eos=runtime.adapter.tokenizer.eos_token_id
  if position is None:
   eligible=[i for i in range(len(targets)) if int(targets[i])!=eos and int(vectors[i].argmax())!=int(targets[i])]
   if not eligible:raise ValueError('No initial unfavored diagnostic token')
   position=eligible[0]
  target=int(targets[position]);v=vectors[position];other=v.clone();other[target]=-torch.inf;j=int(other.argmax());full,fact,weighted=objectives(out.logits,batch.labels,groups)
  data=dict(target_position=position+1,mean_nll=float(full),fact_nll=float(fact),weighted_nll=float(weighted),margin=float(v[target]-v[j]),**token_stat(v,target))
  if competitor is not None:data.update(fixed_competitor_margin=float(v[target]-v[competitor]),strongest_competitor_changed=j!=competitor)
  return data,(position,target,j,int(pos[position]))

def run_checks():
 x=torch.tensor([[[2.,-1.],[0.,1.],[-1.,3.],[1.,0.]]],requires_grad=True);y=torch.tensor([[-100,0,1,1]])
 f,g,z=objectives(x,y,[[0],[1]]);ce=F.cross_entropy(x[:,:-1].reshape(-1,2),y[:,1:].reshape(-1),reduction='none')
 assert torch.allclose(z,.5*ce.mean()+.25*ce[0]+.25*ce[1]);grad=torch.autograd.grad(z,x)[0];assert grad[0,2].abs().sum()>0
 try:objectives(x,y,[[]]);raise AssertionError('Empty accepted')
 except ValueError:pass
 return dict(status='PASS',fixed_half_weights=True,group_mean_not_length_weighted=True,EOS_full_supervision=True,empty_group_rejected=True)
