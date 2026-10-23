"""Fixed inference-only interventions in projected visual-token embeddings."""
import torch

def transform(item,donor,condition):
 assert condition in ('ORIGINAL','ZERO_VISUAL_EMBEDDINGS','FIXED_SOURCE_IMAGE_PERMUTATION')
 from token_hook import layout
 row,batch,raw,expanded,binding=item
 assert batch is not None
 lay=layout(raw,expanded);a=lay['visual_start'];n=lay['visual_count'];p=lay['prefix_length'];assert n>0
 kw,labels,tokens=batch
 assert torch.equal(kw['inputs_embeds'][:,:p],expanded[4])
 if condition=='ORIGINAL':return item
 replacement=torch.zeros_like(expanded[4][:,a:a+n])
 if condition=='FIXED_SOURCE_IMAGE_PERMUTATION':
  dl=layout(donor[2],donor[3]);assert dl['visual_count']==n
  replacement=donor[3][4][:,dl['visual_start']:dl['visual_start']+n].clone()
 assert replacement.shape==(1,n,expanded[4].shape[-1]) and replacement.dtype==expanded[4].dtype
 e=list(expanded);e[4]=expanded[4].clone();e[4][:,a:a+n]=replacement
 k=dict(kw);k['inputs_embeds']=kw['inputs_embeds'].clone();k['inputs_embeds'][:,a:a+n]=replacement
 assert torch.equal(k['inputs_embeds'][:,:p],e[4])
 assert torch.equal(k['inputs_embeds'][:,:a],kw['inputs_embeds'][:,:a]) and torch.equal(k['inputs_embeds'][:,a+n:],kw['inputs_embeds'][:,a+n:])
 return row,(k,labels,tokens),raw,tuple(e),dict(binding,diagnostic_visual_condition=condition,diagnostic_donor_image_sha256=donor[4]['image_sha256'] if condition=='FIXED_SOURCE_IMAGE_PERMUTATION' else None,reference_role='ORIGINAL_TARGET_PROBE_NOT_ALTERED_IMAGE_GROUND_TRUTH')

def selfcheck():
 e=torch.arange(24,dtype=torch.float16).reshape(1,6,4);kw=dict(inputs_embeds=torch.cat([e,torch.ones(1,2,4,dtype=e.dtype)],1),use_cache=False)
 raw=dict(input_ids=torch.tensor([[1,-200,2,3]]));item=({},(kw,torch.tensor([[-100]*6+[1,2]]),(1,2)),raw,(None,None,None,None,e),dict(image_sha256='fixture'))
 donor=({},None,raw,(None,None,None,None,e+7),dict(image_sha256='donor'))
 z=transform(item,donor,'ZERO_VISUAL_EMBEDDINGS');q=transform(item,donor,'FIXED_SOURCE_IMAGE_PERMUTATION')
 assert z[3][4][:,1:4].count_nonzero()==0 and torch.equal(q[3][4][:,1:4],(e+7)[:,1:4])
 assert torch.equal(item[3][4],e) and torch.equal(kw['inputs_embeds'][:,:6],e)
 assert transform(item,donor,'ORIGINAL') is item
 return dict(status='PASS',visual_only=True,prefix_teacher_equal=True,original_unchanged=True,original_identity=True,real_model_calls=0)
