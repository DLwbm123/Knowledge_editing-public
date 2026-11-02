"""Single fixed projected-D arm on the original source/FIT contract."""
from pathlib import Path
import math
import torch
from prefix_cache import PrefixCache
from span_train import terms
from train_pair import save_new,state_digest
import d_gradient

def project(ce,total):
 if ce.shape!=total.shape or not torch.isfinite(ce).all() or not torch.isfinite(total).all():raise ValueError('gradient mismatch or nonfinite')
 extra=total-ce;den=torch.dot(ce.double(),ce.double());dot=torch.dot(extra.double(),ce.double())
 correction=(torch.minimum(dot,dot.new_zeros(()))/den).to(ce) if den.item()>0 else ce.new_zeros(())
 out=ce+extra-correction*ce
 if not torch.isfinite(out).all():raise ValueError('nonfinite projected gradient')
 return out,float(dot),float(correction)

def train(runtime,r,task,rows,config,admission,lock,output,budget):
 owner=task['order']
 if admission.get('status')!='SOURCE_AND_EVALUATION_ROLE_ADMITTED' or owner not in admission['owners'] or lock['order']!=owner or lock['status']!='WEAK_TOKEN_MATCH':raise ValueError('role or span mismatch')
 if [v['audit_role'] for v in rows]!=['NATIVE','FIT','FIT','FIT','FIT'] or any(p.requires_grad for p in runtime.model.parameters()):raise ValueError('training inputs or backbone')
 from m3bench_repro.editors.llava_runtime import seed_everything
 seed_everything(task['seed']);initial=r.p.tr.TT4(task['seed'],8,8).state_dict()
 if sum(v.numel() for v in initial.values())!=7168 or torch.count_nonzero(initial['G1']):raise ValueError('initial state')
 expert=r.p.expert(initial,task['seed'],runtime.device);hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module('model.layers.21.mlp.down_proj'),expert)
 runtime.model.half()
 try:prepared=r.s.prepare(runtime,{'configuration_only':config},task,rows,True)
 finally:r.old.fp32(runtime)
 batches=[r.old.batch_on(runtime,x[1]) for x in prepared];versions={n:p._version for n,p in runtime.model.named_parameters()};hook.attach()
 try:
  optimizer=r.p.optimizer(expert,runtime.model);curve=[];params=list(expert.parameters())
  with PrefixCache(runtime,batches):
   for step in range(1,161):
    budget();expert.zero_grad(set_to_none=True)
    for i in (0,1+(step-1)%4):
     view=lock['views'][i];ce,_=terms(runtime,hook,batches[i],rows[i],lock['answer_tokens'],view['medical'],view)
     (ce*.5).backward();r.COUNTS['backwards']+=1
    sizes=[p.numel() for p in params];gce=torch.cat([p.grad.detach().reshape(-1) for p in params]);d_gradient.source_gradients(r,runtime,hook,expert,batches,rows,lock,step)
    gd=torch.cat([p.grad.detach().reshape(-1) for p in params]);g,dot,coefficient=project(gce,gd);offset=0
    for p,n in zip(params,sizes):p.grad.copy_(g[offset:offset+n].view_as(p));offset+=n
    norm=torch.nn.utils.clip_grad_norm_(params,1.)
    if not torch.isfinite(norm):raise ValueError('nonfinite clip')
    optimizer.step();r.COUNTS['updates']+=1
    curve.append({'step':step,'projected':dot<0,'projection_coefficient':coefficient,'gradient_norm':float(norm)})
  final=r.old.snapshot(expert);path=Path(output)/f'{owner}_D.pt';save_new(path,{'state':final,'module':'model.layers.21.mlp.down_proj','steps':160,'curve':curve})
  loaded=torch.load(path,map_location='cpu',weights_only=True)
  if state_digest(loaded['state'])!=state_digest(final):raise ValueError('saved state mismatch')
  active,tokens=r.q.logits(runtime,hook,batches[0]);expert.load_state_dict(loaded['state']);again,tokens2=r.q.logits(runtime,hook,batches[0])
  if not torch.equal(active,again) or not torch.equal(tokens,tokens2):raise ValueError('reload output mismatch')
  if any(p._version!=versions[n] for n,p in runtime.model.named_parameters()):raise ValueError('backbone changed')
  return {'owner':owner,'steps':160,'projected_steps':sum(x['projected'] for x in curve)}
 finally:hook.detach()
