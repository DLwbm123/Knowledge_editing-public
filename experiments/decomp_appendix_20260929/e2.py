"""Same-W0 structure release checks and in-memory free-rank4 deployment."""
import torch
from structures import convert
from resources import ROOT,write

def structure(method):
 base=method.split('_')[0]
 return 'CP' if base in ['M0','M1','M2','M3'] else 'TK' if base in ['M4','M5'] else 'LR'

def deploy(expert,method,seed):
 return convert(expert,seed) if method.endswith('_STRUCT') else expert

@torch.no_grad()
def parity(runtime,t,expert,seed,stage):
 from methods.medtrace import MedTraceLayerHook
 from scripts.medtrace.run_dev16 import generate
 from diagnostics import generated_ids,LAYER
 import worker_v3 as old
 free=convert(expert,seed);rec=old.record(t);batch=runtime.build_edit_batch(rec)
 h=torch.randn(8,14336,device=next(expert.parameters()).device,generator=torch.Generator(device=next(expert.parameters()).device).manual_seed(seed))
 x=expert.residual(h);y=free.residual(h)
 assert torch.isfinite(x).all() and torch.isfinite(y).all()
 err=float((x-y).abs().max());scale=float(x.abs().max())
 assert err<=1e-4+1e-4*scale,'Structure expansion residual mismatch'
 logits=[];outputs=[]
 for ex in [expert,free]:
  hook=MedTraceLayerHook(runtime.get_module(LAYER),ex);hook.attach();hook.set_teacher_routing(batch.labels)
  try:
   out=runtime.model(**batch.forward_kwargs()).logits
   mask=batch.labels[:,1:]!=-100;logits.append(out[:,:-1][mask].float())
   outputs.append(generate(runtime,dict(query_id=rec.record_id,question=rec.question,image_path=str(rec.image_path)),hook,1024))
  finally:hook.detach()
 assert all(torch.isfinite(x).all() for x in logits)
 write(ROOT/'private/e2_parity'/f'{stage}-{t["order"]}.json',dict(stage=stage,order=t['order'],residual_max_abs=err,residual_relative=float((x-y).norm()/x.norm().clamp_min(1e-12)),predictor_logits_max_abs=float((logits[0]-logits[1]).abs().max()),generation_equal=generated_ids(outputs[0])==generated_ids(outputs[1]),outputs=outputs,rule='FP32 normalized residual; actual FP16 model predictor logits; native free generation. Numerical generation sensitivity is reported, never filtered.'))
 return free

def jobs():
 rows=[]
 for order in range(25,49):
  for kind,method in [('CP','M1'),('TK','M4')]:
   rows.append(dict(id=f'E2-{kind}-{order}',block='E2_REG24',mode='train',kind=kind,methods=[method,method+'_STRUCT'],seed=20260927,order=order,e2_canary=order==25,requires=[] if order==25 else ['E2_CANARY_PASS.json'],status='PENDING'))
 for method in ['M1','M1_STRUCT','M4','M4_STRUCT']:
  rows.append(dict(id=f'E2-REG24-{method}-bank',block='E2_REG24',mode='sequential',seed=20260927,method=method,orders=list(range(25,49)),prefixes=[12,24],requires=['E2_CANARY_PASS.json']+[f'adapters/s20260927/{method}/e{o:03d}.pt' for o in range(25,49)],status='PENDING'))
 return rows
