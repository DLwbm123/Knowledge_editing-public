"""Compare source-only CE and frozen D gradients without updates or evaluation."""
import json,math,os,signal,sys,time,traceback
from pathlib import Path
ROOT=Path(os.environ['DIAG_ROOT']);PARENT=Path(os.environ['PARENT_RUN']);FIXED=Path(os.environ['FIXED_DEPTH_ROOT']);NINE=Path(os.environ['NINE_ROOT']);LOCK=Path(os.environ['SPAN_LOCK']);DEADLINE=float(os.environ['EXPERIMENT_DEADLINE'])
COUNTS={'forwards':0,'backwards':0,'pairs':0,'updates':0,'generations':0}
def write(name,data):
 p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,allow_nan=False));t.replace(p)
def budget():
 if time.time()>DEADLINE-60 or (ROOT/'STOP.json').exists():raise TimeoutError('guard stop or deadline')
 h=json.loads((ROOT/'GUARD_HEALTH.json').read_text())
 if time.time()-h['time']>45 or h['hard_end']!=DEADLINE:raise RuntimeError('stale guard')
 if COUNTS['forwards']>1200 or COUNTS['backwards']>576 or COUNTS['pairs']>72:raise RuntimeError('count ceiling')
def main():
 with (ROOT/'WORKER_ONCE').open('x'):pass
 signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('guard stopped')))
 sys.path[:0]=[str(ROOT/'tools'),str(PARENT/'private/tools'),str(PARENT/'private/source'),str(PARENT/'private/official_llava')]
 import torch,module_run as r,runtime_binding,d_gradient,prior_binding,stream_extract
 from prefix_cache import PrefixCache
 from span_train import terms
 from train_pair import state_digest
 from types import SimpleNamespace
 from scripts.medtrace.run_realmodel_core import load_real_runtime
 from m3bench_repro.editors.llava_runtime import seed_everything
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);r.COUNTS=COUNTS;r.c.budget=budget
 admission=json.loads((FIXED/'ROLE_ADMISSION.json').read_text());source=json.loads((FIXED/'TRAIN_SOURCE_BOUND_PRIVATE.json').read_text());tasks=json.loads((FIXED/'TASK_IDENTITIES_PRIVATE.json').read_text());lk=json.loads(LOCK.read_text())
 spans={x['order']:x for x in lk['records'] if x['status']=='WEAK_TOKEN_MATCH'};owners=admission['owners']
 if admission['status']!='SOURCE_AND_EVALUATION_ROLE_ADMITTED' or lk['status']!='WEAK_TOKEN_PROXY_ONLY' or len(spans)!=9 or len(owners)!=16 or [x['task']['order'] for x in source]!=owners or [x['order'] for x in tasks]!=owners:raise ValueError('training role or span binding')
 frozen=json.loads((NINE/'TRAIN_FROZEN.json').read_text()); exits=[json.loads((NINE/x).read_text())['code'] for x in ('WORKER_EXIT.json','GUARD_EXIT.json')]
 if frozen['status']!='FROZEN' or len(frozen['results'])!=27 or exits!=[0,0]:raise ValueError('D training receipt missing')
 results={x['owner']:x for x in frozen['results'] if x['arm']=='D'}
 if set(results)!=set(spans) or any(x['steps']!=160 for x in results.values()):raise ValueError('D owner receipt mismatch')
 with (PARENT/'private/EVAL_BINDINGS.json').open('rb') as f:config=runtime_binding.extract(f)['configuration_only']
 with (FIXED/'PRIOR_INPUT_BINDINGS_PRIVATE.json').open('rb') as f:
  if stream_extract.nonspace(f)!=b'[':raise ValueError('baseline configuration array')
  old=prior_binding.select(f,stream_extract.nonspace(f),{'input':{'runtime':{'inherited_runtime':True},'generation':True}})
 if old['input']['runtime']['inherited_runtime']!=config['runtime'] or old['input']['generation']!=config['generation']:raise ValueError('historical runtime configuration mismatch')
 runtime=load_real_runtime(SimpleNamespace(cpu_gate=PARENT/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime)
 if runtime.generation_config!=config['generation'] or any(p.requires_grad for p in runtime.model.parameters()):raise ValueError('runtime binding')
 versions={n:p._version for n,p in runtime.model.named_parameters()}
 def count(_m,_a,_k):
  budget();COUNTS['forwards']+=1
 h=runtime.model.register_forward_pre_hook(count,with_kwargs=True)
 out=[]
 try:
  for ent in source:
   task=ent['task'];owner=task['order']
   if owner not in spans:continue
   rows=ent['source_rows'];lock=spans[owner]
   if [x['audit_role'] for x in rows]!=['NATIVE','FIT','FIT','FIT','FIT']:raise ValueError('training roles')
   seed_everything(task['seed']);initial=r.p.tr.TT4(task['seed'],8,8).state_dict()
   saved=torch.load(NINE/'weights'/f'{owner}_D.pt',map_location='cpu',weights_only=True)
   if saved['arm']!='D' or saved['steps']!=160 or state_digest(saved['state'])!=results[owner]['final_digest'] or state_digest(initial)!=saved['initial_digest']:raise ValueError('expert digest binding')
   runtime.model.half()
   try:prepared=r.s.prepare(runtime,{'configuration_only':config},task,rows,True)
   finally:r.old.fp32(runtime)
   batches=[r.old.batch_on(runtime,x[1]) for x in prepared]
   for stage,state in (('initial',initial),('final',saved['state'])):
    expert=r.p.expert(state,task['seed'],runtime.device);hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module('model.layers.21.mlp.down_proj'),expert);hook.attach()
    try:
     before=state_digest(expert.state_dict())
     with PrefixCache(runtime,batches):
      for fit in range(1,5):
       budget();cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
       def restore():torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng)
       def ordinary():
        expert.zero_grad(set_to_none=True)
        for i in (0,fit):
         v=lock['views'][i];ce,_=terms(runtime,hook,batches[i],rows[i],lock['answer_tokens'],v['medical'],v);(ce*0.5).backward();COUNTS['backwards']+=1
        return torch.cat([p.grad.detach().double().reshape(-1).cpu() for p in expert.parameters()])
       def robust():
        d_gradient.source_gradients(r,runtime,hook,expert,batches,rows,lock,fit)
        return torch.cat([p.grad.detach().double().reshape(-1).cpu() for p in expert.parameters()])
       restore();g=ordinary();restore();g2=ordinary();restore();d=robust();restore();d2=robust();restore()
       if not torch.allclose(g,g2,rtol=1e-5,atol=1e-7):raise ValueError('CE gradient repeat mismatch')
       if not torch.allclose(d,d2,rtol=1e-5,atol=1e-7):raise ValueError('gradient repeat mismatch')
       gr=d-g;ng=float(torch.linalg.vector_norm(g));nr=float(torch.linalg.vector_norm(gr));dot=float(torch.dot(g,gr));valid=ng>0 and nr>0
       if not all(math.isfinite(x) for x in (ng,nr,dot)):raise ValueError('nonfinite gradient')
       cos=dot/(ng*nr) if valid else None;correction=max(0.,-dot/ng)/nr if valid else None
       out.append({'owner':owner,'stage':stage,'valid':valid,'negative':bool(dot<0) if valid else None,'cosine':cos,'correction_ratio':correction});COUNTS['pairs']+=1
       if state_digest(expert.state_dict())!=before:raise ValueError('expert changed')
       write('PROGRESS.json',{'pairs':COUNTS['pairs'],'expected':72,'counts':COUNTS})
    finally:hook.detach()
  if len(out)!=72 or COUNTS['backwards']!=576 or any(p._version!=versions[n] for n,p in runtime.model.named_parameters()):raise ValueError('coverage or frozen backbone mismatch')
  # Aggregate by editor before publishing; individual values stay private.
  summary={}
  for stage in ('initial','final'):
   group=[x for x in out if x['stage']==stage]; valid=[x for x in group if x['valid']]
   each=[]
   for owner in owners:
    arr=[x for x in valid if x['owner']==owner]
    if arr:each.append({'negative_rate':sum(x['negative'] for x in arr)/len(arr),'mean_cosine':sum(x['cosine'] for x in arr)/len(arr),'mean_correction':sum(x['correction_ratio'] for x in arr)/len(arr)})
   summary[stage]={'pairs':len(group),'valid':len(valid),'negative':sum(x['negative'] for x in valid),'editor_weighted_negative_rate':sum(x['negative_rate'] for x in each)/len(each) if each else None,'editor_weighted_cosine':sum(x['mean_cosine'] for x in each)/len(each) if each else None,'editor_weighted_correction_ratio':sum(x['mean_correction'] for x in each)/len(each) if each else None,'min_cosine':min((x['cosine'] for x in valid),default=None),'max_cosine':max((x['cosine'] for x in valid),default=None),'sorted_cosine':[x['cosine'] for x in sorted(valid,key=lambda v:v['cosine'])]}
  write('RESULTS.json',{'status':'COMPLETE','counts':COUNTS,'summary':summary,'scope':'training sources only; no curvature; no evaluation'})
 finally:h.remove()
if __name__=='__main__':
 code=0
 try:main()
 except BaseException:
  code=1;write('WORKER_FAILURE.json',{'error':traceback.format_exc(),'counts':COUNTS});raise
 finally:write('WORKER_EXIT.json',{'code':code,'counts':COUNTS})
