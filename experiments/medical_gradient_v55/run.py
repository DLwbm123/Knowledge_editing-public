"""Matched CE versus Base-relative increment constraints on fixed source-local medical packs."""
import os,json,time,signal,sys,traceback,shutil
from pathlib import Path
ROOT=Path(os.environ['V55_ROOT']);PART=int(os.environ['PARTITION']);OUT=ROOT/str(PART)
P=Path(os.environ['PARENT_RUN']);START=time.time()
counts=dict(forwards=0,backwards=0,updates=0,generations=0,admission_backwards=0,vision_forwards=0,cached_base_outputs=0)
limits=json.loads((ROOT/'RESOURCE_LIMITS.json').read_text());last_space=0

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def budget():
 global last_space
 assert time.time()<limits['hard_end']-300 and not (ROOT/'STOP').exists(),'Original deadline or STOP'
 assert counts['forwards']<40 and counts['backwards']<40 and counts['generations']==0
 assert time.time()-read(ROOT/'RESOURCE_GUARD_HEALTH.json')['time']<120
 if time.time()-last_space>30:
  assert shutil.disk_usage(ROOT).free>8*1024**3
  assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())<80000000
  last_space=time.time()

def main():
 with (OUT/'ONCE').open('x'):pass
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Original window deadline')))
 signal.alarm(max(1,int(limits['hard_end']-time.time()-300)))
 sys.path[:0]=[str(ROOT/'tools'),str(P/'private/tools'),str(P/'private/source'),str(P/'private/official_llava')]
 import torch
 import module_run as r
 from scripts.medtrace.run_realmodel_core import load_real_runtime
 from types import SimpleNamespace
 from alignment import TT,Hook,snapshot
 from path_v18 import proposal,construct
 from v51_cache import PrefixCache
 from protocol import brief_row
 from objective import target_score,combine,selfcheck

 r.c.budget=budget
 torch.use_deterministic_algorithms(True);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.matmul.allow_tf32=False
 runtime=load_real_runtime(SimpleNamespace(cpu_gate=P/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime)
 assert not any(v.requires_grad for v in runtime.model.parameters())
 config=read(ROOT/'DATA_ADMISSION.json');all_tasks=config['tasks'];tasks=all_tasks[PART::2]
 lock=read(ROOT/'PROTOCOL_LOCK.json');seed=lock['seed'];assert seed==225034559
 template=read(P/'private/M13_SOURCE_PACKET.json')['original_tasks'][4]
 bindings=read(P/'private/EVAL_BINDINGS.json');assert runtime.generation_config==next(iter(bindings.values()))['generation']
 def count(*_):budget();counts['forwards']+=1
 fh=runtime.model.register_forward_pre_hook(count)
 vh=runtime.model.get_vision_tower().register_forward_pre_hook(lambda *_:counts.__setitem__('vision_forwards',counts['vision_forwards']+1))
 def prepare(rows,training=False):
  runtime.model.half()
  try:return r.s.prepare(runtime,bindings,template,rows,training)
  finally:r.old.fp32(runtime)
 def make():
  ex=TT(seed).to(runtime.device);ex.load_state_dict(r.mm.canonicalize(snapshot(ex))[0]);return ex
 captured={}
 def capture(_module,_args,output):captured['logits']=output.logits
 ch=runtime.model.register_forward_hook(capture)
 def values(ex,h,batch,score=False,grad=True,ce_needed=True):
  if h is not None:h.teacher(batch.labels,batch.forward_kwargs().get('attention_mask'))
  captured.clear();ce=runtime.compute_loss(batch);assert torch.isfinite(ce)
  value=target_score(captured.pop('logits'),batch.labels,runtime.adapter.tokenizer.eos_token_id) if score else None
  def derivative(loss,retain=False):
   gs=torch.autograd.grad(loss,tuple(ex.parameters()),retain_graph=retain);counts['backwards']+=1
   return torch.cat([g.detach().cpu().double().flatten() for g in gs])
  cg=derivative(ce,score) if grad and ce_needed else None
  sg=derivative(value) if grad and score else None
  return float(ce.detach()),cg,float(value.detach()) if score else None,sg
 try:
  write(OUT/'MODEL_READY.json',dict(time=time.time(),objective_selfcheck=selfcheck()))
  for task in tasks:
   folder=OUT/task['key'];folder.mkdir(exist_ok=True)
   rows=task['train_rows'];sources=prepare(rows+[dict(brief_row(q),query_id=q['query_id']+'_CTX') for q in rows],True)
   batches=[r.old.batch_on(runtime,x[1]) for x in sources]
   target_rows=[dict(rows[2],query_id=rows[2]['query_id']+'_TARGET',reference=task['target_answer'])]
   target_rows += [dict(brief_row(target_rows[0]),query_id=target_rows[0]['query_id']+'_CTX')]
   negative_batches=[r.old.batch_on(runtime,x[1]) for x in prepare(target_rows,True)]
   competitor_rows=[dict(rows[1],query_id=rows[1]['query_id']+'_COMPETITOR',reference=rows[2]['reference'])]
   competitor_rows += [dict(brief_row(competitor_rows[0]),query_id=competitor_rows[0]['query_id']+'_CTX')]
   competitor_batches=[r.old.batch_on(runtime,x[1]) for x in prepare(competitor_rows,True)]
   all_batches=batches+negative_batches+competitor_batches
   ex=make();initial=snapshot(ex);h=Hook(runtime.get_module('model.layers.21.mlp.down_proj'),ex,True);h.attach()
   reports={};vectors={};base_scores=None
   try:
    for state_name in ['ZERO','V54_FINAL']:
     if state_name=='V54_FINAL':
      target=Path(os.environ['V54_SOURCE'])/str(PART)/task['key']/'INCREMENT_FLOOR/delta.pt'
      state=torch.load(target,map_location=runtime.device,weights_only=True)
      assert sum(x.numel() for x in state.values())==7168 and all(torch.isfinite(x).all() for x in state.values())
      ex.load_state_dict(state)
     contexts=[];vec=[]
     for c in range(2):
      results=[values(ex,h,batches[4*c+i],i in [1,2]) for i in range(4)]
      negative=values(ex,h,negative_batches[c],True,True,False)
      competitor=values(ex,h,competitor_batches[c],True,True,False)
      positive_score=results[1][2]-competitor[2];negative_score=negative[2]-results[2][2]
      gp=results[1][3]-competitor[3];gn=negative[3]-results[2][3]
      gc=sum(x[1] for x in results)/4;ce=sum(x[0] for x in results)/4
      if state_name=='ZERO':base=(positive_score,negative_score)
      else:base=base_scores[c]
      loss,go,stats=combine(ce,gc,positive_score,gp,negative_score,gn,base,True)
      contexts.append(dict(CE=ce,role_CE=[x[0] for x in results],positive_score=positive_score,negative_score=negative_score,objective=loss,**stats))
      vec.append(dict(role_CE=torch.stack([x[1] for x in results]),positive=gp,negative=gn,CE=gc,objective=go))
     if state_name=='ZERO':base_scores=[(x['positive_score'],x['negative_score']) for x in contexts]
     mean_ce=sum(x['CE'] for x in vec)/2;mean_objective=sum(x['objective'] for x in vec)/2
     def cosine(a,b):
      norm=float(a.norm()*b.norm());return float(a@b)/norm if norm>0 else None
     directions={name:-g/g.norm() for name,g in [('CE',mean_ce),('OBJECTIVE',mean_objective)] if float(g.norm())>0}
     for c,(row,v) in enumerate(zip(contexts,vec)):
      row['role_gradient_norms']=[float(x.norm()) for x in v['role_CE']]
      row['role_gradient_cosines']=[[cosine(a,b) for b in v['role_CE']] for a in v['role_CE']]
      row['positive_negative_gradient_cosine']=cosine(v['positive'],v['negative'])
      row['unit_descent_derivatives']={name:dict(roles_CE=[float(g@direction) for g in v['role_CE']],positive=float(v['positive']@direction),negative=float(v['negative']@direction),pair_violation=float((v['negative']-v['positive'])@direction),positive_floor_violation=float(-v['positive']@direction)) for name,direction in directions.items()}
     reports[state_name]=dict(contexts=contexts,mean_CE_gradient_norm=float(mean_ce.norm()),mean_objective_gradient_norm=float(mean_objective.norm()),CE_objective_cosine=cosine(mean_ce,mean_objective))
     vectors[state_name]=vec
    write(folder/'DIAGNOSTIC.json',reports);torch.save(vectors,folder/'gradient_evidence.pt')
   finally:h.detach()
  assert counts['forwards']==24 and counts['backwards']==32 and counts['updates']==counts['generations']==0
  write(OUT/'MODEL_COMPLETE.json',dict(status='PASS',counts=counts,time=time.time()))
 finally:fh.remove();vh.remove();ch.remove()

if __name__=='__main__':
 code=0
 try:main()
 except BaseException:
  code=1;write(OUT/'FAILURE.json',dict(error=traceback.format_exc(),counts=counts,time=time.time()));write(ROOT/'STOP',dict(reason='V55_FAILURE',time=time.time()));raise
 finally:write(OUT/'EXIT.json',dict(code=code,counts=counts,seconds=time.time()-START))
