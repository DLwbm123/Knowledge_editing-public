"""Matched CE versus Base-relative increment constraints on fixed source-local medical packs."""
import os,json,time,signal,sys,traceback,shutil
from pathlib import Path
ROOT=Path(os.environ['V52_ROOT']);PART=int(os.environ['PARTITION']);OUT=ROOT/str(PART)
P=Path(os.environ['PARENT_RUN']);START=time.time()
counts=dict(forwards=0,backwards=0,updates=0,generations=0,admission_backwards=0,vision_forwards=0,cached_base_outputs=0)
limits=json.loads((ROOT/'RESOURCE_LIMITS.json').read_text());last_space=0

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def budget():
 global last_space
 assert time.time()<limits['hard_end']-300 and not (ROOT/'STOP').exists(),'Original deadline or STOP'
 assert counts['forwards']<80000 and counts['backwards']<10000 and counts['generations']<150
 assert time.time()-read(ROOT/'RESOURCE_GUARD_HEALTH.json')['time']<120
 if time.time()-last_space>30:
  assert shutil.disk_usage(ROOT).free>8*1024**3
  assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())<1500000000
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
 def emit(item,folder,hook=None,zero_smoke=False):
  dest=folder/(item[0]['query_id']+'.json');assert not dest.exists();budget()
  recovered=read(ROOT/'RECOVERY_BASE_OUTPUTS.json').get(str(dest.relative_to(ROOT))) if (ROOT/'RECOVERY_BASE_OUTPUTS.json').exists() and folder.name=='BASE' else None
  if recovered is not None:
   assert recovered['binding']==item[4], 'Recovery exact input binding'
   counts['cached_base_outputs']+=1;write(dest,dict(recovered,recovery_cached=True));return recovered
  counts['generations']+=1;t=time.monotonic()
  if zero_smoke:
   assert hook is not None and hook.expert.G1.count_nonzero()==0
   # Inherited generator requires nonzero residual for ordinary edited evaluation.
   # Zero-only admission owns its hook context while using the Base decode assertion path.
   with hook.generation_request():answer,trace=r.s.fit.short.generate(runtime,None,item[2],item[3])
   assert hook.last_generation_trace and all(x['active_residual_norm']==0 for x in hook.last_generation_trace)
  else:answer,trace=r.s.fit.short.generate(runtime,hook,item[2],item[3])
  tokens=list(answer.raw_token_ids)
  rec=dict(binding=item[4],raw_answer=answer.decoded_text,raw_token_ids=tokens,EOS=tokens[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(tokens)>=1024,seconds=time.monotonic()-t)
  write(dest,rec);return rec
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
  write(OUT/'MODEL_READY.json',dict(time=time.time(),generation=runtime.generation_config,objective_selfcheck=selfcheck()))
  for task in tasks:
   folder=OUT/task['key'];folder.mkdir(exist_ok=True)
   rows=task['train_rows'];sources=prepare(rows+[dict(brief_row(q),query_id=q['query_id']+'_CTX') for q in rows],True)
   batches=[r.old.batch_on(runtime,x[1]) for x in sources]
   target_rows=[dict(rows[2],query_id=rows[2]['query_id']+'_TARGET',reference=task['target_answer'])]
   target_rows += [dict(brief_row(target_rows[0]),query_id=target_rows[0]['query_id']+'_CTX')]
   negative_batches=[r.old.batch_on(runtime,x[1]) for x in prepare(target_rows,True)]
   all_batches=batches+negative_batches
   with torch.no_grad():base_scores=[(values(None,None,batches[1+4*c],True,False)[2],values(None,None,negative_batches[c],True,False)[2]) for c in range(2)]
   write(folder/'BASE_SCORES.json',base_scores)
   evaluation=prepare(task['eval_rows'])+prepare([brief_row(q) for q in task['eval_rows']]);base=[]
   for j,item in enumerate(evaluation):
    context='ORIGINAL' if j<len(task['eval_rows']) else 'SHORT';base.append(emit(item,folder/'outputs'/context/'BASE'))
   initial=snapshot(make());ex=make();h=Hook(runtime.get_module('model.layers.21.mlp.down_proj'),ex,True);h.attach()
   try:
    rec=emit(evaluation[0],folder/'smoke'/'21',h,zero_smoke=True)
    assert rec['raw_token_ids']==base[0]['raw_token_ids'] and rec['raw_answer']==base[0]['raw_answer']
    uncached=[values(ex,h,b,True) for b in all_batches];counts['admission_backwards']+=2*len(all_batches)
    with PrefixCache(runtime,all_batches,21):
     for repetition in range(2):
      for batch,old in zip(all_batches,uncached):
       new=values(ex,h,batch,True);counts['admission_backwards']+=2
       assert new[0]==old[0] and new[2]==old[2] and torch.equal(new[1],old[1]) and torch.equal(new[3],old[3]), 'Real CE/score cache parity'
    for c in range(2):assert (uncached[1+4*c][2],uncached[8+c][2])==tuple(base_scores[c]),'Zero writer Base target-score parity'
   finally:h.detach()
   write(folder/'REAL_SMOKE.json',dict(status='PASS',zero_writer_Base_full_generation=True,zero_score_exact=True,all_batches_CE_score_gradients_exact=True))
   write(OUT/'STAGE0_READY.json',dict(status='PASS',case=task['key'],time=time.time()))
   while not (ROOT/'STAGE1_GO.json').exists():budget();time.sleep(1)
   for arm in ['MATCHED_CE','INCREMENT']:
    ex=make();assert all(torch.equal(v,initial[k]) for k,v in ex.state_dict().items())
    h=Hook(runtime.get_module('model.layers.21.mlp.down_proj'),ex,True);h.attach()
    scalar=dict(step=0,second_moments=[0.,0.]);curve=[];before_counts=dict(counts);started=time.monotonic()
    try:
     with PrefixCache(runtime,all_batches,21):
      for step in range(1,161):
       budget();before=snapshot(ex);losses=[];grads=[];stats=[]
       for c in range(2):
        results=[values(ex,h,batches[4*c+i],i==1) for i in range(4)]
        neg=values(ex,h,negative_batches[c],True,True,False)
        ce=sum(x[0] for x in results)/4;ce_grad=sum(x[1] for x in results)/4
        loss,g,info=combine(ce,ce_grad,results[1][2],results[1][3],neg[2],neg[3],base_scores[c],arm=='INCREMENT')
        losses.append(loss);grads.append(g);stats.append(dict(ce=ce,**info))
       gradient=sum(grads)/2;assert torch.isfinite(gradient).all();keys=list(before)
       raw,scalar,direction=proposal(r,before,gradient,scalar,keys)
       state=construct(r,before,raw,direction,direction,1.,keys);ex.load_state_dict(state)
       counts['updates']+=1;curve.append(dict(step=step,losses=losses,contexts=stats))
       if step%20==0:write(OUT/'PROGRESS.json',dict(case=task['key'],arm=arm,step=step,counts=counts,time=time.time()))
     target_path=folder/arm/'delta.pt';target_path.parent.mkdir(exist_ok=True);torch.save(snapshot(ex),target_path)
     loaded=torch.load(target_path,weights_only=True,map_location=runtime.device);assert all(torch.equal(v,loaded[k]) for k,v in ex.state_dict().items())
     ex.load_state_dict(loaded)
     for j,item in enumerate(evaluation):
      context='ORIGINAL' if j<len(task['eval_rows']) else 'SHORT';emit(item,folder/'outputs'/context/arm,h)
     write(folder/arm/'COMPLETE.json',dict(status='PASS',curve=curve,seconds=time.monotonic()-started,counts={k:counts[k]-before_counts[k] for k in counts},paired_initial_exact=True,save_load_exact=True,writer_parameters=7168))
    finally:h.detach()
   write(folder/'COMPLETE.json',dict(status='PASS',time=time.time(),remaining_consumers='Frozen scoring and successor decision; keep deltas until then'))
  write(OUT/'MODEL_COMPLETE.json',dict(status='PASS',counts=counts,time=time.time()))
 finally:fh.remove();vh.remove();ch.remove()

if __name__=='__main__':
 code=0
 try:main()
 except BaseException:
  code=1;write(OUT/'FAILURE.json',dict(error=traceback.format_exc(),counts=counts,time=time.time()));write(ROOT/'STOP',dict(reason='V52_FAILURE',time=time.time()));raise
 finally:write(OUT/'EXIT.json',dict(code=code,counts=counts,seconds=time.time()-START))
