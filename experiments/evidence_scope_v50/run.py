"""Finite real-model runner. Paths only via environment; launch with neutral entry."""
import os,json,time,signal,sys,traceback,shutil
from pathlib import Path
ROOT=Path(os.environ['V50_ROOT']);PART=int(os.environ['PARTITION']);OUT=ROOT/str(PART)
P=Path(os.environ['PARENT_RUN']);START=time.time()
counts=dict(forwards=0,backwards=0,updates=0,generations=0,probes=0,vision_forwards=0,router_updates=0,conditioner_updates=0)
limits=json.loads((ROOT/'RESOURCE_LIMITS.json').read_text());last_space=0

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def save(p,v):
 import torch
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');torch.save(v,tmp);tmp.replace(p)
def budget():
 global last_space
 assert time.time()<limits['hard_end']-300 and not (ROOT/'STOP').exists(),'Original clock or STOP'
 assert counts['forwards']<80000 and counts['backwards']<5000 and counts['generations']<400
 assert time.time()-read(ROOT/'RESOURCE_GUARD_HEALTH.json')['time']<120
 if time.time()-last_space>30:
  assert shutil.disk_usage(ROOT).free>8*1024**3
  assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())<400000000
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
 from alignment import TT,Hook,snapshot,next_step,TARGETS
 from cache_v18 import PrefixCache
 from protocol import source_rows,brief_row,adaptive_gradient,verify_original
 from components import RadiusRouter,LDKERouter,EvidenceProvider,Evidence,EvidenceGate,Conditioner,ConditionedExpert,pair_features,localize
 r.c.budget=budget
 torch.use_deterministic_algorithms(True);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.matmul.allow_tf32=False
 runtime=load_real_runtime(SimpleNamespace(cpu_gate=P/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime)
 assert not any(v.requires_grad for v in runtime.model.parameters())
 assert len(runtime.model.model.layers)==32
 config=read(ROOT/'DATA_ADMISSION.json');tasks=config['tasks'][PART::2]
 template=read(P/'private/M13_SOURCE_PACKET.json')['original_tasks'][4];seed=template['seed'];assert seed==225034559
 bindings=read(P/'private/EVAL_BINDINGS.json');assert runtime.generation_config==next(iter(bindings.values()))['generation']
 def count(*_):budget();counts['forwards']+=1
 fh=runtime.model.register_forward_pre_hook(count)
 vh=runtime.model.get_vision_tower().register_forward_pre_hook(lambda *_:counts.__setitem__('vision_forwards',counts['vision_forwards']+1))
 def prepare(rows,training=False):
  runtime.model.half()
  try:items=r.s.prepare(runtime,bindings,template,rows,training)
  finally:r.old.fp32(runtime)
  return items
 def prompt_kwargs(item):
  expanded=item[3]
  return dict(inputs_embeds=expanded[4].to(runtime.device,dtype=torch.float32),attention_mask=None if expanded[2] is None else expanded[2].to(runtime.device),position_ids=None if expanded[1] is None else expanded[1].to(runtime.device),use_cache=False)
 def probe(item,first_target=None):
  # Only raw prompt/expanded image embeddings; reference never enters this forward.
  counts['probes']+=1;kwargs=prompt_kwargs(item);stash={};handles=[];pre={};post={}
  handles.append(runtime.get_module(TARGETS['S']).register_forward_pre_hook(lambda m,args:stash.update(radius=args[0].detach().mean(1))))
  handles.append(runtime.model.model.layers[21].register_forward_pre_hook(lambda m,args:stash.update(router=args[0][:,-1].detach())))
  if first_target is not None:
   for i in range(16,32):
    layer=runtime.model.model.layers[i]
    handles.append(layer.post_attention_layernorm.register_forward_pre_hook(lambda m,args,i=i:pre.__setitem__(i,args[0][:,-1].detach().clone())))
    handles.append(layer.mlp.register_forward_hook(lambda m,args,out,i=i:post.__setitem__(i,pre[i]+out[:,-1].detach())))
  try:
   with torch.no_grad():out=runtime.model(**kwargs)
   stash['logits']=out.logits[:,-1].detach();del out
  finally:
   for handle in handles:handle.remove()
  ids=item[2]['input_ids'][0];positions=(ids==-200).nonzero().flatten();assert len(positions)==1
  start=int(positions[0]);emb=kwargs['inputs_embeds'];n=emb.shape[1]-ids.numel()+1;assert n==576
  visual=emb[:,start:start+n];text=torch.cat((emb[:,:start],emb[:,start+n:]),dim=1)
  stash['evidence']=EvidenceProvider()(visual,text)
  if first_target is not None:
   chosen,scores=localize(pre,post,runtime.model.model.norm,runtime.model.lm_head,first_target,16)
   stash['localizer']=dict(selected_layer=chosen,scores=scores,target_rule='first authorized source answer token',residual_before_after_ffn=True)
  return stash
 def make():
  ex=TT(seed).to(runtime.device);ex.load_state_dict(r.mm.canonicalize(snapshot(ex))[0]);return ex
 def emit(item,folder,hook=None,wrapper=None,gate=1.,gamma=None,route=None):
  row=item[0];dest=folder/(row['query_id']+'.json');assert not dest.exists();budget();counts['generations']+=1;t=time.monotonic()
  if wrapper is not None and gate:
   with wrapper.request(torch.tensor([gate],device=runtime.device),gamma):answer,trace=r.s.fit.short.generate(runtime,hook,item[2],item[3])
  else:
   if hook is not None:hook.detach()
   try:answer,trace=r.s.fit.short.generate(runtime,None,item[2],item[3])
   finally:
    if hook is not None:hook.attach()
  tokens=list(answer.raw_token_ids)
  record=dict(binding=item[4],raw_answer=answer.decoded_text,raw_token_ids=tokens,EOS=tokens[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(tokens)>=1024,route=route,seconds=time.monotonic()-t,execution='GENERATED')
  write(dest,record);return record
 def copy_output(original,folder,qid,route):
  record=dict(original,route=route,execution='EXACT_CACHED_FREE_OUTPUT_SELECTION');write(folder/(qid+'.json'),record)
 def extract_evidence(pack):
  e=pack['evidence'];return dict(visual=e.visual.cpu(),global_context=e.global_context.cpu(),relation=e.relation.cpu(),provenance=e.provenance)
 try:
  write(OUT/'MODEL_READY.json',dict(time=time.time(),model_type=runtime.model.config.model_type,hidden_size=runtime.model.config.hidden_size,layers=32,generation=runtime.generation_config))
  for task in tasks:
   folder=OUT/task['key'];folder.mkdir(exist_ok=True)
   sources=prepare(source_rows(task),True);batches=[r.old.batch_on(runtime,x[1]) for x in sources]
   # Black RGB source question; created once before adapter preprocessing, no medical answer.
   black_rows=[]
   for q in source_rows(task):black_rows.append(dict(q,image_path=str(ROOT/'black.png'),reference=''))
   blacks=prepare(black_rows,False)
   features=[];black_features=[]
   for i,item in enumerate(sources):
    target=int(batches[i].labels[batches[i].labels!=-100][0])
    features.append(probe(item,target if i==0 else None));black_features.append(probe(blacks[i]))
   save(folder/'SOURCE_FEATURES.pt',dict(source=[dict(radius=f['radius'].cpu(),router=f['router'].cpu(),evidence=extract_evidence(f)) for f in features],black=[dict(radius=f['radius'].cpu(),router=f['router'].cpu(),evidence=extract_evidence(f)) for f in black_features]))
   write(folder/'LOCALIZER.json',features[0]['localizer'])
   radius_router=RadiusRouter(torch.cat([f['radius'] for f in features[:2]]),torch.cat([f['radius'] for f in features[2:]]),torch.cat([f['radius'] for f in black_features[:2]]))
   torch.manual_seed(seed);ldke=LDKERouter(4096).to(runtime.device);opt=torch.optim.Adam(ldke.parameters(),lr=.01)
   a=torch.cat([f['router'] for f in features[:2]]);p=torch.cat([f['router'] for f in features[2:]]);n=torch.cat([f['router'] for f in black_features[:2]])
   for _ in range(100):
    budget();opt.zero_grad();loss=ldke.objective(a,p,n);assert torch.isfinite(loss);loss.backward();opt.step();counts['router_updates']+=1
   ldke.requires_grad_(False)
   # Query evidence paired with the first UPDATE edit anchor; PRESERVE remains positive support.
   anchor=features[0]['evidence'];xs=[pair_features(f['evidence'],anchor) for f in features];xn=[pair_features(f['evidence'],anchor) for f in black_features]
   torch.manual_seed(seed);gate_model=EvidenceGate().to(runtime.device);opt=torch.optim.Adam(gate_model.parameters(),lr=.01)
   gx=torch.cat(xs+xn);gy=torch.tensor([1.]*4+[0.]*4,device=runtime.device)
   for _ in range(100):
    opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(gate_model(gx),gy);loss.backward();opt.step();counts['router_updates']+=1
   gate_model.requires_grad_(False);source_gates=[float(gate_model.route(x)[0]) for x in xs]
   save(folder/'ROUTERS.pt',dict(radius=radius_router.state_dict(),ldke=ldke.state_dict(),gate=gate_model.state_dict()))
   write(folder/'ROUTER_ADMISSION.json',dict(source_gate=source_gates,proxy_negative_only=True,ldke_source=[bool(ldke.route(f['router'],a)[0][0]) for f in features],radii=radius_router.radii.tolist(),ldke_parameters=sum(v.numel() for v in ldke.parameters()),gate_parameters=sum(v.numel() for v in gate_model.parameters())))
   # Real gate-off/on, teacher-prefill and cached-decode checks use source only.
   ex=make();ex.G1.data.fill_(.001);wrapper=ConditionedExpert(ex);h=Hook(runtime.get_module(TARGETS['S']),wrapper,True);h.attach()
   kwargs=prompt_kwargs(sources[0]);base=features[0]['logits']
   try:
    with torch.no_grad(),wrapper.request(torch.zeros(1,device=runtime.device),torch.ones(1,4,device=runtime.device)),h.generation_request():off=runtime.model(**kwargs).logits[:,-1]
    torch.testing.assert_close(off,base,rtol=0,atol=1e-6)
    with torch.no_grad(),wrapper.request(torch.ones(1,device=runtime.device),torch.ones(1,4,device=runtime.device)),h.generation_request():on=runtime.model(**kwargs).logits[:,-1]
   finally:h.detach()
   original=Hook(runtime.get_module(TARGETS['S']),ex,True);original.attach()
   try:
    with torch.no_grad(),original.generation_request():fixed=runtime.model(**kwargs).logits[:,-1]
   finally:original.detach()
   torch.testing.assert_close(on,fixed,rtol=0,atol=1e-6)
   # Full greedy generation verifies no hidden gate state survives requests.
   emit(sources[0],folder/'smoke'/'BASE')
   h.attach()
   try:
    emit(sources[0],folder/'smoke'/'ON',h,wrapper,1.,torch.ones(1,4,device=runtime.device))
    emit(sources[0],folder/'smoke'/'OFF',h,wrapper,0.,torch.ones(1,4,device=runtime.device))
   finally:h.detach()
   assert read(folder/'smoke'/'BASE'/(sources[0][0]['query_id']+'.json'))['raw_token_ids']==read(folder/'smoke'/'OFF'/(sources[0][0]['query_id']+'.json'))['raw_token_ids']
   assert wrapper.gate is None
   write(folder/'REAL_SMOKE.json',dict(status='PASS',gate0_base_exact=True,gate1_gamma1_fixed_exact=True,real_generation=True,source_only=True,time=time.time()))
   del ex,wrapper,h,original
   write(OUT/'STAGE0_READY.json',dict(status='PASS',case=task['key'],time=time.time()))
   while not (ROOT/'STAGE1_GO.json').exists():budget();time.sleep(1)
   evaluation=prepare(task['eval_rows'])+prepare([brief_row(q) for q in task['eval_rows']])
   # Distinguish original and SHORT duplicate query IDs by separate context folders.
   qfeatures=[];base_outputs=[]
   for j,item in enumerate(evaluation):
    context='ORIGINAL' if j<len(task['eval_rows']) else 'SHORT';rec=emit(item,folder/'outputs'/context/'BASE');base_outputs.append(rec)
    prior_arm='BASE' if context=='ORIGINAL' else 'BASE_SHORT'
    old=read(Path(config['prior_root'])/str(PART)/task['key']/'outputs'/prior_arm/(item[0]['query_id']+'.json'));verify_original(rec,old)
    qfeatures.append(probe(item))
   write(folder/'BASE_PARITY.json',dict(status='PASS',outputs=len(evaluation)))
   # Separate writers start at exactly same canonical initialization, including B_GATE.
   initial=None;w0_records=[];w0_state=None
   for arm in ('W0','B_GATE','B_COND'):
    ex=make();state=snapshot(ex)
    if initial is None:initial=state
    else:assert all(torch.equal(v,initial[k]) for k,v in state.items())
    torch.manual_seed(seed);cond=Conditioner().to(runtime.device);co=torch.optim.Adam(cond.parameters(),lr=.001)
    wrapper=ConditionedExpert(ex);h=Hook(runtime.get_module(TARGETS['S']),wrapper,True);h.attach();scalar=dict(step=0,second_moments=[0.,0.]);curve=[];before_counts=dict(counts);started=time.monotonic()
    try:
     with PrefixCache(runtime,batches):
      for step in range(1,161):
       budget();before=snapshot(ex);grads=[];cgrads=[];losses=[]
       for i,batch in enumerate(batches):
        gamma=cond(xs[i]) if arm=='B_COND' else torch.ones(1,4,device=runtime.device)
        gate=1. if arm=='W0' else source_gates[i]
        with wrapper.request(torch.tensor([gate],device=runtime.device),gamma):
         h.teacher(batch.labels,batch.forward_kwargs().get('attention_mask'));loss=runtime.compute_loss(batch);assert torch.isfinite(loss)
         parameters=list(ex.parameters())+(list(cond.parameters()) if arm=='B_COND' else [])
         gs=torch.autograd.grad(loss,parameters);counts['backwards']+=1
        grads.append(torch.cat([g.detach().cpu().double().flatten() for g in gs[:4]]));losses.append(float(loss.detach()))
        if arm=='B_COND':cgrads.append([g.detach() for g in gs[4:]])
       gradient,weights=adaptive_gradient(grads,losses);state,scalar=next_step(r,before,gradient,scalar);ex.load_state_dict(state)
       if arm=='B_COND':
        co.zero_grad()
        for k,param in enumerate(cond.parameters()):param.grad=sum(cgrads[i][k]*float(weights[i]) for i in range(4))
        co.step();counts['conditioner_updates']+=1
       counts['updates']+=1;curve.append(dict(step=step,losses=losses,weights=weights.tolist()))
       if step%20==0:write(OUT/'PROGRESS.json',dict(case=task['key'],arm=arm,step=step,counts=counts,time=time.time()))
     save(folder/arm/'delta.pt',dict(writer=snapshot(ex),conditioner=cond.state_dict() if arm=='B_COND' else None))
     loaded=torch.load(folder/arm/'delta.pt',weights_only=True,map_location=runtime.device);assert all(torch.equal(v,loaded['writer'][k]) for k,v in ex.state_dict().items())
     if arm=='W0':w0_state=snapshot(ex)
     if arm=='B_GATE' and all(source_gates):assert all(torch.equal(v,w0_state[k]) for k,v in ex.state_dict().items())
     for j,(item,f) in enumerate(zip(evaluation,qfeatures)):
      context='ORIGINAL' if j<len(task['eval_rows']) else 'SHORT';qid=item[0]['query_id'];x=pair_features(f['evidence'],anchor)
      natural=1. if arm=='W0' else float(gate_model.route(x)[0]);gamma=cond(x).detach() if arm=='B_COND' else torch.ones(1,4,device=runtime.device)
      rec=emit(item,folder/'outputs'/context/arm,h,wrapper,natural,gamma,dict(gate=natural,gamma=gamma.tolist()))
      if arm=='W0':
       prior_arm='ADAPT' if context=='ORIGINAL' else 'ADAPT_SHORT';key=(prior_arm,qid)
       control=next(c for c in read(ROOT/'CONTROL_OUTPUTS.json') if c['arm']==prior_arm and c['q']['query_id']==qid)
       assert rec['raw_token_ids']==control['tokens'] and rec['raw_answer']==control['answer'] and rec['EOS']==control['eos'] and rec['at_cap']==control['at_cap'],'W0 original output mismatch'
       w0_records.append(rec)
      if arm!='W0':
       if natural:copy_output(rec,folder/'outputs'/context/(arm+'_FORCED'),qid,dict(gate=1.))
       else:emit(item,folder/'outputs'/context/(arm+'_FORCED'),h,wrapper,1.,gamma,dict(gate=1.))
       copy_output(base_outputs[j],folder/'outputs'/context/(arm+'_OFF'),qid,dict(gate=0.))
     write(folder/arm/'COMPLETE.json',dict(status='PASS',curve=curve,seconds=time.monotonic()-started,counts={k:counts[k]-before_counts[k] for k in counts},writer_parameters=7168,conditioner_parameters=sum(v.numel() for v in cond.parameters()) if arm=='B_COND' else 0,paired_initial_exact=True,save_load_exact=True))
    finally:h.detach()
    del ex,wrapper,h,cond
   for j,(item,f) in enumerate(zip(evaluation,qfeatures)):
    context='ORIGINAL' if j<len(task['eval_rows']) else 'SHORT';qid=item[0]['query_id']
    be,dist=radius_router(f['radius']);ld,sim=ldke.route(f['router'],a)
    for name,on,score in [('BE_RADIUS_TT',bool(be[0]),dist.tolist()),('LDKE_ROUTER_TT_PROXY',bool(ld[0]),sim.tolist())]:
     copy_output(w0_records[j] if on else base_outputs[j],folder/'outputs'/context/name,qid,dict(gate=int(on),score=score,prompt_only=True,proxy_support=True))
     copy_output(w0_records[j],folder/'outputs'/context/(name+'_FORCED'),qid,dict(gate=1))
     copy_output(base_outputs[j],folder/'outputs'/context/(name+'_OFF'),qid,dict(gate=0))
   write(folder/'COMPLETE.json',dict(status='PASS',counts=counts,time=time.time(),registered_weight_consumers_complete=False,remaining_consumers='evidence permutation diagnostic and final report; retain small deltas',proxy_only=True))
  write(OUT/'MODEL_COMPLETE.json',dict(status='PASS',counts=counts,time=time.time()))
 finally:fh.remove();vh.remove()

if __name__=='__main__':
 code=0
 try:main()
 except BaseException:
  code=1;write(OUT/'FAILURE.json',dict(error=traceback.format_exc(),counts=counts,time=time.time()));write(ROOT/'STOP',dict(reason='V50_FAILURE',time=time.time()));raise
 finally:write(OUT/'EXIT.json',dict(code=code,counts=counts,seconds=time.time()-START))
