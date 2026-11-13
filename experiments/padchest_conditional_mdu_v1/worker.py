"""Matched conditional CE/MDU development experiment with protected input space."""
import json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(os.environ['CONDITIONAL_ROOT']);CFG=json.loads((ROOT/'CONFIG.private.json').read_text());CLOCK=CFG['clock']
COUNTS=dict(generations=0,updates=0,backwards=0,forwards=0)
def read(p):return json.loads(Path(p).read_text())
def write(name,x):
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False));t.replace(p)
def save(name,x):
 import torch
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');torch.save(x,t);t.replace(p)
def budget():
 assert time.time()<CLOCK['hard_end']-1200 and not (ROOT/'STOP.json').exists()
 assert time.time()-read(ROOT/'HEALTH.json')['time']<45
 assert COUNTS['generations']<=500 and COUNTS['updates']<=1280 and COUNTS['backwards']<=2560 and COUNTS['forwards']<=600000

def main():
 (ROOT/'WORKER_ONCE').touch(exist_ok=False)
 parent=Path(os.environ['PARENT_RUN']);old=Path(CFG['corrected_root']);sys.path[:0]=[str(ROOT/'tools'),str(old/'tools'),CFG['med150_tools'],str(parent/'private/tools'),str(parent/'private/source'),str(parent/'private/official_llava')]
 import torch,numpy as np,module_run as r
 import geometry as g
 from admission import check
 check(ROOT,CFG)
 from pair_input import install
 from types import SimpleNamespace
 from scripts.medtrace.run_realmodel_core import load_real_runtime
 from m3bench_repro.editors.llava_runtime import seed_everything
 from m3bench_repro.editors.routing import MemoryRouter,decision_as_json
 from span_train import terms
 from robust_objective import weights_and_value,view_loss
 install();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);r.c.budget=budget
 runtime=load_real_runtime(SimpleNamespace(cpu_gate=parent/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime)
 assert not any(p.requires_grad for p in runtime.model.parameters());versions=[(p,p._version) for p in runtime.model.parameters()]
 runtime.run_root=ROOT/'runtime';runtime.run_root.mkdir()
 def count(*_):budget();COUNTS['forwards']+=1
 counter=runtime.model.register_forward_pre_hook(count)
 data=read(old/'T5_INPUTS.private.json');tasks=data['tasks'];assert len(tasks)==4;byid={t['edit_id']:t for t in tasks}
 library=torch.load(old/'MDU/T5_CANONICAL.pt',map_location='cpu',weights_only=True)
 locks={v['order']:v for v in read(old/'MDU/TOKEN_LOCK.private.json')['records']}
 aux=read(ROOT/'AUX_INPUTS.private.json')['queries'];assert len(aux)==4
 config=read(old/'BASE_CONFIGURATION.private.json');assert config['generation']==runtime.generation_config
 rng=torch.load(Path(os.environ['RAW_MED_ROOT'])/'GENERATION_RNG.pt',map_location='cpu',weights_only=False)
 base0={v['query_id']:v for v in [read(p) for p in (old/'base_outputs').glob('*.json')]}
 oldout=[read(p) for p in (old/'MDU/outputs').glob('*.json')]
 basenative={v['query_id']:v for v in oldout if v['mode']=='bank' and v['role']=='T0'}
 prepared={};features={};baseanswers={};hidden={};states={};bases={};source_rows={}
 def prep(task,q,training=False):
  runtime.model.half()
  try:item=r.s.prepare(runtime,{'configuration_only':config},task,[q],training)[0]
  finally:r.old.fp32(runtime)
  return item
 def feature(task,q):
  key=(q['query_id'],q['question'],q['image_path'])
  if key in features:return features[key]
  captured={};handles=[];runtime.model.half()
  try:
   def vision(_m,_a,out):
    assert isinstance(out,torch.Tensor) and out.shape[0]==2;captured['vision']=out.detach().cpu()
   handles.append(runtime.model.get_vision_tower().register_forward_hook(vision))
   batch=runtime.build_question_batch(r.s.record(task,q))
   def first(_m,args):captured['last']=args[0][0,batch.key_token_index].detach().float().cpu()
   def second(_m,args):
    x=args[0][0].detach().float()
    captured['r0']=(x.mean(0) if batch.attention_mask is None else x[batch.attention_mask[0].bool()].mean(0)).cpu()
   handles.extend([runtime.get_module(r.c.LAYER).register_forward_pre_hook(first),runtime.get_module(runtime.target_lock['balancedit']['targets'][0]).register_forward_pre_hook(second)])
   kwargs=batch.forward_kwargs();kwargs['labels']=None
   with torch.no_grad():runtime.model(**kwargs)
   value=dict(r0=captured['r0'],r1=g.evidence_key(captured['last'],captured['vision']))
  finally:
   for h in handles:h.remove()
   r.old.fp32(runtime)
  features[key]=value;return value
 def generate(name,task,item,expert=None,capture=False):
  budget();dest=ROOT/'diagnostics'/f'{name}.json';dest.parent.mkdir(exist_ok=True);dest.with_suffix('.attempt').touch(exist_ok=False)
  vectors=[];handle=None;hook=None
  try:
   if capture:
    def take(_m,args):vectors.append(args[0][0,-1].detach().float().cpu().clone())
    handle=runtime.get_module(r.c.LAYER).register_forward_pre_hook(take)
   if expert is not None:hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert);hook.attach()
   r.c.restore_rng(rng['rng']);np.random.set_state(rng['numpy_rng']);COUNTS['generations']+=1
   value,_=r.s.fit.short.generate(runtime,hook,item[2],item[3]);tokens=list(value.raw_token_ids)
  finally:
   if handle:handle.remove()
   if hook:hook.detach()
  assert 0<len(tokens)<=1024
  answer=dict(raw_answer=value.decoded_text,raw_token_ids=tokens)
  write('diagnostics/'+dest.name,dict(answer=answer,EOS=tokens[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(tokens)>=1024,judge_input=item[4]))
  if capture:
   assert len(vectors)==len(tokens)<=128 and tokens[-1]==runtime.adapter.tokenizer.eos_token_id
   return answer,torch.stack(vectors)
  return answer
 # Training-only material: five positive views for each original source and
 # four patient-disjoint real comparisons. Evaluation probes are not loaded here.
 for t in tasks:
  rows=t['all_rows'][:5];assert [v['audit_role'] for v in rows]==['NATIVE']+['FIT']*4
  source_rows[t['edit_id']]=rows;prepared[t['edit_id']]=[prep(t,q,True) for q in rows]
  for q in rows:feature(t,q)
  q=t['native'];answer,h=generate('base_'+q['query_id'],t,prepared[t['edit_id']][0],capture=True)
  assert answer['raw_token_ids']==base0[q['query_id']]['answer']['raw_token_ids'],'Clean Base source replay differs'
  baseanswers[q['query_id']]=answer;hidden[q['query_id']]=h
 for q in aux:
  task=tasks[0];feature(task,q);item=prep(task,q);answer,h=generate('base_'+q['query_id'],task,item,capture=True)
  baseanswers[q['query_id']]=answer;hidden[q['query_id']]=h
 save('BASE_BEHAVIOR.private.pt',dict(answers=baseanswers,hidden=hidden))
 positive={};negative={};protected={};basis_audit={}
 for t in tasks:
  eid=t['edit_id'];positive[eid]=[feature(t,q)['r1'] for q in source_rows[eid]]
  same=[q for other in tasks if other['edit_id']!=eid and g.concept(other['native']['question'])==g.concept(t['native']['question']) for q in source_rows[other['edit_id']]]
  same+= [q for q in aux if g.concept(q['question'])==g.concept(t['native']['question'])]
  assert len(same)==7;negative[eid]=[feature(t,q)['r1'] for q in same]
  protected[eid]=[other['native'] for other in tasks if other['edit_id']!=eid]+[q for q in aux if g.concept(q['question'])==g.concept(t['native']['question'])]
  assert len(protected[eid])==5
  q,audit=g.basis([hidden[q['query_id']] for q in protected[eid]])
  x=hidden[t['native']['query_id']][0:1];z=g.project_normalized(x,q);retained=float(z.norm()/(x/(x.square().mean(-1,keepdim=True).sqrt()+1e-6)).norm())
  assert retained>1e-4,'Source indistinguishable from protected activation span; stop'
  bases[eid]=q;basis_audit[str(t['order'])]=dict(audit,source_retained_norm_fraction=retained)
 save('PROTECTION_BASES.private.pt',bases)
 entries=g.calibrate(tasks,positive,negative)
 for t in tasks:
  for q in source_rows[t['edit_id']]:assert g.route(entries,feature(t,q)['r1'],q['question'])['logical_edit_id']==t['edit_id']
 # Routing ignores the reference, progression and patient ID by construction.
 save('SCOPE_R1.private.pt',entries)
 write('TRAIN_ONLY_ADMISSION.json',dict(status='PASS',source_patients=4,auxiliary_patients=4,evaluation_probe_patients=6,scope_source_views=20,protected_contexts_per_expert=5,
  bases=basis_audit,scope=[{k:v for k,v in e.items() if k not in ['center','edit_id']} for e in entries],test_queries_used_for_calibration=False,counts=dict(COUNTS),time=time.time()))
 router0=MemoryRouter.from_state(dict(distance='euclidean',entries=library['R0_entries']),device=runtime.device)
 # Verify the original deployment independently before optimizing any new arm.
 for t in tasks:
  q=t['native'];f=feature(t,q);d=router0.route(f['r0'].to(runtime.device));assert d.logical_edit_id==t['edit_id'] and d.activated
  expert=r.p.expert(library['states'][t['edit_id']],t['seed'],runtime.device).requires_grad_(False)
  answer=generate('legacy_replay_'+str(t['order']),t,prepared[t['edit_id']][0],expert)
  assert answer['raw_token_ids']==basenative[q['query_id']]['answer']['raw_token_ids']
 # Both new arms receive the same MED initialization, basis, source batches,
 # optimizer, source/FIT schedule, seeds and 160 updates.
 for arm in ['CE','MDU']:
  states[arm]={}
  for t in tasks:
   budget();eid=t['edit_id'];owner=t['order'];seed_everything(t['seed'])
   initial=torch.load(old/'MDU/candidates'/f'MED_{owner}.pt',map_location='cpu',weights_only=True)
   expert=r.p.expert(initial,t['seed'],runtime.device);g.bind_projection(expert,bases[eid].to(runtime.device))
   initial_cpu={k:v.detach().cpu().clone() for k,v in expert.state_dict().items()}
   hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert);hook.attach();opt=r.p.optimizer(expert,runtime.model)
   batches=[r.old.batch_on(runtime,x[1]) for x in prepared[eid]];rows=source_rows[eid];lock=locks[owner];curve=[]
   try:
    for step in range(1,161):
     budget();indices=(0,1+(step-1)%4);expert.zero_grad(set_to_none=True)
     def calculate(i):return terms(runtime,hook,batches[i],rows[i],lock['answer_tokens'],lock['spans'],lock['views'][i])
     if arm=='MDU':
      with torch.no_grad():first=[calculate(i) for i in indices]
      weights,robust=weights_and_value(first[0][1],first[1][1],1.0)
     losses=[]
     for j,i in enumerate(indices):
      ce,spans=calculate(i)
      if arm=='MDU':
       assert torch.allclose(ce.detach(),first[j][0],atol=1e-5,rtol=1e-5) and torch.allclose(spans.detach(),first[j][1],atol=1e-5,rtol=1e-5)
       loss=view_loss(ce,spans,weights[j],1.0)
      else:loss=.5*ce
      assert torch.isfinite(loss);loss.backward();COUNTS['backwards']+=1;losses.append(float(ce.detach()))
     norm=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.)
     assert torch.isfinite(norm) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in expert.parameters())
     opt.step();COUNTS['updates']+=1;curve.append(dict(step=step,source_CE=losses,grad_norm=float(norm)))
     if step%20==0:
      save(f'resume/{arm}_{owner}.pt',dict(state=r.old.snapshot(expert),optimizer=opt.state_dict(),step=step,curve=curve,rng=r.c.rng(),numpy_rng=np.random.get_state()))
      write('PROGRESS.json',dict(phase='TRAIN',arm=arm,owner=owner,step=step,counts=dict(COUNTS),time=time.time()))
    final={k:v.detach().cpu().clone() for k,v in expert.state_dict().items()};assert sum(x.numel() for x in final.values())==7168
    displacement=float(torch.cat([(final[k]-initial_cpu[k]).flatten() for k in final]).norm())
    save(f'states/{arm}_{owner}.pt',dict(state=final,curve=curve,steps=160,arm=arm,owner=owner,parameter_displacement_from_MED=displacement))
    loaded=torch.load(ROOT/'states'/f'{arm}_{owner}.pt',map_location='cpu',weights_only=True)['state'];assert all(torch.equal(final[k],loaded[k]) for k in final)
    states[arm][eid]=loaded
   finally:hook.detach()
   # Full negative-token replay is an engineering gate for preservation, not a
   # clinical accuracy score. No evaluation probe is used for this check.
   expert.requires_grad_(False)
   for q in protected[eid]:
    answer=generate(f'protected_{arm}_{owner}_{q["query_id"]}',t,prep(t,q),expert)
    assert answer['raw_token_ids']==baseanswers[q['query_id']]['raw_token_ids'],'Protected Base token replay failed'
   if owner==tasks[0]['order']:
    before=generate(f'save_before_{arm}',t,prepared[eid][0],expert)
    restored=r.p.expert(loaded,t['seed'],runtime.device).requires_grad_(False);g.bind_projection(restored,bases[eid].to(runtime.device))
    after=generate(f'save_after_{arm}',t,prepared[eid][0],restored)
    assert before['raw_token_ids']==after['raw_token_ids']
   write('ARM_PROGRESS.json',dict(arm=arm,completed_owner=owner,protected_replay=True,counts=dict(COUNTS),time=time.time()))
 assert COUNTS['updates']==1280 and COUNTS['backwards']==2560
 save('CONDITIONAL_CANONICAL.private.pt',dict(tasks=tasks,states=states,protection_bases=bases,scope_R1=entries,source_commit=CFG['source_commit']))
 write('TRAIN_COMPLETE.json',dict(status='PASS',updates=1280,backwards=2560,protected_replays=40,source_token_replays=4,save_load_generation_checks=2,time=time.time()))
 # Parameters and scope are frozen before any unseen paraphrase/probe feature.
 expectations=data['expected'];cache={};query_items={};query_features={}
 conditions=[('ROUTE_ONLY','LEGACY','R1'),('CONDITIONAL_CE','CE','R1'),('CONDITIONAL_MDU','MDU','R1'),('CE_OLD_ROUTE','CE','R0'),('MDU_OLD_ROUTE','MDU','R0')]
 for condition,arm,scope in conditions:
  outdir=ROOT/condition/'outputs';outdir.mkdir(parents=True)
  for mode in ['single','bank']:
   for row in data['plan']:
    budget();t=byid[row['edit_id']];q=row['query'];qid=q['query_id']
    if qid not in query_items:
     query_items[qid]=prep(t,q);query_features[qid]=feature(t,q)
     assert all(query_items[qid][4][k]==v for k,v in expectations[qid].items())
    item=query_items[qid];f=query_features[qid]
    if scope=='R1':decision=g.route(entries if mode=='bank' else [e for e in entries if e['edit_id']==t['edit_id']],f['r1'],q['question'])
    else:
     router=router0 if mode=='bank' else MemoryRouter.from_state(dict(distance='euclidean',entries=[e for e in library['R0_entries'] if e['logical_edit_id']==t['edit_id']]),device=runtime.device)
     decision=decision_as_json(router.route(f['r0'].to(runtime.device)))
    chosen=decision['logical_edit_id'] if decision['activated'] else None
    identity=(qid,arm if chosen else 'BASE',chosen)
    reused=identity in cache
    if not reused:
     expert=None
     if chosen:
      weights=library['states'][chosen] if arm=='LEGACY' else states[arm][chosen]
      expert=r.p.expert(weights,byid[chosen]['seed'],runtime.device).requires_grad_(False)
      if arm!='LEGACY':g.bind_projection(expert,bases[chosen].to(runtime.device))
     cache[identity]=generate(f'eval_{condition}_{mode}_{row["edit_id"]}_{qid}',t,item,expert)
    answer=cache[identity];cid=r.c.digest([condition,mode,row['edit_id'],row['role'],qid])
    value=dict(consumer_id=cid,condition=condition,mode=mode,edit_id=t['edit_id'],owner=t['order'],role=row['role'],query_id=qid,
      answer=answer,judge_input=item[4],route=decision,effective_expert=chosen,prediction_reused=reused,EOS=answer['raw_token_ids'][-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(answer['raw_token_ids'])>=1024)
    write(f'{condition}/outputs/{cid}.json',value)
   write('PROGRESS.json',dict(phase='EVALUATE',condition=condition,mode=mode,counts=dict(COUNTS),time=time.time()))
  assert len(list(outdir.glob('*.json')))==80
  write(f'{condition}/GENERATION_COMPLETE.json',dict(status='COMPLETE',consumers=80,time=time.time()))
 assert all(p._version==version for p,version in versions),'Frozen backbone changed'
 counter.remove();write('GENERATION_COMPLETE.json',dict(status='ALL_FIVE_CONDITIONS_COMPLETE',counts=COUNTS,consumers=400,time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:write('FAILURE.private.json',dict(time=time.time(),counts=COUNTS,error=traceback.format_exc()));raise
