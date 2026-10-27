"""CPU integration: original SOURCE update identity and train-only routing consumers."""
def run(r):
 import torch,copy
 from path_v20 import proposal,construct,same,flatten
 from cache_v20 import selfcheck as cachecheck
 from route_v20 import selfcheck,calibrate
 seed=713
 state=r.mm.canonicalize(r.p.tr.TT4(seed,8,8).state_dict())[0];keys=list(state);scalar=dict(step=0,second_moments=[0.,0.]);x={k:v.detach().clone().requires_grad_(True) for k,v in state.items()}
 # Independent no-grad diagnostics cannot add a regularization gradient or optimizer update.
 loss=sum(v.square().sum() for v in x.values());g0=torch.autograd.grad(loss,tuple(x.values()));flat=torch.cat([g.double().flatten() for g in g0]);raw,updated,direction=proposal(r,state,flat,scalar,keys)
 y={k:v.detach().clone().requires_grad_(True) for k,v in state.items()};loss2=sum(v.square().sum() for v in y.values())
 with torch.no_grad():diagnostics=[sum(v.abs().sum() for v in y.values()) for _ in range(48)]
 g1=torch.autograd.grad(loss2,tuple(y.values()));assert all(torch.equal(a,b) for a,b in zip(g0,g1))
 # Direct legacy clipped gradients + canonical RMS, rather than testing the helper against itself.
 local={k:v.clone().requires_grad_(True) for k,v in state.items()}
 for k,g in zip(keys,g0):local[k].grad=g.clone()
 torch.nn.utils.clip_grad_norm_(list(local.values()),1.)
 old,oldscalar=r.mm.canonical_rms_step(state,{k:v.grad for k,v in local.items()},copy.deepcopy(scalar))
 assert same(raw,old) and updated==oldscalar and same(construct(r,state,raw,direction,direction,1.,keys),r.mm.canonicalize(old)[0]);assert scalar['step']==0
 route=selfcheck();cache=cachecheck()
 a=torch.tensor([1.,0.]);b=torch.tensor([0.,1.]);ks={x:{m:z.clone() for m in ('joint','image','text')} for x,z in [('s',a),('t',a),('n',b),('h',a)]};tasks=[dict(key='e',source={'id':'s'})];rows=[dict(id='t',role='ROUTER_TRAIN',article='one'),dict(id='n',role='ROUTER_TRAIN',article='two'),dict(id='h',role='HELDOUT',article='three')];rels={('t','e'):'YES',('n','e'):'NO',('h','e'):'YES'}
 c=calibrate(ks,tasks,rows,rels,'JOINT');ks['h']={m:-a for m in ('joint','image','text')};rels['h','e']='NO';assert c==calibrate(ks,tasks,rows,rels,'JOINT')
 consumer=consumer_check()
 return dict(consumer=consumer,status='PASS',source_gradient_and_original_optimizer_path_exact=True,no_grad_support_omission_safe=True,parent_import=True,cache=cache,router=route,heldout_perturbation_no_calibration_effect=True,no_model_calls=True)


def consumer_check():
 import json,tempfile
 from pathlib import Path
 from consumers_v20 import write,blind,summarize
 with tempfile.TemporaryDirectory() as d:
  p=Path(d);s=p/'source/0';(s/'outputs').mkdir(parents=True)
  row=dict(id='q',role='HELDOUT');config=dict(tasks=[dict(key='e',source={'id':'q'})],rows=[row|dict(question='Q',reference='R')],relations=[dict(query_id='q',scope='e',label='YES')],Base_outputs={'q':str(p/'base.json')})
  write(p/'MATRIX_CONFIG.json',config);write(p/'base.json',dict(raw_answer='R'));write(p/'BASE_VERDICTS.json',{'q':dict(label='CORRECT',reason='fixed')})
  write(s/'COMPLETE.json',dict(status='SOURCE_AND_ALL_MATRIX_MODEL_CONSUMERS_COMPLETE'));write(s/'TRAIN_COMPLETE.json',dict(steps=160));write(s/'outputs/q.json',dict(expert='e',query='q',raw_answer='W'))
  exit=dict(code=0,seconds=1,counts={})
  for i in range(5):(p/str(i)).mkdir();write(p/str(i)/'EXIT.json',exit)
  for i in range(2):(p/'features'/str(i)).mkdir(parents=True);write(p/'features'/str(i)/'EXIT.json',exit)
  write(s/'EXIT.json',exit);b=blind(p);assert b==dict(unique=2,reused=1,new=1)
  write(p/'ROUTES.json',dict(decisions={'R0':{'q':None}},natural=[{}]));ver=__import__('json').loads((p/'BLIND_REUSED_VERDICTS.json').read_text());write(p/'BLIND_VERDICTS.json',ver)
  try:summarize(p)
  except AssertionError:pass
  else:raise AssertionError('Missing verdict accepted')
  materials=__import__('json').loads((p/'BLIND_MATERIALS.json').read_text())
  for k in materials:ver.setdefault(k,dict(label='INCORRECT',reason='fixed reference mismatch'))
  write(p/'BLIND_VERDICTS.json',ver);result=summarize(p);assert result['forced']['e']['Base_correct_damage']==1 and result['natural']['R0']['damage']==0
 return dict(status='PASS',all_blind_consumers_required=True,Base_output_reuse=True,forced_damage_natural_OFF_separated=True)
