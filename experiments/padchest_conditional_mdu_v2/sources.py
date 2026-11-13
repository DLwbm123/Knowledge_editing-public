"""Add real-model admission to the never-launched conditional mechanism draft."""
from pathlib import Path

def build(worktree):
 w=Path(worktree);p=w/'experiments/padchest_conditional_mdu_v1'
 out={f.name:f.read_text() for f in p.glob('*.py')}
 marker=' # Both new arms receive the same MED initialization, basis, source batches,'
 assert out['worker.py'].count(marker)==1
 preflight=''' # Test the structural protection on the retained original experts before
 # optimizing any new arm. Projected native equality is recorded, not required:
 # the matched training phase may need to restore source efficacy.
 protected_equal=0;projected_native_equal=0
 for t in tasks:
  eid=t['edit_id'];expert=r.p.expert(library['states'][eid],t['seed'],runtime.device).requires_grad_(False)
  g.bind_projection(expert,bases[eid].to(runtime.device))
  for q in protected[eid]:
   answer=generate(f'preflight_protected_{t["order"]}_{q["query_id"]}',t,prep(t,q),expert)
   protected_equal += answer['raw_token_ids']==baseanswers[q['query_id']]['raw_token_ids']
  answer=generate(f'preflight_native_{t["order"]}',t,prepared[eid][0],expert)
  projected_native_equal += answer['raw_token_ids']==basenative[t['native']['query_id']]['answer']['raw_token_ids']
 write('MECHANISM_PREFLIGHT.json',dict(status='PASS' if protected_equal==20 else 'PROTECTION_PARITY_FAILURE',
  protected_Base_tokens_equal=protected_equal,protected_comparisons=20,projected_original_native_tokens_equal=projected_native_equal,
  projected_original_native_comparisons=4,scope_training_positives_admitted=20,scope_calibration_only=True,
  generations=COUNTS['generations'],updates=0,Judge_calls=0,time=time.time()))
 assert protected_equal==20,'Structural protection did not replay clean Base; no training admission'
'''
 out['worker.py']=out['worker.py'].replace(marker,preflight+marker)
 out['conditional_geometry.py']=out.pop('geometry.py')
 out['conditional_admission.py']=out.pop('admission.py')
 out['worker.py']=out['worker.py'].replace('import geometry as g','import conditional_geometry as g').replace('from admission import check','from conditional_admission import check')
 out['conditional_generate.py']=(w/'experiments/padchest_conditional_mdu_v2/conditional_generate.py').read_text()
 out['conditional_geometry.py']=out['conditional_geometry.py'].replace(" expert.residual=types.MethodType(residual,expert)"," expert._conditional_projection=True\n expert.residual=types.MethodType(residual,expert)")
 worker=out['worker.py']
 worker=worker.replace("COUNTS=dict(generations=0,updates=0,backwards=0,forwards=0)","COUNTS=dict(CFG.get('prior_counts') or dict(generations=0,updates=0,backwards=0,forwards=0))")
 worker=worker.replace("(ROOT/'WORKER_ONCE').touch(exist_ok=False)","(ROOT/('WORKER_RECOVERY1_ONCE' if CFG.get('engineering_recovery') else 'WORKER_ONCE')).touch(exist_ok=False)")
 worker=worker.replace(" import conditional_geometry as g", " import conditional_geometry as g\n from conditional_generate import generate as conditional_generate\n r.s.fit.short.generate=conditional_generate")
 worker=worker.replace("runtime.run_root=ROOT/'runtime';runtime.run_root.mkdir()", "runtime.run_root=ROOT/('runtime_recovery1' if CFG.get('engineering_recovery') else 'runtime');runtime.run_root.mkdir()")
 worker=worker.replace(" prepared={};features={};baseanswers={};hidden={};states={};bases={};source_rows={}"," prepared={};features={};baseanswers={};hidden={};states={};bases={};source_rows={}\n recovered_base=torch.load(ROOT/'BASE_BEHAVIOR.private.pt',map_location='cpu',weights_only=True) if CFG.get('engineering_recovery') else None\n recovered_bases=torch.load(ROOT/'PROTECTION_BASES.private.pt',map_location='cpu',weights_only=True) if CFG.get('engineering_recovery') else None")
 old="  budget();dest=ROOT/'diagnostics'/f'{name}.json';dest.parent.mkdir(exist_ok=True);dest.with_suffix('.attempt').touch(exist_ok=False)"
 new="""  budget();dest=ROOT/'diagnostics'/f'{name}.json';dest.parent.mkdir(exist_ok=True)
  if CFG.get('engineering_recovery') and dest.exists():
   saved=read(dest)
   assert all(saved['judge_input'][k]==item[4][k] for k in ['question','reference','image_sha256','prompt_ids','attention_mask'])
   if capture:
    qid=item[0]['query_id'];assert saved['answer']==recovered_base['answers'][qid]
    return saved['answer'],recovered_base['hidden'][qid]
   return saved['answer']
  attempt=dest.with_suffix('.attempt')
  if attempt.exists():
   assert CFG.get('engineering_recovery') and name==CFG['retry_lost_generation_name'],'No unregistered generation replay'
   attempt=dest.with_suffix('.recovery1.attempt')
  attempt.touch(exist_ok=False)"""
 assert worker.count(old)==1;worker=worker.replace(old,new)
 worker=worker.replace(" save('BASE_BEHAVIOR.private.pt',dict(answers=baseanswers,hidden=hidden))"," if not CFG.get('engineering_recovery'):save('BASE_BEHAVIOR.private.pt',dict(answers=baseanswers,hidden=hidden))")
 worker=worker.replace("  q,audit=g.basis([hidden[q['query_id']] for q in protected[eid]])", "  if recovered_bases is not None:\n   q=recovered_bases[eid];audit={k:v for k,v in read(ROOT/'engineering_recovery1/TRAIN_ONLY_ADMISSION.json')['bases'][str(t['order'])].items() if k!='source_retained_norm_fraction'}\n  else:q,audit=g.basis([hidden[q['query_id']] for q in protected[eid]])")
 worker=worker.replace(" save('PROTECTION_BASES.private.pt',bases)"," if not CFG.get('engineering_recovery'):save('PROTECTION_BASES.private.pt',bases)")
 worker=worker.replace(" entries=g.calibrate(tasks,positive,negative)"," entries=torch.load(ROOT/'SCOPE_R1.private.pt',map_location='cpu',weights_only=True) if CFG.get('engineering_recovery') else g.calibrate(tasks,positive,negative)")
 worker=worker.replace(" save('SCOPE_R1.private.pt',entries)"," if not CFG.get('engineering_recovery'):save('SCOPE_R1.private.pt',entries)")
 worker=worker.replace("write('FAILURE.private.json',dict(time=time.time(),counts=COUNTS,error=traceback.format_exc()))", "write('FAILURE_RECOVERY1.private.json' if CFG.get('engineering_recovery') else 'FAILURE.private.json',dict(time=time.time(),counts=COUNTS,error=traceback.format_exc()))")
 out['worker.py']=worker
 admission=out['conditional_admission.py'];needle2=" old=Path(cfg['corrected_root']);"
 admission=admission.replace(needle2," if cfg.get('engineering_recovery'):\n  result=json.loads((root/'DATA_ADMISSION.json').read_text());assert result['status']=='PASS';return result\n"+needle2)
 out['conditional_admission.py']=admission
 guard=out['guard.py']
 for old,new in [('GUARD_ONCE','GUARD_RECOVERY1_ONCE'),('worker.log','worker_recovery1.log'),('IDENTITY.private.json','IDENTITY_RECOVERY1.private.json'),('GUARD_EXIT.json','GUARD_EXIT_RECOVERY1.json'),('GUARD_FAILURE.private.json','GUARD_FAILURE_RECOVERY1.private.json')]:guard=guard.replace(old,new)
 out['guard_recovery1.py']=guard
 closeout=out['closeout.py']
 closeout=closeout.replace("(control/'CLOSEOUT_ONCE')","(control/'CLOSEOUT_RECOVERY1_ONCE')")
 for old,new in [('FAILURE.private.json','FAILURE_RECOVERY1.private.json'),('GUARD_FAILURE.private.json','GUARD_FAILURE_RECOVERY1.private.json'),('GUARD_EXIT.json','GUARD_EXIT_RECOVERY1.json')]:closeout=closeout.replace(old,new)
 out['closeout_recovery1.py']=closeout
 for name,source in out.items():compile(source,name,'exec')
 return out
