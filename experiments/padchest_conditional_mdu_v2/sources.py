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
 for name,source in out.items():compile(source,name,'exec')
 return out
