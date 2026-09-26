"""Explain fixed R3 errors without changing supports, samples or Judge verdicts."""
import os,sys,time
from pathlib import Path
from dataclasses import replace
ROOT=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(ROOT))
from worker_r4 import shared,old,rt,R3,SEED
from budget import read,write

def main():
 import torch
 from methods.medtrace import MedTraceLayerHook
 job=read(Path(os.environ['RUN_JOB_JSON']));folder=ROOT/'jobs'/job['id'];start=time.time();write(folder/'STATUS.json',dict(status='RUNNING',job=job,pid=os.getpid()))
 try:
  changes=read(R3/'private/reports/R2_VERIFY24_EXTENSION_single_1_S_R0_changes.json');targets=[c for c in changes if c['task']=='T2G' and c['delta']]
  tasks=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'];rows=[]
  for p in (R3/'jobs').glob('verify-single-*/*/p*/CONSUMERS.json'):rows+=read(p)
  private=[]
  with rt.gpu_session(job['id']):
   runtime=rt.load_runtime(folder/'runtime',SEED);handle=runtime.model.register_forward_pre_hook(lambda m,a:rt.check_budget())
   for c in targets:
    t=next(t for t in tasks if t['canonical_edit_id']==c['edit']);q=next(q for q in t['evaluation'] if q['query_id']==c['query_id']);rec=replace(old.record(t),question=q['question'],target=q['reference'],image_path=Path(q['image_path']))
    batch=runtime.build_edit_batch(rec);probs={}
    for writer in ['Base_OFF','P','P+S']:
     hook=None
     if writer!='Base_OFF':
      ex,_=shared.load(runtime,t,writer);hook=MedTraceLayerHook(runtime.get_module(rt.LAYER),ex);hook.attach();hook.set_teacher_routing(batch.labels)
     try:
      with torch.no_grad():probs[writer]=dict(target_mean_log_probability=-float(runtime.compute_loss(batch)),target_tokens=len(batch.target_token_ids))
     finally:
      if hook:hook.detach()
    outputs={a:next(r for r in rows if r['arm']==a+'@80_R0' and r['edit']==c['edit'] and r['query_id']==c['query_id'])['output'] for a in ['P','P+S']}
    category=('anatomic_entity/location substitution' if t['order']==35 else 'incomplete spatial relation' if t['order']==42 else 'size hallucination and repetition' if c['delta']<0 else 'size/density target restored despite repetition')
    siblings=[x for x in changes if x['edit']==c['edit'] and x['task']=='T2G']
    teachers=[];run=R3/'runs/s20260925/P+S'/f'e{t["order"]:03d}'
    for path in (run/'private/base').glob('*/*.json'):
     value=read(path);binding=value.get('binding',{})
     for u in t['U_fit']+t['U_new']:
      if binding.get('question')==u['question'] and binding.get('image_source_sha256')==u['image_sha256']:teachers.append(dict(source_support=u,Base_teacher_output=value))
    private.append(dict(edit=t['canonical_edit_id'],order=t['order'],query_id=q['query_id'],category=category,delta=c['delta'],outputs=outputs,reference=q['reference'],target_log_probability=probs,other_paraphrase_deltas=[x['delta'] for x in siblings],teachers=teachers,supports=t['U_fit']+t['U_new'],support_source_images_distinct=all(u['image_sha256']!=t['native']['image_sha256'] for u in t['U_fit']+t['U_new']),conflict_interpretation='different source-image facts can compete lexically, but are not a proven contradictory target on the same input; no supports removed'))
   handle.remove();assert runtime.base_guard.verify()['unchanged']
  write(ROOT/'private/R3_ERROR_ATTRIBUTION.json',private)
  public=[dict(category=x['category'],direction='regression' if x['delta']<0 else 'recovery',target_log_probability=x['target_log_probability'],other_paraphrase_deltas=x['other_paraphrase_deltas'],PS_cap_hit=x['outputs']['P+S']['cap_hit'],different_support_images=x['support_source_images_distinct']) for x in private]
  # All formal T1G samples are image variants; count R1 rejections separately.
  mapping={(r['arm'],r['edit'],r['query_id']):r for r in rows if r['task']=='T1G'};scored={p.stem:read(p)['is_correct'] for p in (ROOT/'private/judge/scores').glob('*.json')};damage=[]
  for key,a in mapping.items():
   if key[0]!='P+S@80_R0':continue
   b=mapping[('P+S@80_R1',key[1],key[2])]
   if scored.get(a['judge_key']) is True and scored.get(b['judge_key']) is False:damage.append(dict(rejected=not b['route']['activated'],ratio=b['route_diagnostics']['ratio']))
  write(ROOT/'public/R3_ERROR_ATTRIBUTION.json',dict(T2G_cases=public,T1G_new_errors=len(damage),T1G_new_errors_rejected=sum(x['rejected'] for x in damage),causal_claim=False,note='Post-hoc explanation only; never used to select formal rows, supports, beta or gate. Teacher-forced log probability and free generation are distinct observations.'))
  write(folder/'STATUS.json',dict(status='GPU_COMPLETE',job=job,seconds=time.time()-start))
 except Exception as e:
  import traceback
  write(folder/'STATUS.json',dict(status='FAILED',job=job,error=str(e),traceback=traceback.format_exc(),seconds=time.time()-start));raise
if __name__=='__main__':main()
