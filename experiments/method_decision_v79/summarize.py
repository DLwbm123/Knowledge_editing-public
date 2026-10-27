"""Aggregate frozen V79 labels and visual interventions; no new model calls."""
import collections,json,statistics
from pathlib import Path
from score import read,write,rows,summary

def finish(root,parent,base,audit):
 s=summary(root,parent,base);tasks,rr=rows(root,parent,base)
 v=read(root/'BLIND_VERDICTS.json');lab={(m['arm'],m['owner'],m['query']):v[m['blind_id']]['label'] for m in read(root/'BLIND_MAPPING.json')}
 historical=read(audit/'PAIRED_FULL_PRIVATE.json');hist={(x['owner'],x['query']):x for x in historical}
 assert len(hist)==219
 pair={};oldtable=collections.defaultdict(collections.Counter);per={}
 for t in tasks:
  per[t['order']]=collections.defaultdict(collections.Counter)
  for q in t['eval_rows']:
   h=hist[t['order'],q['query_id']];assert h['question']==q['question'] and h['reference']==q['reference']
   for a,old in [('A_NATIVE','TT_NATIVE'),('B_VIEW_CE','TT_VIEW'),('R_V69','TT_SUPERVISED')]:assert lab[a,t['order'],q['query_id']]==h['outputs'][old]['label']
   for role in q['roles']:
    label=h['outputs']['LORA_NATIVE']['label'];oldtable[role][label]+=1;per[t['order']][role][label]+=1
 oldtable={role:dict(total=sum(c.values()),correct=c['CORRECT'],labels=dict(c)) for role,c in oldtable.items()};s['table']['LORA_NATIVE']=oldtable
 s['per_edit']['LORA_NATIVE']={o:{r:dict(total=sum(c.values()),correct=c['CORRECT'],labels=dict(c)) for r,c in d.items()} for o,d in per.items()}
 for a in ['R_V69','L_VIEW_LORA']:
  pair[a]={}
  for role in oldtable:
   old={(h['owner'],h['query']) for h in historical if role in h['roles'] and h['outputs']['LORA_NATIVE']['label']=='CORRECT'}
   new={(t['order'],q['query_id']) for t in tasks for q in t['eval_rows'] if role in q['roles'] and lab[a,t['order'],q['query_id']]=='CORRECT'}
   pair[a][role]=dict(lost=len(old-new),gained=len(new-old))
 s['versus_native_lora']=pair
 s['unsupported']['LORA_NATIVE']=sum(bool(h['outputs']['LORA_NATIVE']['scope']['unsupported_source_assertion']) for h in historical if set(h['roles']) & {'T1L','T2L'})
 s['capped']['LORA_NATIVE']=sum(read(base/'1'/t['key']/'FIXED160/outputs'/(q['query_id']+'.json'))['at_cap'] for t in tasks for q in t['eval_rows'])
 # Public identities are within-experiment ordinal numbers, never raw patient/query IDs.
 paired=[]
 for t in tasks:
  for idx,q in enumerate(t['eval_rows'],1):
   h=hist[t['order'],q['query_id']];labels={a:lab[a,t['order'],q['query_id']] for a in ['A_NATIVE','B_VIEW_CE','R_V69','L_VIEW_LORA']};labels['LORA_NATIVE']=h['outputs']['LORA_NATIVE']['label'];labels['BASE']=h['base']['verdict']['label']
   paired.append(dict(owner=t['order'],consumer=idx,roles=q['roles'],labels=labels,capped={x['arm']:x['output']['at_cap'] for x in rr if x['owner']==t['order'] and x['query']==q['query_id']}))
 s['paired_anonymous']=paired
 s['R_original_gate']=read(root.parent.parent/'output_init_v69_20261023/completed/SUMMARY.json')['gate']
 diag={};diagnostic_rows=[]
 for a in ['BASE','R_V69','L_VIEW_LORA']:
  records=[]
  for t in tasks:
   done=read(root/'diagnostic'/a/t['key']/'COMPLETE.json');assert done['status']=='PASS' and len(done['records'])==12
   records+=done['records']
  assert len(records)==120 and all(r['teacher_KV_pass'] for r in records)
  originals={(r['owner'],r['query']):r for r in records if r['condition']=='ORIGINAL'};assert len(originals)==40
  diag[a]={}
  for group in ['ALL','DIRECT_IMAGE_FINDING','CASE_CONDITIONED_RELATION']:
   grp=[r for r in records if group=='ALL' or (r['owner']%2==1)==(group=='DIRECT_IMAGE_FINDING')];diag[a][group]={}
   for condition in ['ORIGINAL','ZERO_VISUAL_EMBEDDINGS','FIXED_SOURCE_IMAGE_PERMUTATION']:
    z=[r for r in grp if r['condition']==condition];orig=[originals[r['owner'],r['query']] for r in z]
    diag[a][group][condition]=dict(n=len(z),source_answer_exact=sum(r['source_answer_exact'] for r in z),changed_tokens=sum(r['raw_token_ids']!=o['raw_token_ids'] for r,o in zip(z,orig)),changed_first_token=sum(r['raw_token_ids'][:1]!=o['raw_token_ids'][:1] for r,o in zip(z,orig)),caps=sum(r['at_cap'] for r in z),mean_nll=statistics.mean(r['mean_target_nll'] for r in z),mean_delta_nll=statistics.mean(r['mean_target_nll']-o['mean_target_nll'] for r,o in zip(z,orig)),source_exact_retained=sum(r['source_answer_exact'] and o['source_answer_exact'] for r,o in zip(z,orig)),source_exact_original=sum(o['source_answer_exact'] for o in orig))
  for t in tasks:
   qq=[q for q in t['eval_rows'] if 'T2G' in q['roles']]
   for i,q in enumerate(qq,1):
    o=originals[t['order'],q['query_id']]
    for r in records:
     if (r['owner'],r['query'])!=(t['order'],q['query_id']):continue
     diagnostic_rows.append(dict(arm=a,owner=t['order'],within_owner_T2G=i,condition=r['condition'],mean_target_nll=r['mean_target_nll'],sum_target_nll=r['sum_target_nll'],target_length=r['target_length'],source_answer_exact=r['source_answer_exact'],at_cap=r['at_cap'],changed_tokens=r['raw_token_ids']!=o['raw_token_ids'],changed_first_token=r['raw_token_ids'][:1]!=o['raw_token_ids'][:1]))
 s['visual_diagnostic']=diag;s['visual_rows_anonymous']=diagnostic_rows
 assert len(diagnostic_rows)==360
 s['training_cache']={a:{str(t['order']):{k:read(root/a/t['key']/'TRAIN.json')[k] for k in ['cache_hits','cache_misses','train_seconds']} for t in tasks} for a in ['R_V69','L_VIEW_LORA']}
 assert all(z['cache_hits']==317 and z['cache_misses']==3 for d in s['training_cache'].values() for z in d.values())
 s['status']='FINITE_PLAN_COMPLETE_NO_JOINT_PROMOTION';s['new_algorithm_calls_after_completion']=0
 s['limitations']=['Exposed DEV and one fixed seed; no independent confirmation or population claim.','Assistant full-text blind scoring, not independent clinical annotation.','Zero visual embeddings are OOD; fixed donor is one permutation. Original targets are not clinical truth for altered images.','Source exact is a strict string measure, not semantic correctness.','LoRA uses18432 trainable parameters vs TT7168; matched information/updates, not equal capacity.','Historical locality scope ambiguity preserved; T1L Base correct0 and T2L Base correct1.']
 w=read(root.parent/'WINDOW.json')
 s['resource_accounting']=dict(original_start=w['start'],hard_end=w['hard_end'],new_GPU_process_seconds=s['GPU_process_seconds'],inherited_V63_V78_seconds=w['inherited_V63_V78_GPU_process_seconds'],combined_seconds=w['inherited_V63_V78_GPU_process_seconds']+s['GPU_process_seconds'],remote_budget_bytes=4000000000,local_budget_bytes=1000000000,last_guard_owned_bytes=read(root/'RESOURCE_GUARD_HEALTH.json')['owned_bytes'])
 if (root/'RETIREMENT.json').exists() and (root/'LOCAL_RETIREMENT.json').exists():
  remote=read(root/'RETIREMENT.json');local=read(root/'LOCAL_RETIREMENT.json');assert remote['status']==local['status']=='RETIRED' and remote['bytes']==local['bytes']
  s['lifecycle']=dict(remote_states_retired=len(remote['files']),local_copies_retired=len(local['files']),bytes_each_host=remote['bytes'],no_registered_successor=True,untrained_initializers_and_reproducibility_evidence_retained=True,weights_reconstruction_requires_training=True)
 write(root/'RESULTS.json',s);return s
if __name__=='__main__':
 s=finish(Path('.private/method_decision_v79_20261023/completed'),Path('.private/source_view_v64_20261022/completed'),Path('.private/source_stop_v62_20261022/completed'),Path('.private/method_decision_v79_20261023/audit'))
 print(json.dumps({k:s[k] for k in ['versus_native_lora','R_original_gate','visual_diagnostic']},indent=2))
