"""Read-only historical output audit; no new labels and no model calls."""
import json,collections
from pathlib import Path

def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def audit(private,out):
 roots={name:private/folder/'completed' for name,folder in [('v62','source_stop_v62_20261022'),('v64','source_view_v64_20261022'),('v69','output_init_v69_20261023'),('v75','worst_view_v75_20261023')]}
 lookup={};scopes={};packets={}
 for name,root in roots.items():
  v=read(root/'BLIND_VERDICTS.json');p={x['id']:x for x in read(root/'BLIND_PACKET.json')};packets[name]=p
  for m in read(root/'BLIND_MAPPING.json'):lookup[(name,m['arm'],m['owner'],m['query'])]=dict(p[m['blind_id']],verdict=v[m['blind_id']])
  if (root/'SCOPE_MAPPING.json').exists():
   sv=read(root/'SCOPE_VERDICTS.json')
   for m in read(root/'SCOPE_MAPPING.json'):scopes[(name,m['arm'],m['owner'],m['query'])]=sv[m['id']]
 arms=[('TT_NATIVE','v64','A_NATIVE'),('TT_VIEW','v64','B_VIEW_CE'),('TT_SUPERVISED','v69','G_OUTPUT_INIT'),('TT_WORST_VIEW','v75','M_WORST_VIEW'),('LORA_NATIVE','v62','LORA_FIXED160')]
 # Verify the original name rather than silently accepting another reference arm.
 available={k[1] for k in lookup if k[0]=='v62'}
 if 'LORA_FIXED160' not in available:
  assert 'NORMALIZED_LORA_R1_FIXED160' in available
  arms[-1]=('LORA_NATIVE','v62','NORMALIZED_LORA_R1_FIXED160')
 for row in read(private/'v62_followup_20261022/LOCALITY_REVIEW_VERDICTS.json'):
  for q in row['queries']:scopes[('v62',arms[-1][2],row['owner'],q)]=row['arms']['LORA160']
 tasks=read(roots['v69']/'DATA_ADMISSION.json')['tasks'];rows=[];lookup_correct=oracle_correct=0
 for t in tasks:
  for q in t['eval_rows']:
   owner=t['order'];qid=q['query_id'];base=lookup[('v69','BASE',None,qid)];outputs={}
   for display,root,arm in arms:
    x=lookup[(root,arm,owner,qid)];assert x['question']==q['question'] and x['reference']==q['reference']
    outputs[display]=dict(answer=x['answer'],label=x['verdict']['label'],scope=scopes.get((root,arm,owner,qid)))
   rows.append(dict(owner=owner,query=qid,image_path=q['image_path'],question=q['question'],reference=q['reference'],roles=q['roles'],base=base,outputs=outputs))
   if 'T2G' in q['roles']:
    lookup_correct+=q['reference']==t['source']['reference']
    oracle_correct+=base['verdict']['label']=='CORRECT' or outputs['TT_SUPERVISED']['label']=='CORRECT'
 assert len(rows)==219 and lookup_correct==40 and oracle_correct==37
 # Only the declared native source + two frozen paraphrases are training support.
 views=read(roots['v69']/'VIEWS_FROZEN.json')['views'];assert len(views)==20
 for t in tasks:
  vv=[v for v in views if v['owner']==t['order']];assert len(vv)==2 and all(v['target_reference']==t['source']['reference'] for v in vv)
 classified=read(private/'v62_followup_20261022/SCOPE_CLASSIFICATION.json');categories=dict(collections.Counter(x['category'] for x in classified))
 write(out/'PAIRED_FULL_PRIVATE.json',rows)
 metrics={a:{role:sum(role in row['roles'] and row['outputs'][a]['label']=='CORRECT' for row in rows) for role in ['T0','T1G','T2G','T1L','T2L','REPHRASE_DIAGNOSTIC']} for a,_,_ in arms}
 result=dict(status='PASS',consumers=219,arms=5,metrics=metrics,source_lookup_T2G_exact=lookup_correct,Base_or_V69_hard_selection_T2G_upper_bound=oracle_correct,locality_categories=categories,legal_source_training_bindings=30,explicit_inapplicable_source_support=0,gate_training='NOT_ADMITTED_NO_EXPLICIT_LEGAL_NEGATIVE_SUPPORT',historical_labels_unchanged=True,model_calls=0,limitation='No new clinical annotation; no inferred negatives from different organ, random image or missing reference. LoRA scope labels reused from original V62 followup; known historical scope ambiguity retained.')
 write(out/'OUTPUT_AUDIT.json',result);return result
if __name__=='__main__':print(json.dumps(audit(Path('.private'),Path('.private/method_decision_v79_20261023/audit')),ensure_ascii=False,indent=2))
