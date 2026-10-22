"""Frozen three-arm source-local evaluation with blinded verdicts."""
import json,random,statistics
from collections import Counter
from pathlib import Path
ARMS=('BASE','MATCHED_CE','INCREMENT_FLOOR')
INSTRUCTION='Answer the question briefly with only the requested factual finding. Do not add explanations or information not requested.'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def signature(r):return (r['context'],r['q']['query_id'],r['answer'],tuple(r['tokens']))
def rows(root):
 rr=[]
 for i,t in enumerate(read(root/'DATA_ADMISSION.json')['tasks']):
  for c in ('ORIGINAL','SHORT'):
   for a in ARMS:
    for q in t['eval_rows']:
     v=read(root/str(i%2)/t['key']/'outputs'/c/a/(q['query_id']+'.json'))
     expected=q['question'] if c=='ORIGINAL' else q['question']+'\n'+INSTRUCTION
     assert v['binding']['question']==expected and v['binding']['reference']==q['reference'] and v['binding']['image_path']==q['image_path']
     rr.append(dict(case=t['key'],context=c,arm=a,q=q,answer=v['raw_answer'],tokens=v['raw_token_ids'],eos=v['EOS'],cap=v['at_cap']))
 return rr

def validate(root):
 for part in ('0','1'):
  x=read(root/part/'EXIT.json');assert x['code']==0 and x['counts']['updates']==160 and x['counts']['backwards']==2632 and x['counts']['admission_backwards']==72
 for i,t in enumerate(read(root/'DATA_ADMISSION.json')['tasks']):
  f=root/str(i%2)/t['key'];assert read(f/'REAL_SMOKE.json')['status']=='PASS'
  for a in ARMS[1:]:
   x=read(f/a/'COMPLETE.json');assert x['paired_initial_exact'] and x['save_load_exact'] and x['counts']['updates']==160 and x['counts']['backwards']==(1920 if a=='MATCHED_CE' else 2560)
 rr=rows(root);assert len(rr)==96 and len({(r['context'],r['arm'],r['q']['query_id']) for r in rr})==96
 return dict(status='PASS',bindings=96,source_clusters=2,independent_confirmation=False)

def blind(root):
 validate(root);rr=rows(root);known={}
 for old in read(root/'PRIOR_LABELS.json'):
  key=signature(old)
  if key in known:assert known[key]==old['label']
  known[key]=old['label']
 reuse=[dict(index=i,label=known[signature(r)]) for i,r in enumerate(rr) if signature(r) in known]
 unique={signature(r):r for r in rr if signature(r) not in known};keys=list(unique);random.Random(540022).shuffle(keys)
 packet=[];mapping={}
 for i,key in enumerate(keys):
  r=unique[key];code=f'Q{i+1:03d}';packet.append(dict(code=code,question=r['q']['question'],reference=r['q']['reference'],answer=r['answer']));mapping[code]=[j for j,x in enumerate(rr) if signature(x)==key]
 assert not (root/'BLIND_PACKET.json').exists()
 write(root/'BLIND_PACKET.json',packet);write(root/'BLIND_MAPPING.json',mapping);write(root/'EXACT_LABEL_REUSE.json',reuse)
 return dict(unique_outputs=len(packet),total_bindings=len(rr),reused_bindings=len(reuse))

def summarize(root):
 validate(root);assert (root/'VERDICTS_FROZEN.json').exists();rr=rows(root)
 for item in read(root/'EXACT_LABEL_REUSE.json'):rr[item['index']]['label']=item['label']
 mapping=read(root/'BLIND_MAPPING.json');verdicts=read(root/'BLIND_VERDICTS.json');assert set(mapping)==set(verdicts)
 for code,ids in mapping.items():
  v=verdicts[code];assert v['label'] in ['CORRECT','INCORRECT','UNCERTAIN','ABSTAIN'] and v['reason']
  for i in ids:rr[i]['label']=v['label']
 idx={(r['context'],r['arm'],r['q']['query_id']):r for r in rr}
 def correct(c,a,q):return idx[c,a,q]['label']=='CORRECT'
 out=dict(status='COMPLETE',independent_confirmation=False,clinical_expert_review=False,contexts={},common_correct={},paired={},development_recommendation=False)
 safety=[]
 for c in ('ORIGINAL','SHORT'):
  base=[r for r in rr if r['context']==c and r['arm']=='BASE'];groups={}
  for group in sorted({r['q']['eval_group'] for r in base}):
   chosen=[r for r in base if r['q']['eval_group']==group];role_metrics={}
   for role in sorted({r['q']['relation_role'] for r in chosen}):
    ids=[r['q']['query_id'] for r in chosen if r['q']['relation_role']==role];metrics={}
    for a in ARMS:
     metrics[a]=dict(correct=sum(correct(c,a,q) for q in ids),labels=dict(Counter(idx[c,a,q]['label'] for q in ids)),base_C_support=sum(correct(c,'BASE',q) for q in ids),base_C_damage=sum(correct(c,'BASE',q) and not correct(c,a,q) for q in ids),repairs_vs_CE=sum(not correct(c,'MATCHED_CE',q) and correct(c,a,q) for q in ids),losses_vs_CE=sum(correct(c,'MATCHED_CE',q) and not correct(c,a,q) for q in ids))
    role_metrics[role]=dict(n=len(ids),arms=metrics)
    if role in ['SOURCE','PROTECTION']:
     safety.append(metrics['INCREMENT_FLOOR']['losses_vs_CE']==0)
     safety.append(metrics['INCREMENT_FLOOR']['base_C_damage']<=metrics['MATCHED_CE']['base_C_damage'])
   groups[group]=role_metrics
  out['contexts'][c]=dict(groups=groups,stability={a:dict(eos=sum(idx[c,a,r['q']['query_id']]['eos'] for r in base),cap=sum(idx[c,a,r['q']['query_id']]['cap'] for r in base),mean_tokens=statistics.mean(len(idx[c,a,r['q']['query_id']]['tokens']) for r in base)) for a in ARMS})
 tasks=read(root/'DATA_ADMISSION.json')['tasks'];repaired=[]
 for task in tasks:
  pairs={}
  for pair in ['T','H']:
   ids=[q['query_id'] for q in task['eval_rows'] if q['pair_id']==pair and q['relation_role'] in ['POSITIVE','NEGATIVE']];assert len(ids)==2
   pairs[pair]={a:{c:all(correct(c,a,q) for q in ids) for c in ['ORIGINAL','SHORT']} for a in ARMS}
  common={a:all(pairs['H'][a].values()) for a in ARMS}
  repaired.append(common['INCREMENT_FLOOR'] and not common['MATCHED_CE'])
  out['paired'][task['key']]=dict(train_vs_paraphrase=pairs,paraphrase_common=common,unseen_complete_pair=False)
 for group in ['TRAIN_FIT','SAME_PIXEL_PARAPHRASE','UNSEEN_PANEL']:
  ids=[q['query_id'] for t in tasks for q in t['eval_rows'] if q['eval_group']==group]
  out['common_correct'][group]=dict(n=len(ids),arms={a:sum(correct('ORIGINAL',a,q) and correct('SHORT',a,q) for q in ids) for a in ARMS})
 out['repaired_source_clusters_vs_CE']=sum(repaired);out['source_protection_safety']=all(safety)
 out['development_recommendation']=sum(repaired)>=2 and all(safety)
 out['decision']='SOURCE_LOCAL_INCREMENT_CANDIDATE' if out['development_recommendation'] else 'NO_REGISTERED_DEVELOPMENT_PROMOTION'
 write(root/'LABELED_PRIVATE.json',rr);write(root/'RESULTS.json',out);return out
if __name__=='__main__':
 import sys
 op,p=sys.argv[1:];print(json.dumps(globals()[op](Path(p)),indent=2))
