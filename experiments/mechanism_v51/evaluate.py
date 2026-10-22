"""Three-arm paired evaluation; exact previous verdict reuse and blinded new outputs."""
import json,random,statistics
from collections import Counter
from pathlib import Path
ARMS=('BASE','W0_FIXED21','W0_LOC')
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
  x=read(root/part/'EXIT.json');assert x['code']==0 and x['counts']['updates']==640 and x['counts']['backwards']==2568 and x['counts']['admission_backwards']==8
 for i,t in enumerate(read(root/'DATA_ADMISSION.json')['tasks']):
  f=root/str(i%2)/t['key'];assert read(f/'REAL_SMOKE.json')['status']=='PASS'
  for a in ARMS[1:]:
   x=read(f/a/'COMPLETE.json');assert x['paired_initial_exact'] and x['save_load_exact'] and x['counts']['updates']==160
   assert read(f/a/'CACHE_PARITY.json')['value_and_gradient_exact']
 rr=rows(root);assert len(rr)==186 and len({(r['context'],r['arm'],r['q']['query_id']) for r in rr})==186
 return dict(status='PASS',bindings=186,cases=4,independent_confirmation=False)

def blind(root):
 validate(root);rr=rows(root);prior=read(root/'PRIOR_LABELS.json');known={}
 for r in rr:
  for a,labels in prior.items():
   if a.endswith('_SHORT')!=(r['context']=='SHORT'):continue
   x=labels.get(r['q']['query_id'])
   if x and x['tokens']==r['tokens'] and x['answer']==r['answer']:
    key=signature(r)
    if key in known:assert known[key]==x['label']
    known[key]=x['label']
 unique={signature(r):r for r in rr if signature(r) not in known};keys=list(unique);random.Random(510022).shuffle(keys)
 packet=[];mapping={};reuse=[dict(index=i,label=known[signature(r)]) for i,r in enumerate(rr) if signature(r) in known]
 for i,key in enumerate(keys):
  r=unique[key];code=f'Q{i+1:03d}';packet.append(dict(code=code,question=r['q']['question'],reference=r['q']['reference'],answer=r['answer']));mapping[code]=[j for j,x in enumerate(rr) if signature(x)==key]
 assert not (root/'BLIND_PACKET.json').exists()
 write(root/'BLIND_PACKET.json',packet);write(root/'BLIND_MAPPING.json',mapping);write(root/'EXACT_LABEL_REUSE.json',reuse)
 return dict(new_unique=len(packet),reused_bindings=len(reuse),total_bindings=186)

def summarize(root):
 validate(root);assert (root/'VERDICTS_FROZEN.json').exists();rr=rows(root)
 for x in read(root/'EXACT_LABEL_REUSE.json'):rr[x['index']]['label']=x['label']
 mapping=read(root/'BLIND_MAPPING.json');verdicts=read(root/'BLIND_VERDICTS.json');assert set(mapping)==set(verdicts)
 for code,ids in mapping.items():
  x=verdicts[code];assert x['label'] in ['CORRECT','INCORRECT','UNCERTAIN','ABSTAIN'] and x['reason']
  for i in ids:rr[i]['label']=x['label']
 assert all('label' in r for r in rr)
 idx={(r['context'],r['arm'],r['q']['query_id']):r for r in rr}
 def correct(c,a,q):return idx[c,a,q]['label']=='CORRECT'
 out=dict(status='COMPLETE',independent_confirmation=False,clinical_expert_review=False,history_adoption_gate=False,contexts={},common_correct={},case_complete={},development_recommendation=False)
 conditions=[];strict=[];changes=[]
 for c in ('ORIGINAL','SHORT'):
  base=[r for r in rr if r['context']==c and r['arm']=='BASE'];groups={}
  for g in sorted({r['q']['eval_group'] for r in base}):
   ids=[r['q']['query_id'] for r in base if r['q']['eval_group']==g];arms={}
   for a in ARMS:
    chosen=[idx[c,a,q] for q in ids]
    arms[a]=dict(labels=dict(Counter(r['label'] for r in chosen)),base_C_support=sum(correct(c,'BASE',q) for q in ids),base_C_damage=sum(correct(c,'BASE',q) and not correct(c,a,q) for q in ids),repairs_vs_fixed=sum(not correct(c,'W0_FIXED21',q) and correct(c,a,q) for q in ids),losses_vs_fixed=sum(correct(c,'W0_FIXED21',q) and not correct(c,a,q) for q in ids))
   groups[g]=dict(n=len(ids),arms=arms)
   if g in ('SOURCE','HELDOUT','LOCALITY_CLINICAL'):
    a,b=arms['W0_FIXED21'],arms['W0_LOC'];conditions.extend([b['losses_vs_fixed']==0,b['base_C_damage']<=a['base_C_damage']]);strict.append(b['repairs_vs_fixed']>0 or b['base_C_damage']<a['base_C_damage'])
  out['contexts'][c]=dict(groups=groups,stability={a:dict(eos=sum(idx[c,a,r['q']['query_id']]['eos'] for r in base),cap=sum(idx[c,a,r['q']['query_id']]['cap'] for r in base),empty=sum(not idx[c,a,r['q']['query_id']]['answer'].strip() for r in base),mean_tokens=statistics.mean(len(idx[c,a,r['q']['query_id']]['tokens']) for r in base)) for a in ARMS})
  for r in base:
   q=r['q']['query_id'];a,b=idx[c,'W0_FIXED21',q],idx[c,'W0_LOC',q]
   if (a['label']=='CORRECT')!=(b['label']=='CORRECT'):changes.append(dict(context=c,query_id=q,group=r['q']['eval_group'],fixed_label=a['label'],selected_label=b['label'],fixed_answer=a['answer'],selected_answer=b['answer']))
 ids=[r['q']['query_id'] for r in rr if r['context']=='ORIGINAL' and r['arm']=='BASE' and r['q']['eval_group']=='HELDOUT']
 for a in ARMS:
  out['common_correct'][a]=sum(correct('ORIGINAL',a,q) and correct('SHORT',a,q) for q in ids)
  out['case_complete'][a]={c:sum(all(correct(c,a,q) for q in ids if idx[c,a,q]['case']==t['key']) for t in read(root/'DATA_ADMISSION.json')['tasks']) for c in ('ORIGINAL','SHORT')}
 out['development_recommendation']=all(conditions) and any(strict);out['decision']='SOURCE_SELECTED_LAYER_DEV_CANDIDATE' if out['development_recommendation'] else 'NO_PARETO_IMPROVEMENT'
 out['scope']='forced-on layer placement only; first-target-token selector; no routing/medical grounding claim'
 write(root/'LABELED_PRIVATE.json',rr);write(root/'PAIRED_CHANGES_PRIVATE.json',changes);write(root/'RESULTS.json',out);return out
if __name__=='__main__':
 import sys
 op,p=sys.argv[1:];print(json.dumps(globals()[op](Path(p)),indent=2))
