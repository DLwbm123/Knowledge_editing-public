"""Exact output reuse, blinded novel verdicts, and paired development readouts."""
import json,random,statistics
from collections import Counter
from pathlib import Path
NATURAL=('BASE','W0','BE_RADIUS_TT','LDKE_ROUTER_TT_PROXY','B_GATE','B_COND')
ROUTED=NATURAL[2:]
ARMS=NATURAL+tuple(a+'_FORCED' for a in ROUTED)+tuple(a+'_OFF' for a in ROUTED)
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def signature(r):return (r['context'],r['q']['query_id'],r['answer'],tuple(r['tokens']))
def rows(root):
 data=read(root/'DATA_ADMISSION.json');result=[]
 for i,t in enumerate(data['tasks']):
  folder=root/str(i%2)/t['key']
  for context in ('ORIGINAL','SHORT'):
   for arm in ARMS:
    for q in t['eval_rows']:
     d=read(folder/'outputs'/context/arm/(q['query_id']+'.json'))
     expected=q['question'] if context=='ORIGINAL' else q['question']+'\n'+read(root/'PROTOCOL_LOCK.json')['short_instruction']
     assert d['binding']['question']==expected and d['binding']['reference']==q['reference'] and d['binding']['image_path']==q['image_path']
     result.append(dict(case=t['key'],context=context,arm=arm,q=q,answer=d['raw_answer'],tokens=d['raw_token_ids'],eos=d['EOS'],cap=d['at_cap'],route=d['route'],execution=d['execution']))
 return result

def validate(root):
 for part in ('0','1'):
  assert read(root/part/'EXIT.json')['code']==0
  assert read(root/part/'MODEL_COMPLETE.json')['counts']['updates']==960
 rr=rows(root);assert len(rr)==868 and len({(r['context'],r['arm'],r['q']['query_id']) for r in rr})==868
 for i,t in enumerate(read(root/'DATA_ADMISSION.json')['tasks']):
  f=root/str(i%2)/t['key'];assert read(f/'REAL_SMOKE.json')['status']=='PASS'
  for arm in ('W0','B_GATE','B_COND'):
   d=read(f/arm/'COMPLETE.json');assert d['paired_initial_exact'] and d['save_load_exact'] and d['counts']['updates']==160
 return dict(status='PASS',bindings=len(rr),cases=4,independent_confirmation=False)

def blind(root):
 validate(root);rr=rows(root);prior=read(root/'PRIOR_LABELS.json');known={}
 for r in rr:
  for a,labels in prior.items():
   if (a.endswith('_SHORT'))!=(r['context']=='SHORT'):continue
   p=labels.get(r['q']['query_id'])
   if p and p['tokens']==r['tokens'] and p['answer']==r['answer']:
    k=signature(r)
    if k in known:assert known[k]==p['label'],'Prior label conflict; no silent reassignment'
    known[k]=p['label']
 unique={signature(r):r for r in rr if signature(r) not in known};keys=list(unique);random.Random(500022).shuffle(keys)
 packet=[];mapping={};reused=[]
 for i,r in enumerate(rr):
  if signature(r) in known:reused.append(dict(index=i,label=known[signature(r)]))
 for i,k in enumerate(keys):
  r=unique[k];code=f'Q{i+1:03d}';packet.append(dict(code=code,question=r['q']['question'],reference=r['q']['reference'],answer=r['answer']));mapping[code]=[j for j,x in enumerate(rr) if signature(x)==k]
 assert not (root/'BLIND_PACKET.json').exists()
 write(root/'BLIND_PACKET.json',packet);write(root/'BLIND_MAPPING.json',mapping);write(root/'EXACT_LABEL_REUSE.json',reused)
 return dict(unique_new_verdicts=len(packet),reused_bindings=len(reused),bindings=len(rr))

def summarize(root):
 validate(root);rr=rows(root);assert (root/'VERDICTS_FROZEN.json').exists()
 for x in read(root/'EXACT_LABEL_REUSE.json'):rr[x['index']]['label']=x['label']
 mapping=read(root/'BLIND_MAPPING.json');verdicts=read(root/'BLIND_VERDICTS.json');assert set(mapping)==set(verdicts)
 for code,indices in mapping.items():
  v=verdicts[code];assert v['label'] in ('CORRECT','INCORRECT','UNCERTAIN','ABSTAIN') and v['reason']
  for i in indices:rr[i]['label']=v['label']
 assert all('label' in r for r in rr)
 index={(r['context'],r['arm'],r['q']['query_id']):r for r in rr};out=dict(status='COMPLETE',independent_confirmation=False,clinical_expert_review=False,medical_grounding_confirmed=False,history_adoption_gate=False,contexts={},common_correct={},oracle={},pareto_candidates=[],scope_false_activation='UNKNOWN: no verified true-negative medical boundary set')
 def correct(c,a,q):return index[c,a,q]['label']=='CORRECT'
 for c in ('ORIGINAL','SHORT'):
  crows=[r for r in rr if r['context']==c];base=[r for r in crows if r['arm']=='BASE'];groups={}
  for group in sorted({r['q']['eval_group'] for r in base}):
   ids=[r['q']['query_id'] for r in base if r['q']['eval_group']==group]
   groups[group]=dict(n=len(ids),arms={})
   for a in ARMS:
    chosen=[index[c,a,q] for q in ids];support=sum(correct(c,'BASE',q) for q in ids)
    groups[group]['arms'][a]=dict(labels=dict(Counter(r['label'] for r in chosen)),base_C_support=support,base_C_damage=sum(correct(c,'BASE',q) and not correct(c,a,q) for q in ids),repairs_vs_W0=sum(not correct(c,'W0',q) and correct(c,a,q) for q in ids),losses_vs_W0=sum(correct(c,'W0',q) and not correct(c,a,q) for q in ids),activation=sum(int(r['route']['gate']) for r in chosen if r['route'] is not None),activation_denominator=sum(r['route'] is not None for r in chosen))
  ids=[r['q']['query_id'] for r in base if r['q']['eval_group']=='HELDOUT']
  out['oracle'][c]=dict(n=len(ids),base_or_W0_correct=sum(correct(c,'BASE',q) or correct(c,'W0',q) for q in ids),both_wrong=sum(not correct(c,'BASE',q) and not correct(c,'W0',q) for q in ids),purpose='posthoc selector upper bound only; labels never route')
  out['contexts'][c]=dict(groups=groups,stability={a:dict(eos=sum(index[c,a,r['q']['query_id']]['eos'] for r in base),cap=sum(index[c,a,r['q']['query_id']]['cap'] for r in base),empty=sum(not index[c,a,r['q']['query_id']]['answer'].strip() for r in base),mean_tokens=statistics.mean(len(index[c,a,r['q']['query_id']]['tokens']) for r in base)) for a in ARMS})
 ids=[r['q']['query_id'] for r in rr if r['arm']=='BASE' and r['context']=='ORIGINAL' and r['q']['eval_group']=='HELDOUT']
 for a in NATURAL:out['common_correct'][a]=sum(correct('ORIGINAL',a,q) and correct('SHORT',a,q) for q in ids)
 for a in ROUTED:
  conditions=[];strict=[]
  for c in ('ORIGINAL','SHORT'):
   for group in ('SOURCE','HELDOUT','LOCALITY_CLINICAL'):
    x=out['contexts'][c]['groups'][group]['arms'];wc=x['W0']['labels'].get('CORRECT',0);ac=x[a]['labels'].get('CORRECT',0)
    conditions.extend([ac>=wc,x[a]['base_C_damage']<=x['W0']['base_C_damage'],x[a]['losses_vs_W0']==0]);strict.extend([ac>wc,x[a]['base_C_damage']<x['W0']['base_C_damage']])
  if all(conditions) and any(strict):out['pareto_candidates'].append(a)
 out['B_gate_coverage_matched']=all(index[c,'B_GATE',q]['route']['gate']==index[c,'B_COND',q]['route']['gate'] for c in ('ORIGINAL','SHORT') for q in [r['q']['query_id'] for r in rr if r['context']=='ORIGINAL' and r['arm']=='BASE'])
 assert out['B_gate_coverage_matched']
 out['scope_recall']={c:{a:{g:dict(activated=out['contexts'][c]['groups'][g]['arms'][a]['activation'],n=out['contexts'][c]['groups'][g]['n']) for g in ('SOURCE','HELDOUT')} for a in ROUTED} for c in ('ORIGINAL','SHORT')}
 out['coverage_comparison_note']='B_GATE/B_COND use identical frozen gate per query; BE/LDKE coverage reported separately, no threshold fitting or same-coverage superiority claim without calibration support.'
 out['finalists']=out['pareto_candidates'][:2]
 out['finalist_rule']='predeclared strict no-new-W0-loss Pareto; at most two by natural-arm order; no clinical adoption'
 out['case_complete']={a:{c:sum(all(correct(c,a,q) for q in ids if index[c,a,q]['case']==t['key']) for t in read(root/'DATA_ADMISSION.json')['tasks']) for c in ('ORIGINAL','SHORT')} for a in NATURAL}
 write(root/'LABELED_PRIVATE.json',rr);write(root/'RESULTS.json',out);return out

if __name__=='__main__':
 import sys
 op,path=sys.argv[1:];print(json.dumps(globals()[op](Path(path)),ensure_ascii=False,indent=2))
