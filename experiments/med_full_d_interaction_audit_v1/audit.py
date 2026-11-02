import collections,json,sys
from pathlib import Path
w=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(w/'experiments/med_fixed_depth_v1'),str(w/'experiments/med150_source_review_v1'),str(w/'experiments/med_proposition_generalization_v1'),str(w/'experiments/med_source_teacher_v1')]
import stream_extract as s,score_index,prior_binding,ledger

def rows(path,field,allowed):
 out=[]
 with path.open('rb') as f:
  if s.nonspace(f)!=b'{':raise ValueError('expected object')
  while True:
   t=s.nonspace(f)
   if t==b'}':break
   name=s.string(f,True)
   if s.nonspace(f)!=b':':raise ValueError('colon')
   t=s.nonspace(f)
   if name==field:
    if t!=b'[':raise ValueError('array')
    while True:
     t=s.nonspace(f)
     if t==b']':break
     out.append(prior_binding.select(f,t,allowed))
     t=s.nonspace(f)
     if t==b']':break
     if t!=b',':raise ValueError('delimiter')
   else:score_index.skip(f,t)
   t=s.nonspace(f)
   if t==b'}':break
   if t!=b',':raise ValueError('object delimiter')
 return out

def judgments(path,keys):
 return ledger.selected_decisions(path,keys)

nine=Path('/private/tmp/.r.20261101.09');full=Path('/private/tmp/.r.20261102.02')
fields={'owner':True,'query_id':True,'arm':True,'key':True}
a=[x for x in rows(nine/'PACKET_PRIVATE.json','consumers',fields) if x.get('arm')=='D']
b=rows(full/'PACKET_PRIVATE.json','consumers',fields)
if len(a)!=371 or len(b)!=371:raise ValueError('consumer count')
A={(x['owner'],x['query_id']):x['key'] for x in a};B={(x['owner'],x['query_id']):x['key'] for x in b}
if len(A)!=371 or set(A)!=set(B):raise ValueError('paired identities')
owners={x[0] for x in A}
if len(owners)!=9:raise ValueError('owners')
da=judgments(nine/'DECISIONS.json',set(A.values()));db=judgments(full/'DECISIONS.json',set(B.values()))
private=w/'.private/med_fixed_depth_v1_20261030'
base=rows(private/'BASE_M0_SCORE_BINDINGS.json','rows',{'owner':True,'query_id':True,'MED':{'core_correct':True,'full_supported':True}})
med={(x['owner'],x['query_id']):x['MED'] for x in base if x['owner'] in owners}
if set(med)!=set(A):raise ValueError('baseline binding')
roles=json.loads((private/'EVALUATION_ROLES.json').read_text());out={}
for role in ('T0','FIT','T1G','T2G','T1L','T2L'):
 selected=[(int(o),q) for o,g in roles.items() if int(o) in owners for q in g[role]]
 if any(x not in A for x in selected):raise ValueError('role binding')
 out[role]={}
 for field in ('core_correct','full_supported'):
  combo=collections.Counter();trans=[collections.Counter() for _ in range(3)];edit=[set() for _ in range(6)];unknown=0
  for e in selected:
   vals=[med[e].get(field),da.get(A[e],{}).get(field),db.get(B[e],{}).get(field)]
   if any(v is not True and v is not False for v in vals):unknown+=1;continue
   combo[''.join('1' if v else '0' for v in vals)]+=1
   for i,(x,y) in enumerate(((vals[0],vals[1]),(vals[0],vals[2]),(vals[1],vals[2]))):
    if not x and y:trans[i]['gain']+=1;edit[2*i].add(e[0])
    if x and not y:trans[i]['loss']+=1;edit[2*i+1].add(e[0])
  out[role][field]={'events':len(selected),'unknown':unknown,'patterns_MED_D_FULLMEDD':dict(sorted(combo.items())),
   'transitions':{k:{'gains':trans[i]['gain'],'losses':trans[i]['loss'],'gain_editors':len(edit[2*i]),'loss_editors':len(edit[2*i+1])} for i,k in enumerate(('MED_to_D','MED_to_FULLMEDD','D_to_FULLMEDD'))}}
result={'status':'COMPLETE','scope':'posthoc exposed nine-editor DEV; association not causation','events_total':371,'editors':9,'model_calls':0,'new_judge':0,'panels':out}
print(json.dumps(result,ensure_ascii=False,indent=2))
