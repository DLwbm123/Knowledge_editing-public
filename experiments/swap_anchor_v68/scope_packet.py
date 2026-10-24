import json,sys,random
from pathlib import Path
sys.path.insert(0,'experiments/swap_anchor_v68');import score
r=Path('.private/swap_anchor_v68_20261023/completed');p=Path('.private/source_view_v64_20261022/completed');b=Path('.private/source_stop_v62_20261022/completed');oldroot=Path('.private/gradient_guard_v67_20261023/completed')
tasks=score.read(r/'DATA_ADMISSION.json')['tasks']
# Reconstruct V65 old output bindings using its arm names.
old={};tmap={t['order']:t for t in tasks}
for m in score.read(oldroot/'SCOPE_MAPPING.json'):
 t=tmap[m['owner']];folder='B_VIEW_CE' if m['arm']=='E_GRADIENT' else m['arm'];origin=oldroot if m['arm']=='E_GRADIENT' else p
 o=score.read(origin/'0'/t['key']/folder/'outputs'/(m['query']+'.json'));k=(t['source']['reference'],o['binding']['question'],o['binding']['reference'],o['raw_answer']);val=score.read(oldroot/'SCOPE_VERDICTS.json')[m['id']]
 if k in old:assert old[k]==val
 old[k]=val
_,rr=score.rows(r,p,b);selected=[]
for x in rr:
 if x['arm']=='BASE':continue
 t=tmap[x['owner']];q=next(q for q in t['eval_rows'] if q['query_id']==x['query'])
 if not set(q['roles']) & {'T1L','T2L'}:continue
 o=x['output'];k=(t['source']['reference'],o['binding']['question'],o['binding']['reference'],o['raw_answer']);selected.append((x,k))
keys=sorted({k for _,k in selected});random.Random(680023).shuffle(keys);ids={k:f'S{i:03}' for i,k in enumerate(keys)};vv={ids[k]:old[k] for k in keys if k in old};review=[dict(zip(['source_reference','question','reference','answer'],k),id=ids[k]) for k in keys if k not in old]
score.write(r/'SCOPE_REUSED.json',vv);score.write(r/'SCOPE_REVIEW.json',review);score.write(r/'SCOPE_MAPPING.json',[dict(arm=x['arm'],owner=x['owner'],query=x['query'],id=ids[k]) for x,k in selected]);print('reuse',len(vv),'new',len(review),'consumers',len(selected))
