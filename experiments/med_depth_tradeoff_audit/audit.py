import json, math, os, statistics
from collections import Counter
from pathlib import Path
def ranks(values):
    order=sorted(range(len(values)), key=values.__getitem__)
    result=[0.0]*len(values)
    i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]: j+=1
        for k in order[i:j]: result[k]=(i+j-1)/2
        i=j
    return result

def spearman(x,y):
    a,b=ranks(x),ranks(y); ma,mb=statistics.mean(a),statistics.mean(b)
    den=math.sqrt(sum((v-ma)**2 for v in a)*sum((v-mb)**2 for v in b))
    return sum((u-ma)*(v-mb) for u,v in zip(a,b))/den if den else None
p=Path(os.environ['FIXED_SCORE_ROOT']); w=Path(os.environ['FIXED_WORKTREE'])
packet=json.loads((p/'PACKET_PRIVATE.json').read_text())['consumers']
scores=json.loads((p/'DECISIONS.json').read_text())['decisions']
roles=json.loads((w/'.private/med_fixed_depth_v1_20261030/EVALUATION_ROLES.json').read_text())
resp=json.loads(Path(os.environ['DEPTH_RESPONSE_FILE']).read_text())
vals={(r['owner'],r['query_id'],r['arm']):scores[r['key']]['core_correct'] for r in packet}
if len(vals)!=1016 or len(roles)!=16 or len(resp)!=32: raise ValueError('coverage')
rmap={(r['owner'],r['arm']):r['response_norm'] for r in resp}
items=[]
for owner,group in roles.items():
 o=int(owner); d=[]
 for role in ('T2G','T2L'):
  qs=group[role]
  if not qs: raise ValueError('empty support')
  d.append(statistics.mean(int(vals[(o,q,'CDEEP')])-int(vals[(o,q,'C21')]) for q in qs))
 a,b=rmap[(o,'C21')],rmap[(o,'CDEEP')]
 if min(a,b)<=0 or not all(map(math.isfinite,(a,b,*d))): raise ValueError('nonfinite')
 items.append((d[0],d[1],math.log(b/a)))
def sign(x): return 'positive' if x>0 else 'negative' if x<0 else 'zero'
result={'editors':len(items),'t2g_net':dict(Counter(sign(a) for a,b,c in items)),'t2l_net':dict(Counter(sign(b) for a,b,c in items)), 'cross':dict(Counter(sign(a)+'_T2G__'+sign(b)+'_T2L' for a,b,c in items)),'correlations':{}}
for j,name in enumerate(('t2g','t2l')):
 x=[r[2] for r in items]; y=[r[j] for r in items]
 c=spearman(x,y) if len(set(x))>1 and len(set(y))>1 else None
 result['correlations'][name]={'n':len(x),'spearman':float(c) if c is not None and math.isfinite(c) else None}
result['limitations']=['post hoc descriptive association, not causation','response norm measured on source not T2L','DEV cohort, no patient independence or clinical truth','no new model or score calls']
output=w/'reports/med_depth_tradeoff_audit_20261031/RESULTS.json'
output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(result,ensure_ascii=False))
