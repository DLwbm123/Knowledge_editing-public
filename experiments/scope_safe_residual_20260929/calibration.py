"""Frozen global scope rejection grid; calibration rows must be data-policy vetted."""
from mechanisms import scope_gate,source_folds

def calibrate(rows,edits):
 roles=['native','same_image','cross_image'];counts={(e,role):sum(r['edit']==e and r['role']==role for r in rows) for e in edits for role in roles}
 if any(counts[e,'cross_image']<2 or counts[e,'same_image']<4 or not counts[e,'native'] for e in edits):return dict(status='UNSUPPORTED_CAL_SCOPE',selected='R0')
 if any(r['purpose']!='calibration' for r in rows):raise ValueError('Calibration cannot read fit/regression/holdout rows')
 folds=source_folds(rows)
 def measure(data,tau):
  out=[]
  for prefix in sorted({r['prefix'] for r in data}):
   subset=[r for r in data if r['prefix']==prefix];hit={}
   for role in roles+['negative']:
    rs=[r for r in subset if r['role']==role]
    if not rs:return None
    active=[r['r0_active'] if tau is None else scope_gate(r['feature'],r['candidate'],r['r0_active'],r['positive'],r['negative'],tau)['active'] for r in rs]
    hit[role]=sum(a and (r['candidate'] in r['legal_experts'] if role!='negative' else True) for a,r in zip(active,rs))/len(rs)
   out.append(hit)
  return out
 def select(data):
  base=measure(data,None)
  if not base:return None
  choices=[]
  for tau in [-.2,0,.1,.2,.3]:
   result=measure(data,tau)
   legal=all(a['native']>=b['native'] and all(a[k]>=b[k]-.01-1e-12 for k in roles[1:]) for a,b in zip(result,base))
   if legal:choices.append((sum(x['negative'] for x in result)/len(result),tau))
  base_neg=sum(x['negative'] for x in base)/len(base)
  if not choices or min(choices)[0]>=base_neg:return None
  return min(choices)[1]
 checks=[]
 for fold in range(3):
  train=[r for r,f in zip(rows,folds) if f!=fold];test=[r for r,f in zip(rows,folds) if f==fold];tau=select(train);checks.append(dict(fold=fold,train_sources=len({r['source'] for r in train}),validation_sources=len({r['source'] for r in test}),tau=tau,baseline=measure(test,None),validation=measure(test,tau)))
 tau=select(rows)
 return dict(status='NO_ROUTING_GAIN' if tau is None else 'CALIBRATED_PENDING_CROSS_FIT_REVIEW',selected='R0' if tau is None else 'Rneg',tau=tau,cross_fit=checks,not_confirmatory=True)
