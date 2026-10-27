"""Frozen same-support centroid routers. No answer, article ID or heldout tuning."""
TAUS=(-.5,0.,.25,.5,.6,.7,.8,.85,.9,.95,.98,.99)
def unit(x):
 import torch
 x=x.double();n=x.norm();assert float(n)>0;return x/n

def scores(query,support,kind):
 import torch
 def cosine(mode):
  prototype=unit(torch.stack([unit(x[mode]) for x in support]).mean(0))
  return float(unit(query[mode])@prototype)
 assert support and kind in ('JOINT','CONTEXT_MIN')
 return cosine('joint') if kind=='JOINT' else min(cosine('image'),cosine('text'))

def choose(values,tau):
 assert values
 best=max(range(len(values)),key=lambda i:values[i])
 return best if values[best]>=tau else None

def calibrate(keys,tasks,rows,relations,kind):
 # Leave the query's complete PMC group out of support, while source remains fixed.
 from collections import defaultdict
 train=[x for x in rows if x['role']=='ROUTER_TRAIN'];trials=[]
 for tau in TAUS:
  groups=defaultdict(list)
  for q in train:
   labels=[relations[q['id'],t['key']] for t in tasks]
   if 'UNKNOWN' in labels:continue
   supports=[]
   for t in tasks:
    ids=[t['source']['id']]+[x['id'] for x in train if x['article']!=q['article'] and relations[x['id'],t['key']]=='YES']
    supports.append([keys[i] for i in ids])
   selected=choose([scores(keys[q['id']],s,kind) for s in supports],tau)
   yes={i for i,x in enumerate(labels) if x=='YES'}
   correct=selected in yes if yes else selected is None
   group='|'.join(str(i) for i in sorted(yes)) if yes else 'OFF'
   groups[group].append(float(correct))
  value=sum(sum(v)/len(v) for v in groups.values())/len(groups) if groups else None
  trials.append(dict(tau=tau,macro_legal_action_accuracy=value,groups={k:len(v) for k,v in groups.items()}))
 valid=[x for x in trials if x['macro_legal_action_accuracy'] is not None]
 if not valid:return dict(status='NA_NO_CALIBRATION_DENOMINATOR',trials=trials)
 best=max(valid,key=lambda x:(x['macro_legal_action_accuracy'],x['tau']))
 return dict(status='PASS',tau=best['tau'],trials=trials,selection='ROUTER_TRAIN_only_leave_source_group_out')

def validate_matrix(records,experts,queries):
 expected={(e,q) for e in experts for q in queries};keys=[(r['expert'],r['query']) for r in records]
 assert len(keys)==len(set(keys)) and set(keys)==expected,'Missing/duplicate/extra capability consumer'
 assert all(r.get('path') for r in records)
 return len(keys)

def selfcheck():
 import torch
 a=torch.tensor([1.,0.]);b=torch.tensor([0.,1.]);k={'joint':a,'image':a,'text':a};disagree={'joint':a,'image':a,'text':b}
 assert scores(k,[k],'JOINT')==1 and scores(k,[k],'CONTEXT_MIN')==1
 assert scores(disagree,[k],'CONTEXT_MIN')==0 and scores(disagree,[k],'JOINT')==1
 assert choose([.5,.5],.5)==0 and choose([.5,.5],.6) is None
 good=[dict(expert='e',query='q',path='p')];assert validate_matrix(good,['e'],['q'])==1
 for bad in [[],good+good]:
  try:validate_matrix(bad,['e'],['q'])
  except AssertionError:pass
  else:raise AssertionError('Missing/duplicate accepted')
 return dict(status='PASS',same_support=True,multi_modal_disagreement=True,OFF_and_tie=True,complete_matrix_required=True,no_model_calls=True)
