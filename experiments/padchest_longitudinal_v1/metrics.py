"""Fixed-denominator CORE bounds; source/probe dependence remains explicit."""
from collections import defaultdict

def bounds(v):
 return dict(n=len(v),correct=sum(x is True for x in v),wrong=sum(x is False for x in v),unknown=sum(x is None for x in v),
  lower_pct=100*sum(x is True for x in v)/len(v) if v else None,upper_pct=100*sum(x is not False for x in v)/len(v) if v else None)
def conjunction(v):return False if any(x is False for x in v) else None if any(x is None for x in v) else True

def summarize(values,core,task_inputs,base_core):
 plan={(r['edit_id'],r['query']['query_id']):r['query'] for r in task_inputs['plan']}
 out={}
 for mode in ['single','bank']:
  rows=[v for v in values if v['mode']==mode];roles=defaultdict(list);classes=defaultdict(list);joint=defaultdict(list);retention=[];pre=[];post=[];activation=defaultdict(int)
  for v in rows:
   q=plan[(v['edit_id'],v['query_id'])];answer=core[v['consumer_id']];roles[v['role']].append(answer)
   activation[v['role']]+=v.get('route',{}).get('activated') is True
   if v['role']=='T0':joint[v['edit_id']].append(answer)
   if v['role'] in ['SAME_STATE','CONTRAST']:
    classes[q['progression']].append(answer);joint[v['edit_id']].append(answer)
    before=base_core[q['query_id']];pre.append(before);post.append(answer)
    if before is True:retention.append(answer)
  byclass={k:bounds(classes[k]) for k in ['improving','stable','worsening']}
  macro={name:sum(v[name] for v in byclass.values())/3 if all(v['n'] for v in byclass.values()) else None for name in ['lower_pct','upper_pct']}
  assert all(len(v)==4 for v in joint.values()),'Joint source plus three probes must retain four events per edit'
  out[mode]=dict(roles={k:bounds(v) for k,v in roles.items()},probe_classes=byclass,probe_balanced_accuracy_pct=macro,
   joint_native_and_three_probes=bounds([conjunction(v) for v in joint.values()]),pre_probe_accuracy=bounds(pre),post_probe_accuracy=bounds(post),
   pre_correct_probe_retention=bounds(retention),route_ON=dict(activation),dependent_repeated_probes=True)
 return out
