"""Score-only pairwise diagnostic, with frozen identity and role checks."""
import json,os,sys,collections,statistics
from pathlib import Path
w=Path(os.environ['WORKTREE']); out=Path(os.environ['PRIVATE_OUT'])
sys.path[:0]=[str(w/'experiments'/x) for x in ('med_common25_loss_ablation_v1','med_full_d150_v1','med150_source_review_v1','med_fixed_depth_v1','med_medinit_101_v1')]
from selected_records import identities
import ledger,summary,prior_binding,stream_extract
be=w/'.private/med150_balancedit_v1_20261028/scoring'
base=json.loads(Path('/private/tmp/.r.20261104.102prep/BASELINE.private.json').read_text())
first=json.loads(Path('/private/tmp/.r.20261104.102/DECISIONS.json').read_text())['decisions']
second=json.loads(Path('/private/tmp/.r.20261104.103/DECISIONS.json').read_text())['decisions']
if set(first)&set(second) or (set(first)|set(second))&set(base['decisions']): raise ValueError('judgment overlap')
judgments={**base['decisions'],**first,**second}
routing=base['routing']; assert routing['status']=='ROUTING_FROZEN' and len(routing['records'])==2148
roles=json.loads((w/'.private/med_full_d150_v1_20261102/EVALUATION_ROLES_PRIVATE.json').read_text())
rows=identities(be/'PACKET_PRIVATE.json','consumers',{'mode','prefix','query_id','key'})
bank=[r for r in rows if r.get('mode')=='bank' and r.get('prefix')==150]
assert len(bank)==2148 and len({r['query_id'] for r in bank})==2148
routes={r['query_id']:r for r in routing['records']};assert set(routes)=={r['query_id'] for r in bank}
decisions=ledger.selected_decisions(be/'DECISIONS.json',{r['key'] for r in bank});assert len(decisions)>0
cons=base['consumers']; assert len(cons)==2148*3
keys={(r['arm'],r['query_id']):r['key'] for r in cons};assert len(keys)==len(cons)
bekeys={r['query_id']:r['key'] for r in bank}
def conflicts_only(path):
 with Path(path).open('rb') as f:
  data=prior_binding.select(f,stream_extract.nonspace(f),{'categories'})
 return set(data['categories']['conflict_unknown'])
conf=conflicts_only('/private/tmp/.r.20261104.102/SUPPLEMENT_INPUT.private.json') | conflicts_only('/private/tmp/.r.20261104.103/SUPPLEMENT_INPUT.private.json')
result={}
for sensitivity in ('adjudicated','conflicts_unknown'):
 d=judgments if sensitivity=='adjudicated' else {k:v for k,v in judgments.items() if k not in conf}
 result[sensitivity]={}
 for g in ('all','original25','new76','med49'):
  result[sensitivity][g]={}
  for role in summary.ROLES:
   events=[(o,q) for o in range(1,151) for q in roles['roles'][str(o)][role] if q in routes and (g=='all' or routes[q]['group']==g)]
   result[sensitivity][g][role]={}
   for field in summary.FIELDS:
    values={a:[] for a in ('MED','MCE','BE')}; table=collections.Counter()
    for owner,q in events:
     m=d.get(keys['MED',q],{}).get(field);c=d.get(keys['MCE',q],{}).get(field);b=decisions.get(bekeys[q],{}).get(field)
     for a,v in (('MED',m),('MCE',c),('BE',b)):values[a].append((owner,v))
     table['unknown' if c is None or b is None else ('both_correct' if c and b else 'mce_only' if c else 'be_only' if b else 'both_wrong')]+=1
    result[sensitivity][g][role][field]={'arms':{a:summary.summarize(v) for a,v in values.items()},'mce_be_pair_counts':dict(table)}
published=json.loads((w/'reports/med_medinit_101_supplement_v1_20261104/RESULTS.json').read_text())
be_published=json.loads((w/'reports/med150_balancedit_v1_20261028/RESULTS.json').read_text())
for role in summary.ROLES:
 for field in summary.FIELDS:
  v=result['adjudicated']['all'][role][field]['arms']
  for a in ('MED','MCE'):
   if v[a]!=published['groups']['all'][role][field]['arms'][a]:raise ValueError('published method support mismatch')
  ref=be_published['panels']['bank'][field][role]['BALANCEDIT']
  if v['BE']['unknown'] or v['BE']['events']!=ref['n'] or abs(v['BE']['bounds'][0]-ref['macro_percent'])>1e-7:raise ValueError('published comparator support mismatch')
with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({'status':'COMPLETE','events':2148,'new_judge':0,'qa_decoded':0,'output':str(out.name)}))
